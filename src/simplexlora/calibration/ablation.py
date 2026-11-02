"""Layer-level causal ablation for calibrated Simplex LoRA importance."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Subset

from simplexlora.calibration.models import (
    LayerPlan,
    LayerRegistry,
    atomic_write_text,
    dump_json,
    utc_now_iso,
)
from simplexlora.peft_plugin import SimplexLoraLinear

ABLATION_JSON_FILENAME = "ablation_report.json"
ABLATION_MARKDOWN_FILENAME = "ablation_report.md"


@dataclass(frozen=True)
class LayerAblationRecord:
    """Observed loss increase when every adapter of one transformer layer is disabled."""

    layer_index: int
    module_names: tuple[str, ...]
    calibrated_importance: float
    rank_sum: int
    weight_gradient_norm: float
    baseline_loss: float
    ablated_loss: float
    delta_loss: float
    relative_delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "module_names": list(self.module_names),
            "calibrated_importance": self.calibrated_importance,
            "rank_sum": self.rank_sum,
            "weight_gradient_norm": self.weight_gradient_norm,
            "baseline_loss": self.baseline_loss,
            "ablated_loss": self.ablated_loss,
            "delta_loss": self.delta_loss,
            "relative_delta": self.relative_delta,
        }


@dataclass(frozen=True)
class LayerAblationResult:
    """Comparable layer-level ablation measurements for one calibration run."""

    run_id: str
    adapter_name: str
    sample_count: int
    max_samples: int
    batch_size: int
    baseline_loss: float
    layers: tuple[LayerAblationRecord, ...]
    importance_delta_spearman: float | None
    gradient_delta_spearman: float | None
    top_k: int
    top_k_overlap: int | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "adapter_name": self.adapter_name,
            "sample_count": self.sample_count,
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "baseline_loss": self.baseline_loss,
            "importance_delta_spearman": self.importance_delta_spearman,
            "gradient_delta_spearman": self.gradient_delta_spearman,
            "top_k": self.top_k,
            "top_k_overlap": self.top_k_overlap,
            "created_at": self.created_at,
            "layers": [record.to_dict() for record in self.layers],
        }


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    try:
        from scipy.stats import spearmanr
    except ImportError:  # pragma: no cover - scipy is a project dependency
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _top_k_names(records: list[LayerAblationRecord], key: Callable[[LayerAblationRecord], float], k: int) -> set[int]:
    ordered = sorted(records, key=lambda record: (-key(record), record.layer_index))
    return {record.layer_index for record in ordered[: min(k, len(ordered))]}


class LayerAblationEvaluator:
    """Evaluate calibrated importance with held-out layer-disable interventions.

    The evaluator is designed to run from the final projection observer, before
    ``_final_lora_rank_update`` replaces adapters. It restores every gate and
    model mode after each intervention.
    """

    def __init__(
        self,
        *,
        model: Any,
        dataset: Any,
        data_collator: Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]],
        registry: LayerRegistry,
        adapter_name: str,
        max_samples: int = 64,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        top_k: int = 5,
    ) -> None:
        if max_samples < 1:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if top_k < 1:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if len(dataset) == 0:
            raise ValueError("ablation dataset must not be empty")

        self.model = model
        self.dataset = dataset
        self.data_collator = data_collator
        self.registry = registry
        self.adapter_name = adapter_name
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.top_k = top_k
        self.result: LayerAblationResult | None = None

    def evaluate(self, plan: LayerPlan) -> LayerAblationResult:
        """Measure held-out loss after disabling each layer as a group."""
        if self.result is not None:
            return self.result

        sample_count = min(self.max_samples, len(self.dataset))
        loader = DataLoader(
            Subset(self.dataset, range(sample_count)),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
        )
        grouped = self._group_plan_modules(plan)
        if not grouped:
            raise ValueError("ablation plan contains no transformer layers with stable layer indices")

        was_training = bool(self.model.training)
        self.model.eval()
        try:
            baseline_loss = self._evaluate_loss(loader, sample_count)
            records: list[LayerAblationRecord] = []
            for layer_index, entries in grouped:
                modules = [
                    cast(SimplexLoraLinear, self.model.get_submodule(entry.peft_module_name)) for entry in entries
                ]
                saved = [
                    cast(torch.Tensor, module.lora_weight[self.adapter_name]).detach().clone() for module in modules
                ]
                try:
                    with torch.no_grad():
                        for module in modules:
                            cast(torch.Tensor, module.lora_weight[self.adapter_name]).zero_()
                    ablated_loss = self._evaluate_loss(loader, sample_count)
                finally:
                    with torch.no_grad():
                        for module, value in zip(modules, saved, strict=True):
                            cast(torch.Tensor, module.lora_weight[self.adapter_name]).copy_(value)

                delta_loss = ablated_loss - baseline_loss
                records.append(
                    LayerAblationRecord(
                        layer_index=layer_index,
                        module_names=tuple(entry.module_name for entry in entries),
                        calibrated_importance=math.fsum(entry.normalized_importance for entry in entries),
                        rank_sum=sum(entry.rank for entry in entries),
                        weight_gradient_norm=self._layer_weight_gradient_norm(modules),
                        baseline_loss=baseline_loss,
                        ablated_loss=ablated_loss,
                        delta_loss=delta_loss,
                        relative_delta=(delta_loss / baseline_loss if baseline_loss != 0.0 else None),
                    )
                )
        finally:
            self.model.train(was_training)

        importance_top = _top_k_names(records, lambda record: record.calibrated_importance, self.top_k)
        delta_top = _top_k_names(records, lambda record: record.delta_loss, self.top_k)
        importance_values = [record.calibrated_importance for record in records]
        delta_values = [record.delta_loss for record in records]
        meaningful_importance_count = sum(value > 1e-12 for value in importance_values)
        meaningful_delta_count = sum(value > 1e-12 for value in delta_values)
        top_k_overlap = (
            len(importance_top & delta_top)
            if meaningful_importance_count >= self.top_k and meaningful_delta_count >= self.top_k
            else None
        )
        result = LayerAblationResult(
            run_id=plan.run_id,
            adapter_name=self.adapter_name,
            sample_count=sample_count,
            max_samples=self.max_samples,
            batch_size=self.batch_size,
            baseline_loss=baseline_loss,
            layers=tuple(records),
            importance_delta_spearman=_spearman(importance_values, delta_values),
            gradient_delta_spearman=_spearman(
                [record.weight_gradient_norm for record in records],
                delta_values,
            ),
            top_k=self.top_k,
            top_k_overlap=top_k_overlap,
            created_at=utc_now_iso(),
        )
        self.result = result
        return result

    def _group_plan_modules(self, plan: LayerPlan) -> list[tuple[int, tuple[Any, ...]]]:
        grouped: dict[int, list[Any]] = {}
        for entry in plan.modules:
            if entry.layer_index >= 0:
                grouped.setdefault(entry.layer_index, []).append(entry)
        return [
            (layer_index, tuple(sorted(entries, key=lambda entry: entry.module_name)))
            for layer_index, entries in sorted(grouped.items())
        ]

    def _evaluate_loss(self, loader: DataLoader[Any], sample_count: int) -> float:
        total_loss = 0.0
        seen = 0
        with torch.no_grad():
            for raw_batch in loader:
                batch = {
                    name: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for name, value in raw_batch.items()
                }
                output = self.model(**batch)
                loss = getattr(output, "loss", None)
                if loss is None:
                    raise ValueError("ablation model forward did not return a loss")
                input_ids = batch.get("input_ids")
                if not isinstance(input_ids, torch.Tensor):
                    raise ValueError("ablation batches must contain a tensor named 'input_ids'")
                batch_size = int(input_ids.shape[0])
                total_loss += float(loss.detach()) * batch_size
                seen += batch_size
        if seen != sample_count:
            raise RuntimeError(f"ablation evaluated {seen} examples, expected {sample_count}")
        return total_loss / seen

    def _layer_weight_gradient_norm(self, modules: list[SimplexLoraLinear]) -> float:
        squared: list[torch.Tensor] = []
        for module in modules:
            grad = cast(torch.Tensor, module.lora_weight[self.adapter_name]).grad
            if grad is not None:
                squared.append(grad.detach().pow(2).sum())
        return float(torch.stack(squared).sum().sqrt().item()) if squared else 0.0


def render_ablation_markdown(result: LayerAblationResult) -> str:
    """Render a concise human-readable ablation report."""
    overlap = (
        "n/a" if result.top_k_overlap is None else f"{result.top_k_overlap}/{min(result.top_k, len(result.layers))}"
    )
    lines = [
        "# SimplexLoRA layer ablation",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Adapter: `{result.adapter_name}`",
        f"- Held-out examples: {result.sample_count}",
        f"- Baseline loss: {result.baseline_loss:.8f}",
        f"- Spearman importance vs delta loss: {result.importance_delta_spearman}",
        f"- Spearman weight-grad norm vs delta loss: {result.gradient_delta_spearman}",
        f"- Top-{result.top_k} overlap: {overlap}",
        "",
        "| Layer | Calibrated importance | Rank sum | Weight grad norm | Ablated loss | Delta loss | Relative delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in sorted(result.layers, key=lambda item: (-item.delta_loss, item.layer_index)):
        relative = "n/a" if record.relative_delta is None else f"{record.relative_delta:.6f}"
        lines.append(
            f"| {record.layer_index} | {record.calibrated_importance:.8f} | {record.rank_sum} | "
            f"{record.weight_gradient_norm:.8f} | {record.ablated_loss:.8f} | "
            f"{record.delta_loss:.8f} | {relative} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_ablation_artifacts(result: LayerAblationResult, output_dir: Path) -> tuple[Path, Path]:
    """Atomically write JSON and Markdown ablation artifacts."""
    output_dir = Path(output_dir)
    json_path = output_dir / ABLATION_JSON_FILENAME
    markdown_path = output_dir / ABLATION_MARKDOWN_FILENAME
    atomic_write_text(json_path, dump_json(result.to_dict()) + "\n", overwrite=True)
    atomic_write_text(markdown_path, render_ablation_markdown(result), overwrite=True)
    return json_path, markdown_path

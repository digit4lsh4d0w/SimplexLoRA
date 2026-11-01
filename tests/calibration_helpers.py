"""Shared helpers for calibration tests (tiny CPU models, registries, events)."""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

from simplexlora.calibration.models import CalibrationRunMetadata, ProjectionEvent, utc_now_iso
from simplexlora.calibration.registry import build_layer_registry
from simplexlora.peft_plugin import SimplexLoraLinear, get_simplex_peft_model, make_simplex_lora_config


class TinyLayeredModel(nn.Module):
    """Two transformer-like blocks with q/v projections (stable layer indices)."""

    def __init__(self) -> None:
        super().__init__()
        # `Any` keeps type checkers away from nn.Module's dynamic attribute magic.
        self.model: Any = _TinyBody()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    forward: Callable[..., Any] = _forward


class _TinyBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict({"q_proj": nn.Linear(8, 8), "v_proj": nn.Linear(8, 8)}) for _ in range(2)
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            block = cast(nn.ModuleDict, module)
            q_proj = cast(nn.Linear, block["q_proj"])
            v_proj = cast(nn.Linear, block["v_proj"])
            x = torch.tanh(q_proj(x) + v_proj(x))
        return x.sum()

    forward: Callable[..., Any] = _forward


def build_tiny_peft_model(seed: int = 7) -> Any:
    torch.manual_seed(seed)
    base = TinyLayeredModel()
    config = make_simplex_lora_config(
        r=4,
        lora_alpha=8,
        lora_dropout=0,
        target_modules=["q_proj", "v_proj"],
    )
    return get_simplex_peft_model(base, config)


def tiny_registry(model: Any) -> Any:
    return build_layer_registry(model, "default")


def unzero_adapters(model: Any, seed: int = 3) -> None:
    """Give lora_B non-zero weights so simplex gradients differ per module."""
    generator = torch.Generator().manual_seed(seed)
    for _, module in model.named_modules():
        if isinstance(module, SimplexLoraLinear):
            weight = module.lora_B["default"].weight
            with torch.no_grad():
                weight.copy_(torch.randn(weight.shape, generator=generator))


def run_metadata(adapter_name: str = "default", **overrides: Any) -> CalibrationRunMetadata:
    values: dict[str, Any] = {
        "run_id": "calib-test",
        "created_at": utc_now_iso(),
        "model_id": "tiny-model",
        "model_revision": "abc123",
        "architecture": "TinyLayeredModel",
        "device": "cpu",
        "adapter_name": adapter_name,
        "peft_config": {"r": 4, "lora_alpha": 8},
        "default_lora_rank": 4,
        "lora_alpha": 8,
        "fat_step": 1,
        "max_fat_steps": 2,
        "weight_lr": 5e-3,
        "weight_decay": 0.01,
        "seed": 11,
        "effective_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "library_versions": {"torch": torch.__version__},
        "data": {"dataset": "unit"},
    }
    values.update(overrides)
    return CalibrationRunMetadata(**values)


def make_event(
    registry: Any,
    *,
    projection_index: int,
    step: int,
    weights: list[float],
    is_final: bool = False,
    loss: float | None = 0.5,
    default_lora_rank: int = 4,
) -> ProjectionEvent:
    import math

    ranks = [math.ceil(weight * default_lora_rank - 1e-9) for weight in weights]
    return ProjectionEvent(
        step=step,
        projection_index=projection_index,
        is_final_projection=is_final,
        weights_before_step=tuple(weights),
        weights_after_gradient_step=tuple(weights),
        weights_projected=tuple(weights),
        ranks=tuple(ranks),
        selected_mask=tuple(rank > 0 for rank in ranks),
        loss_at_projection=loss,
        loss_window=loss,
        weight_grad_norms=tuple(0.0 for _ in weights),
        lora_grad_norms=tuple(0.0 for _ in weights),
        created_at=utc_now_iso(),
    )


def calibration_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "calibration_output_dir": None,
        "calibration_top_k": 5,
        "calibration_only": False,
        "calibration_no_artifacts": False,
        "calibration_rank_floor": 1,
        "calibration_alpha_mode": "preserve",
        "calibration_min_weight": 0.0,
        "calibration_unsloth_top_k": None,
        "calibration_overwrite": False,
        "calibration_run_id": None,
        "fat_step": 1,
        "max_fat_steps": 2,
        "max_steps": -1,
        "lora_r": 4,
        "lora_alpha": 8,
        "weight_lr": 5e-3,
        "weight_decay": 0.01,
        "seed": 11,
        "model_id": "tiny-model",
        "output_dir": "outputs/unit-tests",
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_plan(
    module_specs: list[tuple[str, float]],
    *,
    run_id: str = "calib-plan-test",
    default_lora_rank: int = 16,
    lora_alpha: int = 32,
) -> Any:
    """Build a structurally valid LayerPlan from (module_name, weight) specs.

    Weights are normalized so that normalized_importance sums to exactly 1.
    """
    from simplexlora.calibration.models import (
        LayerPlan,
        LayerPlanAdapterInfo,
        LayerPlanEntry,
        LayerPlanModelInfo,
    )

    total = sum(weight for _, weight in module_specs)
    entries = []
    for module_name, weight in module_specs:
        importance = weight / total
        rank = max(1, math.ceil(importance * default_lora_rank - 1e-9)) if weight > 0 else 0
        layer_index = int(module_name.split(".")[2]) if module_name.startswith("model.layers.") else -1
        component = module_name.rsplit(".", 1)[-1]
        entries.append(
            LayerPlanEntry(
                module_name=module_name,
                peft_module_name="base_model.model." + module_name,
                layer_index=layer_index,
                component=component,
                in_features=8,
                out_features=8,
                simplex_weight=weight,
                normalized_importance=importance,
                rank=rank,
                selected=rank > 0,
                lora_alpha=lora_alpha,
                scaling=lora_alpha / rank if rank > 0 else 0.0,
                is_top_k=False,
            )
        )
    entries.sort(key=lambda entry: (entry.layer_index, entry.component, entry.module_name))

    top = sorted((e for e in entries if e.selected), key=lambda e: (-e.normalized_importance, e.module_name))[:3]
    top_names = {e.module_name for e in top}
    entries = [dataclasses.replace(entry, is_top_k=entry.module_name in top_names) for entry in entries]

    return LayerPlan(
        schema_version="1.0",
        run_id=run_id,
        model=LayerPlanModelInfo(model_id="tiny-model", revision="rev1", architecture="TinyLayeredModel"),
        adapter=LayerPlanAdapterInfo(
            name="default",
            projection_step=10,
            default_lora_rank=default_lora_rank,
            lora_alpha=lora_alpha,
            total_rank=sum(entry.rank for entry in entries),
            selected_module_count=sum(1 for entry in entries if entry.selected),
        ),
        modules=tuple(entries),
        top_k=3,
    )

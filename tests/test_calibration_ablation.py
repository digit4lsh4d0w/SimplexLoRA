"""Held-out layer ablation tests for Simplex calibration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import torch
from datasets import Dataset

from simplexlora.calibration.ablation import (
    ABLATION_JSON_FILENAME,
    ABLATION_MARKDOWN_FILENAME,
    LayerAblationEvaluator,
    write_ablation_artifacts,
)
from simplexlora.calibration.models import (
    LayerPlan,
    LayerPlanAdapterInfo,
    LayerPlanEntry,
    LayerPlanModelInfo,
)
from simplexlora.calibration.registry import build_layer_registry
from simplexlora.experiments.decompile import CausalLMCollator
from simplexlora.peft_plugin import SimplexLoraLinear
from tests.test_calibration_integration import build_tiny_causal_lm


def build_eval_dataset(size: int = 4) -> Dataset:
    sequences = [[1 + (index * 7 + offset) % 63 for offset in range(8)] for index in range(size)]
    return Dataset.from_dict({"input_ids": sequences, "labels": [row[:] for row in sequences]})


def build_plan(model: torch.nn.Module) -> LayerPlan:
    registry = build_layer_registry(model, "default")
    entries = []
    for identity in registry.identities:
        rank = identity.initial_rank
        entries.append(
            LayerPlanEntry(
                module_name=identity.module_name,
                peft_module_name=identity.peft_module_name,
                layer_index=identity.layer_index,
                component=identity.component,
                in_features=identity.in_features,
                out_features=identity.out_features,
                simplex_weight=1.0,
                normalized_importance=1.0 / len(registry),
                rank=rank,
                selected=True,
                lora_alpha=identity.lora_alpha,
                scaling=identity.lora_alpha / rank,
            )
        )
    return LayerPlan(
        schema_version="1.0",
        run_id="calib-ablation-test",
        model=LayerPlanModelInfo(model_id="tiny", revision="local", architecture="Qwen2ForCausalLM"),
        adapter=LayerPlanAdapterInfo(
            name="default",
            projection_step=10,
            default_lora_rank=4,
            lora_alpha=8,
            total_rank=sum(entry.rank for entry in entries),
            selected_module_count=len(entries),
        ),
        modules=tuple(entries),
        top_k=3,
    )


class LayerAblationEvaluatorTests(unittest.TestCase):
    def test_ablation_measures_active_layer_and_restores_gates(self) -> None:
        torch.manual_seed(3)
        model = build_tiny_causal_lm(seed=3)
        model.config.use_cache = False
        layers = [module for _, module in model.named_modules() if isinstance(module, SimplexLoraLinear)]
        generator = torch.Generator().manual_seed(12)
        for layer in layers:
            with torch.no_grad():
                layer.lora_A["default"].weight.normal_(0.0, 0.2, generator=generator)
                layer.lora_B["default"].weight.normal_(0.0, 0.2, generator=generator)
                name = next(name for name, module in model.named_modules() if module is layer)
                cast(torch.Tensor, layer.lora_weight["default"]).fill_(1.0 if "layers.0." in name else 0.0)

        saved = {id(layer): cast(torch.Tensor, layer.lora_weight["default"]).detach().clone() for layer in layers}
        registry = build_layer_registry(model, "default")
        evaluator = LayerAblationEvaluator(
            model=model,
            dataset=build_eval_dataset(),
            data_collator=CausalLMCollator(0),
            registry=registry,
            adapter_name="default",
            max_samples=4,
            batch_size=2,
            top_k=1,
        )

        result = evaluator.evaluate(build_plan(model))

        self.assertEqual(result.sample_count, 4)
        self.assertEqual(len(result.layers), 2)
        by_layer = {record.layer_index: record for record in result.layers}
        self.assertEqual(by_layer[1].delta_loss, 0.0)
        self.assertGreater(abs(by_layer[0].delta_loss), 1e-8)
        for layer in layers:
            torch.testing.assert_close(cast(torch.Tensor, layer.lora_weight["default"]), saved[id(layer)])
        self.assertTrue(model.training)

    def test_ablation_artifacts_are_machine_and_human_readable(self) -> None:
        torch.manual_seed(5)
        model = build_tiny_causal_lm(seed=5)
        model.config.use_cache = False
        registry = build_layer_registry(model, "default")
        evaluator = LayerAblationEvaluator(
            model=model,
            dataset=build_eval_dataset(2),
            data_collator=CausalLMCollator(0),
            registry=registry,
            adapter_name="default",
            max_samples=2,
            batch_size=1,
        )
        result = evaluator.evaluate(build_plan(model))
        self.assertIsNone(result.top_k_overlap)

        with TemporaryDirectory() as tmp:
            json_path, markdown_path = write_ablation_artifacts(result, Path(tmp))
            document = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(document["run_id"], result.run_id)
            self.assertEqual(len(document["layers"]), 2)
            self.assertEqual(json_path.name, ABLATION_JSON_FILENAME)
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Spearman importance vs delta loss", markdown)
            self.assertIn("Top-5 overlap: n/a", markdown)
            self.assertEqual(markdown_path.name, ABLATION_MARKDOWN_FILENAME)


if __name__ == "__main__":
    unittest.main()

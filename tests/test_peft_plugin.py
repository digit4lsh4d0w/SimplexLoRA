from __future__ import annotations

import json
import math
import unittest
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from torch import nn

from simplexlora import optimizers
from simplexlora.peft_plugin import (
    SimplexLoraLinear,
    get_simplex_peft_model,
    load_simplex_peft_model,
    make_simplex_lora_config,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(6, 4)

    def _forward(self, *args: Any) -> Any:
        x = cast(torch.Tensor, args[0])
        return torch.tanh(self.proj(x))

    forward: Callable[..., Any] = _forward


def build_model() -> tuple[Any, TinyModel, dict[str, torch.Tensor]]:
    torch.manual_seed(7)
    base_model = TinyModel()
    base_state = deepcopy(base_model.state_dict())
    config = make_simplex_lora_config(
        r=2,
        lora_alpha=4,
        lora_dropout=0,
        target_modules=["proj"],
    )
    model = get_simplex_peft_model(base_model, config)
    return model, base_model, base_state


def linear_weight(module: Any) -> torch.Tensor:
    return cast(nn.Linear, module).weight


class SimplexPeftPluginTests(unittest.TestCase):
    def test_custom_layer_exposes_simplex_parameter(self) -> None:
        model, _, _ = build_model()
        layer = cast(SimplexLoraLinear, model.base_model.model.proj)
        parameter_names = {name for name, _ in model.named_parameters()}

        self.assertIsInstance(layer, SimplexLoraLinear)
        self.assertEqual(layer.r["default"], 2)
        self.assertIn("base_model.model.proj.lora_weight.default", parameter_names)
        self.assertEqual(layer.lora_weight["default"].item(), 1)

    def test_forward_uses_learned_weight_and_backpropagates(self) -> None:
        model, _, _ = build_model()
        layer = cast(SimplexLoraLinear, model.base_model.model.proj)
        inputs = torch.randn(5, 6)

        with torch.no_grad():
            linear_weight(layer.lora_A["default"]).normal_()
            linear_weight(layer.lora_B["default"]).normal_()
            cast(torch.Tensor, layer.lora_weight["default"]).fill_(0.25)

        base_output = layer.get_base_layer()(inputs)
        expected_delta = layer.lora_B["default"](layer.lora_A["default"](inputs)) * layer.scaling["default"] * 0.25
        expected = torch.tanh(base_output + expected_delta)
        actual = model(inputs)

        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()

        self.assertIsNotNone(linear_weight(layer.lora_A["default"]).grad)
        self.assertIsNotNone(linear_weight(layer.lora_B["default"]).grad)
        self.assertIsNotNone(cast(torch.Tensor, layer.lora_weight["default"]).grad)

    def test_bitsandbytes_4bit_base_layer_on_cpu(self) -> None:
        try:
            import bitsandbytes as bnb
        except ImportError:
            self.skipTest("bitsandbytes is not installed")

        base_model = nn.Sequential(
            bnb.nn.Linear4bit(
                6,
                4,
                compute_dtype=torch.float32,
                quant_type="nf4",
                device="cpu",
            )
        ).to("cpu")
        config = make_simplex_lora_config(
            r=2,
            lora_alpha=4,
            target_modules=["0"],
        )
        model = get_simplex_peft_model(base_model, config)
        layer = cast(SimplexLoraLinear, model.base_model.model[0])

        output = model(torch.randn(3, 6))
        output.sum().backward()

        self.assertIsInstance(layer, SimplexLoraLinear)
        self.assertIsNotNone(cast(torch.Tensor, layer.lora_weight["default"]).grad)

    def test_merge_unmerge_accounts_for_weight(self) -> None:
        model, _, base_state = build_model()
        layer = cast(SimplexLoraLinear, model.base_model.model.proj)
        inputs = torch.randn(3, 6)

        with torch.no_grad():
            linear_weight(layer.lora_A["default"]).normal_()
            linear_weight(layer.lora_B["default"]).normal_()
            cast(torch.Tensor, layer.lora_weight["default"]).fill_(0.4)

        unmerged_output = model(inputs)
        model.merge_adapter()
        merged_output = model(inputs)
        model.unmerge_adapter()
        restored_output = model(inputs)

        torch.testing.assert_close(merged_output, unmerged_output)
        torch.testing.assert_close(restored_output, unmerged_output)
        torch.testing.assert_close(layer.get_base_layer().weight, base_state["proj.weight"])

    def test_save_and_load_restores_plugin_parameters(self) -> None:
        model, _, base_state = build_model()
        layer = cast(SimplexLoraLinear, model.base_model.model.proj)
        inputs = torch.randn(4, 6)

        with torch.no_grad():
            linear_weight(layer.lora_A["default"]).normal_()
            linear_weight(layer.lora_B["default"]).normal_()
            cast(torch.Tensor, layer.lora_weight["default"]).fill_(0.7)
        expected = model(inputs)

        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            adapter_config = json.loads(Path(directory, "adapter_config.json").read_text(encoding="utf-8"))
            self.assertEqual(adapter_config["peft_type"], "LORA")

            restored_base = TinyModel()
            restored_base.load_state_dict(base_state)
            restored = load_simplex_peft_model(restored_base, directory)
            restored_layer = cast(SimplexLoraLinear, restored.base_model.model.proj)

            torch.testing.assert_close(restored(inputs), expected)
            self.assertIsInstance(restored_layer, SimplexLoraLinear)
            self.assertAlmostEqual(cast(torch.Tensor, restored_layer.lora_weight["default"]).item(), 0.7)

    def test_dynamic_rank_updates_match_simplex_optimizer_contract(self) -> None:
        model, _, _ = build_model()
        layer = cast(SimplexLoraLinear, model.base_model.model.proj)
        original_weight = cast(torch.Tensor, layer.lora_weight["default"])

        layer._update_lora_rank_QR(3, "default")
        self.assertIs(layer.lora_weight["default"], original_weight)
        self.assertEqual(layer.r["default"], 3)
        self.assertEqual(layer.lora_A["default"].out_features, 3)
        self.assertEqual(layer.lora_B["default"].in_features, 3)
        self.assertTrue(cast(torch.Tensor, layer.lora_weight["default"]).requires_grad)
        self.assertTrue(math.isclose(layer.scaling["default"], 4 / 3))

        layer._final_lora_rank_update(1, "default")
        self.assertEqual(layer.r["default"], 1)
        self.assertEqual(cast(torch.Tensor, layer.lora_weight["default"]).item(), 1)
        self.assertFalse(cast(torch.Tensor, layer.lora_weight["default"]).requires_grad)

    def test_simplex_optimizer_survives_multiple_rank_updates(self) -> None:
        model, _, _ = build_model()
        layer = cast(SimplexLoraLinear, model.base_model.model.proj)
        lora_ab_params = [
            linear_weight(layer.lora_A["default"]),
            linear_weight(layer.lora_B["default"]),
        ]
        weight_params = [cast(torch.Tensor, layer.lora_weight["default"])]
        optimizer = optimizers.SimplexAdamW(
            [
                {"params": lora_ab_params, "name": "loraAB"},
                {"params": [], "name": "other_params"},
                {
                    "params": weight_params,
                    "proj": optimizers.proj_simplex_euclidean,
                    "lr": 5e-3,
                    "wd": 0.01,
                    "name": "weight_params",
                },
            ],
            lora_layers=[layer],
            lr=1e-3,
            num_adapters=1,
            fat_step=1,
            max_fat_steps=2,
            default_lora_rank=2,
        )

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(3, 6)).square().sum().backward()
            optimizer.step()

        self.assertEqual(layer.r["default"], 2)
        self.assertFalse(cast(torch.Tensor, layer.lora_weight["default"]).requires_grad)


if __name__ == "__main__":
    unittest.main()

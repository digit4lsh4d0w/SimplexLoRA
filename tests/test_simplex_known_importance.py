"""Known-answer test: Simplex calibration must recover causally active layers."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

from simplexlora import optimizers
from simplexlora.peft_plugin import SimplexLoraLinear, get_simplex_peft_model, make_simplex_lora_config


class _KnownLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)


class _KnownBody(nn.Module):
    def __init__(self, *, layer_count: int, active_layers: tuple[int, ...], width: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_KnownLayer(width) for _ in range(layer_count))
        self.active_layers = active_layers

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x
        for index, layer in enumerate(self.layers):
            if index in self.active_layers:
                output = output + cast(_KnownLayer, layer).q_proj(x)
        return output

    forward: Callable[..., Any] = _forward


class KnownImportanceModel(nn.Module):
    """Six interchangeable layers; only instances in ``active_layers`` affect the loss."""

    def __init__(self, *, active_layers: tuple[int, ...], width: int = 8) -> None:
        super().__init__()
        self.model = _KnownBody(layer_count=6, active_layers=active_layers, width=width)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    forward: Callable[..., Any] = _forward


class KnownImportanceTests(unittest.TestCase):
    def _run(self, active_layers: tuple[int, ...], seed: int) -> tuple[set[int], dict[int, int]]:
        torch.manual_seed(seed)
        width = 8
        base = KnownImportanceModel(active_layers=active_layers, width=width)
        for layer in base.model.layers:
            with torch.no_grad():
                cast(_KnownLayer, layer).q_proj.weight.zero_()

        config = make_simplex_lora_config(
            r=4,
            lora_alpha=8,
            lora_dropout=0,
            target_modules=["q_proj"],
        )
        model = get_simplex_peft_model(base, config)
        named_layers = [
            (name, module) for name, module in model.named_modules() if isinstance(module, SimplexLoraLinear)
        ]
        named_layers.sort(key=lambda item: item[0])
        layers = [module for _, module in named_layers]
        self.assertEqual(len(layers), 6)

        generator = torch.Generator().manual_seed(seed + 100)
        inputs = torch.randn(12, width, generator=generator)
        for layer in layers:
            with torch.no_grad():
                layer.lora_A["default"].weight.normal_(mean=0.0, std=0.1, generator=generator)
                layer.lora_B["default"].weight.normal_(mean=0.0, std=0.1, generator=generator)
        source = layers[active_layers[0]]
        for layer_index in active_layers[1:]:
            with torch.no_grad():
                layers[layer_index].lora_A["default"].weight.copy_(source.lora_A["default"].weight)
                layers[layer_index].lora_B["default"].weight.copy_(source.lora_B["default"].weight)

        with torch.no_grad():
            current = cast(torch.Tensor, model(inputs))
            target = inputs + 2.0 * (current - inputs)

        lora_layers = layers
        lora_ab_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
        ]
        weight_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_weight" in name
        ]
        optimizer = optimizers.SimplexAdamW(
            [
                {"params": lora_ab_params, "name": "loraAB"},
                {"params": [], "name": "other_params"},
                {
                    "params": weight_params,
                    "proj": optimizers.proj_simplex_euclidean,
                    "lr": 1.0,
                    "wd": 0.0,
                    "name": "weight_params",
                },
            ],
            lora_layers=lora_layers,
            lr=1e-3,
            weight_decay=0.0,
            num_adapters=len(weight_params),
            fat_step=2,
            max_fat_steps=3,
            default_lora_rank=4,
        )

        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            loss = (cast(torch.Tensor, model(inputs)) - target).square().mean()
            loss.backward()
            optimizer.step()

        selected = set(optimizer.chosen_layers)
        ranks = {
            layer_index: int(optimizer.lora_ranks[position].item())
            for position, layer_index in enumerate(optimizer.chosen_layers)
        }
        return selected, ranks

    def test_recovers_first_known_layer_set(self) -> None:
        selected, ranks = self._run(active_layers=(1, 4), seed=7)
        self.assertEqual(selected, {1, 4})
        self.assertTrue(all(rank > 0 for rank in ranks.values()))

    def test_recovers_second_known_layer_set(self) -> None:
        selected, ranks = self._run(active_layers=(0, 3), seed=11)
        self.assertEqual(selected, {0, 3})
        self.assertTrue(all(rank > 0 for rank in ranks.values()))


if __name__ == "__main__":
    unittest.main()

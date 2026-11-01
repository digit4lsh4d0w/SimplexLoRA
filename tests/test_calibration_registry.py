"""Unit tests for ``simplexlora.calibration.registry``."""

from __future__ import annotations

import unittest

import torch

from tests.calibration_helpers import build_tiny_peft_model, tiny_registry

from simplexlora.calibration.models import CalibrationRegistryError
from simplexlora.calibration.registry import build_layer_registry
from simplexlora.peft_plugin import SimplexLoraLinear


class BuildLayerRegistryTests(unittest.TestCase):
    def test_registry_ordering_is_stable(self) -> None:
        model = build_tiny_peft_model()
        registry = tiny_registry(model)
        names = list(registry.module_names)
        self.assertEqual(
            names,
            sorted(
                names,
                key=lambda name: (
                    registry.identity_by_name(name).layer_index,
                    registry.identity_by_name(name).component,
                    name,
                ),
            ),
        )
        # sorted by layer index first, then component, then name
        self.assertEqual(names[0], "model.layers.0.q_proj")
        self.assertEqual(names[1], "model.layers.0.v_proj")
        self.assertEqual(names[-1], "model.layers.1.v_proj")

    def test_identities_are_unique_and_consistent(self) -> None:
        model = build_tiny_peft_model()
        registry = tiny_registry(model)
        self.assertEqual(len(registry), 4)
        self.assertEqual(len(set(registry.module_names)), 4)
        self.assertEqual(set(registry.module_name_to_index), set(registry.module_names))
        for index, identity in enumerate(registry.identities):
            self.assertEqual(identity.module_index, index)
            self.assertEqual(identity.layer_index, index // 2)
            self.assertIn(identity.component, ("q_proj", "v_proj"))
            self.assertEqual(identity.in_features, 8)
            self.assertEqual(identity.out_features, 8)
            self.assertEqual(identity.initial_rank, 4)
            self.assertEqual(identity.lora_alpha, 8)
            self.assertTrue(identity.peft_module_name.startswith("base_model.model."))

    def test_canonical_name_resolves_to_base_module(self) -> None:
        model = build_tiny_peft_model()
        registry = tiny_registry(model)
        base = model.base_model.model
        for identity in registry.identities:
            module = base.get_submodule(identity.module_name)
            self.assertIsInstance(module, SimplexLoraLinear)

    def test_wrong_adapter_name_rejected(self) -> None:
        model = build_tiny_peft_model()
        with self.assertRaises(CalibrationRegistryError):
            build_layer_registry(model, "missing-adapter")

    def test_multiple_peft_adapters_rejected(self) -> None:
        model = build_tiny_peft_model()
        model.peft_config["other"] = model.peft_config["default"]
        from simplexlora.calibration.registry import build_layer_registry

        with self.assertRaises(CalibrationRegistryError):
            build_layer_registry(model, "default")

    def test_no_simplex_layers_rejected(self) -> None:
        from torch import nn

        from simplexlora.calibration.registry import build_layer_registry

        plain = nn.Sequential(nn.Linear(4, 4))
        with self.assertRaises(CalibrationRegistryError):
            build_layer_registry(plain, "default")

    def test_target_modules_filter(self) -> None:
        from simplexlora.calibration.registry import build_layer_registry

        model = build_tiny_peft_model()
        registry = build_layer_registry(model, "default", target_modules=["q_proj"])
        self.assertEqual(set(registry.module_names), {"model.layers.0.q_proj", "model.layers.1.q_proj"})

    def test_lora_dimension_mismatch_rejected(self) -> None:
        model = build_tiny_peft_model()
        layer = model.base_model.model.model.layers[0]["q_proj"]
        with torch.no_grad():
            layer.lora_A["default"].weight = torch.nn.Parameter(torch.zeros(3, 8))
        with self.assertRaises(CalibrationRegistryError):
            build_layer_registry(model, "default")


if __name__ == "__main__":
    unittest.main()

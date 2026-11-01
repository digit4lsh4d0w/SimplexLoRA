"""Unit tests for the Unsloth handoff (spec sections 8.10, 6.8, 14)."""

from __future__ import annotations

import dataclasses
import re
import unittest
from typing import Literal, cast

from tests.calibration_helpers import make_plan
from tests.calibration_helpers import TinyLayeredModel

from simplexlora.calibration.handoff import build_unsloth_handoff
from simplexlora.calibration.models import CalibrationValidationError, WarningCode

PLAN_SPECS = [
    ("model.layers.0.q_proj", 4.0),
    ("model.layers.0.v_proj", 2.0),
    ("model.layers.1.q_proj", 1.0),
    ("model.layers.1.v_proj", 0.0),
]


class BuildUnslothHandoffTests(unittest.TestCase):
    def test_basic_handoff_structure(self) -> None:
        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan)
        self.assertEqual(handoff.schema_version, "1.0")
        self.assertEqual(handoff.source_plan_run_id, plan.run_id)
        self.assertEqual(handoff.alpha_mode, "preserve")
        self.assertEqual(handoff.rank_floor, 1)
        self.assertEqual(handoff.min_weight, 0.0)
        self.assertIsNone(handoff.top_k)
        # rank-0 module must never appear in target_modules
        self.assertNotIn("model.layers.1.v_proj", handoff.target_modules)
        self.assertEqual(len(handoff.target_modules), 3)
        self.assertEqual(len(handoff.rank_pattern), 3)
        self.assertEqual(len(handoff.alpha_pattern), 3)

    def test_target_modules_are_literal_canonical_names(self) -> None:
        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan)
        selected_names = [name for name, _ in PLAN_SPECS if name != "model.layers.1.v_proj"]
        self.assertEqual(list(handoff.target_modules), selected_names)

    def test_rank_pattern_keys_are_escaped(self) -> None:
        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan)
        selected_names = [name for name, _ in PLAN_SPECS if name != "model.layers.1.v_proj"]
        for name in selected_names:
            key = re.escape(name)
            self.assertIn(key, handoff.rank_pattern)
            # compiling proves the key is a valid regex
            re.compile(key)

    def test_rank0_modules_excluded(self) -> None:
        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan)
        self.assertIn("model.layers.1.v_proj", handoff.excluded_modules)
        self.assertNotIn("model.layers.1.v_proj", handoff.target_modules)

    def test_rank_floor_produces_warning(self) -> None:
        # importance 1/1001 -> rank ceil((1/1001) * 16) = 1, floored to 4
        tiny = make_plan([("model.layers.0.q_proj", 1000.0), ("model.layers.0.v_proj", 1.0)])
        handoff = build_unsloth_handoff(tiny, rank_floor=4)
        codes = [warning.code for warning in handoff.warnings]
        self.assertIn(WarningCode.HANDOFF_RANK_FLOOR.value, codes)
        # the floored module got rank 4 in the pattern
        floored_key = re.escape("model.layers.0.v_proj")
        self.assertEqual(handoff.rank_pattern[floored_key], 4)
        self.assertGreater(handoff.applied_rank_total, handoff.requested_rank_total)

    def test_top_k_produces_warning(self) -> None:
        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan, top_k=2)
        codes = [warning.code for warning in handoff.warnings]
        self.assertIn(WarningCode.HANDOFF_TOP_K.value, codes)
        self.assertEqual(len(handoff.target_modules), 2)
        # the two most important modules are kept
        self.assertIn("model.layers.0.q_proj", handoff.target_modules)

    def test_handoff_applies_to_plain_peft_model(self) -> None:
        from peft import LoraConfig, get_peft_model

        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan)
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=list(handoff.target_modules),
            rank_pattern=dict(handoff.rank_pattern),
            alpha_pattern=dict(handoff.alpha_pattern),
            bias="none",
        )
        model = get_peft_model(TinyLayeredModel(), config)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertEqual(trainable, handoff.estimated_trainable_parameters)

    def test_min_weight_excludes_and_warns(self) -> None:
        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan, min_weight=0.4)
        codes = [warning.code for warning in handoff.warnings]
        self.assertIn(WarningCode.HANDOFF_MIN_WEIGHT.value, codes)
        self.assertLess(len(handoff.target_modules), 3)

    def test_twice_rank_alpha_mode_warns_about_scaling(self) -> None:
        plan = make_plan(PLAN_SPECS)
        preserve = build_unsloth_handoff(plan, alpha_mode="preserve")
        twice = build_unsloth_handoff(plan, alpha_mode="twice_rank")
        codes = [warning.code for warning in twice.warnings]
        self.assertIn(WarningCode.HANDOFF_ALPHA_CHANGED.value, codes)
        self.assertNotIn(WarningCode.HANDOFF_ALPHA_CHANGED.value, [w.code for w in preserve.warnings])
        for module_name, applied_rank in twice.rank_pattern.items():
            self.assertEqual(twice.alpha_pattern[module_name], 2 * applied_rank)

    def test_preserve_keeps_original_alpha(self) -> None:
        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan, alpha_mode="preserve")
        for module_name, applied_rank in handoff.rank_pattern.items():
            self.assertEqual(handoff.alpha_pattern[module_name], 32)  # lora_alpha from make_plan

    def test_estimated_trainable_parameters(self) -> None:
        plan = make_plan(PLAN_SPECS)
        handoff = build_unsloth_handoff(plan)
        # all modules are 8x8, ranks are ceil(importance*16) >= 1
        expected = sum(rank * (8 + 8) for rank in handoff.rank_pattern.values())
        self.assertEqual(handoff.estimated_trainable_parameters, expected)
        self.assertEqual(
            handoff.applied_rank_total,
            sum(handoff.rank_pattern.values()),
        )

    def test_damaged_plan_never_becomes_handoff(self) -> None:
        plan = make_plan(PLAN_SPECS)
        wrong = dataclasses.replace(plan.modules[0], selected=False)
        broken = dataclasses.replace(plan, modules=(wrong, *plan.modules[1:]))
        with self.assertRaises(CalibrationValidationError):
            build_unsloth_handoff(broken)

    def test_invalid_options_rejected(self) -> None:
        plan = make_plan(PLAN_SPECS)
        with self.assertRaises(ValueError):
            build_unsloth_handoff(plan, rank_floor=0)
        with self.assertRaises(ValueError):
            build_unsloth_handoff(plan, min_weight=2.0)
        with self.assertRaises(ValueError):
            build_unsloth_handoff(plan, alpha_mode=cast(Literal["preserve", "twice_rank"], "nonsense"))
        with self.assertRaises(ValueError):
            build_unsloth_handoff(plan, top_k=0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for plan finalize/validate/load/verify (spec section 8.7-8.9, 8.13)."""

from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.calibration_helpers import build_tiny_peft_model, make_event, make_plan, run_metadata, tiny_registry

from simplexlora.calibration.models import (
    CalibrationPlanError,
    CalibrationValidationError,
)
from simplexlora.calibration.plan import (
    finalize_layer_plan,
    load_layer_plan,
    validate_layer_plan,
    verify_applied_plan,
)
from simplexlora.calibration.recorder import CalibrationRecorder


class ValidateLayerPlanTests(unittest.TestCase):
    def test_valid_plan_passes(self) -> None:
        plan = make_plan(
            [("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0), ("model.layers.1.q_proj", 0.0)]
        )
        report = validate_layer_plan(plan)
        self.assertTrue(report.ok)
        self.assertEqual(report.statistics["selected_count"], 2)

    def test_duplicate_module_names_rejected(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        modules = plan.modules + (plan.modules[0],)
        tampered = dataclasses.replace(plan, modules=modules)
        with self.assertRaises(CalibrationValidationError) as context:
            validate_layer_plan(tampered, strict=True)
        codes = [issue.code for issue in context.exception.report.errors]
        self.assertIn("duplicate_module", codes)

    def test_dimension_mismatch_detected(self) -> None:
        model = build_tiny_peft_model()
        plan = make_plan(
            [("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0), ("model.layers.1.q_proj", 1.0)]
        )
        wrong = dataclasses.replace(plan.modules[0], in_features=64)
        tampered = dataclasses.replace(plan, modules=(wrong,) + plan.modules[1:])
        with self.assertRaises(CalibrationValidationError) as context:
            validate_layer_plan(tampered, model=model, adapter_name="default", strict=True)
        codes = [issue.code for issue in context.exception.report.errors]
        self.assertIn("dimension_mismatch", codes)

    def test_selected_mask_mismatch_detected(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        wrong = dataclasses.replace(plan.modules[0], selected=False)
        tampered = dataclasses.replace(plan, modules=(wrong,) + plan.modules[1:])
        with self.assertRaises(CalibrationValidationError) as context:
            validate_layer_plan(tampered, strict=True)
        codes = [issue.code for issue in context.exception.report.errors]
        self.assertIn("selected_mask_mismatch", codes)

    def test_importance_sum_mismatch_detected(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        wrong = dataclasses.replace(plan.modules[0], normalized_importance=0.5)
        tampered = dataclasses.replace(plan, modules=(wrong,) + plan.modules[1:])
        with self.assertRaises(CalibrationValidationError) as context:
            validate_layer_plan(tampered, strict=True)
        codes = [issue.code for issue in context.exception.report.errors]
        self.assertIn("importance_sum", codes)

    def test_adapter_totals_mismatch_detected(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        tampered = dataclasses.replace(plan, adapter=dataclasses.replace(plan.adapter, total_rank=999))
        with self.assertRaises(CalibrationValidationError) as context:
            validate_layer_plan(tampered, strict=True)
        codes = [issue.code for issue in context.exception.report.errors]
        self.assertIn("adapter_total_rank", codes)

    def test_negative_rank_detected(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        wrong = dataclasses.replace(plan.modules[0], rank=-1, selected=False)
        tampered = dataclasses.replace(plan, modules=(wrong,) + plan.modules[1:])
        with self.assertRaises(CalibrationValidationError) as context:
            validate_layer_plan(tampered, strict=True)
        codes = [issue.code for issue in context.exception.report.errors]
        self.assertIn("negative_rank", codes)

    def test_non_strict_mode_returns_report(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        tampered = dataclasses.replace(plan, modules=plan.modules + (plan.modules[0],))
        report = validate_layer_plan(tampered, strict=False)
        self.assertFalse(report.ok)


class VerifyAppliedPlanTests(unittest.TestCase):
    def _run_calibration(self, output_dir: Path | None = None):
        """Run a real 2-projection calibration with the observer attached.

        Returns (model, session, output_dir).
        """
        import torch

        from simplexlora import optimizers
        from tests.calibration_helpers import calibration_args, unzero_adapters

        from simplexlora.calibration.recorder import create_calibration_session
        from simplexlora.calibration.cli import build_run_metadata

        model = build_tiny_peft_model()
        unzero_adapters(model)
        layers = [m for _, m in model.named_modules() if type(m).__name__ == "SimplexLoraLinear"]
        lora_ab = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
        weight_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_weight" in n]
        optimizer = optimizers.SimplexAdamW(
            [
                {"params": lora_ab, "name": "loraAB"},
                {"params": [], "name": "other_params"},
                {
                    "params": weight_params,
                    "proj": optimizers.proj_simplex_euclidean,
                    "lr": 5e-2,
                    "wd": 0.0,
                    "name": "weight_params",
                },
            ],
            lora_layers=layers,
            lr=1e-3,
            num_adapters=len(weight_params),
            fat_step=1,
            max_fat_steps=2,
            # Force the intermediate projection to resize adapters before the
            # final projection and verification pass.
            default_lora_rank=5,
        )
        args = calibration_args(
            calibration_output_dir=str(output_dir) if output_dir else None,
            lora_r=5,
        )
        metadata = build_run_metadata(model, adapter_name="default", args=args)
        session = create_calibration_session(model=model, optimizer=optimizer, args=args, run_metadata=metadata)
        torch.manual_seed(5)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.randn(3, 8))
            loss.backward()
            optimizer.add_microbatch_loss(loss.detach())
            optimizer.step()
        return model, session, Path(args.calibration_output_dir) if args.calibration_output_dir else None

    def test_verify_ok_after_real_final_rank_update(self) -> None:
        with TemporaryDirectory() as tmp:
            model, session, _ = self._run_calibration(Path(tmp) / "calibration")
            self.assertIsNotNone(session.plan)
            # The session verified the plan right after the final rank update.
            self.assertTrue(session.recorder.calibration_completed)
            self.assertNotIn(
                "plan_verification_failed",
                [warning.code for warning in session.recorder.warnings],
            )
            report = session.recorder.verification_report
            self.assertIsNotNone(report)
            self.assertTrue(report.ok, msg=str([issue.message for issue in report.errors]))
            session.close()

    def test_plan_survives_final_reset_of_adapters(self) -> None:
        """The frozen plan must not change when _final_lora_rank_update resets A/B."""
        with TemporaryDirectory() as tmp:
            model, session, _ = self._run_calibration(Path(tmp) / "calibration")
            plan = session.plan
            snapshot_before = [(m.module_name, m.rank, m.simplex_weight) for m in plan.modules]
            # Any post-hoc mutation of the model must not leak into the plan.
            for identity in session.registry.identities:
                layer = model.base_model.model.get_submodule(identity.module_name)
                if int(layer.r["default"]) > 0:
                    layer._update_lora_rank_QR(1, "default")
            snapshot_after = [(m.module_name, m.rank, m.simplex_weight) for m in plan.modules]
            self.assertEqual(snapshot_before, snapshot_after)
            session.close()

    def test_substituted_rank_detected(self) -> None:
        model = build_tiny_peft_model()
        registry = tiny_registry(model)
        # Simulate the final state: q_proj modules finalized at rank 4.
        for identity in registry.identities:
            if identity.component == "q_proj":
                layer = model.base_model.model.get_submodule(identity.module_name)
                layer._final_lora_rank_update(4, "default")
        # A rank-0 module of the plan was (wrongly) finalized: freeze its weight.
        model.base_model.model.get_submodule("model.layers.1.q_proj").lora_weight["default"].requires_grad_(False)
        plan = make_plan(
            [("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0), ("model.layers.1.q_proj", 0.0)]
        )
        report = verify_applied_plan(model, plan, adapter_name="default")
        self.assertFalse(report.ok)
        codes = [issue.code for issue in report.errors]
        self.assertIn("rank_mismatch", codes)
        self.assertIn("zero_rank_weight_nonzero", codes)


class LoadLayerPlanTests(unittest.TestCase):
    def test_roundtrip_preserves_plan(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "layer_plan.json"
            path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
            loaded = load_layer_plan(path)
            self.assertEqual(loaded.run_id, plan.run_id)
            self.assertEqual([m.module_name for m in loaded.modules], [m.module_name for m in plan.modules])
            self.assertEqual(loaded.adapter.total_rank, plan.adapter.total_rank)
            self.assertEqual(loaded.top_k, plan.top_k)
            self.assertTrue(validate_layer_plan(loaded).ok)

    def test_unsupported_schema_rejected(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0)])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "layer_plan.json"
            document = plan.to_dict()
            document["schema_version"] = "0.9"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(CalibrationPlanError):
                load_layer_plan(path)

    def test_missing_required_field_rejected_in_strict_mode(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0)])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "layer_plan.json"
            document = plan.to_dict()
            entry = document["modules"][0]
            del entry["rank"]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(CalibrationPlanError):
                load_layer_plan(path)

    def test_structurally_invalid_document_rejected(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "layer_plan.json"
            document = plan.to_dict()
            document["modules"][0]["normalized_importance"] = 0.123
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(CalibrationValidationError):
                load_layer_plan(path)

    def test_unknown_top_level_fields_preserved(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 2.0)])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "layer_plan.json"
            document = plan.to_dict()
            document["custom_future_field"] = {"anything": [1, 2, 3]}
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = load_layer_plan(path)
            self.assertEqual(loaded.extra_fields["custom_future_field"], {"anything": [1, 2, 3]})


class FinalizeLayerPlanTests(unittest.TestCase):
    def test_finalize_requires_exactly_one_final_projection(self) -> None:
        with TemporaryDirectory() as tmp:
            model = build_tiny_peft_model()
            registry = tiny_registry(model)
            recorder = CalibrationRecorder(
                registry=registry, output_dir=Path(tmp), run_metadata=run_metadata(), top_k=5
            )
            # no projections at all
            with self.assertRaises(CalibrationPlanError):
                finalize_layer_plan(recorder, registry, top_k=5)
            recorder.record_projection(make_event(registry, projection_index=1, step=1, weights=[1.0] * 4))
            # one non-final projection
            with self.assertRaises(CalibrationPlanError):
                finalize_layer_plan(recorder, registry, top_k=5)
            recorder.close()

    def test_finalize_marks_top_k_and_includes_zero_ranks(self) -> None:
        with TemporaryDirectory() as tmp:
            model = build_tiny_peft_model()
            registry = tiny_registry(model)
            recorder = CalibrationRecorder(
                registry=registry, output_dir=Path(tmp), run_metadata=run_metadata(), top_k=2
            )
            recorder.record_projection(
                make_event(registry, projection_index=1, step=2, weights=[3.0, 1.0, 0.0, 0.0], is_final=True)
            )
            plan = finalize_layer_plan(recorder, registry, top_k=2)
            self.assertEqual(len(plan.modules), 4)  # zero-rank modules stay in the plan
            self.assertEqual(plan.adapter.selected_module_count, 2)
            self.assertEqual(plan.adapter.projection_step, 2)
            self.assertEqual(len(plan.top_k_modules), 2)
            for entry in plan.modules:
                expected_scaling = entry.lora_alpha / entry.rank if entry.rank > 0 else 0.0
                self.assertEqual(entry.scaling, expected_scaling)
            # idempotent: second call returns the same plan
            self.assertIs(finalize_layer_plan(recorder, registry, top_k=2), plan)
            recorder.close()


if __name__ == "__main__":
    unittest.main()

"""Unit tests for ``simplexlora.calibration.recorder``."""

from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import torch

from tests.calibration_helpers import build_tiny_peft_model, make_event, run_metadata, tiny_registry

from simplexlora.calibration.models import (
    PROJECTIONS_FILENAME,
    STEPS_FILENAME,
    CalibrationRecorderError,
    WarningCode,
)
from simplexlora.calibration.recorder import CalibrationRecorder, CalibrationSession


def make_recorder(output_dir: Path, *, overwrite: bool = False) -> tuple[CalibrationRecorder, Any]:
    model = build_tiny_peft_model()
    registry = tiny_registry(model)
    recorder = CalibrationRecorder(
        registry=registry,
        output_dir=output_dir,
        run_metadata=run_metadata(),
        top_k=5,
        overwrite=overwrite,
    )
    return recorder, model


def _fake_optimizer(model) -> torch.optim.Optimizer:
    """A minimal optimizer carrying the weight_params group (no observer)."""
    weight_params = [
        parameter for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_weight" in name
    ]
    return torch.optim.SGD(
        [
            {"params": [], "name": "loraAB"},
            {"params": [], "name": "other_params"},
            {"params": weight_params, "name": "weight_params"},
        ],
        lr=0.0,
    )


class CalibrationRecorderStepTests(unittest.TestCase):
    def test_record_step_once_per_optimizer_step(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            recorder.record_step(
                step=1,
                epoch=None,
                loss=1.5,
                learning_rate=1e-3,
                lora_grad_norm=0.5,
                weight_grad_norm=0.1,
                elapsed_seconds=0.2,
                is_projection_step=False,
            )
            with self.assertRaises(CalibrationRecorderError):
                recorder.record_step(
                    step=1,
                    epoch=None,
                    loss=1.0,
                    learning_rate=1e-3,
                    lora_grad_norm=0.5,
                    weight_grad_norm=0.1,
                    elapsed_seconds=0.2,
                    is_projection_step=False,
                )
            self.assertEqual(len(recorder.step_records), 1)
            recorder.close()

    def test_steps_jsonl_is_flushed_per_step(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            for step in range(1, 4):
                recorder.record_step(
                    step=step,
                    epoch=None,
                    loss=float(step),
                    learning_rate=1e-3,
                    lora_grad_norm=0.1 * step,
                    weight_grad_norm=0.0,
                    elapsed_seconds=0.0,
                    is_projection_step=step % 2 == 0,
                )
                content = (Path(tmp) / STEPS_FILENAME).read_text(encoding="utf-8")
                self.assertEqual(content.count("\n"), step)
            recorder.close()

    def test_unavailable_loss_records_null_and_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            recorder.record_step(
                step=1,
                epoch=None,
                loss=None,
                learning_rate=1e-3,
                lora_grad_norm=0.5,
                weight_grad_norm=0.1,
                elapsed_seconds=0.1,
                is_projection_step=False,
            )
            self.assertIn(WarningCode.LOSS_UNAVAILABLE.value, [warning.code for warning in recorder.warnings])
            line = json.loads((Path(tmp) / STEPS_FILENAME).read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(line["loss"])
            recorder.close()

    def test_non_finite_values_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            with self.assertRaises(CalibrationRecorderError):
                recorder.record_step(
                    step=1,
                    epoch=None,
                    loss=float("nan"),
                    learning_rate=1e-3,
                    lora_grad_norm=0.5,
                    weight_grad_norm=0.1,
                    elapsed_seconds=0.1,
                    is_projection_step=False,
                )
            recorder.close()


class CalibrationRecorderProjectionTests(unittest.TestCase):
    def test_record_projection_once_per_index(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            event = make_event(recorder.registry, projection_index=1, step=1, weights=[1.0, 1.0, 1.0, 1.0])
            recorder.record_projection(event)
            with self.assertRaises(CalibrationRecorderError):
                recorder.record_projection(event)
            self.assertEqual(len(recorder.snapshots), 1)
            recorder.close()

    def test_rank_formula_is_validated(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            event = make_event(recorder.registry, projection_index=1, step=1, weights=[1.0, 1.0, 1.0, 1.0])
            bad = dataclasses.replace(event, ranks=(2, 4, 4, 4))
            with self.assertRaises(CalibrationRecorderError):
                recorder.record_projection(bad)
            recorder.close()

    def test_selected_mask_mismatch_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            event = make_event(recorder.registry, projection_index=1, step=1, weights=[2.0, 1.0, 1.0, 0.0])
            bad = dataclasses.replace(event, selected_mask=(True, True, True, True))
            with self.assertRaises(CalibrationRecorderError):
                recorder.record_projection(bad)
            recorder.close()

    def test_wrong_array_length_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            event = make_event(recorder.registry, projection_index=1, step=1, weights=[1.0, 1.0, 1.0, 1.0])
            bad = dataclasses.replace(event, weights_before_step=event.weights_before_step[:2])
            with self.assertRaises(CalibrationRecorderError):
                recorder.record_projection(bad)
            recorder.close()

    def test_projections_jsonl_written_with_per_layer_records(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            recorder.record_projection(
                make_event(recorder.registry, projection_index=1, step=1, weights=[1.0, 1.0, 1.0, 1.0])
            )
            lines = (Path(tmp) / PROJECTIONS_FILENAME).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            document = json.loads(lines[0])
            self.assertEqual(len(document["layers"]), 4)
            self.assertEqual(document["schema_version"], "1.0")
            self.assertIn("metrics", document)
            recorder.close()

    def test_previous_ranks_and_rank_delta(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            recorder.record_projection(
                make_event(recorder.registry, projection_index=1, step=1, weights=[1.0, 1.0, 1.0, 1.0])
            )
            recorder.record_projection(
                make_event(recorder.registry, projection_index=2, step=2, weights=[2.0, 1.0, 1.0, 0.0])
            )
            second = recorder.snapshots[1]
            self.assertEqual([record.previous_rank for record in second.records], [4, 4, 4, 4])
            self.assertEqual([record.rank_delta for record in second.records], [4, 0, 0, -4])
            self.assertEqual([record.rank for record in second.records], [8, 4, 4, 0])
            recorder.close()

    def test_csv_order_is_deterministic(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            # Record projections out of "natural" module order to prove sorting.
            recorder.record_projection(
                make_event(recorder.registry, projection_index=2, step=2, weights=[1.0, 1.0, 1.0, 1.0])
            )
            recorder.record_projection(
                make_event(recorder.registry, projection_index=1, step=1, weights=[1.0, 1.0, 1.0, 1.0])
            )
            text = recorder.csv_text()
            lines = text.splitlines()
            self.assertEqual(
                lines[0],
                "projection_index,step,module_name,layer_index,component,weight_before,"
                "weight_after,weight_projected,normalized_importance,rank,previous_rank,"
                "rank_delta,selected,lora_alpha,scaling",
            )
            # rows sorted by (projection_index, layer_index, component, module_name)
            keys = [tuple(line.split(",")[index] for index in (0, 3, 4, 2)) for line in lines[1:]]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(keys[0][0], "1")
            # the projection recorded second (index 1) still has initial previous_rank
            row_proj1 = [line for line in lines[1:] if line.split(",")[0] == "1"][0]
            self.assertEqual(row_proj1.split(",")[10], "4")
            recorder.close()

    def test_final_projection_freezes_plan_and_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, model = make_recorder(Path(tmp))
            recorder.record_projection(
                make_event(
                    recorder.registry,
                    projection_index=1,
                    step=2,
                    weights=[2.0, 1.0, 1.0, 0.0],
                    is_final=True,
                )
            )
            session = CalibrationSession(
                model=model,
                optimizer=_fake_optimizer(model),
                registry=recorder.registry,
                recorder=recorder,
                adapter_name="default",
                top_k=5,
                enabled=True,
            )
            session.on_final_projection()
            plan = recorder.layer_plan
            assert plan is not None
            self.assertEqual(plan.adapter.total_rank, 16)
            self.assertEqual(plan.adapter.selected_module_count, 3)
            session.on_final_projection()
            self.assertIs(recorder.layer_plan, plan)
            recorder.close()

    def test_final_rank_update_verification_failure_marks_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, model = make_recorder(Path(tmp))
            recorder.record_projection(
                make_event(
                    recorder.registry,
                    projection_index=1,
                    step=2,
                    weights=[2.0, 1.0, 1.0, 0.0],
                    is_final=True,
                )
            )
            session = CalibrationSession(
                model=model,
                optimizer=_fake_optimizer(model),
                registry=recorder.registry,
                recorder=recorder,
                adapter_name="default",
                top_k=5,
                enabled=True,
            )
            session.on_final_projection()
            # Tamper with the model so verify_applied_plan must fail.
            model.base_model.model.model.layers[0]["q_proj"].lora_weight["default"].data.fill_(1.0)
            model.base_model.model.model.layers[0]["q_proj"].lora_weight["default"].requires_grad_(False)
            from simplexlora.calibration.models import CalibrationValidationError

            with self.assertRaises(CalibrationValidationError):
                session.on_final_rank_update()
            self.assertIn(
                WarningCode.PLAN_VERIFICATION_FAILED.value,
                [warning.code for warning in recorder.warnings],
            )
            self.assertFalse(recorder.manifest.extra.get("handoff_usable", True))
            recorder.close()


class RecorderArtifactPolicyTests(unittest.TestCase):
    def test_atomic_write_preserves_previous_file_on_injected_error(self) -> None:
        from simplexlora.calibration.models import CalibrationArtifactError, atomic_write_text

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.txt"
            atomic_write_text(path, "original")
            with mock.patch("os.replace", side_effect=OSError("boom")):
                with self.assertRaises(CalibrationArtifactError):
                    atomic_write_text(path, "replacement", overwrite=True)
            self.assertEqual(path.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_rerun_into_completed_directory_requires_overwrite(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, _ = make_recorder(Path(tmp))
            recorder.close()
            with self.assertRaises(CalibrationRecorderError):
                make_recorder(Path(tmp))
            # overwrite resets all artifact streams instead of appending
            recorder2, _ = make_recorder(Path(tmp), overwrite=True)
            recorder2.close()
            model = build_tiny_peft_model()
            registry = tiny_registry(model)
            recorder3 = CalibrationRecorder(
                registry=registry,
                output_dir=Path(tmp),
                run_metadata=run_metadata(),
                top_k=5,
                overwrite=True,
            )
            self.assertEqual((Path(tmp) / STEPS_FILENAME).read_text(encoding="utf-8"), "")
            self.assertEqual((Path(tmp) / PROJECTIONS_FILENAME).read_text(encoding="utf-8"), "")
            recorder3.close()

    def test_json_never_contains_nan_or_infinity(self) -> None:
        from simplexlora.calibration.models import dump_json

        with self.assertRaises(Exception):
            dump_json({"value": float("nan")})
        with self.assertRaises(Exception):
            dump_json({"value": float("inf")})


if __name__ == "__main__":
    unittest.main()

"""Unit tests for run comparison, report rendering and CLI plumbing."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.calibration_helpers import build_tiny_peft_model, calibration_args, make_plan, tiny_registry

from simplexlora.calibration.cli import (
    add_calibration_arguments,
    deterministic_run_id,
    validate_calibration_arguments,
)
from simplexlora.calibration.compare import compare_calibration_runs
from simplexlora.calibration.handoff import build_unsloth_handoff
from simplexlora.calibration.plan import finalize_layer_plan, load_layer_plan
from simplexlora.calibration.recorder import CalibrationRecorder
from simplexlora.calibration.report import (
    render_calibration_report,
    render_console_summary,
    render_stability_report,
)


class CompareCalibrationRunsTests(unittest.TestCase):
    def _plans(self):
        specs_a = [
            ("model.layers.0.q_proj", 4.0),
            ("model.layers.0.v_proj", 2.0),
            ("model.layers.1.q_proj", 1.0),
            ("model.layers.1.v_proj", 0.0),
        ]
        specs_b = [
            ("model.layers.0.q_proj", 3.5),
            ("model.layers.0.v_proj", 2.5),
            ("model.layers.1.q_proj", 0.5),
            ("model.layers.1.v_proj", 1.0),
        ]
        plan_a = make_plan(specs_a, run_id="run-a")
        plan_b = make_plan(specs_b, run_id="run-b")
        return plan_a, plan_b

    def test_jaccard_and_frequencies(self) -> None:
        plan_a, plan_b = self._plans()
        report = compare_calibration_runs([plan_a, plan_b], top_k=3)
        # top-3 of A: q0, v0, q1 (v1 has rank 0); top-3 of B: q0, v0, v1
        self.assertEqual(
            set(report.top_k_intersection),
            {"model.layers.0.q_proj", "model.layers.0.v_proj"},
        )
        self.assertEqual(len(report.top_k_union), 4)
        self.assertAlmostEqual(report.jaccard_top_k, 2 / 4)
        stats = {item.module_name: item for item in report.module_stats}
        self.assertEqual(stats["model.layers.0.q_proj"].top_k_frequency, 2)
        self.assertEqual(stats["model.layers.1.q_proj"].top_k_frequency, 1)
        self.assertEqual(stats["model.layers.1.v_proj"].top_k_frequency, 1)
        frequencies = [item.top_k_frequency for item in report.module_stats]
        self.assertEqual(frequencies, sorted(frequencies, reverse=True))

    def test_fewer_than_two_plans_rejected(self) -> None:
        plan = make_plan([("model.layers.0.q_proj", 1.0)])
        with self.assertRaises(ValueError):
            compare_calibration_runs([plan])

    def test_incomparable_modules_rejected(self) -> None:
        plan_a = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        plan_b = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.1.q_proj", 1.0)])
        with self.assertRaises(ValueError):
            compare_calibration_runs([plan_a, plan_b])

    def test_dimension_mismatch_rejected(self) -> None:
        plan_a = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        plan_b = make_plan([("model.layers.0.q_proj", 2.0), ("model.layers.0.v_proj", 1.0)])
        wrong = type(plan_b.modules[0])(
            **{
                **{field: getattr(plan_b.modules[0], field) for field in plan_b.modules[0].__dataclass_fields__},
                "in_features": 99,
            }
        )
        tampered = type(plan_b)(
            **{
                **{field: getattr(plan_b, field) for field in plan_b.__dataclass_fields__},
                "modules": (wrong, plan_b.modules[1]),
            }
        )
        with self.assertRaises(ValueError):
            compare_calibration_runs([plan_a, tampered])

    def test_different_models_produce_warning(self) -> None:
        plan_a, plan_b = self._plans()
        other_model = type(plan_b)(
            **{
                **{field: getattr(plan_b, field) for field in plan_b.__dataclass_fields__},
                "model": type(plan_b.model)(model_id="other-model", revision="rev1", architecture="X"),
            }
        )
        report = compare_calibration_runs([plan_a, other_model])
        self.assertTrue(report.warnings)

    def test_stability_report_renders(self) -> None:
        plan_a, plan_b = self._plans()
        report = compare_calibration_runs([plan_a, plan_b])
        markdown = render_stability_report(report)
        self.assertIn("# SimplexLoRA calibration stability report", markdown)
        self.assertIn("Jaccard", markdown)


def _build_manifest_and_plan(output_dir: Path):
    """Write a complete calibration run and return (recorder, plan, handoff)."""
    from tests.calibration_helpers import make_event, run_metadata

    model = build_tiny_peft_model()
    registry = tiny_registry(model)
    recorder = CalibrationRecorder(
        registry=registry,
        output_dir=output_dir,
        run_metadata=run_metadata(),
        top_k=3,
    )
    recorder.record_projection(make_event(registry, projection_index=1, step=1, weights=[1.0, 1.0, 1.0, 1.0]))
    recorder.record_projection(
        make_event(registry, projection_index=2, step=2, weights=[3.0, 2.0, 1.0, 0.0], is_final=True)
    )
    plan = finalize_layer_plan(recorder, registry, top_k=3)
    handoff = build_unsloth_handoff(plan)
    recorder.write_artifacts(plan=plan, handoff=handoff)
    return recorder, plan, handoff


class RenderCalibrationReportTests(unittest.TestCase):
    def test_report_contains_required_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, plan, handoff = _build_manifest_and_plan(Path(tmp))
            report = render_calibration_report(
                manifest=recorder.manifest,
                snapshots=recorder.snapshots,
                plan=plan,
                handoff=handoff,
                top_k=3,
            )
            for heading in (
                "# SimplexLoRA calibration report",
                "## Run parameters",
                "## Projection metrics",
                "## Top-3 modules",
                "## Rank dynamics",
                "## Component activity",
                "## Zero-rank modules",
                "## Concentration diagnostics",
                "## Unsloth handoff",
                "## Artifacts",
            ):
                self.assertIn(heading, report)
            self.assertNotIn(str(Path(tmp).resolve()), report)  # no absolute local paths
            recorder.close()

    def test_report_without_handoff_marks_section(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, plan, _ = _build_manifest_and_plan(Path(tmp))
            report = render_calibration_report(
                manifest=recorder.manifest,
                snapshots=recorder.snapshots,
                plan=plan,
                handoff=None,
                top_k=3,
            )
            self.assertIn("Handoff was not generated", report)
            recorder.close()

    def test_top_k_in_markdown_matches_plan_json(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, plan, handoff = _build_manifest_and_plan(Path(tmp))
            report = render_calibration_report(
                manifest=recorder.manifest,
                snapshots=recorder.snapshots,
                plan=plan,
                handoff=handoff,
                top_k=3,
            )
            top_modules = [entry.module_name for entry in plan.top_k_modules]
            for name in top_modules:
                self.assertIn(f"`{name}`", report)
            loaded = load_layer_plan(Path(tmp) / "layer_plan.json")
            self.assertEqual([m.module_name for m in loaded.top_k_modules], top_modules)
            recorder.close()


class RenderConsoleSummaryTests(unittest.TestCase):
    def test_summary_lists_top_modules_and_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, plan, _ = _build_manifest_and_plan(Path(tmp))
            summary = render_console_summary(plan, top_k=3, artifact_paths=recorder.artifact_paths)
            self.assertIn("active modules: 3 / 4", summary)
            self.assertIn("total rank budget:", summary)
            self.assertIn("model.layers.0.q_proj", summary)
            self.assertIn("full report:", summary)
            self.assertIn("unsloth handoff:", summary)
            recorder.close()

    def test_uncompleted_calibration_returns_explicit_message(self) -> None:
        summary = render_console_summary(None)
        self.assertIn("did not complete", summary)
        self.assertNotIn("weight=", summary)

    def test_summary_does_not_contain_full_matrix(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder, plan, _ = _build_manifest_and_plan(Path(tmp))
            summary = render_console_summary(plan, top_k=3)
            self.assertEqual(summary.count("weight="), 3)
            recorder.close()


class CalibrationCliArgumentsTests(unittest.TestCase):
    def _parser(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--fat-step", type=int, default=10)
        parser.add_argument("--max-fat-steps", type=int, default=5)
        parser.add_argument("--max-steps", type=int, default=-1)
        add_calibration_arguments(parser)
        return parser

    def test_defaults_are_valid(self) -> None:
        args = self._parser().parse_args([])
        validate_calibration_arguments(args)

    def test_calibration_only_with_wrong_max_steps_rejected(self) -> None:
        args = self._parser().parse_args(["--calibration-only", "--max-steps", "999"])
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)

    def test_calibration_only_matching_max_steps_accepted(self) -> None:
        args = self._parser().parse_args(["--calibration-only", "--max-steps", "50"])
        validate_calibration_arguments(args)

    def test_calibration_only_requires_artifacts(self) -> None:
        args = self._parser().parse_args(["--calibration-only", "--calibration-no-artifacts"])
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)

    def test_bad_rank_floor_rejected(self) -> None:
        args = self._parser().parse_args(["--calibration-rank-floor", "0"])
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)

    def test_bad_min_weight_rejected(self) -> None:
        args = self._parser().parse_args(["--calibration-min-weight", "1.5"])
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)

    def test_bad_top_k_rejected(self) -> None:
        args = self._parser().parse_args(["--calibration-top-k", "0"])
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)

    def test_zero_fat_step_rejected(self) -> None:
        args = self._parser().parse_args(["--fat-step", "0"])
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)

    def test_zero_max_fat_steps_rejected(self) -> None:
        args = self._parser().parse_args(["--max-fat-steps", "0"])
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)

    def test_output_dir_in_sources_rejected(self) -> None:
        args = calibration_args(calibration_output_dir="src/simplexlora/calibration")
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)

    def test_explicit_run_id_validated(self) -> None:
        args = self._parser().parse_args(["--calibration-run-id", "bad/id!"])
        with self.assertRaises(ValueError):
            validate_calibration_arguments(args)
        args = self._parser().parse_args(["--calibration-run-id", "ok-run_id.1"])
        validate_calibration_arguments(args)

    def test_deterministic_run_id_is_stable(self) -> None:
        first = deterministic_run_id(
            seed=1,
            model_id="m",
            model_revision="r",
            adapter_name="default",
            default_lora_rank=16,
            lora_alpha=32,
            fat_step=10,
            max_fat_steps=5,
            weight_lr=0.001,
            weight_decay=0.01,
        )
        second = deterministic_run_id(
            seed=1,
            model_id="m",
            model_revision="r",
            adapter_name="default",
            default_lora_rank=16,
            lora_alpha=32,
            fat_step=10,
            max_fat_steps=5,
            weight_lr=0.001,
            weight_decay=0.01,
        )
        changed = deterministic_run_id(
            seed=2,
            model_id="m",
            model_revision="r",
            adapter_name="default",
            default_lora_rank=16,
            lora_alpha=32,
            fat_step=10,
            max_fat_steps=5,
            weight_lr=0.001,
            weight_decay=0.01,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertTrue(first.startswith("calib-"))


if __name__ == "__main__":
    unittest.main()

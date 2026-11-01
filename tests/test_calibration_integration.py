"""Integration tests: tiny causal LM, full calibration pipeline on CPU (spec 15.2)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import AutoModelForCausalLM

from simplexlora import optimizers
from simplexlora.calibration.cli import build_run_metadata
from simplexlora.calibration.models import RunStatus, StopReason
from simplexlora.calibration.plan import load_layer_plan, validate_layer_plan, verify_applied_plan
from simplexlora.calibration.recorder import create_calibration_session
from simplexlora.calibration.trainer import SimplexCalibrationTrainer
from simplexlora.peft_plugin import SimplexLoraLinear, get_simplex_peft_model, make_simplex_lora_config

from tests.calibration_helpers import calibration_args

TARGET_MODULES = ["q_proj", "v_proj", "o_proj"]


def build_tiny_causal_lm(seed: int = 0):
    """Tiny Qwen2 built entirely from a local config — no downloads, no network."""
    from transformers import Qwen2Config

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config)
    peft_config = make_simplex_lora_config(
        r=4,
        lora_alpha=8,
        lora_dropout=0,
        target_modules=TARGET_MODULES,
    )
    return get_simplex_peft_model(model, peft_config)


def build_optimizer(model, *, fat_step: int = 1, max_fat_steps: int = 2, default_lora_rank: int = 4):
    layers = [m for _, m in model.named_modules() if isinstance(m, SimplexLoraLinear)]
    lora_ab = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    weight_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_weight" in n]
    return optimizers.SimplexAdamW(
        [
            {"params": lora_ab, "name": "loraAB"},
            {"params": [], "name": "other_params"},
            {
                "params": weight_params,
                "proj": optimizers.proj_simplex_euclidean,
                "lr": 5e-3,
                "wd": 0.0,
                "name": "weight_params",
            },
        ],
        lora_layers=layers,
        lr=1e-3,
        num_adapters=len(weight_params),
        fat_step=fat_step,
        max_fat_steps=max_fat_steps,
        default_lora_rank=default_lora_rank,
    )


def run_manual_calibration(
    output_dir: Path | None,
    *,
    seed: int,
    steps: int = 2,
    fat_step: int = 1,
    max_fat_steps: int = 2,
    with_artifacts: bool = True,
):
    """A faithful standalone calibration loop: forward/backward/step.

    ``output_dir`` is required when ``with_artifacts=True`` (artifacts must
    never land in the repository by accident).
    """
    if with_artifacts and output_dir is None:
        raise ValueError("output_dir is required when with_artifacts=True")
    model = build_tiny_causal_lm(seed=seed)
    optimizer = build_optimizer(model, fat_step=fat_step, max_fat_steps=max_fat_steps)
    args = calibration_args(
        calibration_output_dir=str(output_dir) if output_dir else "outputs/unused",
        calibration_no_artifacts=not with_artifacts,
        seed=seed,
        fat_step=fat_step,
        max_fat_steps=max_fat_steps,
        calibration_run_id=f"calib-int-{seed}",
    )
    metadata = build_run_metadata(model, adapter_name="default", args=args)
    session = create_calibration_session(model=model, optimizer=optimizer, args=args, run_metadata=metadata)

    generator = torch.Generator().manual_seed(seed + 100)
    losses = []
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randint(0, 63, (2, 8), generator=generator)
            loss = model(input_ids=inputs, labels=inputs).loss
            loss.backward()
            if with_artifacts:
                optimizer.add_microbatch_loss(loss.detach())
            losses.append(loss.item())
            optimizer.step()
        session.finalize()
    finally:
        session.close()
    return model, session, losses


def registry_names(model) -> list[str]:
    return [
        name.replace("base_model.model.", "")
        for name, module in model.named_modules()
        if isinstance(module, SimplexLoraLinear)
    ]


class FullArtifactPipelineTests(unittest.TestCase):
    def test_two_calibration_cycles_produce_full_artifact_set(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "calibration"
            model, session, losses = run_manual_calibration(output_dir, seed=3, steps=2)

            self.assertEqual(len(losses), 2)
            assert session.recorder is not None
            self.assertTrue(session.recorder.calibration_completed)

            for name in (
                "manifest.json",
                "steps.jsonl",
                "projections.jsonl",
                "calibration_layers.csv",
                "calibration_report.md",
                "layer_plan.json",
                "unsloth_handoff.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["optimizer_steps"], 2)
            self.assertEqual(len(manifest["projections"]), 2)
            self.assertEqual(manifest["projections"][-1]["is_final_projection"], True)

            # final projection's global step matches fat_step * max_fat_steps
            projections = [
                json.loads(line)
                for line in (output_dir / "projections.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(projections[-1]["step"], 1 * 2)
            self.assertTrue(projections[-1]["is_final_projection"])
            assert session.plan is not None
            self.assertEqual(session.plan.adapter.projection_step, projections[-1]["step"])

            steps = [
                json.loads(line) for line in (output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(steps), 2)
            self.assertTrue(steps[-1]["is_projection_step"])
            for record in steps:
                self.assertIsNotNone(record["loss"])

            # plan on disk validates and matches the model
            plan = load_layer_plan(output_dir / "layer_plan.json")
            self.assertTrue(validate_layer_plan(plan, model=model, adapter_name="default").ok)
            self.assertEqual(
                set(m.module_name for m in plan.modules),
                set(registry_names(model)),
            )
            # every target module is present, including rank-0 ones
            # 2 layers x (q_proj, v_proj, o_proj)
            self.assertEqual(len(plan.modules), 6)
            session.close()

    def test_plan_is_frozen_before_final_reset_and_survives_it(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "calibration"
            model, session, _ = run_manual_calibration(output_dir, seed=3, steps=2)
            plan = session.plan
            frozen = [entry.rank for entry in plan.modules]

            # _final_lora_rank_update already ran inside optimizer.step(); the
            # saved plan must be untouched by the reset.
            on_disk = load_layer_plan(output_dir / "layer_plan.json")
            self.assertEqual([entry.rank for entry in on_disk.modules], frozen)
            self.assertTrue(verify_applied_plan(model, on_disk, adapter_name="default").ok)
            session.close()

    def test_observer_does_not_change_loss_trajectory(self) -> None:
        """Identical seeds: with-observer and without-observer runs match exactly."""
        with TemporaryDirectory() as tmp:
            _, _, losses_without = run_manual_calibration(Path(tmp) / "plain", seed=7, steps=3, with_artifacts=False)
            _, _, losses_with = run_manual_calibration(Path(tmp) / "observed", seed=7, steps=3, with_artifacts=True)
        self.assertEqual(len(losses_without), len(losses_with))
        for index, (left, right) in enumerate(zip(losses_without, losses_with, strict=True)):
            self.assertAlmostEqual(left, right, places=6, msg=f"loss diverged at step {index}")

    def test_two_seeds_compare_via_stability_report(self) -> None:
        from simplexlora.calibration.compare import compare_calibration_runs
        from simplexlora.calibration.report import render_stability_report

        with TemporaryDirectory() as tmp:
            _, session_a, _ = run_manual_calibration(Path(tmp) / "a", seed=3, steps=2)
            _, session_b, _ = run_manual_calibration(Path(tmp) / "b", seed=4, steps=2)
            report = compare_calibration_runs([session_a.plan, session_b.plan], top_k=5)
            self.assertEqual(report.run_count, 2)
            self.assertTrue(0.0 <= report.jaccard_top_k <= 1.0)
            markdown = render_stability_report(report)
            self.assertIn("stability report", markdown)
            session_a.close()
            session_b.close()


class CalibrationOnlyTrainerTests(unittest.TestCase):
    def test_calibration_only_stops_at_exact_step_count(self) -> None:
        """Trainer stops exactly at fat_step * max_fat_steps optimizer steps."""
        from datasets import Dataset
        from transformers import TrainingArguments

        fat_step, max_fat_steps = 2, 2
        required_steps = fat_step * max_fat_steps

        with TemporaryDirectory() as tmp:
            model = build_tiny_causal_lm(seed=1)
            optimizer = build_optimizer(model, fat_step=fat_step, max_fat_steps=max_fat_steps)
            args = calibration_args(
                calibration_output_dir=str(Path(tmp) / "calibration"),
                seed=1,
                fat_step=fat_step,
                max_fat_steps=max_fat_steps,
                calibration_run_id="calib-only-test",
            )
            metadata = build_run_metadata(model, adapter_name="default", args=args)
            session = create_calibration_session(model=model, optimizer=optimizer, args=args, run_metadata=metadata)

            sequences = [torch.randint(0, 63, (8,)).tolist() for _ in range(8)]
            dataset = Dataset.from_dict({"input_ids": sequences, "labels": [row[:] for row in sequences]})
            training_args = TrainingArguments(
                output_dir=tmp,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                max_steps=required_steps,
                logging_steps=1,
                eval_strategy="no",
                save_strategy="no",
                report_to=[],
                seed=1,
                use_cpu=True,
            )
            trainer = SimplexCalibrationTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=None,
                optimizers=(optimizer, None),
                calibration_optimizer=optimizer,
            )
            try:
                trainer.train()
            finally:
                session.finalize()
                session.close(stop_reason=StopReason.CALIBRATION_ONLY)

            self.assertEqual(trainer.state.global_step, required_steps)
            assert session.recorder is not None
            self.assertTrue(session.recorder.calibration_completed)
            self.assertEqual(session.recorder.manifest.optimizer_steps, required_steps)
            self.assertEqual(session.recorder.manifest.status, RunStatus.COMPLETED)
            assert session.plan is not None
            self.assertEqual(session.plan.adapter.projection_step, required_steps)
            for entry in session.plan.modules:
                expected_scaling = entry.lora_alpha / entry.rank if entry.rank > 0 else 0.0
                self.assertEqual(entry.scaling, expected_scaling)

            steps = [
                json.loads(line)
                for line in (Path(tmp) / "calibration" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(steps), required_steps)
            self.assertTrue(steps[-1]["is_projection_step"])
            # Trainer used the exact microbatch loss path: every step has a loss
            for record in steps:
                self.assertIsNotNone(record["loss"])

    def test_accumulated_step_loss_matches_trainer(self) -> None:
        from datasets import Dataset
        from transformers import TrainingArguments

        with TemporaryDirectory() as tmp:
            model = build_tiny_causal_lm(seed=2)
            optimizer = build_optimizer(model, fat_step=10, max_fat_steps=1)
            args = calibration_args(
                calibration_output_dir=str(Path(tmp) / "calibration"),
                seed=2,
                fat_step=10,
                max_fat_steps=1,
                calibration_run_id="calib-ga-loss",
            )
            metadata = build_run_metadata(model, adapter_name="default", args=args)
            session = create_calibration_session(model=model, optimizer=optimizer, args=args, run_metadata=metadata)

            sequences = [torch.randint(0, 63, (8,)).tolist() for _ in range(8)]
            dataset = Dataset.from_dict({"input_ids": sequences, "labels": [row[:] for row in sequences]})
            training_args = TrainingArguments(
                output_dir=tmp,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                max_steps=1,
                logging_steps=1,
                eval_strategy="no",
                save_strategy="no",
                report_to=[],
                seed=2,
                use_cpu=True,
            )
            trainer = SimplexCalibrationTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=None,
                optimizers=(optimizer, None),
                calibration_optimizer=optimizer,
            )
            try:
                trainer.train()
            finally:
                session.close()

            logged = [row["loss"] for row in trainer.state.log_history if "loss" in row][-1]
            recorded = json.loads(
                (Path(tmp) / "calibration" / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )["loss"]
            self.assertAlmostEqual(recorded, logged, places=6)

    def test_no_artifacts_mode_creates_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "calibration"
            model, session, losses = run_manual_calibration(base, seed=7, steps=2, with_artifacts=False)
            self.assertFalse(session.enabled)
            self.assertIsNone(session.recorder)
            self.assertFalse(base.exists())
            self.assertEqual(len(losses), 2)
            session.close()


class LocalQwenRealModelTests(unittest.TestCase):
    """Calibration pipeline on the locally cached Qwen2.5-Coder-0.5B-Instruct.

    Strictly offline: ``HF_HUB_OFFLINE=1`` and ``local_files_only=True``.
    Skipped when the model is not present in the local HF cache.
    """

    MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def setUpClass(cls) -> None:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cache_root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
        repo_dir = cache_root / "models--Qwen--Qwen2.5-Coder-0.5B-Instruct"
        ref = repo_dir / "refs" / "main"
        if not ref.is_file():
            raise unittest.SkipTest(f"local cache for {cls.MODEL_ID} is missing")
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = repo_dir / "snapshots" / revision
        if not (snapshot / "model.safetensors").is_file():
            raise unittest.SkipTest("model.safetensors is not cached locally")
        cls.snapshot = snapshot
        cls.revision = revision

    def _build_model(self, seed: int = 0):
        from transformers import AutoModelForCausalLM

        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(
            self.snapshot,
            local_files_only=True,
            dtype=torch.float32,
            trust_remote_code=False,
        )
        model.config.use_cache = False
        peft_config = make_simplex_lora_config(
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=self.TARGET_MODULES,
        )
        return get_simplex_peft_model(model, peft_config)

    def test_full_calibration_on_local_qwen(self) -> None:

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "calibration"
            model = self._build_model()
            optimizer = build_optimizer(
                model,
                fat_step=1,
                max_fat_steps=2,
                default_lora_rank=8,
            )
            args = calibration_args(
                calibration_output_dir=str(output_dir),
                seed=42,
                fat_step=1,
                max_fat_steps=2,
                lora_r=8,
                lora_alpha=16,
                weight_lr=5e-3,
                weight_decay=0.0,
                model_id=self.MODEL_ID,
                calibration_run_id="calib-qwen-local",
            )
            metadata = build_run_metadata(
                model,
                adapter_name="default",
                args=args,
                data={"dataset": "local-integration"},
                device=next(model.parameters()).device,
            )
            session = create_calibration_session(model=model, optimizer=optimizer, args=args, run_metadata=metadata)

            generator = torch.Generator().manual_seed(42)
            try:
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    inputs = torch.randint(0, 150000, (1, 16), generator=generator)
                    labels = inputs.clone()
                    loss = model(input_ids=inputs, labels=labels).loss
                    loss.backward()
                    optimizer.add_microbatch_loss(loss.detach())
                    optimizer.step()
                session.finalize()
            finally:
                session.close()

            # all artifacts exist
            for name in (
                "manifest.json",
                "steps.jsonl",
                "projections.jsonl",
                "calibration_layers.csv",
                "calibration_report.md",
                "layer_plan.json",
                "unsloth_handoff.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["model"]["model_id"], self.MODEL_ID)

            # plan: every target module of the Qwen model is present, rank 0 included
            plan = load_layer_plan(output_dir / "layer_plan.json")
            expected_modules = len(registry_names(model))
            self.assertEqual(expected_modules % len(self.TARGET_MODULES), 0)
            self.assertEqual(len(plan.modules), expected_modules)
            module_names = {module.module_name for module in plan.modules}
            self.assertTrue(validate_layer_plan(plan, model=model, adapter_name="default").ok)
            self.assertTrue(verify_applied_plan(model, plan, adapter_name="default").ok)

            # per-layer CSV: 2 projections x all modules
            csv_lines = (output_dir / "calibration_layers.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(csv_lines) - 1, 2 * len(plan.modules))

            # handoff references only positive ranks
            handoff = json.loads((output_dir / "unsloth_handoff.json").read_text(encoding="utf-8"))
            self.assertTrue(handoff["target_modules"])
            for module_name in handoff["target_modules"]:
                self.assertFalse(module_name.startswith("^"))
                self.assertIn(module_name, set(module_names))
            self.assertEqual(handoff["requested_rank_total"], plan.adapter.total_rank)


if __name__ == "__main__":
    unittest.main()

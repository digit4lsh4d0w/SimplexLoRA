"""CLI plumbing: argument definitions, validation and run metadata assembly."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from simplexlora.calibration.models import (
    CalibrationRunMetadata,
    collect_library_versions,
    utc_now_iso,
    validate_run_id,
)

DEFAULT_CALIBRATION_OUTPUT_DIR = None


def calibration_output_dir(args: Any) -> Path:
    """Resolve the calibration directory: ``<output-dir>/calibration`` by default."""
    explicit = getattr(args, "calibration_output_dir", None)
    if explicit:
        return Path(explicit)
    output_dir = getattr(args, "output_dir", None)
    return Path(output_dir) / "calibration" if output_dir else Path("outputs") / "calibration"


def add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the standard calibration arguments to an existing parser."""
    parser.add_argument(
        "--calibration-output-dir",
        default=DEFAULT_CALIBRATION_OUTPUT_DIR,
        help="directory for calibration artifacts (default: <output-dir>/calibration)",
    )
    parser.add_argument(
        "--calibration-top-k",
        type=int,
        default=20,
        help="size of the Top-K reported in artifacts and warnings",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="stop right after fat_step * max_fat_steps optimizer steps, after the final projection",
    )
    parser.add_argument(
        "--calibration-no-artifacts",
        action="store_true",
        help="disable calibration observation entirely (no recorder, no JSONL, no observer)",
    )
    parser.add_argument(
        "--calibration-rank-floor",
        type=int,
        default=1,
        help="minimum applied rank in the Unsloth handoff",
    )
    parser.add_argument(
        "--calibration-alpha-mode",
        choices=("preserve", "twice_rank"),
        default="preserve",
        help="alpha policy of the Unsloth handoff",
    )
    parser.add_argument(
        "--calibration-min-weight",
        type=float,
        default=0.0,
        help="minimum normalized importance for a module to enter the handoff",
    )
    parser.add_argument(
        "--calibration-unsloth-top-k",
        type=int,
        default=None,
        help="optional Top-K restriction of the Unsloth handoff",
    )
    parser.add_argument(
        "--calibration-overwrite",
        action="store_true",
        help="allow rerunning into a directory that already holds calibration artifacts",
    )
    parser.add_argument(
        "--calibration-run-id",
        default=None,
        help="explicit run id (otherwise derived deterministically from seed/model/optimizer)",
    )


def validate_calibration_arguments(args: argparse.Namespace) -> None:
    """Reject incompatible calibration configurations before loading any model."""
    calibration_only = bool(getattr(args, "calibration_only", False))
    no_artifacts = bool(getattr(args, "calibration_no_artifacts", False))
    if no_artifacts and calibration_only:
        raise ValueError(
            "--calibration-only requires artifacts and cannot be combined with --calibration-no-artifacts"
        )

    fat_step = int(getattr(args, "fat_step", 1))
    raw_max_fat_steps = getattr(args, "max_fat_steps", None)
    if raw_max_fat_steps is None:
        raw_max_fat_steps = getattr(args, "mfs", None)
    max_fat_steps = int(raw_max_fat_steps) if raw_max_fat_steps is not None else 1
    if not no_artifacts:
        if fat_step <= 0:
            raise ValueError(f"fat_step must be > 0, got {fat_step}")
        if max_fat_steps <= 0:
            raise ValueError(f"max_fat_steps must be > 0 when calibration is enabled, got {max_fat_steps}")

    top_k = int(getattr(args, "calibration_top_k", 20))
    if top_k <= 0:
        raise ValueError(f"--calibration-top-k must be > 0, got {top_k}")

    rank_floor = int(getattr(args, "calibration_rank_floor", 1))
    if rank_floor < 1:
        raise ValueError(f"--calibration-rank-floor must be >= 1, got {rank_floor}")

    min_weight = float(getattr(args, "calibration_min_weight", 0.0))
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(f"--calibration-min-weight must be within [0, 1], got {min_weight}")

    unsloth_top_k = getattr(args, "calibration_unsloth_top_k", None)
    if unsloth_top_k is not None and int(unsloth_top_k) < 1:
        raise ValueError(f"--calibration-unsloth-top-k must be >= 1, got {unsloth_top_k}")

    if calibration_only:
        required_steps = fat_step * max_fat_steps
        if required_steps <= 0:
            raise ValueError(
                "--calibration-only requires fat_step * max_fat_steps to be positive, "
                f"got {fat_step} * {max_fat_steps}"
            )
        explicit_max_steps = int(getattr(args, "max_steps", -1))
        if explicit_max_steps > 0 and explicit_max_steps != required_steps:
            raise ValueError(
                f"--calibration-only runs exactly fat_step * max_fat_steps = {required_steps} "
                f"optimizer steps and is incompatible with an explicit --max-steps of {explicit_max_steps}"
            )

    output_dir = calibration_output_dir(args)
    if _points_into_project_sources(output_dir):
        raise ValueError(f"--calibration-output-dir must not point into project sources, got {output_dir}")

    run_id = getattr(args, "calibration_run_id", None)
    if run_id is not None:
        validate_run_id(str(run_id))


def _points_into_project_sources(output_dir: Path) -> bool:
    """The output directory must not overlap the project's own sources."""
    try:
        resolved = output_dir.resolve()
    except OSError:  # pragma: no cover - defensive
        return False
    cwd = Path.cwd().resolve()
    if resolved == cwd:
        return True
    src = (cwd / "src").resolve()
    if resolved == src or src in resolved.parents:
        return True
    tests = (cwd / "tests").resolve()
    if resolved == tests:
        return True
    return False


def deterministic_run_id(
    *,
    seed: int,
    model_id: str,
    model_revision: str,
    adapter_name: str,
    default_lora_rank: int,
    lora_alpha: int,
    fat_step: int,
    max_fat_steps: int,
    weight_lr: float,
    weight_decay: float,
) -> str:
    """Stable run id derived from everything that changes the calibration result."""
    payload = "|".join(
        str(part)
        for part in (
            seed,
            model_id,
            model_revision,
            adapter_name,
            default_lora_rank,
            lora_alpha,
            fat_step,
            max_fat_steps,
            weight_lr,
            weight_decay,
        )
    )
    return "calib-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_run_metadata(
    model: Any,
    *,
    adapter_name: str,
    args: argparse.Namespace,
    data: dict[str, Any] | None = None,
    device: Any = None,
) -> CalibrationRunMetadata:
    """Assemble the versioned run metadata for a calibration session."""
    explicit_run_id = getattr(args, "calibration_run_id", None)
    seed = int(getattr(args, "seed", 0))
    model_id = str(getattr(args, "model_id", getattr(model, "name_or_path", "unknown")))
    model_revision = str(
        getattr(getattr(model, "config", None), "_commit_hash", None) or getattr(args, "model_revision", "") or ""
    )
    architecture = type(model).__name__
    weight_lr = float(
        getattr(args, "weight_lr", None) or getattr(args, "learning_rate_w", None) or getattr(args, "lr_w", 0.0)
    )
    weight_decay = float(
        getattr(args, "weight_decay", None) or getattr(args, "weight_decay_w", None) or getattr(args, "wd_w", 0.0)
    )
    fat_step = int(getattr(args, "fat_step", 1))
    raw_max_fat_steps = getattr(args, "max_fat_steps", None)
    if raw_max_fat_steps is None:
        raw_max_fat_steps = getattr(args, "mfs", None)
    max_fat_steps = int(raw_max_fat_steps) if raw_max_fat_steps is not None else 1
    default_lora_rank = int(getattr(args, "lora_r", getattr(args, "lora_rank", 0)))
    lora_alpha = int(getattr(args, "lora_alpha", 0))

    peft_config: dict[str, Any] = {}
    peft_configs = getattr(model, "peft_config", None)
    if peft_configs and adapter_name in peft_configs:
        peft_config = peft_configs[adapter_name]
        peft_config = (
            {
                key: (value if isinstance(value, (str, int, float, bool, list, tuple, type(None))) else str(value))
                for key, value in sorted(vars(peft_config).items())
            }
            if hasattr(peft_config, "__dict__")
            else {}
        )

    if explicit_run_id is None:
        explicit_run_id = deterministic_run_id(
            seed=seed,
            model_id=model_id,
            model_revision=model_revision,
            adapter_name=adapter_name,
            default_lora_rank=default_lora_rank,
            lora_alpha=lora_alpha,
            fat_step=fat_step,
            max_fat_steps=max_fat_steps,
            weight_lr=weight_lr,
            weight_decay=weight_decay,
        )
    validate_run_id(str(explicit_run_id))

    return CalibrationRunMetadata(
        run_id=str(explicit_run_id),
        created_at=utc_now_iso(),
        model_id=model_id,
        model_revision=model_revision,
        architecture=architecture,
        device=str(getattr(device, "type", device)) if device is not None else "unknown",
        adapter_name=adapter_name,
        peft_config=peft_config,
        default_lora_rank=default_lora_rank,
        lora_alpha=lora_alpha,
        fat_step=fat_step,
        max_fat_steps=max_fat_steps,
        weight_lr=weight_lr,
        weight_decay=weight_decay,
        seed=seed,
        effective_batch_size=int(
            getattr(args, "effective_batch_size", 0)
            or (
                int(getattr(args, "batch_size", 1))
                * int(
                    getattr(
                        args,
                        "gradient_accumulation_steps",
                        getattr(args, "grad_acc_steps", 1),
                    )
                )
            )
        ),
        gradient_accumulation_steps=int(
            getattr(args, "gradient_accumulation_steps", getattr(args, "grad_acc_steps", 1))
        ),
        library_versions=collect_library_versions(device),
        data=dict(data or {}),
    )


def handoff_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Extract handoff policies from parsed CLI arguments."""
    return {
        "rank_floor": int(getattr(args, "calibration_rank_floor", 1)),
        "min_weight": float(getattr(args, "calibration_min_weight", 0.0)),
        "top_k": getattr(args, "calibration_unsloth_top_k", None),
        "alpha_mode": str(getattr(args, "calibration_alpha_mode", "preserve")),
    }

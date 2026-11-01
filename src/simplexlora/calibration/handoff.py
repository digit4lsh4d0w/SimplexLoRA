"""Unsloth/PEFT handoff configuration derived from a validated layer plan."""

from __future__ import annotations

import re
from typing import Literal

from simplexlora.calibration.models import (
    SCHEMA_VERSION,
    CalibrationWarning,
    LayerPlan,
    UnslothHandoff,
    WarningCode,
)
from simplexlora.calibration.plan import validate_layer_plan


def _anchored_regex(module_name: str) -> str:
    """Build an unambiguous, escaped PEFT target-module regex."""
    return "^" + re.escape(module_name)


def _suffix_pattern(module_name: str) -> str:
    """Escaped pattern key for ``rank_pattern``/``alpha_pattern`` (suffix matched)."""
    return re.escape(module_name)


def build_unsloth_handoff(
    plan: LayerPlan,
    *,
    rank_floor: int = 1,
    min_weight: float = 0.0,
    top_k: int | None = None,
    alpha_mode: Literal["preserve", "twice_rank"] = "preserve",
) -> UnslothHandoff:
    """Convert a layer plan into a plain LoRA configuration for Unsloth/PEFT.

    Does not import Unsloth and does not claim that Unsloth accepted anything;
    acceptance is verified by an integration test. All deviations from the source
    plan (floor, top-k, min weight, alpha policy) are reported explicitly.
    """
    if alpha_mode not in ("preserve", "twice_rank"):
        raise ValueError(f"alpha_mode must be 'preserve' or 'twice_rank', got {alpha_mode!r}")
    if rank_floor < 1:
        raise ValueError(f"rank_floor must be >= 1, got {rank_floor}")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(f"min_weight must be within [0, 1], got {min_weight}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")

    # A damaged or incomplete plan must never become a downstream handoff.
    validate_layer_plan(plan, strict=True)

    warnings: list[CalibrationWarning] = []

    candidates = [entry for entry in plan.modules if entry.rank > 0]
    requested_rank_total = sum(entry.rank for entry in candidates)

    included = [entry for entry in candidates if entry.normalized_importance >= min_weight]
    excluded_by_min_weight = [entry for entry in candidates if entry.normalized_importance < min_weight]
    if excluded_by_min_weight:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.HANDOFF_MIN_WEIGHT,
                message=f"{len(excluded_by_min_weight)} module(s) excluded by min_weight={min_weight}",
            )
        )

    if top_k is not None:
        ranked = sorted(included, key=lambda entry: (-entry.normalized_importance, entry.module_name))
        kept = {entry.module_name for entry in ranked[:top_k]}
        excluded_by_top_k = [entry for entry in included if entry.module_name not in kept]
        included = [entry for entry in included if entry.module_name in kept]
        if excluded_by_top_k:
            warnings.append(
                CalibrationWarning(
                    code=WarningCode.HANDOFF_TOP_K,
                    message=f"{len(excluded_by_top_k)} module(s) excluded by top_k={top_k}",
                )
            )

    floored = [entry for entry in included if entry.rank < rank_floor]
    if floored:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.HANDOFF_RANK_FLOOR,
                message=f"{len(floored)} module(s) raised to rank_floor={rank_floor}",
            )
        )

    if alpha_mode == "twice_rank":
        warnings.append(
            CalibrationWarning(
                code=WarningCode.HANDOFF_ALPHA_CHANGED,
                message="alpha_mode='twice_rank' changes scaling of every module "
                "(alpha = 2 * applied rank instead of the calibrated lora_alpha)",
            )
        )

    target_modules: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    applied_rank_total = 0
    estimated_trainable_parameters = 0
    excluded_modules: list[str] = []

    included_names = {entry.module_name for entry in included}
    for entry in plan.modules:
        if entry.module_name in included_names:
            applied_rank = max(entry.rank, rank_floor)
            alpha = entry.lora_alpha if alpha_mode == "preserve" else 2 * applied_rank
            target_modules.append(_anchored_regex(entry.module_name))
            rank_pattern[_suffix_pattern(entry.module_name)] = applied_rank
            alpha_pattern[_suffix_pattern(entry.module_name)] = alpha
            applied_rank_total += applied_rank
            estimated_trainable_parameters += applied_rank * (entry.in_features + entry.out_features)
        else:
            # rank-0 modules and modules excluded by min_weight/top_k are all
            # reported explicitly so the downstream user sees the full picture.
            excluded_modules.append(entry.module_name)

    selection_policy = (
        "rank>0"
        + (f",min_weight>={min_weight}" if min_weight > 0 else "")
        + (f",top_k={top_k}" if top_k is not None else "")
        + f",rank_floor={rank_floor}"
        + f",alpha_mode={alpha_mode}"
    )

    return UnslothHandoff(
        schema_version=SCHEMA_VERSION,
        source_plan_run_id=plan.run_id,
        selection_policy=selection_policy,
        alpha_mode=alpha_mode,
        rank_floor=rank_floor,
        min_weight=min_weight,
        top_k=top_k,
        target_modules=tuple(target_modules),
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        excluded_modules=tuple(sorted(excluded_modules)),
        requested_rank_total=requested_rank_total,
        applied_rank_total=applied_rank_total,
        estimated_trainable_parameters=estimated_trainable_parameters,
        warnings=tuple(warnings),
    )

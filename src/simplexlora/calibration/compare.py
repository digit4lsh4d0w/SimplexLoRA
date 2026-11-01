"""Stability comparison of several calibration runs (different seeds/subsets)."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from simplexlora.calibration.models import (
    CalibrationWarning,
    StabilityModuleStats,
    StabilityReport,
    WarningCode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from simplexlora.calibration.models import LayerPlan


def _top_k_names(plan: LayerPlan, top_k: int) -> list[str]:
    return [entry.module_name for entry in plan.top_k_modules[:top_k]]


def compare_calibration_runs(
    plans: Sequence[LayerPlan],
    *,
    top_k: int = 20,
) -> StabilityReport:
    """Compare two or more plans of the same model to assess rank stability.

    Modules are matched by canonical name; incomparable dimensions raise.
    Results are sorted by Top-K frequency, then mean rank, then module name.
    """
    if len(plans) < 2:
        raise ValueError(f"compare_calibration_runs needs at least two plans, got {len(plans)}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    model_ids = {plan.model.model_id for plan in plans}
    architectures = {plan.model.architecture for plan in plans}
    adapter_names = {plan.adapter.name for plan in plans}

    warnings: list[CalibrationWarning] = []
    if len(model_ids) > 1:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.INCOMPLETE_RUN,
                message=f"plans refer to different models: {sorted(model_ids)}; "
                "comparison is meaningless across models",
            )
        )
    if len(architectures) > 1:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.INCOMPLETE_RUN,
                message=f"plans refer to different architectures: {sorted(architectures)}",
            )
        )
    if len(adapter_names) > 1:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.INCOMPLETE_RUN,
                message=f"plans refer to different adapters: {sorted(adapter_names)}",
            )
        )

    reference_names = [entry.module_name for entry in plans[0].modules]
    reference_dimensions = {entry.module_name: (entry.in_features, entry.out_features) for entry in plans[0].modules}
    for plan in plans[1:]:
        names = [entry.module_name for entry in plan.modules]
        if names != reference_names:
            missing = sorted(set(reference_names) - set(names))
            extra = sorted(set(names) - set(reference_names))
            raise ValueError(
                f"plans have incomparable module sets (missing in second plan: {missing[:5]}, extra: {extra[:5]})"
            )
        for entry in plan.modules:
            if reference_dimensions[entry.module_name] != (entry.in_features, entry.out_features):
                raise ValueError(
                    f"dimension mismatch for module {entry.module_name!r}: "
                    f"{reference_dimensions[entry.module_name]} != {(entry.in_features, entry.out_features)}"
                )

    effective_top_k = min(top_k, len(reference_names))
    top_k_sets = [set(_top_k_names(plan, effective_top_k)) for plan in plans]
    union = sorted(set().union(*top_k_sets)) if top_k_sets else []
    intersection = sorted(set.intersection(*top_k_sets)) if top_k_sets else []
    jaccard = len(intersection) / len(union) if union else 0.0

    module_names = set(reference_names)
    for plan in plans[1:]:
        module_names &= {entry.module_name for entry in plan.modules}
    module_names |= set(union)

    stats: list[StabilityModuleStats] = []
    for name in module_names:
        ranks: list[int] = []
        importances: list[float] = []
        hits = 0
        for plan, top_set in zip(plans, top_k_sets, strict=True):
            entry = next(entry for entry in plan.modules if entry.module_name == name)
            ranks.append(entry.rank)
            importances.append(entry.normalized_importance)
            if name in top_set:
                hits += 1
        mean_rank = statistics.fmean(ranks)
        std_rank = statistics.pstdev(ranks) if len(ranks) > 1 else 0.0
        mean_importance = statistics.fmean(importances)
        std_importance = statistics.pstdev(importances) if len(importances) > 1 else 0.0
        stats.append(
            StabilityModuleStats(
                module_name=name,
                top_k_frequency=hits,
                top_k_presence=len(plans),
                mean_rank=mean_rank,
                std_rank=std_rank,
                min_rank=min(ranks),
                max_rank=max(ranks),
                mean_importance=mean_importance,
                std_importance=std_importance,
                stable=hits == len(plans) and std_rank <= 1.0,
            )
        )

    stats.sort(key=lambda item: (-item.top_k_frequency, item.mean_rank, item.module_name))

    return StabilityReport(
        run_count=len(plans),
        top_k=effective_top_k,
        model_ids=tuple(sorted(model_ids)),
        architectures=tuple(sorted(architectures)),
        adapter_names=tuple(sorted(adapter_names)),
        module_names=tuple(reference_names),
        top_k_union=tuple(union),
        top_k_intersection=tuple(intersection),
        jaccard_top_k=jaccard,
        module_stats=tuple(stats),
        warnings=tuple(warnings),
    )

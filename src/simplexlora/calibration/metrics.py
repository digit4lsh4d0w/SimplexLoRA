"""Concentration and coverage metrics for simplex weight distributions."""

from __future__ import annotations

import math
from collections.abc import Sequence

from simplexlora.calibration.models import (
    CalibrationArtifactError,
    CalibrationMetrics,
    CalibrationRecorderError,
    check_finite,
)


def _finite_sequence(values: Sequence[float], name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise CalibrationRecorderError(f"{name} must be a sequence of numbers")
    try:
        result = tuple(check_finite(value, f"{name}[{index}]") for index, value in enumerate(values))
    except CalibrationArtifactError as error:
        raise CalibrationRecorderError(str(error)) from error
    if any(value < 0 for value in result):
        raise CalibrationRecorderError(f"{name} must be non-negative")
    return result


def _rank_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise CalibrationRecorderError(f"{name} must be a sequence of integers")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CalibrationRecorderError(f"{name}[{index}] must be an int, got {value!r}")
        if value < 0:
            raise CalibrationRecorderError(f"{name}[{index}] must be non-negative, got {value!r}")
        result.append(int(value))
    return tuple(result)


def compute_calibration_metrics(
    weights: Sequence[float],
    ranks: Sequence[int],
    *,
    default_lora_rank: int,
    components: Sequence[str],
    top_k: int,
) -> CalibrationMetrics:
    """Turn a projected simplex weight distribution into comparable numbers.

    ``weights`` are the projected simplex weights, ``ranks`` the assigned ranks
    (``ceil(weight * default_lora_rank)``), ``components`` one component label
    per module, ``top_k`` the ``k`` for ``top_k_share``.

    Pure function: no model, GPU or file system access.
    """
    weight_values = _finite_sequence(weights, "weights")
    rank_values = _rank_sequence(ranks, "ranks")
    component_values = tuple(str(component) for component in components)

    if len(weight_values) != len(rank_values) or len(weight_values) != len(component_values):
        raise CalibrationRecorderError(
            f"length mismatch: weights={len(weight_values)}, ranks={len(rank_values)}, "
            f"components={len(component_values)}"
        )
    if not weight_values:
        raise CalibrationRecorderError("weights must not be empty")
    if default_lora_rank <= 0:
        raise CalibrationRecorderError(f"default_lora_rank must be positive, got {default_lora_rank}")
    if top_k < 1:
        raise CalibrationRecorderError(f"top_k must be >= 1, got {top_k}")

    total_weight = math.fsum(weight_values)
    if total_weight <= 0:
        raise CalibrationRecorderError("sum of projected weights must be positive")

    module_count = len(weight_values)
    probabilities = [weight / total_weight for weight in weight_values]

    positive = [p for p in probabilities if p > 0]
    entropy = -math.fsum(p * math.log(p) for p in positive)
    normalized_entropy = 1.0 if module_count == 1 else entropy / math.log(module_count)
    effective_module_count = math.exp(entropy) if entropy <= 700 else float(module_count)
    if effective_module_count < 1.0 and positive:
        effective_module_count = 1.0

    hhi = math.fsum(p * p for p in probabilities)
    # hhi is provably positive (probabilities sum to 1), but guard division anyway.
    inverse_hhi = 1.0 / hhi if hhi > 0 else 0.0

    ordered = sorted(probabilities, reverse=True)
    top_1_share = ordered[0] if ordered else 0.0
    top_5_share = math.fsum(ordered[: min(5, module_count)])
    top_k_share = math.fsum(ordered[: min(top_k, module_count)])

    component_shares: dict[str, float] = {}
    rank_by_component: dict[str, int] = {}
    for probability, rank, component in zip(probabilities, rank_values, component_values, strict=True):
        component_shares[component] = component_shares.get(component, 0.0) + probability
        rank_by_component[component] = rank_by_component.get(component, 0) + rank

    selected_ranks = [rank for rank in rank_values if rank > 0]
    zero_rank_count = module_count - len(selected_ranks)
    total_rank = sum(selected_ranks)
    rank_utilization = total_rank / (default_lora_rank * module_count)
    max_rank = max(rank_values) if rank_values else 0
    mean_selected_rank = total_rank / len(selected_ranks) if selected_ranks else 0.0

    return CalibrationMetrics(
        module_count=module_count,
        selected_count=len(selected_ranks),
        zero_rank_count=zero_rank_count,
        total_rank=total_rank,
        rank_utilization=rank_utilization,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        effective_module_count=effective_module_count,
        hhi=hhi,
        inverse_hhi=inverse_hhi,
        top_1_share=top_1_share,
        top_5_share=top_5_share,
        top_k_share=top_k_share,
        component_shares=component_shares,
        rank_by_component=rank_by_component,
        max_rank=max_rank,
        mean_selected_rank=mean_selected_rank,
    )

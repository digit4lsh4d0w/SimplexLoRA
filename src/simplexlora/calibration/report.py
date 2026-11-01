"""Pure Markdown and console rendering for calibration runs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from simplexlora.calibration.models import (
    CSV_FILENAME,
    LAYER_PLAN_FILENAME,
    MANIFEST_FILENAME,
    PROJECTIONS_FILENAME,
    STEPS_FILENAME,
    UNSLOTH_HANDOFF_FILENAME,
    CalibrationWarning,
    LayerPlan,
    ReportThresholds,
    StabilityReport,
    WarningCode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from simplexlora.calibration.models import (
        CalibrationManifest,
        ProjectionSnapshot,
        UnslothHandoff,
    )


def fmt_number(value: float | int | None, digits: int = 4) -> str:
    """Uniform number formatting; ``None`` renders as ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    if number != 0 and (abs(number) >= 1e5 or abs(number) < 1e-4):
        return f"{number:.3e}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".,")


def _collect_threshold_warnings(
    snapshots: Sequence[ProjectionSnapshot],
    *,
    thresholds: ReportThresholds,
    default_lora_rank: int,
    top_k: int,
) -> list[CalibrationWarning]:
    warnings: list[CalibrationWarning] = []
    if not snapshots:
        warnings.append(CalibrationWarning(code=WarningCode.INCOMPLETE_RUN, message="no projections were recorded"))
        return warnings

    final = snapshots[-1]
    final_metrics = final.metrics
    if final_metrics.top_1_share > thresholds.top_share_warning:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.HIGH_CONCENTRATION,
                message=(
                    f"top-1 normalized importance {final_metrics.top_1_share:.3f} exceeds "
                    f"{thresholds.top_share_warning}; the distribution is dominated by one module"
                ),
            )
        )
    if final_metrics.effective_module_count < thresholds.min_effective_modules:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.LOW_EFFECTIVE_MODULES,
                message=(
                    f"effective module count {final_metrics.effective_module_count:.2f} < "
                    f"{thresholds.min_effective_modules}; the importance mass is concentrated"
                ),
            )
        )
    zero_fraction = final_metrics.zero_rank_count / final_metrics.module_count
    if zero_fraction > thresholds.zero_rank_fraction_warning:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.ZERO_RANK_MAJORITY,
                message=f"{zero_fraction:.1%} of modules have zero rank",
            )
        )
    if final_metrics.max_rank > thresholds.max_rank_multiple_warning * default_lora_rank:
        warnings.append(
            CalibrationWarning(
                code=WarningCode.RANK_ABOVE_BUDGET,
                message=(
                    f"maximum rank {final_metrics.max_rank} exceeds "
                    f"{thresholds.max_rank_multiple_warning}x default_lora_rank {default_lora_rank}"
                ),
            )
        )
    if len(snapshots) >= 2 and top_k <= final_metrics.module_count:
        previous = snapshots[-2].metrics
        if abs(previous.top_k_share - final_metrics.top_k_share) < thresholds.winner_take_all_delta:
            warnings.append(
                CalibrationWarning(
                    code=WarningCode.WINNER_TAKE_ALL,
                    message=(
                        "top-k share changed by less than "
                        f"{thresholds.winner_take_all_delta} between the last two projections; "
                        "the distribution looks static (winner-take-all)"
                    ),
                )
            )
    if any(snapshot.loss_at_projection is None for snapshot in snapshots):
        warnings.append(
            CalibrationWarning(
                code=WarningCode.LOSS_UNAVAILABLE,
                message="loss was unavailable for at least one projection event",
            )
        )
    if not any(snapshot.is_final_projection for snapshot in snapshots):
        warnings.append(
            CalibrationWarning(
                code=WarningCode.INCOMPLETE_RUN,
                message="the run stopped before the final projection",
            )
        )
    return warnings


def render_calibration_report(
    *,
    manifest: CalibrationManifest,
    snapshots: Sequence[ProjectionSnapshot],
    plan: LayerPlan,
    handoff: UnslothHandoff | None,
    top_k: int,
    thresholds: ReportThresholds | None = None,
) -> str:
    """Render the human-readable Markdown report (pure function).

    The report contains no hidden computations: every number comes directly from
    the manifest, the recorded snapshots, the plan or the handoff. Large arrays
    stay in JSONL/CSV; the report shows aggregates and Top-K tables only.
    """
    thresholds = thresholds or ReportThresholds()
    metadata = manifest.metadata
    lines: list[str] = []

    final_snapshots = [snapshot for snapshot in snapshots if snapshot.is_final_projection]
    completed = bool(final_snapshots) and manifest.calibration_completed
    warnings = _collect_threshold_warnings(
        snapshots,
        thresholds=thresholds,
        default_lora_rank=metadata.default_lora_rank,
        top_k=top_k,
    )

    lines.append("# SimplexLoRA calibration report")
    lines.append("")
    lines.append(f"- Run ID: `{manifest.run_id}`")
    lines.append(f"- Status: `{manifest.status.value}` (stop reason: `{manifest.stop_reason.value}`)")
    lines.append(f"- Optimizer steps recorded: {manifest.optimizer_steps}")
    lines.append(f"- Projections recorded: {len(snapshots)}")
    lines.append(f"- Calibration completed: {'yes' if completed else 'no'}")
    lines.append(f"- Final rank budget: {plan.adapter.total_rank}")
    lines.append(f"- Selected modules: {plan.adapter.selected_module_count} / {len(plan.modules)}")
    lines.append("")

    lines.append("## Run parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Model | `{metadata.model_id}` (revision `{metadata.model_revision}`) |")
    lines.append(f"| Architecture | `{metadata.architecture}` |")
    lines.append(f"| Device | `{metadata.device}` |")
    lines.append(f"| Adapter | `{metadata.adapter_name}` |")
    lines.append(f"| default_lora_rank | {metadata.default_lora_rank} |")
    lines.append(f"| lora_alpha | {metadata.lora_alpha} |")
    lines.append(f"| fat_step | {metadata.fat_step} |")
    lines.append(f"| max_fat_steps | {metadata.max_fat_steps} |")
    lines.append(f"| weight_lr | {fmt_number(metadata.weight_lr)} |")
    lines.append(f"| weight_decay | {fmt_number(metadata.weight_decay)} |")
    lines.append(f"| seed | {metadata.seed} |")
    lines.append(f"| effective batch size | {metadata.effective_batch_size} |")
    lines.append(f"| gradient accumulation | {metadata.gradient_accumulation_steps} |")
    if metadata.data:
        data_summary = "; ".join(f"{key}={value}" for key, value in metadata.data.items() if value is not None)[:5000]
        lines.append(f"| data | {data_summary} |")
    lines.append("")

    lines.append("## Projection metrics")
    lines.append("")
    lines.append(
        "| # | Step | Final | Loss | Mean loss since prev | Entropy | Norm. entropy | "
        "Eff. modules | HHI | Top-1 share | Total rank | Selected |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for snapshot in snapshots:
        metrics = snapshot.metrics
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                snapshot.projection_index,
                snapshot.step,
                "yes" if snapshot.is_final_projection else "no",
                fmt_number(snapshot.loss_at_projection),
                fmt_number(snapshot.mean_loss_since_previous_projection),
                fmt_number(metrics.entropy),
                fmt_number(metrics.normalized_entropy),
                fmt_number(metrics.effective_module_count, 2),
                fmt_number(metrics.hhi),
                fmt_number(metrics.top_1_share),
                metrics.total_rank,
                metrics.selected_count,
            )
        )
    lines.append("")

    top_k_module_count = len(plan.top_k_modules)
    lines.append(
        f"## Top-{min(top_k, top_k_module_count) if top_k_module_count else 'n/a'} modules of the final projection"
    )
    lines.append("")
    lines.append("| Module | Weight | Importance | Rank | Alpha | Scaling |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for entry in plan.top_k_modules:
        lines.append(
            f"| `{entry.module_name}` | {fmt_number(entry.simplex_weight)} | "
            f"{fmt_number(entry.normalized_importance)} | {entry.rank} | {entry.lora_alpha} | "
            f"{fmt_number(entry.scaling)} |"
        )
    if not plan.top_k_modules:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")

    lines.append("## Rank dynamics of the Top-K modules")
    lines.append("")
    top_k_names = {entry.module_name for entry in plan.top_k_modules}
    if snapshots and top_k_names:
        header = "| Module | " + " | ".join(f"@{snapshot.projection_index}" for snapshot in snapshots) + " |"
        lines.append(header)
        lines.append("| --- | " + " | ".join("---" for _ in snapshots) + " |")
        for name in sorted(top_k_names):
            identity_index = None
            for index, entry in enumerate(plan.modules):
                if entry.module_name == name:
                    identity_index = index
                    break
            if identity_index is None:
                continue
            cells = [str(snapshot.records[identity_index].rank) for snapshot in snapshots]
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    else:
        lines.append("_No projections recorded._")
    lines.append("")

    lines.append("## Component activity (final projection)")
    lines.append("")
    final_metrics = snapshots[-1].metrics if snapshots else None
    lines.append("| Component | Importance share | Total rank |")
    lines.append("| --- | --- | --- |")
    if final_metrics is not None:
        for component in sorted(final_metrics.component_shares):
            lines.append(
                f"| {component} | {fmt_number(final_metrics.component_shares[component])} | "
                f"{final_metrics.rank_by_component.get(component, 0)} |"
            )
    else:
        lines.append("| n/a | n/a | n/a |")
    lines.append("")

    lines.append("## Zero-rank modules")
    lines.append("")
    zero_rank = [entry.module_name for entry in plan.modules if entry.rank == 0]
    if zero_rank:
        preview = ", ".join(f"``{name}``" for name in zero_rank[:20])
        suffix = f" (+{len(zero_rank) - 20} more)" if len(zero_rank) > 20 else ""
        lines.append(f"{len(zero_rank)} module(s): {preview}{suffix}")
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## Concentration diagnostics and warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- **{warning.code}**: {warning.message}")
    else:
        lines.append("_No warnings._")
    lines.append("")
    lines.append(
        "Interpretation limits: the simplex weight is the calibrated importance of a module; "
        "the normalized importance sums to one across all modules; the rank is the actually "
        "allocated `ceil(weight * default_lora_rank)`; `scaling = lora_alpha / rank`. "
        "Gradient norms are optimization diagnostics only and are never importance measures."
    )
    lines.append("")

    lines.append("## Unsloth handoff")
    lines.append("")
    if handoff is None:
        lines.append("_Handoff was not generated (disabled or failed)._")
    else:
        lines.append(f"- Source plan run: `{handoff.source_plan_run_id}`")
        lines.append(f"- Selection policy: `{handoff.selection_policy}`")
        lines.append(f"- Alpha mode: `{handoff.alpha_mode}`")
        lines.append(f"- Requested rank total: {handoff.requested_rank_total}")
        lines.append(f"- Applied rank total: {handoff.applied_rank_total}")
        lines.append(f"- Estimated trainable parameters: {handoff.estimated_trainable_parameters:,}")
        lines.append(f"- Target modules (escaped, anchored regex): {len(handoff.target_modules)}")
        if handoff.excluded_modules:
            lines.append(f"- Excluded modules: {len(handoff.excluded_modules)}")
        if handoff.warnings:
            lines.append("- Handoff warnings:")
            for warning in handoff.warnings:
                lines.append(f"  - **{warning.code}**: {warning.message}")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{MANIFEST_FILENAME}`: run metadata, status and checksums")
    lines.append(f"- `{STEPS_FILENAME}`: one line per optimizer step")
    lines.append(f"- `{PROJECTIONS_FILENAME}`: one line per projection with per-layer records")
    lines.append(f"- `{CSV_FILENAME}`: denormalized (projection, module) table")
    lines.append(f"- `{LAYER_PLAN_FILENAME}`: frozen handoff plan (all modules, including rank 0)")
    lines.append(
        f"- `{UNSLOTH_HANDOFF_FILENAME}`: derived Unsloth/PEFT configuration"
        if handoff is not None
        else f"- `{UNSLOTH_HANDOFF_FILENAME}`: not generated"
    )
    lines.append("")
    lines.append(
        "All paths above are file names relative to the calibration directory of this run. "
        "Full data lives in the JSONL and CSV artifacts; this report shows aggregates only."
    )
    lines.append("")

    return "\n".join(lines)


def render_console_summary(
    plan: LayerPlan | None,
    *,
    top_k: int = 10,
    thresholds: ReportThresholds | None = None,
    artifact_paths: Any = None,
) -> str:
    """Short plain-text summary printed right after a run.

    ``plan`` is ``None`` when calibration did not reach its final projection;
    in that case an explicit message is returned instead of a fake Top-K.
    """
    thresholds = thresholds or ReportThresholds()
    if plan is None:
        return (
            "Calibration did not complete: no final projection was recorded, "
            "so there is no layer plan or Top-K to report."
        )

    lines: list[str] = []
    lines.append("SimplexLoRA calibration summary")
    lines.append(f"  run: {plan.run_id}")
    lines.append(f"  active modules: {plan.adapter.selected_module_count} / {len(plan.modules)}")
    lines.append(f"  total rank budget: {plan.adapter.total_rank}")
    top_modules = sorted(plan.top_k_modules, key=lambda entry: (-entry.normalized_importance, entry.module_name))[
        :top_k
    ]
    if top_modules:
        lines.append(f"  top {len(top_modules)} modules:")
        for entry in top_modules:
            lines.append(f"    {entry.module_name:<60} weight={fmt_number(entry.simplex_weight)} rank={entry.rank}")
    if plan.modules and plan.top_k_modules:
        top_share = max((entry.normalized_importance for entry in plan.top_k_modules), default=0.0)
        if top_share > thresholds.top_share_warning:
            lines.append(
                f"  warning: distribution is highly concentrated "
                f"(top normalized importance {top_share:.3f} > {thresholds.top_share_warning})"
            )
    if artifact_paths is not None:
        report_path = getattr(artifact_paths, "report", None)
        handoff_path = getattr(artifact_paths, "unsloth_handoff", None)
        if report_path is not None:
            lines.append(f"  full report: {report_path}")
        if handoff_path is not None:
            lines.append(f"  unsloth handoff: {handoff_path}")
    return "\n".join(lines)


def render_stability_report(report: StabilityReport) -> str:
    """Markdown rendering of a :class:`StabilityReport`."""
    lines: list[str] = []
    lines.append("# SimplexLoRA calibration stability report")
    lines.append("")
    lines.append(f"- Runs compared: {report.run_count}")
    lines.append(f"- Top-K: {report.top_k}")
    lines.append(f"- Jaccard similarity of Top-K sets: {fmt_number(report.jaccard_top_k)}")
    lines.append(f"- Top-K union: {len(report.top_k_union)} module(s)")
    lines.append(f"- Top-K intersection: {len(report.top_k_intersection)} module(s)")
    for warning in report.warnings:
        lines.append(f"- warning **{warning.code}**: {warning.message}")
    lines.append("")
    lines.append("| Module | Top-K hits | Mean rank | Std rank | Min | Max | Mean importance | Stable |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for stats in report.module_stats:
        lines.append(
            "| `{}` | {}/{} | {} | {} | {} | {} | {} | {} |".format(
                stats.module_name,
                stats.top_k_frequency,
                report.run_count,
                fmt_number(stats.mean_rank),
                fmt_number(stats.std_rank),
                stats.min_rank,
                stats.max_rank,
                fmt_number(stats.mean_importance),
                "yes" if stats.stable else "no",
            )
        )
    lines.append("")
    return "\n".join(lines)

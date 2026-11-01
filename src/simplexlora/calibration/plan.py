"""Freezing, validating, loading and verifying calibration layer plans."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

from simplexlora.calibration.models import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    CalibrationArtifactError,
    CalibrationPlanError,
    CalibrationRegistryError,
    CalibrationValidationError,
    LayerPlan,
    LayerPlanAdapterInfo,
    LayerPlanEntry,
    LayerPlanModelInfo,
    ValidationIssue,
    ValidationReport,
    check_finite,
    load_json_document,
)

if TYPE_CHECKING:
    from pathlib import Path

    from simplexlora.calibration.recorder import CalibrationRecorder

NORMALIZED_IMPORTANCE_TOLERANCE = 1e-6


def _entry_from_arrays(
    identity: Any,
    record: Any,
) -> LayerPlanEntry:
    return LayerPlanEntry(
        module_name=identity.module_name,
        peft_module_name=identity.peft_module_name,
        layer_index=identity.layer_index,
        component=identity.component,
        in_features=identity.in_features,
        out_features=identity.out_features,
        simplex_weight=record.weight_projected,
        normalized_importance=record.normalized_importance,
        rank=record.rank,
        selected=record.selected,
        lora_alpha=record.lora_alpha,
        scaling=record.scaling,
        is_top_k=False,
    )


def _mark_top_k(plan_modules: tuple[LayerPlanEntry, ...], top_k: int) -> tuple[LayerPlanEntry, ...]:
    selected = [entry for entry in plan_modules if entry.selected]
    chosen = sorted(selected, key=lambda entry: (-entry.normalized_importance, entry.module_name))[
        : min(top_k, len(selected))
    ]
    chosen_names = {entry.module_name for entry in chosen}
    return tuple(dataclasses.replace(entry, is_top_k=entry.module_name in chosen_names) for entry in plan_modules)


def finalize_layer_plan(
    recorder: CalibrationRecorder,
    registry: Any,
    *,
    top_k: int,
) -> LayerPlan:
    """Freeze the final projection into an immutable, comparable :class:`LayerPlan`.

    Must be called before ``_final_lora_rank_update`` resets ``lora_A``/``lora_B``.
    The plan is built from the recorded final :class:`ProjectionSnapshot`, never
    from the optimizer's live state after the reset. Repeated calls return the
    same cached plan.
    """
    cached = getattr(recorder, "layer_plan", None)
    if cached is not None:
        return cached

    final_snapshots = [snapshot for snapshot in recorder.snapshots if snapshot.is_final_projection]
    if len(final_snapshots) != 1:
        raise CalibrationPlanError(
            f"expected exactly one final projection, found {len(final_snapshots)}; "
            "the plan can only be frozen after the final projection"
        )
    snapshot = final_snapshots[0]

    if len(snapshot.records) != len(registry.identities):
        raise CalibrationPlanError(
            f"final snapshot has {len(snapshot.records)} records, registry has {len(registry.identities)}"
        )

    entries = tuple(
        _entry_from_arrays(identity, record)
        for identity, record in zip(registry.identities, snapshot.records, strict=True)
    )
    entries = _mark_top_k(entries, top_k)

    metadata = recorder.run_metadata
    plan = LayerPlan(
        schema_version=SCHEMA_VERSION,
        run_id=metadata.run_id,
        model=LayerPlanModelInfo(
            model_id=metadata.model_id,
            revision=metadata.model_revision,
            architecture=metadata.architecture,
        ),
        adapter=LayerPlanAdapterInfo(
            name=registry.adapter_name,
            projection_step=snapshot.step,
            default_lora_rank=metadata.default_lora_rank,
            lora_alpha=metadata.lora_alpha,
            total_rank=sum(entry.rank for entry in entries),
            selected_module_count=sum(1 for entry in entries if entry.selected),
        ),
        modules=entries,
        top_k=top_k,
    )
    recorder.store_layer_plan(plan)
    return plan


def _resolve_layer(model: Any, peft_module_name: str) -> Any:
    try:
        return model.get_submodule(peft_module_name)
    except AttributeError:
        return None


def validate_layer_plan(
    plan: LayerPlan,
    *,
    model: Any = None,
    adapter_name: str | None = None,
    strict: bool = True,
) -> ValidationReport:
    """Check a plan for structural consistency before any downstream use.

    With ``model`` given, module names are resolved against the live model and
    feature dimensions are compared. In ``strict`` mode any error raises
    :class:`CalibrationValidationError`.
    """
    from simplexlora.peft_plugin import SimplexLoraLinear

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if plan.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            ValidationIssue(
                code="unsupported_schema_version",
                message=f"schema_version {plan.schema_version!r} is not supported "
                f"(supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)})",
            )
        )

    module_names = [entry.module_name for entry in plan.modules]
    duplicate_names = sorted({name for name in module_names if module_names.count(name) > 1})
    for name in duplicate_names:
        errors.append(
            ValidationIssue(code="duplicate_module", message=f"duplicate module_name {name!r}", module_name=name)
        )

    total_rank = 0
    selected_count = 0
    importance_sum = 0.0
    for entry in plan.modules:
        label = entry.module_name
        if entry.rank < 0:
            errors.append(ValidationIssue(code="negative_rank", message="rank must be >= 0", module_name=label))
        if entry.selected != (entry.rank > 0):
            errors.append(
                ValidationIssue(
                    code="selected_mask_mismatch",
                    message=f"selected={entry.selected} does not match rank={entry.rank}",
                    module_name=label,
                )
            )
        for field_name in ("simplex_weight", "normalized_importance", "scaling"):
            value = getattr(entry, field_name)
            try:
                check_finite(value, f"{label}.{field_name}")
            except CalibrationArtifactError as error:
                errors.append(ValidationIssue(code="non_finite", message=str(error), module_name=label))
            if field_name != "scaling" and value < 0:
                errors.append(
                    ValidationIssue(
                        code="negative_weight",
                        message=f"{field_name} must be non-negative, got {value}",
                        module_name=label,
                    )
                )
        if not isinstance(entry.lora_alpha, int) or not math.isfinite(entry.lora_alpha):
            errors.append(ValidationIssue(code="non_finite", message="lora_alpha is not finite", module_name=label))
        total_rank += entry.rank
        selected_count += int(entry.selected)
        importance_sum += entry.normalized_importance

    if plan.modules and not math.isclose(importance_sum, 1.0, abs_tol=NORMALIZED_IMPORTANCE_TOLERANCE):
        errors.append(
            ValidationIssue(
                code="importance_sum",
                message=f"sum of normalized_importance is {importance_sum:.8f}, expected 1.0 "
                f"(tolerance {NORMALIZED_IMPORTANCE_TOLERANCE})",
            )
        )

    if plan.adapter.total_rank != total_rank:
        errors.append(
            ValidationIssue(
                code="adapter_total_rank",
                message=f"adapter.total_rank={plan.adapter.total_rank} != sum of module ranks {total_rank}",
            )
        )
    if plan.adapter.selected_module_count != selected_count:
        errors.append(
            ValidationIssue(
                code="adapter_selected_count",
                message=f"adapter.selected_module_count={plan.adapter.selected_module_count} != {selected_count}",
            )
        )

    resolved = 0
    model_module_names: set[str] | None = None
    if model is not None:
        if adapter_name is not None:
            from simplexlora.calibration.registry import build_layer_registry

            try:
                model_registry = build_layer_registry(model, adapter_name)
            except CalibrationRegistryError as error:
                errors.append(
                    ValidationIssue(
                        code="registry_mismatch",
                        message=f"could not enumerate model calibration modules: {error}",
                    )
                )
            else:
                model_module_names = {identity.peft_module_name for identity in model_registry.identities}
                plan_module_names = {entry.peft_module_name for entry in plan.modules}
                missing = sorted(model_module_names - plan_module_names)
                extra = sorted(plan_module_names - model_module_names)
                if missing:
                    errors.append(
                        ValidationIssue(
                            code="missing_modules",
                            message=f"plan omits {len(missing)} model module(s), including {missing[:3]}",
                        )
                    )
                if extra:
                    errors.append(
                        ValidationIssue(
                            code="extra_modules",
                            message=f"plan contains {len(extra)} module(s) absent from the model, including {extra[:3]}",
                        )
                    )
        for entry in plan.modules:
            layer = _resolve_layer(model, entry.peft_module_name)
            if layer is None:
                errors.append(
                    ValidationIssue(
                        code="module_not_found",
                        message=f"{entry.peft_module_name!r} does not resolve in the model",
                        module_name=entry.module_name,
                    )
                )
                continue
            resolved += 1
            if not isinstance(layer, SimplexLoraLinear):
                errors.append(
                    ValidationIssue(
                        code="not_simplex_layer",
                        message=f"{entry.peft_module_name!r} is {type(layer).__name__}, expected SimplexLoraLinear",
                        module_name=entry.module_name,
                    )
                )
                continue
            base = layer.get_base_layer()
            in_features = int(getattr(base, "in_features", getattr(layer, "in_features")))
            out_features = int(getattr(base, "out_features", getattr(layer, "out_features")))
            if in_features != entry.in_features or out_features != entry.out_features:
                errors.append(
                    ValidationIssue(
                        code="dimension_mismatch",
                        message=f"model features ({in_features}, {out_features}) != plan features "
                        f"({entry.in_features}, {entry.out_features})",
                        module_name=entry.module_name,
                    )
                )
            if adapter_name is not None and adapter_name not in getattr(layer, "lora_A", {}):
                errors.append(
                    ValidationIssue(
                        code="missing_adapter",
                        message=f"adapter {adapter_name!r} missing on {entry.peft_module_name!r}",
                        module_name=entry.module_name,
                    )
                )

    statistics = {
        "module_count": len(plan.modules),
        "selected_count": selected_count,
        "total_rank": total_rank,
        "modules_resolved_in_model": resolved,
        "model_module_count": len(model_module_names) if model_module_names is not None else None,
    }
    report = ValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        statistics=statistics,
    )
    if strict and errors:
        raise CalibrationValidationError(report)
    return report


def verify_applied_plan(
    model: Any,
    plan: LayerPlan,
    *,
    adapter_name: str,
) -> ValidationReport:
    """Prove that the saved plan matches the model's actually applied ranks.

    Read-only: only inspects ``lora_A``/``lora_B``/``lora_weight``/``r`` of each
    module after ``_final_lora_rank_update``.
    """
    from simplexlora.peft_plugin import SimplexLoraLinear

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    checked = 0
    zero_checked = 0

    for entry in plan.modules:
        layer = _resolve_layer(model, entry.peft_module_name)
        if layer is None or not isinstance(layer, SimplexLoraLinear):
            errors.append(
                ValidationIssue(
                    code="module_not_found",
                    message=f"{entry.peft_module_name!r} does not resolve to a SimplexLoraLinear",
                    module_name=entry.module_name,
                )
            )
            continue
        if adapter_name not in getattr(layer, "lora_A", {}):
            errors.append(
                ValidationIssue(
                    code="missing_adapter",
                    message=f"adapter {adapter_name!r} missing on {entry.peft_module_name!r}",
                    module_name=entry.module_name,
                )
            )
            continue

        weight = layer.lora_weight[adapter_name]
        if entry.rank > 0:
            checked += 1
            applied_rank = int(layer.r[adapter_name])
            if applied_rank != entry.rank:
                errors.append(
                    ValidationIssue(
                        code="rank_mismatch",
                        message=f"applied rank {applied_rank} != plan rank {entry.rank}",
                        module_name=entry.module_name,
                    )
                )
            lora_a = layer.lora_A[adapter_name].weight
            lora_b = layer.lora_B[adapter_name].weight
            if tuple(lora_a.shape) != (entry.rank, entry.in_features):
                errors.append(
                    ValidationIssue(
                        code="lora_a_dimension_mismatch",
                        message=f"lora_A shape {tuple(lora_a.shape)} != {(entry.rank, entry.in_features)}",
                        module_name=entry.module_name,
                    )
                )
            if tuple(lora_b.shape) != (entry.out_features, entry.rank):
                errors.append(
                    ValidationIssue(
                        code="lora_b_dimension_mismatch",
                        message=f"lora_B shape {tuple(lora_b.shape)} != {(entry.out_features, entry.rank)}",
                        module_name=entry.module_name,
                    )
                )
            if float(weight.detach().reshape(()).item()) != 1.0:
                errors.append(
                    ValidationIssue(
                        code="weight_not_finalized",
                        message="lora_weight is not fixed at 1.0 after the final rank update",
                        module_name=entry.module_name,
                    )
                )
            if weight.requires_grad:
                errors.append(
                    ValidationIssue(
                        code="weight_still_trainable",
                        message="lora_weight still requires grad after the final rank update",
                        module_name=entry.module_name,
                    )
                )
        else:
            zero_checked += 1
            # Excluded adapters are disabled rather than merely omitted from
            # the rank plan, otherwise their original A/B matrices continue
            # training and contribute despite rank=0 in the report.
            if float(weight.detach().reshape(()).item()) != 0.0:
                errors.append(
                    ValidationIssue(
                        code="zero_rank_weight_nonzero",
                        message="rank-0 module has a non-zero lora_weight after finalization",
                        module_name=entry.module_name,
                    )
                )
            if weight.requires_grad:
                errors.append(
                    ValidationIssue(
                        code="zero_rank_weight_trainable",
                        message="rank-0 module still has a trainable lora_weight",
                        module_name=entry.module_name,
                    )
                )
            if layer.lora_A[adapter_name].weight.requires_grad or layer.lora_B[adapter_name].weight.requires_grad:
                errors.append(
                    ValidationIssue(
                        code="zero_rank_adapter_trainable",
                        message="rank-0 module still has trainable lora_A/lora_B parameters",
                        module_name=entry.module_name,
                    )
                )

    statistics = {
        "module_count": len(plan.modules),
        "positive_rank_checked": checked,
        "zero_rank_checked": zero_checked,
        "total_rank": sum(entry.rank for entry in plan.modules),
    }
    return ValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        statistics=statistics,
    )


_KNOWN_PLAN_KEYS = {"schema_version", "run_id", "model", "adapter", "modules", "top_k"}


def _parse_model_info(document: dict[str, Any]) -> LayerPlanModelInfo:
    for key in ("model_id", "revision", "architecture"):
        if key not in document:
            raise CalibrationPlanError(f"plan model info is missing required key {key!r}")
    return LayerPlanModelInfo(
        model_id=str(document["model_id"]),
        revision=str(document["revision"]),
        architecture=str(document["architecture"]),
    )


def _parse_adapter_info(document: dict[str, Any]) -> LayerPlanAdapterInfo:
    for key in ("name", "projection_step", "default_lora_rank", "lora_alpha", "total_rank", "selected_module_count"):
        if key not in document:
            raise CalibrationPlanError(f"plan adapter info is missing required key {key!r}")
    return LayerPlanAdapterInfo(
        name=str(document["name"]),
        projection_step=int(document["projection_step"]),
        default_lora_rank=int(document["default_lora_rank"]),
        lora_alpha=int(document["lora_alpha"]),
        total_rank=int(document["total_rank"]),
        selected_module_count=int(document["selected_module_count"]),
    )


def _parse_entry(document: dict[str, Any]) -> LayerPlanEntry:
    required = (
        "module_name",
        "peft_module_name",
        "layer_index",
        "component",
        "in_features",
        "out_features",
        "simplex_weight",
        "normalized_importance",
        "rank",
        "selected",
        "lora_alpha",
        "scaling",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise CalibrationPlanError(f"plan module entry is missing keys: {', '.join(missing)}")
    return LayerPlanEntry(
        module_name=str(document["module_name"]),
        peft_module_name=str(document["peft_module_name"]),
        layer_index=int(document["layer_index"]),
        component=str(document["component"]),
        in_features=int(document["in_features"]),
        out_features=int(document["out_features"]),
        simplex_weight=check_finite(document["simplex_weight"], "simplex_weight"),
        normalized_importance=check_finite(document["normalized_importance"], "normalized_importance"),
        rank=int(document["rank"]),
        selected=bool(document["selected"]),
        lora_alpha=int(document["lora_alpha"]),
        scaling=check_finite(document["scaling"], "scaling"),
        is_top_k=bool(document.get("is_top_k", False)),
    )


def load_layer_plan(path: Path, *, strict: bool = True) -> LayerPlan:
    """Load a ``layer_plan.json`` safely (data only, no code execution).

    Unknown top-level fields are preserved but do not affect any computation.
    In strict mode a full structural validation runs before returning.
    """
    document = load_json_document(path)

    for key in ("schema_version", "run_id", "model", "adapter", "modules"):
        if key not in document:
            if strict:
                raise CalibrationPlanError(f"layer plan at {path} is missing required key {key!r}")
            continue

    schema_version = str(document.get("schema_version", ""))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise CalibrationPlanError(
            f"layer plan at {path} has unsupported schema_version {schema_version!r} "
            f"(supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    modules_document = document.get("modules", [])
    if not isinstance(modules_document, list):
        raise CalibrationPlanError(f"layer plan at {path}: 'modules' must be a list")

    plan = LayerPlan(
        schema_version=schema_version,
        run_id=str(document.get("run_id", "")),
        model=_parse_model_info(document.get("model", {})),
        adapter=_parse_adapter_info(document.get("adapter", {})),
        modules=tuple(_parse_entry(entry) for entry in modules_document),
        top_k=int(document.get("top_k", 0)),
        extra_fields={key: value for key, value in document.items() if key not in _KNOWN_PLAN_KEYS},
    )

    if strict:
        validate_layer_plan(plan, strict=True)
    return plan


__all__ = [
    "NORMALIZED_IMPORTANCE_TOLERANCE",
    "finalize_layer_plan",
    "validate_layer_plan",
    "verify_applied_plan",
    "load_layer_plan",
]

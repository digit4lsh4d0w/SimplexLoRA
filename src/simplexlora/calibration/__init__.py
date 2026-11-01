"""Calibration observability: registry, recorder, plans, reports and handoff.

Public API follows ``docs/ai/simplexlora-calibration-observability-spec.md``
(schema version 1.0). CLI integrations should only use this package's public
functions and never format JSON themselves.
"""

from __future__ import annotations

from simplexlora.calibration.cli import (
    add_calibration_arguments,
    build_run_metadata,
    deterministic_run_id,
    handoff_options_from_args,
    validate_calibration_arguments,
)
from simplexlora.calibration.compare import compare_calibration_runs
from simplexlora.calibration.handoff import build_unsloth_handoff
from simplexlora.calibration.metrics import compute_calibration_metrics
from simplexlora.calibration.models import (
    SCHEMA_VERSION,
    ArtifactPaths,
    CalibrationError,
    CalibrationManifest,
    CalibrationMetrics,
    CalibrationPlanError,
    CalibrationRecorderError,
    CalibrationRegistryError,
    CalibrationRunMetadata,
    CalibrationValidationError,
    CalibrationWarning,
    LayerIdentity,
    LayerPlan,
    LayerPlanEntry,
    LayerProjectionRecord,
    LayerRegistry,
    ProjectionEvent,
    ProjectionSnapshot,
    ReportThresholds,
    RunStatus,
    StabilityReport,
    StepRecord,
    StopReason,
    ValidationReport,
    WarningCode,
)
from simplexlora.calibration.plan import (
    finalize_layer_plan,
    load_layer_plan,
    validate_layer_plan,
    verify_applied_plan,
)
from simplexlora.calibration.recorder import (
    CalibrationRecorder,
    CalibrationSession,
    attach_calibration_session,
    create_calibration_session,
    disabled_calibration_session,
    write_calibration_artifacts,
)
from simplexlora.calibration.registry import build_layer_registry
from simplexlora.calibration.report import (
    render_calibration_report,
    render_console_summary,
    render_stability_report,
)
from simplexlora.calibration.trainer import SimplexCalibrationTrainer

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactPaths",
    "CalibrationError",
    "CalibrationManifest",
    "CalibrationMetrics",
    "CalibrationPlanError",
    "CalibrationRecorder",
    "CalibrationRecorderError",
    "CalibrationRegistryError",
    "CalibrationRunMetadata",
    "CalibrationSession",
    "CalibrationValidationError",
    "CalibrationWarning",
    "LayerIdentity",
    "LayerPlan",
    "LayerPlanEntry",
    "LayerProjectionRecord",
    "LayerRegistry",
    "ProjectionEvent",
    "ProjectionSnapshot",
    "ReportThresholds",
    "RunStatus",
    "StabilityReport",
    "StepRecord",
    "StopReason",
    "ValidationReport",
    "WarningCode",
    "add_calibration_arguments",
    "attach_calibration_session",
    "build_layer_registry",
    "build_run_metadata",
    "build_unsloth_handoff",
    "compare_calibration_runs",
    "compute_calibration_metrics",
    "create_calibration_session",
    "deterministic_run_id",
    "disabled_calibration_session",
    "finalize_layer_plan",
    "handoff_options_from_args",
    "load_layer_plan",
    "render_calibration_report",
    "render_console_summary",
    "render_stability_report",
    "validate_calibration_arguments",
    "validate_layer_plan",
    "verify_applied_plan",
    "write_calibration_artifacts",
    "SimplexCalibrationTrainer",
]

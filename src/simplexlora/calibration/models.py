"""Data models, enums and serialization helpers for calibration observability.

Schema version ``1.0`` follows
``docs/ai/simplexlora-calibration-observability-spec.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = ("1.0",)

MANIFEST_FILENAME = "manifest.json"
STEPS_FILENAME = "steps.jsonl"
PROJECTIONS_FILENAME = "projections.jsonl"
CSV_FILENAME = "calibration_layers.csv"
REPORT_FILENAME = "calibration_report.md"
LAYER_PLAN_FILENAME = "layer_plan.json"
UNSLOTH_HANDOFF_FILENAME = "unsloth_handoff.json"

CSV_COLUMNS = (
    "projection_index",
    "step",
    "module_name",
    "layer_index",
    "component",
    "weight_before",
    "weight_after",
    "weight_projected",
    "normalized_importance",
    "rank",
    "previous_rank",
    "rank_delta",
    "selected",
    "lora_alpha",
    "scaling",
)


class CalibrationError(Exception):
    """Base class for calibration observability errors."""


class CalibrationRegistryError(CalibrationError):
    """Raised when the layer registry cannot be built consistently."""


class CalibrationRecorderError(CalibrationError):
    """Raised when recorder invariants are violated."""


class CalibrationArtifactError(CalibrationError):
    """Raised when artifacts cannot be written or read safely."""


class CalibrationPlanError(CalibrationError):
    """Raised when a layer plan is missing, ambiguous or malformed."""


class CalibrationValidationError(CalibrationPlanError):
    """Aggregated validation failure carrying a full :class:`ValidationReport`."""

    def __init__(self, report: ValidationReport) -> None:
        # ValidationReport is defined below; dataclass ordering does not matter here.
        super().__init__(f"layer plan validation failed with {len(report.errors)} error(s)")
        self.report = report


class RunStatus(str, Enum):
    """Lifecycle status persisted in ``manifest.json``."""

    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class StopReason(str, Enum):
    """Why a calibration run stopped producing events."""

    TRAINING_COMPLETE = "training_complete"
    CALIBRATION_ONLY = "calibration_only"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    NOT_STOPPED = "not_stopped"


class WarningCode(str, Enum):
    """Machine-readable warning codes shared by manifest, report and handoff."""

    LOSS_UNAVAILABLE = "loss_unavailable"
    HIGH_CONCENTRATION = "high_concentration"
    LOW_EFFECTIVE_MODULES = "low_effective_modules"
    ZERO_RANK_MAJORITY = "zero_rank_majority"
    RANK_ABOVE_BUDGET = "rank_above_budget"
    WINNER_TAKE_ALL = "winner_take_all"
    INCOMPLETE_RUN = "incomplete_run"
    HANDOFF_RANK_FLOOR = "handoff_rank_floor"
    HANDOFF_TOP_K = "handoff_top_k"
    HANDOFF_MIN_WEIGHT = "handoff_min_weight"
    HANDOFF_ALPHA_CHANGED = "handoff_alpha_changed"
    PLAN_VERIFICATION_FAILED = "plan_verification_failed"
    WRITE_ERROR = "write_error"


class CalibrationObserver(Protocol):
    """Event sink implemented by the calibration bridge and called by ``SimplexAdamW``.

    Implementations must never change parameters, RNG, gradients or schedulers.
    """

    def on_optimizer_step(
        self,
        *,
        step: int,
        loss: float | None,
        learning_rate: float,
        lora_grad_norm: float,
        weight_grad_norm: float,
        is_projection_step: bool,
    ) -> None: ...

    def on_projection(self, event: ProjectionEvent) -> None: ...

    def on_before_final_rank_update(self) -> None: ...

    def on_final_rank_update(self) -> None: ...


@dataclass(frozen=True)
class CalibrationWarning:
    """A single warning attached to a calibration run."""

    code: str
    message: str
    step: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.code, Enum):
            object.__setattr__(self, "code", str(self.code.value))

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "step": self.step}


@dataclass(frozen=True)
class ValidationIssue:
    """A single error or warning produced by plan validation."""

    code: str
    message: str
    module_name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.code, Enum):
            object.__setattr__(self, "code", str(self.code.value))

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "module_name": self.module_name}


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of ``validate_layer_plan`` or ``verify_applied_plan``."""

    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    statistics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "statistics": dict(self.statistics),
        }


@dataclass(frozen=True)
class LayerIdentity:
    """Stable identity of one calibrated LoRA module."""

    module_name: str
    peft_module_name: str
    module_index: int
    layer_index: int
    component: str
    in_features: int
    out_features: int
    initial_rank: int
    lora_alpha: int
    scaling: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayerRegistry:
    """Immutable registry of all ``SimplexLoraLinear`` modules of one adapter."""

    adapter_name: str
    identities: tuple[LayerIdentity, ...]
    single_adapter: bool = True
    _module_name_to_index: Mapping[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.identities)

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(identity.module_name for identity in self.identities)

    @property
    def module_name_to_index(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._module_name_to_index))

    def identity_by_index(self, index: int) -> LayerIdentity:
        return self.identities[index]

    def identity_by_name(self, module_name: str) -> LayerIdentity:
        try:
            return self.identities[self._module_name_to_index[module_name]]
        except KeyError:
            raise CalibrationRegistryError(f"Unknown module name: {module_name!r}") from None


@dataclass(frozen=True)
class ProjectionEvent:
    """Immutable snapshot of one simplex projection, produced by ``SimplexAdamW``.

    All arrays are full-length (``len(registry)``) plain Python sequences.
    ``weights_before_step`` are the simplex weights before this optimizer step's
    gradient update, ``weights_after_gradient_step`` after the gradient step but
    before the simplex projection, ``weights_projected`` the projection result.
    """

    step: int
    projection_index: int
    is_final_projection: bool
    weights_before_step: tuple[float, ...]
    weights_after_gradient_step: tuple[float, ...]
    weights_projected: tuple[float, ...]
    ranks: tuple[int, ...]
    selected_mask: tuple[bool, ...]
    loss_at_projection: float | None
    loss_window: float | None
    weight_grad_norms: tuple[float, ...]
    lora_grad_norms: tuple[float, ...]
    created_at: str


@dataclass(frozen=True)
class CalibrationMetrics:
    """Aggregated description of one importance distribution."""

    module_count: int
    selected_count: int
    zero_rank_count: int
    total_rank: int
    rank_utilization: float
    entropy: float
    normalized_entropy: float
    effective_module_count: float
    hhi: float
    inverse_hhi: float
    top_1_share: float
    top_5_share: float
    top_k_share: float
    component_shares: Mapping[str, float]
    rank_by_component: Mapping[str, int]
    max_rank: int
    mean_selected_rank: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_count": self.module_count,
            "selected_count": self.selected_count,
            "zero_rank_count": self.zero_rank_count,
            "total_rank": self.total_rank,
            "rank_utilization": self.rank_utilization,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "effective_module_count": self.effective_module_count,
            "hhi": self.hhi,
            "inverse_hhi": self.inverse_hhi,
            "top_1_share": self.top_1_share,
            "top_5_share": self.top_5_share,
            "top_k_share": self.top_k_share,
            "component_shares": dict(self.component_shares),
            "rank_by_component": dict(self.rank_by_component),
            "max_rank": self.max_rank,
            "mean_selected_rank": self.mean_selected_rank,
        }


@dataclass(frozen=True)
class StepRecord:
    """One optimizer step within the calibration phase."""

    schema_version: str
    step: int
    epoch: float | None
    loss: float | None
    learning_rate: float
    lora_grad_norm: float
    weight_grad_norm: float
    elapsed_seconds: float
    is_projection_step: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayerProjectionRecord:
    """Per-layer row of one projection (CSV and JSONL representation)."""

    projection_index: int
    step: int
    module_name: str
    layer_index: int
    component: str
    weight_before: float
    weight_after: float
    weight_projected: float
    normalized_importance: float
    rank: int
    previous_rank: int
    rank_delta: int
    selected: bool
    lora_alpha: int
    scaling: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_row(self) -> tuple[Any, ...]:
        return (
            self.projection_index,
            self.step,
            self.module_name,
            self.layer_index,
            self.component,
            self.weight_before,
            self.weight_after,
            self.weight_projected,
            self.normalized_importance,
            self.rank,
            self.previous_rank,
            self.rank_delta,
            int(self.selected),
            self.lora_alpha,
            self.scaling,
        )


@dataclass(frozen=True)
class ProjectionSnapshot:
    """Validated projection event enriched with metrics and per-layer records."""

    schema_version: str
    projection_index: int
    step: int
    is_final_projection: bool
    loss_at_projection: float | None
    mean_loss_since_previous_projection: float | None
    metrics: CalibrationMetrics
    records: tuple[LayerProjectionRecord, ...]
    weights_before_gradient_step: tuple[float, ...]
    weights_after_gradient_step: tuple[float, ...]
    projected_weights: tuple[float, ...]
    ranks: tuple[int, ...]
    selected_mask: tuple[bool, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "projection_index": self.projection_index,
            "step": self.step,
            "is_final_projection": self.is_final_projection,
            "loss_at_projection": self.loss_at_projection,
            "mean_loss_since_previous_projection": self.mean_loss_since_previous_projection,
            "weights_before_gradient_step": list(self.weights_before_gradient_step),
            "weights_after_gradient_step": list(self.weights_after_gradient_step),
            "projected_weights": list(self.projected_weights),
            "ranks": list(self.ranks),
            "selected_mask": list(self.selected_mask),
            "metrics": self.metrics.to_dict(),
            "layers": [record.to_dict() for record in self.records],
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class LayerPlanModelInfo:
    """Model fingerprint stored in ``layer_plan.json``."""

    model_id: str
    revision: str
    architecture: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayerPlanAdapterInfo:
    """Adapter metadata stored in ``layer_plan.json``."""

    name: str
    projection_step: int
    default_lora_rank: int
    lora_alpha: int
    total_rank: int
    selected_module_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayerPlanEntry:
    """One module of the frozen final layer plan."""

    module_name: str
    peft_module_name: str
    layer_index: int
    component: str
    in_features: int
    out_features: int
    simplex_weight: float
    normalized_importance: float
    rank: int
    selected: bool
    lora_alpha: int
    scaling: float
    is_top_k: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "peft_module_name": self.peft_module_name,
            "layer_index": self.layer_index,
            "component": self.component,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "simplex_weight": self.simplex_weight,
            "normalized_importance": self.normalized_importance,
            "rank": self.rank,
            "selected": self.selected,
            "lora_alpha": self.lora_alpha,
            "scaling": self.scaling,
            "is_top_k": self.is_top_k,
        }


@dataclass(frozen=True)
class LayerPlan:
    """Frozen, validated result of the final projection for downstream training."""

    schema_version: str
    run_id: str
    model: LayerPlanModelInfo
    adapter: LayerPlanAdapterInfo
    modules: tuple[LayerPlanEntry, ...]
    top_k: int
    extra_fields: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), compare=False)

    @property
    def selected_modules(self) -> tuple[LayerPlanEntry, ...]:
        return tuple(entry for entry in self.modules if entry.selected)

    @property
    def top_k_modules(self) -> tuple[LayerPlanEntry, ...]:
        return tuple(entry for entry in self.modules if entry.is_top_k)

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model": self.model.to_dict(),
            "adapter": self.adapter.to_dict(),
            "modules": [entry.to_dict() for entry in self.modules],
            "top_k": self.top_k,
        }
        document.update(dict(self.extra_fields))
        return document


@dataclass(frozen=True)
class UnslothHandoff:
    """Derived LoRA configuration for Unsloth/PEFT downstream training."""

    schema_version: str
    source_plan_run_id: str
    selection_policy: str
    alpha_mode: str
    rank_floor: int
    min_weight: float
    top_k: int | None
    target_modules: tuple[str, ...]
    rank_pattern: Mapping[str, int]
    alpha_pattern: Mapping[str, int]
    excluded_modules: tuple[str, ...]
    requested_rank_total: int
    applied_rank_total: int
    estimated_trainable_parameters: int
    warnings: tuple[CalibrationWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_plan_run_id": self.source_plan_run_id,
            "selection_policy": self.selection_policy,
            "alpha_mode": self.alpha_mode,
            "rank_floor": self.rank_floor,
            "min_weight": self.min_weight,
            "top_k": self.top_k,
            "target_modules": list(self.target_modules),
            "rank_pattern": dict(self.rank_pattern),
            "alpha_pattern": dict(self.alpha_pattern),
            "excluded_modules": list(self.excluded_modules),
            "requested_rank_total": self.requested_rank_total,
            "applied_rank_total": self.applied_rank_total,
            "estimated_trainable_parameters": self.estimated_trainable_parameters,
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass(frozen=True)
class ArtifactPaths:
    """Absolute paths and SHA-256 checksums of all written artifacts."""

    output_dir: Path
    manifest: Path
    steps: Path
    projections: Path
    csv: Path
    report: Path
    layer_plan: Path
    unsloth_handoff: Path | None
    checksums: Mapping[str, str]

    @property
    def relative_paths(self) -> dict[str, str]:
        paths = {
            MANIFEST_FILENAME: self.manifest,
            STEPS_FILENAME: self.steps,
            PROJECTIONS_FILENAME: self.projections,
            CSV_FILENAME: self.csv,
            REPORT_FILENAME: self.report,
            LAYER_PLAN_FILENAME: self.layer_plan,
        }
        if self.unsloth_handoff is not None:
            paths[UNSLOTH_HANDOFF_FILENAME] = self.unsloth_handoff
        return {name: path.relative_to(self.output_dir).as_posix() for name, path in paths.items()}


@dataclass
class CalibrationRunMetadata:
    """Static, versioned description of one calibration run."""

    run_id: str
    created_at: str
    model_id: str
    model_revision: str
    architecture: str
    device: str
    adapter_name: str
    peft_config: dict[str, Any]
    default_lora_rank: int
    lora_alpha: int
    fat_step: int
    max_fat_steps: int
    weight_lr: float
    weight_decay: float
    seed: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    library_versions: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationManifest:
    """The master structured document of one calibration run."""

    schema_version: str
    run_id: str
    status: RunStatus
    created_at: str
    updated_at: str
    metadata: CalibrationRunMetadata
    optimizer_steps: int = 0
    projections: tuple[dict[str, Any], ...] = ()
    calibration_completed: bool = False
    stop_reason: StopReason = StopReason.NOT_STOPPED
    warnings: tuple[CalibrationWarning, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "versions": dict(self.metadata.library_versions),
            "model": {
                "model_id": self.metadata.model_id,
                "model_revision": self.metadata.model_revision,
                "architecture": self.metadata.architecture,
                "device": self.metadata.device,
            },
            "adapter": {
                "name": self.metadata.adapter_name,
                "peft_config": dict(self.metadata.peft_config),
            },
            "calibration": {
                "default_lora_rank": self.metadata.default_lora_rank,
                "lora_alpha": self.metadata.lora_alpha,
                "fat_step": self.metadata.fat_step,
                "max_fat_steps": self.metadata.max_fat_steps,
                "weight_lr": self.metadata.weight_lr,
                "weight_decay": self.metadata.weight_decay,
                "seed": self.metadata.seed,
                "effective_batch_size": self.metadata.effective_batch_size,
                "gradient_accumulation_steps": self.metadata.gradient_accumulation_steps,
            },
            "optimizer_steps": self.optimizer_steps,
            "data": dict(self.metadata.data),
            "projections": [dict(item) for item in self.projections],
            "calibration_completed": self.calibration_completed,
            "stop_reason": self.stop_reason.value,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "artifacts": dict(self.artifacts),
            **dict(self.extra),
        }


@dataclass(frozen=True)
class StabilityModuleStats:
    """Per-module statistics aggregated across compared runs."""

    module_name: str
    top_k_frequency: int
    top_k_presence: int
    mean_rank: float
    std_rank: float
    min_rank: int
    max_rank: int
    mean_importance: float
    std_importance: float
    stable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StabilityReport:
    """Result of ``compare_calibration_runs``."""

    run_count: int
    top_k: int
    model_ids: tuple[str, ...]
    architectures: tuple[str, ...]
    adapter_names: tuple[str, ...]
    module_names: tuple[str, ...]
    top_k_union: tuple[str, ...]
    top_k_intersection: tuple[str, ...]
    jaccard_top_k: float
    module_stats: tuple[StabilityModuleStats, ...]
    warnings: tuple[CalibrationWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_count": self.run_count,
            "top_k": self.top_k,
            "model_ids": list(self.model_ids),
            "architectures": list(self.architectures),
            "adapter_names": list(self.adapter_names),
            "top_k_union": list(self.top_k_union),
            "top_k_intersection": list(self.top_k_intersection),
            "jaccard_top_k": self.jaccard_top_k,
            "modules": [stats.to_dict() for stats in self.module_stats],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass(frozen=True)
class ReportThresholds:
    """Configurable interpretation thresholds used by reports and warnings."""

    top_share_warning: float = 0.5
    min_effective_modules: float = 2.0
    zero_rank_fraction_warning: float = 0.9
    max_rank_multiple_warning: float = 4.0
    winner_take_all_delta: float = 0.005


def utc_now_iso() -> str:
    """Current UTC time as a sortable ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def validate_run_id(run_id: str) -> str:
    """Allowlist-validation for identifiers serialized into artifact paths.

    Raises ``ValueError`` per the CLI-validation contract (spec 8.16).
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id or ""):
        raise ValueError(f"run_id {run_id!r} contains unsafe characters; allowed: letters, digits, '.', '_', '-'")
    return run_id


def check_finite(value: float, name: str) -> float:
    """Reject NaN/inf before anything is serialized."""
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationArtifactError(f"{name} is not finite: {value!r}")
    return number


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/enums/mappings to JSON-safe structures.

    Raises :class:`CalibrationArtifactError` on NaN, infinity or unsupported types,
    so no ``NaN``/``Infinity`` can ever reach a JSON artifact.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return check_finite(value, "float value")
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if (
        isinstance(value, (list, tuple, set, frozenset))
        or isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
    ):
        if isinstance(value, (set, frozenset)):
            raise CalibrationArtifactError("sets are not allowed in JSON artifacts (unstable ordering)")
        return [to_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return to_jsonable(value.to_dict())
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    raise CalibrationArtifactError(f"Value of type {type(value).__name__} is not JSON-serializable")


def dump_json(value: Any, *, indent: int | None = 2) -> str:
    """Deterministic JSON dump without NaN/Infinity and without escaped unicode."""
    return json.dumps(to_jsonable(value), indent=indent, ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of artifact content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_symlink_free(path: Path) -> Path:
    """Resolve ``path`` and reject symlink components escaping the project root."""
    resolved = path.resolve(strict=False)
    if path.is_symlink() or str(resolved) != str(path.absolute() if not path.is_absolute() else path):
        # Reject only symlinks, not plain relative-path normalization.
        if path.is_symlink():
            raise CalibrationArtifactError(
                f"output path {path} is a symbolic link; pass allow_symlink_output=True to permit it"
            )
    return resolved


def atomic_write_text(path: Path, text: str, *, overwrite: bool = True) -> str:
    """Write ``text`` to ``path`` via a temp file in the same directory + atomic replace."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise CalibrationArtifactError(f"refusing to overwrite existing artifact: {path}")
    _resolve_symlink_free(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_text(text)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except CalibrationArtifactError:
        raise
    except OSError as error:
        temp_path.unlink(missing_ok=True)
        raise CalibrationArtifactError(f"failed to write artifact {path}: {error}") from error
    return digest


def load_json_document(path: Path) -> dict[str, Any]:
    """Read a JSON document without any code execution."""
    try:
        with open(Path(path), encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationArtifactError(f"failed to read JSON document {path}: {error}") from error
    if not isinstance(document, dict):
        raise CalibrationArtifactError(f"expected a JSON object in {path}, got {type(document).__name__}")
    return document


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all lines of a JSONL file produced by the recorder."""
    documents: list[dict[str, Any]] = []
    with open(Path(path), encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise CalibrationArtifactError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return documents


def collect_library_versions(device: Any = None) -> dict[str, Any]:
    """Best-effort library versions for the manifest fingerprint."""
    import torch

    import simplexlora

    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "accelerator": getattr(device, "type", None) if device is not None else None,
    }
    for name in ("peft", "transformers", "bitsandbytes", "accelerate", "datasets"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    versions["simplexlora"] = getattr(simplexlora, "__version__", None)
    return versions

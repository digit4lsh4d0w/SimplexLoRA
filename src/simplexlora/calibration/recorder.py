"""Single-owner state, event recording and artifact writing for calibration runs.

The :class:`CalibrationRecorder` owns all mutable run state, the
:class:`_OptimizerObserverBridge` adapts optimizer events to it, and
:class:`CalibrationSession` wires registry, recorder, optimizer and trainer
together.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simplexlora.calibration.metrics import compute_calibration_metrics
from simplexlora.calibration.models import (
    CSV_COLUMNS,
    CSV_FILENAME,
    LAYER_PLAN_FILENAME,
    MANIFEST_FILENAME,
    PROJECTIONS_FILENAME,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    STEPS_FILENAME,
    UNSLOTH_HANDOFF_FILENAME,
    CalibrationManifest,
    CalibrationRecorderError,
    CalibrationRunMetadata,
    CalibrationWarning,
    LayerPlan,
    LayerProjectionRecord,
    ProjectionEvent,
    ProjectionSnapshot,
    ReportThresholds,
    RunStatus,
    StepRecord,
    StopReason,
    WarningCode,
    dump_json,
    utc_now_iso,
    validate_run_id,
)
from simplexlora.calibration.plan import finalize_layer_plan, verify_applied_plan
from simplexlora.calibration.cli import calibration_output_dir
from simplexlora.calibration.registry import LayerRegistry
from simplexlora.calibration.report import render_calibration_report
from simplexlora.calibration.models import atomic_write_text
from simplexlora.peft_plugin import SimplexLoraLinear

if TYPE_CHECKING:
    from collections.abc import Sequence

    from simplexlora.calibration.models import ValidationReport


class _JsonlWriter:
    """Append-only JSONL writer that flushes on every line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = open(self.path, "a", encoding="utf-8", newline="\n")

    def write(self, document: dict[str, Any]) -> None:
        self._handle.write(dump_json(document, indent=None) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    @property
    def closed(self) -> bool:
        return self._handle.closed


class CalibrationRecorder:
    """Owns the state of one calibration run and writes all its artifacts.

    The recorder never holds tensor objects or references to model parameters:
    every event is converted to plain Python types by the optimizer bridge
    before it reaches this class.
    """

    def __init__(
        self,
        *,
        registry: LayerRegistry,
        output_dir: Path,
        run_metadata: CalibrationRunMetadata,
        top_k: int,
        fail_on_write_error: bool = True,
        overwrite: bool = False,
        allow_symlink_output: bool = False,
        thresholds: ReportThresholds | None = None,
    ) -> None:
        if top_k < 1:
            raise CalibrationRecorderError(f"top_k must be >= 1, got {top_k}")
        validate_run_id(run_metadata.run_id)

        self.registry = registry
        self.output_dir = Path(output_dir)
        self.run_metadata = run_metadata
        self.top_k = top_k
        self.fail_on_write_error = fail_on_write_error
        self.overwrite = overwrite
        self.allow_symlink_output = allow_symlink_output
        self.thresholds = thresholds or ReportThresholds()

        if self.output_dir.is_symlink() and not allow_symlink_output:
            raise CalibrationRecorderError(
                f"output directory {self.output_dir} is a symbolic link; pass allow_symlink_output=True to permit it"
            )

        leftovers = (
            self.output_dir / MANIFEST_FILENAME,
            self.output_dir / STEPS_FILENAME,
            self.output_dir / PROJECTIONS_FILENAME,
            self.output_dir / CSV_FILENAME,
            self.output_dir / REPORT_FILENAME,
            self.output_dir / LAYER_PLAN_FILENAME,
            self.output_dir / UNSLOTH_HANDOFF_FILENAME,
        )
        if any(path.exists() for path in leftovers) and not self.overwrite:
            raise CalibrationRecorderError(
                f"calibration directory {self.output_dir} already contains artifacts from a previous run; "
                "pass --calibration-overwrite to replace them (auto-incrementing the directory is not allowed)"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            # A rerun into the same directory must never mix artifact lines.
            for path in leftovers:
                path.unlink(missing_ok=True)

        self.warnings: list[CalibrationWarning] = []
        self.step_records: list[StepRecord] = []
        self.snapshots: list[ProjectionSnapshot] = []
        self.csv_rows: list[tuple[Any, ...]] = []
        self.layer_plan: LayerPlan | None = None
        self.handoff: Any | None = None
        self.calibration_completed = False
        self.verification_report: ValidationReport | None = None
        self._closed = False
        self._loss_unavailable_warned = False
        self._recorded_steps: set[int] = set()
        self._recorded_projections: set[int] = set()
        self._artifact_checksums: dict[str, str] = {}

        self.manifest = CalibrationManifest(
            schema_version=SCHEMA_VERSION,
            run_id=run_metadata.run_id,
            status=RunStatus.RUNNING,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            metadata=run_metadata,
        )

        self._steps_writer = _JsonlWriter(self.output_dir / STEPS_FILENAME)
        self._projections_writer = _JsonlWriter(self.output_dir / PROJECTIONS_FILENAME)
        self._write_manifest()

    # ------------------------------------------------------------------ warnings

    def add_warning(self, code: Any, message: str, *, step: int | None = None) -> CalibrationWarning:
        warning = CalibrationWarning(code=code, message=message, step=step)
        self.warnings.append(warning)
        self.manifest.warnings = tuple(self.warnings)
        return warning

    # ------------------------------------------------------------------ steps

    def record_step(
        self,
        *,
        step: int,
        epoch: float | None,
        loss: float | None,
        learning_rate: float,
        lora_grad_norm: float,
        weight_grad_norm: float,
        elapsed_seconds: float,
        is_projection_step: bool,
    ) -> StepRecord:
        """Record exactly one optimizer step of the calibration phase."""
        if self._closed:
            raise CalibrationRecorderError("recorder is closed")
        if not isinstance(step, int) or isinstance(step, bool):
            raise CalibrationRecorderError(f"step must be an int, got {step!r}")
        if step in self._recorded_steps:
            raise CalibrationRecorderError(f"step {step} was already recorded")
        if loss is None:
            if not self._loss_unavailable_warned:
                self._loss_unavailable_warned = True
                self.add_warning(
                    WarningCode.LOSS_UNAVAILABLE,
                    f"training loss is unavailable at step {step}; recording loss: null",
                    step=step,
                )
        elif not math.isfinite(float(loss)):
            raise CalibrationRecorderError(f"loss must be finite or None, got {loss!r}")
        for name, value in (
            ("learning_rate", learning_rate),
            ("lora_grad_norm", lora_grad_norm),
            ("weight_grad_norm", weight_grad_norm),
            ("elapsed_seconds", elapsed_seconds),
        ):
            if not math.isfinite(float(value)):
                raise CalibrationRecorderError(f"{name} must be finite, got {value!r}")
        if float(lora_grad_norm) < 0 or float(weight_grad_norm) < 0:
            raise CalibrationRecorderError("gradient norms must be non-negative")
        if float(elapsed_seconds) < 0:
            raise CalibrationRecorderError("elapsed_seconds must be non-negative")

        record = StepRecord(
            schema_version=SCHEMA_VERSION,
            step=step,
            epoch=None if epoch is None else float(epoch),
            loss=None if loss is None else float(loss),
            learning_rate=float(learning_rate),
            lora_grad_norm=float(lora_grad_norm),
            weight_grad_norm=float(weight_grad_norm),
            elapsed_seconds=float(elapsed_seconds),
            is_projection_step=bool(is_projection_step),
            created_at=utc_now_iso(),
        )
        self.step_records.append(record)
        self._recorded_steps.add(step)
        self.manifest.optimizer_steps = len(self.step_records)
        self._try_write(self._steps_writer.write, record.to_dict())
        return record

    # ------------------------------------------------------------------ projections

    def record_projection(self, event: ProjectionEvent) -> ProjectionSnapshot:
        """Record one projection; the event must come from ``SimplexAdamW`` itself."""
        if self._closed:
            raise CalibrationRecorderError("recorder is closed")
        if event.projection_index in self._recorded_projections:
            raise CalibrationRecorderError(f"projection {event.projection_index} was already recorded")

        module_count = len(self.registry)
        for name in (
            "weights_before_step",
            "weights_after_gradient_step",
            "weights_projected",
            "ranks",
            "selected_mask",
            "weight_grad_norms",
            "lora_grad_norms",
        ):
            if len(getattr(event, name)) != module_count:
                raise CalibrationRecorderError(
                    f"{name} has length {len(getattr(event, name))}, expected {module_count} (len(registry))"
                )

        for name in ("weights_before_step", "weights_after_gradient_step", "weights_projected"):
            values = getattr(event, name)
            if any(not math.isfinite(float(value)) for value in values):
                raise CalibrationRecorderError(f"{name} contains non-finite values")
        projected = tuple(float(value) for value in event.weights_projected)
        if any(value < 0 for value in projected):
            raise CalibrationRecorderError("projected weights must be non-negative")
        if math.fsum(projected) <= 0:
            raise CalibrationRecorderError("projected weights must have a positive sum")

        default_lora_rank = self.run_metadata.default_lora_rank
        ranks = tuple(int(value) for value in event.ranks)
        for index, (weight, rank) in enumerate(zip(projected, ranks, strict=True)):
            # The optimizer computes ranks in float32; the tiny epsilon absorbs
            # the float32/float64 ceil boundary difference without ever
            # recomputing the rank from a different formula.
            expected = math.ceil(weight * default_lora_rank - 1e-9)
            if expected != rank:
                raise CalibrationRecorderError(
                    f"rank mismatch at module index {index}: {rank} != ceil({weight} * {default_lora_rank})"
                )
        selected_mask = tuple(bool(value) for value in event.selected_mask)
        if selected_mask != tuple(rank > 0 for rank in ranks):
            raise CalibrationRecorderError("selected_mask does not equal rank > 0")

        for name in ("loss_at_projection", "loss_window"):
            value = getattr(event, name)
            if value is not None and not math.isfinite(float(value)):
                raise CalibrationRecorderError(f"{name} must be finite or None")

        previous_ranks = (
            tuple(record.rank for record in self.snapshots[-1].records)
            if self.snapshots
            else tuple(identity.initial_rank for identity in self.registry.identities)
        )

        total_weight = math.fsum(projected)
        metrics = compute_calibration_metrics(
            projected,
            ranks,
            default_lora_rank=default_lora_rank,
            components=[identity.component for identity in self.registry.identities],
            top_k=self.top_k,
        )

        records = []
        csv_rows: list[tuple[Any, ...]] = []

        for index, identity in enumerate(self.registry.identities):
            record = LayerProjectionRecord(
                projection_index=event.projection_index,
                step=event.step,
                module_name=identity.module_name,
                layer_index=identity.layer_index,
                component=identity.component,
                weight_before=float(event.weights_before_step[index]),
                weight_after=float(event.weights_after_gradient_step[index]),
                weight_projected=projected[index],
                normalized_importance=projected[index] / total_weight,
                rank=ranks[index],
                previous_rank=int(previous_ranks[index]),
                rank_delta=ranks[index] - int(previous_ranks[index]),
                selected=selected_mask[index],
                lora_alpha=identity.lora_alpha,
                scaling=(identity.lora_alpha / ranks[index] if ranks[index] > 0 else 0.0),
            )
            records.append(record)
            csv_rows.append(record.to_csv_row())

        snapshot = ProjectionSnapshot(
            schema_version=SCHEMA_VERSION,
            projection_index=event.projection_index,
            step=event.step,
            is_final_projection=bool(event.is_final_projection),
            loss_at_projection=None if event.loss_at_projection is None else float(event.loss_at_projection),
            mean_loss_since_previous_projection=None if event.loss_window is None else float(event.loss_window),
            metrics=metrics,
            records=tuple(records),
            weights_before_gradient_step=tuple(float(value) for value in event.weights_before_step),
            weights_after_gradient_step=tuple(float(value) for value in event.weights_after_gradient_step),
            projected_weights=projected,
            ranks=ranks,
            selected_mask=selected_mask,
            created_at=event.created_at or utc_now_iso(),
        )
        self.snapshots.append(snapshot)
        self._recorded_projections.add(event.projection_index)
        self.csv_rows.extend(csv_rows)
        self.manifest.projections = self.manifest.projections + (
            {
                "projection_index": snapshot.projection_index,
                "step": snapshot.step,
                "is_final_projection": snapshot.is_final_projection,
            },
        )
        self._diagnose_projection(snapshot)
        self._try_write(self._projections_writer.write, snapshot.to_dict())
        return snapshot

    def _diagnose_projection(self, snapshot: ProjectionSnapshot) -> None:
        """Threshold-based concentration warnings (thresholds are parameters)."""
        metrics = snapshot.metrics
        thresholds = self.thresholds
        if metrics.top_1_share > thresholds.top_share_warning:
            self.add_warning(
                WarningCode.HIGH_CONCENTRATION,
                f"projection {snapshot.projection_index}: top-1 normalized importance "
                f"{metrics.top_1_share:.3f} exceeds {thresholds.top_share_warning}",
                step=snapshot.step,
            )
        if metrics.effective_module_count < thresholds.min_effective_modules:
            self.add_warning(
                WarningCode.LOW_EFFECTIVE_MODULES,
                f"projection {snapshot.projection_index}: effective module count "
                f"{metrics.effective_module_count:.2f} < {thresholds.min_effective_modules}",
                step=snapshot.step,
            )
        zero_fraction = metrics.zero_rank_count / metrics.module_count if metrics.module_count else 0.0
        if zero_fraction > thresholds.zero_rank_fraction_warning:
            self.add_warning(
                WarningCode.ZERO_RANK_MAJORITY,
                f"projection {snapshot.projection_index}: {zero_fraction:.1%} of modules have zero rank",
                step=snapshot.step,
            )
        if metrics.max_rank > thresholds.max_rank_multiple_warning * self.run_metadata.default_lora_rank:
            self.add_warning(
                WarningCode.RANK_ABOVE_BUDGET,
                f"projection {snapshot.projection_index}: max rank {metrics.max_rank} exceeds "
                f"{thresholds.max_rank_multiple_warning}x default_lora_rank",
                step=snapshot.step,
            )
        if len(self.snapshots) >= 2:
            previous = self.snapshots[-2]
            if (
                metrics.top_1_share > thresholds.top_share_warning
                or metrics.effective_module_count < thresholds.min_effective_modules
            ) and abs(previous.metrics.top_k_share - metrics.top_k_share) < thresholds.winner_take_all_delta:
                self.add_warning(
                    WarningCode.WINNER_TAKE_ALL,
                    f"top-k share barely changed between projections "
                    f"{previous.projection_index} and {snapshot.projection_index}; "
                    "the distribution looks static (winner-take-all)",
                    step=snapshot.step,
                )

    # ------------------------------------------------------------------ plan/artifacts

    def store_layer_plan(self, plan: LayerPlan) -> None:
        if self.layer_plan is None:
            self.layer_plan = plan

    def build_layer_plan(self, *, model: Any = None) -> LayerPlan:
        """Freeze the final projection into a plan (cached, idempotent)."""
        if self.layer_plan is not None:
            return self.layer_plan
        plan = finalize_layer_plan(self, self.registry, top_k=self.top_k)
        from simplexlora.calibration.plan import validate_layer_plan

        validate_layer_plan(plan, model=model, adapter_name=self.registry.adapter_name, strict=True)
        return plan

    def csv_text(self) -> str:
        """Deterministic CSV rendering of all recorded rows."""
        rows = sorted(self.csv_rows, key=lambda row: (row[0], row[3], row[4], row[2]))
        lines = [",".join(CSV_COLUMNS)]
        for row in rows:
            lines.append(",".join(_csv_cell(value) for value in row))
        return "\n".join(lines) + "\n"

    def register_artifact(self, name: str, text: str) -> None:
        """Atomically write one artifact and remember its SHA-256."""
        from simplexlora.calibration.models import atomic_write_text

        try:
            digest = atomic_write_text(self.output_dir / name, text, overwrite=True)
        except Exception as error:
            self.add_warning(WarningCode.WRITE_ERROR, f"failed to write {name}: {error}")
            if self.fail_on_write_error:
                raise
            return
        self._artifact_checksums[name] = digest

    def _atomic_write(self, path: Path, text: str) -> str:
        try:
            return atomic_write_text(path, text, overwrite=self.overwrite or path.name == MANIFEST_FILENAME)
        except Exception as error:
            self.add_warning(WarningCode.WRITE_ERROR, f"failed to write {path.name}: {error}")
            if self.fail_on_write_error:
                raise
            return ""

    def _try_write(self, writer: Any, document: dict[str, Any]) -> None:
        try:
            writer(document)
        except CalibrationRecorderError:
            raise
        except Exception as error:
            self.add_warning(WarningCode.WRITE_ERROR, f"JSONL write failed: {error}")
            if self.fail_on_write_error:
                raise CalibrationRecorderError(f"JSONL write failed: {error}") from error

    def _write_manifest(self) -> None:
        self.manifest.updated_at = utc_now_iso()
        self.manifest.artifacts = dict(self._artifact_checksums)
        self._atomic_write(self.output_dir / MANIFEST_FILENAME, dump_json(self.manifest.to_dict()))

    def write_artifacts(
        self,
        *,
        plan: LayerPlan,
        handoff: Any | None,
    ) -> Any:
        """Write all final artifacts atomically; the manifest is updated last."""
        report = render_calibration_report(
            manifest=self.manifest,
            snapshots=self.snapshots,
            plan=plan,
            handoff=handoff,
            top_k=self.top_k,
            thresholds=self.thresholds,
        )
        self.register_artifact(CSV_FILENAME, self.csv_text())
        self.register_artifact(LAYER_PLAN_FILENAME, dump_json(plan.to_dict()))
        if handoff is not None:
            self.register_artifact(UNSLOTH_HANDOFF_FILENAME, dump_json(handoff.to_dict()))
        self.register_artifact(REPORT_FILENAME, report)
        self.handoff = handoff
        self._write_manifest()

    # ------------------------------------------------------------------ lifecycle

    @property
    def artifact_paths(self) -> Any:
        from simplexlora.calibration.models import ArtifactPaths

        return ArtifactPaths(
            output_dir=self.output_dir,
            manifest=self.output_dir / MANIFEST_FILENAME,
            steps=self.output_dir / STEPS_FILENAME,
            projections=self.output_dir / PROJECTIONS_FILENAME,
            csv=self.output_dir / CSV_FILENAME,
            report=self.output_dir / REPORT_FILENAME,
            layer_plan=self.output_dir / LAYER_PLAN_FILENAME,
            unsloth_handoff=self.output_dir / UNSLOTH_HANDOFF_FILENAME if self.handoff is not None else None,
            checksums=dict(self._artifact_checksums),
        )

    def close(
        self,
        *,
        failed: bool = False,
        interrupted: bool = False,
        stop_reason: StopReason | None = None,
    ) -> None:
        """Flush JSONL, compute checksums and persist the final manifest."""
        if self._closed:
            return
        self._closed = True

        if stop_reason is None:
            if failed:
                status = RunStatus.PARTIAL if self.snapshots else RunStatus.FAILED
                stop_reason = StopReason.ERROR
            elif interrupted:
                status = RunStatus.PARTIAL if self.snapshots else RunStatus.FAILED
                stop_reason = StopReason.INTERRUPTED
            else:
                status = (
                    RunStatus.COMPLETED
                    if self.calibration_completed
                    else (RunStatus.PARTIAL if self.snapshots else RunStatus.FAILED)
                )
                stop_reason = StopReason.TRAINING_COMPLETE
        else:
            status = (
                RunStatus.COMPLETED
                if self.calibration_completed
                else (RunStatus.PARTIAL if self.snapshots else RunStatus.FAILED)
            )

        self.manifest.status = status
        self.manifest.stop_reason = stop_reason or StopReason.NOT_STOPPED
        self.manifest.calibration_completed = self.calibration_completed
        self.manifest.warnings = tuple(self.warnings)
        if not self.calibration_completed and status != RunStatus.COMPLETED:
            self.add_warning(
                WarningCode.INCOMPLETE_RUN,
                "the run ended before the final projection was recorded",
            )

        self._steps_writer.close()
        self._projections_writer.close()

        if self.layer_plan is not None:
            # Re-render the report with the final status and warnings.
            self.register_artifact(
                REPORT_FILENAME,
                render_calibration_report(
                    manifest=self.manifest,
                    snapshots=self.snapshots,
                    plan=self.layer_plan,
                    handoff=self.handoff,
                    top_k=self.top_k,
                    thresholds=self.thresholds,
                ),
            )
            if not self.calibration_completed:
                self.manifest.extra["handoff_usable"] = False

        from simplexlora.calibration.models import sha256_text

        for name, path in (
            (STEPS_FILENAME, self.output_dir / STEPS_FILENAME),
            (PROJECTIONS_FILENAME, self.output_dir / PROJECTIONS_FILENAME),
        ):
            try:
                self._artifact_checksums[name] = sha256_text(path.read_text(encoding="utf-8"))
            except OSError:
                continue
        self._write_manifest()


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(character in text for character in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_calibration_artifacts(
    *,
    recorder: CalibrationRecorder,
    plan: LayerPlan,
    handoff: Any | None,
    output_dir: Path,
) -> Any:
    """Write all final artifacts of a finished calibration atomically.

    Centralizes serialization, ordering and checksums; existing successful
    artifacts are only replaced when the recorder was created with
    ``overwrite=True``. The manifest is written last.
    """
    if Path(output_dir) != recorder.output_dir:
        raise CalibrationRecorderError(
            f"output_dir {output_dir} does not match the recorder directory {recorder.output_dir}"
        )
    recorder.write_artifacts(plan=plan, handoff=handoff)
    return recorder.artifact_paths


class _OptimizerObserverBridge:
    """Adapts ``SimplexAdamW`` events to the recorder and session.

    Reorders optimizer-position arrays into registry order and tracks losses
    between projections. Does not touch tensors after the constructor.
    """

    def __init__(self, session: CalibrationSession) -> None:
        self._session = session
        self._last_step_time: float | None = None
        self._losses_since_projection: list[float] = []

    # -- optimizer position -> registry index mapping ------------------------

    def _reorder(self, values: Sequence[float], positions: Sequence[int]) -> tuple[Any, ...]:
        registry = self._session.registry
        assert registry is not None, "observer bridge requires an enabled session"
        registry_length = len(registry)
        reordered = [None] * registry_length
        for value, registry_index in zip(values, positions, strict=True):
            reordered[registry_index] = value
        if any(item is None for item in reordered):
            raise CalibrationRecorderError("event arrays do not cover all registry modules")
        return tuple(reordered)

    def on_optimizer_step(
        self,
        *,
        step: int,
        loss: float | None,
        learning_rate: float,
        lora_grad_norm: float,
        weight_grad_norm: float,
        is_projection_step: bool,
    ) -> None:
        now = time.monotonic()
        elapsed = 0.0 if self._last_step_time is None else now - self._last_step_time
        self._last_step_time = now
        if loss is not None:
            self._losses_since_projection.append(float(loss))
        recorder = self._session.recorder
        assert recorder is not None, "observer bridge requires an enabled session"
        recorder.record_step(
            step=step,
            epoch=None,
            loss=loss,
            learning_rate=learning_rate,
            lora_grad_norm=lora_grad_norm,
            weight_grad_norm=weight_grad_norm,
            elapsed_seconds=elapsed,
            is_projection_step=is_projection_step,
        )

    def on_projection(self, event: ProjectionEvent) -> None:
        positions = self._session.optimizer_module_positions
        losses = [value for value in self._losses_since_projection if value is not None]
        loss_at = self._losses_since_projection[-1] if self._losses_since_projection else None
        loss_window = math.fsum(losses) / len(losses) if losses else None

        event = dataclasses.replace(
            event,
            weights_before_step=self._reorder(event.weights_before_step, positions),
            weights_after_gradient_step=self._reorder(event.weights_after_gradient_step, positions),
            weights_projected=self._reorder(event.weights_projected, positions),
            ranks=self._reorder(event.ranks, positions),
            selected_mask=self._reorder(event.selected_mask, positions),
            weight_grad_norms=self._reorder(event.weight_grad_norms, positions),
            lora_grad_norms=self._reorder(event.lora_grad_norms, positions),
            loss_at_projection=loss_at,
            loss_window=loss_window,
        )
        self._losses_since_projection = []
        recorder = self._session.recorder
        assert recorder is not None, "observer bridge requires an enabled session"
        snapshot = recorder.record_projection(event)
        if snapshot.is_final_projection:
            self._session.on_final_projection()

    def on_final_rank_update(self) -> None:
        self._session.on_final_rank_update()


class CalibrationSession:
    """Wires registry, recorder, optimizer observer and trainer callback together."""

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        registry: LayerRegistry | None,
        recorder: CalibrationRecorder | None,
        adapter_name: str,
        top_k: int,
        handoff_options: dict[str, Any] | None = None,
        thresholds: ReportThresholds | None = None,
        enabled: bool = True,
    ) -> None:
        if enabled:
            if recorder is None or registry is None:
                raise CalibrationRecorderError("an enabled session requires a recorder and a registry")
            if getattr(optimizer, "calibration_observer", None) is not None:
                raise CalibrationRecorderError(
                    "the optimizer already has a calibration observer; attaching twice is forbidden"
                )

        self.model = model
        self.optimizer = optimizer
        self.registry = registry
        self.recorder = recorder
        self.adapter_name = adapter_name
        self.top_k = top_k
        self.handoff_options = dict(handoff_options or {})
        self.thresholds = thresholds or ReportThresholds()
        self.enabled = enabled
        self.plan: LayerPlan | None = None
        self._closed = False

        if enabled:
            assert registry is not None
            self.optimizer_module_positions = _resolve_optimizer_module_positions(
                optimizer, model, registry, adapter_name
            )
            self.bridge = _OptimizerObserverBridge(self)
            optimizer.calibration_observer = self.bridge
        else:
            self.optimizer_module_positions = ()
            self.bridge = None

    # ------------------------------------------------------------------ events

    def on_final_projection(self) -> None:
        """Freeze and persist the plan before ``_final_lora_rank_update``."""
        if not self.enabled or self.plan is not None:
            return
        assert self.recorder is not None, "session is enabled"
        self.plan = self.recorder.build_layer_plan(model=self.model)

        from simplexlora.calibration.handoff import build_unsloth_handoff

        handoff = build_unsloth_handoff(self.plan, **self.handoff_options)
        self.recorder.write_artifacts(plan=self.plan, handoff=handoff)

    def on_final_rank_update(self) -> None:
        """Verify the applied plan right after ``_final_lora_rank_update``."""
        if not self.enabled or self.plan is None:
            return
        assert self.recorder is not None, "session is enabled"
        report = verify_applied_plan(self.model, self.plan, adapter_name=self.adapter_name)
        self.recorder.verification_report = report
        self.recorder.manifest.extra["verification"] = report.to_dict()
        if not report.ok:
            self.recorder.add_warning(
                WarningCode.PLAN_VERIFICATION_FAILED,
                "the plan does not match the applied model ranks; the downstream handoff is unusable",
            )
            self.recorder.manifest.extra["handoff_usable"] = False
            from simplexlora.calibration.models import CalibrationValidationError

            raise CalibrationValidationError(report)
        self.recorder.manifest.extra["handoff_usable"] = True
        self.recorder.calibration_completed = True

    # ------------------------------------------------------------------ lifecycle

    @property
    def artifact_paths(self) -> Any:
        if self.recorder is None:
            return None
        return self.recorder.artifact_paths

    def finalize(self) -> LayerPlan | None:
        """Public, idempotent entry point used after training ends."""
        if not self.enabled:
            return None
        assert self.recorder is not None, "session is enabled"
        if self.plan is None and self.recorder.layer_plan is not None:
            self.plan = self.recorder.layer_plan
        return self.plan

    def close(
        self,
        *,
        failed: bool = False,
        interrupted: bool = False,
        stop_reason: StopReason | None = None,
    ) -> None:
        """Persist the final/partial manifest and detach the observer."""
        if self._closed or not self.enabled:
            self._closed = True
            return
        self._closed = True
        assert self.recorder is not None, "session is enabled"
        try:
            self.optimizer.calibration_observer = None
        except AttributeError:  # pragma: no cover - optimizers always accept the attribute
            pass
        self.recorder.close(failed=failed, interrupted=interrupted, stop_reason=stop_reason)

    def __enter__(self) -> CalibrationSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is None:
            self.close()
        elif issubclass(exc_type, KeyboardInterrupt):
            self.close(interrupted=True)
        else:
            self.close(failed=True)
        return False


def _resolve_optimizer_module_positions(
    optimizer: Any,
    model: Any,
    registry: LayerRegistry,
    adapter_name: str,
) -> tuple[int, ...]:
    """Map each ``weight_params`` entry to its registry module index."""
    weight_groups = [group for group in optimizer.param_groups if group.get("name") == "weight_params"]
    if len(weight_groups) != 1:
        raise CalibrationRecorderError(
            f"expected exactly one optimizer group named 'weight_params', found {len(weight_groups)}"
        )
    params = list(weight_groups[0]["params"])

    parameter_to_name: dict[int, str] = {}
    for name, module in model.named_modules():
        if not isinstance(module, SimplexLoraLinear):
            continue
        if not hasattr(module, "lora_weight"):
            continue
        for adapter_key, weight in module.lora_weight.items():
            if adapter_key != adapter_name:
                continue
            canonical_name = _registry_name_for_module(model, name, registry)
            parameter_to_name[id(weight)] = canonical_name

    positions: list[int] = []
    name_to_index = dict(registry.module_name_to_index)
    for parameter in params:
        module_name = parameter_to_name.get(id(parameter))
        if module_name is None or module_name not in name_to_index:
            raise CalibrationRecorderError(
                "an optimizer weight parameter does not map to any registry module; "
                "the optimizer and the registry disagree"
            )
        positions.append(name_to_index[module_name])
    if len(positions) != len(registry):
        raise CalibrationRecorderError(
            f"optimizer has {len(positions)} weight parameters, registry has {len(registry)} modules"
        )
    return tuple(positions)


def _registry_name_for_module(model: Any, peft_module_name: str, registry: LayerRegistry) -> str:
    for identity in registry.identities:
        if identity.peft_module_name == peft_module_name:
            return identity.module_name
    raise CalibrationRecorderError(f"module {peft_module_name!r} is a SimplexLoraLinear but missing from the registry")


def create_calibration_session(
    *,
    model: Any,
    optimizer: Any,
    args: Any,
    run_metadata: CalibrationRunMetadata,
) -> CalibrationSession:
    """Build a ready-to-use session from CLI arguments (thin wiring helper).

    With ``--calibration-no-artifacts`` no observer is created and the optimizer
    keeps its exact original behavior.
    """
    from simplexlora.calibration.cli import handoff_options_from_args

    no_artifacts = bool(getattr(args, "calibration_no_artifacts", False))
    adapter_name = run_metadata.adapter_name
    top_k = int(getattr(args, "calibration_top_k", 20))

    if no_artifacts:
        # Observation disabled: no recorder, no observer, no JSONL.
        return CalibrationSession(
            model=model,
            optimizer=optimizer,
            registry=None,  # type: ignore[arg-type]
            recorder=None,
            adapter_name=adapter_name,
            top_k=top_k,
            enabled=False,
        )

    registry = build_registry_for_model(model, adapter_name)
    recorder = CalibrationRecorder(
        registry=registry,
        output_dir=calibration_output_dir(args),
        run_metadata=run_metadata,
        top_k=top_k,
        fail_on_write_error=bool(getattr(args, "calibration_fail_on_write_error", True)),
        overwrite=bool(getattr(args, "calibration_overwrite", False)),
    )
    return CalibrationSession(
        model=model,
        optimizer=optimizer,
        registry=registry,
        recorder=recorder,
        adapter_name=adapter_name,
        top_k=top_k,
        handoff_options=handoff_options_from_args(args),
    )


def build_registry_for_model(model: Any, adapter_name: str) -> LayerRegistry:
    """Public registry builder used by sessions and experiments."""
    from simplexlora.calibration.registry import build_layer_registry

    return build_layer_registry(model, adapter_name)


def attach_calibration_session(
    model: Any,
    optimizer: Any,
    training_args: Any,
    *,
    model_id: str,
    data: dict[str, Any] | None = None,
    device: Any = None,
) -> CalibrationSession:
    """Thin wiring helper for dataclass-based experiment CLIs (glue/nlg/squad/llm).

    Validates the calibration arguments, builds run metadata and attaches the
    observer to the optimizer. Returns a disabled no-op session when artifacts
    are switched off.
    """
    from simplexlora.calibration.cli import (
        build_run_metadata,
        configure_calibration_only,
        validate_calibration_arguments,
    )

    validate_calibration_arguments(training_args)
    configure_calibration_only(training_args)
    training_args.model_id = model_id
    metadata = build_run_metadata(
        model,
        adapter_name="default",
        args=training_args,
        data=data,
        device=device,
    )
    return create_calibration_session(model=model, optimizer=optimizer, args=training_args, run_metadata=metadata)


def disabled_calibration_session(
    model: Any,
    optimizer: Any,
    *,
    adapter_name: str = "default",
) -> CalibrationSession:
    """A no-op session for strategies without SimplexAdamW calibration."""
    return CalibrationSession(
        model=model,
        optimizer=optimizer,
        registry=None,  # type: ignore[arg-type]
        recorder=None,
        adapter_name=adapter_name,
        top_k=20,
        enabled=False,
    )

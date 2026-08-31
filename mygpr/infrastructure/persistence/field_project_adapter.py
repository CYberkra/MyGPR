#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infrastructure adapter over the verified hybrid FieldProjectStore."""
from __future__ import annotations

from pathlib import Path
import re
from threading import RLock
from typing import Any, Mapping, Sequence

import numpy as np

from core.field_project_backup import backup_project_archive, restore_project_archive
from core.field_project_models import FieldLineRecord, validate_line_id
from core.field_project_operations import update_project_metadata, delete_project_line, batch_import_line_data
from core.field_project_source_ops import check_project_source_files, relink_project_line_source, export_project_source_manifest_csv
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.field_report_export import generate_project_report_package
from core.processing_artifact_index import ProcessingArtifactRecord, index_processing_artifacts
from core.sensor_sync import SensorSyncConfig
from core.sensor_sync_service import synchronize_project_line_sensors
from core.security_paths import resolve_managed_path
from core.project_integrity import ProjectIntegrityAuditor
from mygpr.application.jobs.context import ExecutionContext
from mygpr.infrastructure.persistence.interpretation_adapter import InterpretationPersistenceMixin
from mygpr.infrastructure.persistence.spatial_adapter import SpatialPersistenceMixin
from mygpr.application.project.ports import ProjectRepositoryPort, ProjectSessionPort
from mygpr.domain.reporting.models import ReportPackage
from mygpr.domain.acquisition.models import ProjectSensorSyncResult, SensorSyncSettings
from mygpr.domain.project.models import (
    IntegrityIssue,
    IntegrityReport,
    LineDatasetInfo,
    ProjectArtifact,
    ProjectBackup,
    ProjectLine,
    ProjectLineData,
    ProjectRestore,
    ProjectSummary,
    ProjectMetadata, LineQualityIssue, LineQualityReport, SourceFileStatus, LineDeleteResult,
    BatchImportItemResult, BatchImportSummary,
)


def _report_legacy_progress(
    context: ExecutionContext,
    first: object,
    second: object,
    third: object,
) -> None:
    """Bridge historical (stage,done,total) and (done,total,message) callbacks."""
    if isinstance(first, str):
        context.report_progress(int(second), int(third), first)
    else:
        context.report_progress(int(first), int(second), str(third))


def _load_trace_metadata(store: FieldProjectStore, line_id: str) -> dict[str, np.ndarray]:
    """Load synchronized trace metadata through a validated project-relative path."""
    try:
        record = store.get_line(line_id)
    except KeyError:
        return {}
    relative = str(getattr(record, "trace_metadata_path", "") or "").strip()
    if not relative:
        return {}
    path = resolve_managed_path(store.root, relative, require_exists=True, require_file=True)
    with np.load(path, allow_pickle=False) as payload:
        result = {str(key): np.array(payload[key], copy=True) for key in payload.files}
    expected = int(getattr(record, "trace_count", 0) or 0)
    if expected > 0:
        invalid = [key for key, value in result.items() if np.asarray(value).ndim != 1 or np.asarray(value).size != expected]
        if invalid:
            raise ValueError(f"trace metadata length mismatch for {line_id}: {invalid}")
    return result


def _line_record(record: FieldLineRecord) -> ProjectLine:
    quality = record.data_quality.value if hasattr(record.data_quality, "value") else record.data_quality
    return ProjectLine(
        line_id=record.line_id,
        name=record.name,
        length_m=record.length_m,
        sample_count=record.raw_rows,
        trace_count=record.trace_count,
        data_quality=str(quality or ""),
        processing_status=record.processing_status,
        data_format=record.data_format,
        raw_size_mb=record.raw_size_mb,
        updated_at=record.updated_at,
    )



def _quality_report(record: Any) -> LineQualityReport:
    return LineQualityReport(
        line_id=record.line_id, status=str(getattr(record.status, "value", record.status)), status_label=record.status_label, checked_at=record.checked_at,
        sample_count=record.sample_count, trace_count=record.trace_count, time_window_ns=record.time_window_ns,
        length_m=record.length_m, amplitude_min=record.amplitude_min, amplitude_max=record.amplitude_max,
        amplitude_p995=record.amplitude_p995, nan_ratio=record.nan_ratio, finite_ratio=record.finite_ratio,
        trajectory_points=record.trajectory_points, orientation=str(getattr(record.orientation, "value", record.orientation)),
        orientation_message=record.orientation_message, suggested_action=record.suggested_action,
        issues=tuple(LineQualityIssue(i.severity, i.code, i.message, i.suggestion) for i in record.issues),
        sampled=record.sampled, evaluated_value_count=record.evaluated_value_count,
    )

def _source_status(record: Any) -> SourceFileStatus:
    return SourceFileStatus(
        line_id=record.line_id, role=record.role, source_path=record.source_path, source_filename=record.source_filename,
        source_size_bytes=int(record.source_size_bytes), import_mode=record.import_mode, project_raw_path=record.project_raw_path,
        status=record.status, status_label=record.status_label, last_checked_at=record.last_checked_at, warning=record.warning,
        hash_policy=record.hash_policy, full_hash_status=record.full_hash_status,
    )

def _artifact(record: ProcessingArtifactRecord) -> ProjectArtifact:
    return ProjectArtifact(
        artifact_id=record.artifact_id,
        line_id=record.line_id,
        name=record.method_name or record.method_id or record.artifact_id,
        data_reference=record.data_path,
        branch_id=record.branch_id,
        parent_artifact_id=record.parent_artifact_id,
        method_id=record.method_id,
        method_name=record.method_name,
        status=record.status,
        shape=record.output_shape,
        sha256=record.output_data_sha256,
        created_at=record.created_at,
        manifest=record.to_dict(),
    )


class LegacyFieldProjectSession(InterpretationPersistenceMixin, SpatialPersistenceMixin, ProjectSessionPort):
    """One open project; concrete storage objects remain private to this adapter."""

    def __init__(self, store: FieldProjectStore) -> None:
        self._store = store
        self._lock = RLock()

    @property
    def summary(self) -> ProjectSummary:
        manifest = self._store.manifest
        return ProjectSummary(
            project_id=manifest.project_id,
            name=manifest.name,
            root_path=str(self._store.root),
            schema=manifest.schema,
            storage_backend=manifest.storage_backend,
            revision=manifest.revision,
            status=manifest.status,
            read_only=self._store.read_only,
            line_count=len(manifest.lines),
        )

    def close(self) -> None:
        with self._lock:
            self._store.close()


    def get_metadata(self) -> ProjectMetadata:
        manifest = self._store.manifest
        return ProjectMetadata(manifest.project_id, manifest.name, manifest.location, manifest.operator, manifest.project_no, manifest.device_model, manifest.coordinate_system, manifest.vertical_datum)

    def update_metadata(self, **changes: str | None) -> ProjectMetadata:
        with self._lock:
            update_project_metadata(self._store, **changes)
            return self.get_metadata()

    def list_quality_reports(self) -> Sequence[LineQualityReport]:
        with self._lock:
            return tuple(_quality_report(report) for line in self._store.list_lines() if (report := self._store.load_quality_report(line.line_id)) is not None)

    def run_line_quality_check(self, line_id: str) -> LineQualityReport:
        with self._lock:
            return _quality_report(self._store.run_line_quality_check(validate_line_id(line_id)))

    def run_project_quality_check(self, *, context: ExecutionContext) -> Sequence[LineQualityReport]:
        with self._lock:
            reports = self._store.run_project_quality_check(cancel_requested=context.is_cancelled, progress_callback=context.report_progress)
        return tuple(_quality_report(item) for item in reports)

    def check_source_files(self, *, context: ExecutionContext) -> Sequence[SourceFileStatus]:
        with self._lock:
            records = check_project_source_files(self._store, cancel_requested=context.is_cancelled, progress_callback=context.report_progress)
        return tuple(_source_status(item) for item in records)

    def relink_line_source(self, line_id: str, new_source: Path, *, allow_mismatch: bool, context: ExecutionContext) -> SourceFileStatus:
        with self._lock:
            record = relink_project_line_source(self._store, validate_line_id(line_id), new_source, allow_mismatch=allow_mismatch, cancel_requested=context.is_cancelled, progress_callback=context.report_progress)
        return _source_status(record)

    def export_source_manifest(self, destination: Path | None = None) -> Path:
        with self._lock:
            return export_project_source_manifest_csv(self._store, destination)

    def transpose_line_dataset(self, line_id: str, *, context: ExecutionContext) -> LineQualityReport:
        with self._lock:
            report = self._store.transpose_gpr_dataset(validate_line_id(line_id), cancel_requested=context.is_cancelled, progress_callback=lambda current, total, message: context.report_progress(current, total, message))
        return _quality_report(report)

    def delete_line(self, line_id: str, *, reason: str) -> LineDeleteResult:
        with self._lock:
            result = delete_project_line(self._store, validate_line_id(line_id), reason=reason)
        return LineDeleteResult(result.line_id, result.line_name, tuple(result.deleted_paths), result.remaining_line_count)

    def batch_import_lines(self, sources: Sequence[Path], *, context: ExecutionContext) -> BatchImportSummary:
        def progress(current: int, total: int, item: Any) -> None:
            context.report_progress(current, total, f"{item.line_id}: {item.message}")
        with self._lock:
            result = batch_import_line_data(self._store, list(sources), progress_callback=progress, cancel_requested=context.is_cancelled)
        rows = tuple(BatchImportItemResult(**item.__dict__) for item in result.results)
        return BatchImportSummary(result.total, result.succeeded, result.failed, rows)

    def list_lines(self) -> Sequence[ProjectLine]:
        with self._lock:
            return tuple(_line_record(record) for record in self._store.list_lines())

    def get_line(self, line_id: str) -> ProjectLine:
        safe = validate_line_id(line_id)
        with self._lock:
            return _line_record(self._store.get_line(safe))

    def get_dataset_info(self, line_id: str) -> LineDatasetInfo:
        safe = validate_line_id(line_id)
        with self._lock:
            dataset = self._store.load_gpr_dataset(safe)
            return LineDatasetInfo(
                line_id=safe,
                shape=tuple(int(value) for value in dataset.matrix.shape),
                dtype=str(dataset.matrix.dtype),
                length_m=dataset.length_m,
                time_window_ns=dataset.time_window_ns,
                dielectric_constant=dataset.dielectric_constant,
                source_path=dataset.source_path,
                format_name=dataset.format_name,
                metadata=dataset.to_metadata(),
            )

    def import_line_source(
        self,
        line_id: str,
        source: Path,
        *,
        name: str,
        copy_into_project: bool,
        dielectric_constant: float,
        context: ExecutionContext,
    ) -> ProjectLine:
        safe = validate_line_id(line_id)
        context.raise_if_cancelled()
        with self._lock:
            record = self._store.import_line_file(
                safe,
                source,
                name=name or safe,
                copy_into_project=bool(copy_into_project),
                dielectric_constant=float(dielectric_constant),
                cancel_requested=context.is_cancelled,
                progress_callback=lambda first, second, third: _report_legacy_progress(context, first, second, third),
            )
        return _line_record(record)

    def synchronize_line_sensors(
        self,
        line_id: str,
        *,
        rtk_path: Path,
        trace_timestamps_path: Path | None,
        imu_path: Path | None,
        altimeter_path: Path | None,
        settings: SensorSyncSettings,
        context: ExecutionContext,
    ) -> ProjectSensorSyncResult:
        safe = validate_line_id(line_id)
        context.raise_if_cancelled()
        config = SensorSyncConfig(**settings.to_mapping())
        with self._lock:
            payload = synchronize_project_line_sensors(
                self._store,
                line_id=safe,
                rtk_path=rtk_path,
                trace_timestamps_path=trace_timestamps_path,
                imu_path=imu_path,
                altimeter_path=altimeter_path,
                config=config,
                progress=context.report_progress,
                cancel_checker=context.is_cancelled,
            )
        return ProjectSensorSyncResult(
            project_id=self.summary.project_id,
            line_id=safe,
            trajectory_path=str(payload["trajectory_path"]),
            manifest_path=str(payload["manifest_path"]),
            trace_metadata_path=str(payload["trace_metadata_path"]),
            diagnostics=dict(payload.get("diagnostics") or {}),
            config=dict(payload.get("config") or {}),
            summary=str(payload.get("summary") or ""),
        )

    def save_line_dataset(
        self,
        line_id: str,
        data: np.ndarray,
        *,
        name: str,
        length_m: float,
        time_window_ns: float,
        dielectric_constant: float,
        metadata: Mapping[str, Any],
        context: ExecutionContext,
    ) -> ProjectLine:
        safe = validate_line_id(line_id)
        matrix = np.asarray(data)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError("line dataset must be a non-empty 2D array")
        context.raise_if_cancelled()
        dataset = GPRDataSet.from_matrix(
            safe,
            matrix,
            length_m=length_m if length_m > 0 else None,
            time_window_ns=time_window_ns,
            dielectric_constant=dielectric_constant,
            format_name=str(metadata.get("format_name") or "backend-api"),
            metadata=dict(metadata),
        )
        with self._lock:
            try:
                record = self._store.get_line(safe)
                record.name = str(name or safe)
            except KeyError:
                record = FieldLineRecord(line_id=safe, name=str(name or safe))
            self._store.upsert_line(record)
            self._store.save_gpr_dataset(
                safe,
                dataset,
                cancel_requested=context.is_cancelled,
                progress_callback=context.report_progress,
            )
            return _line_record(self._store.get_line(safe))

    def read_dataset(self, line_id: str) -> ProjectLineData:
        safe = validate_line_id(line_id)
        with self._lock:
            dataset = self._store.load_gpr_dataset(safe)
            matrix = np.asarray(dataset.matrix, dtype=np.float32)
            header = dataset.to_metadata()
            header.update(
                distance_axis_m=np.asarray(dataset.distance_axis_m, dtype=np.float32),
                time_axis_ns=np.asarray(dataset.time_axis_ns, dtype=np.float32),
                depth_axis_m=np.asarray(dataset.depth_axis_m, dtype=np.float32),
            )
            trace_metadata = _load_trace_metadata(self._store, safe)
        return ProjectLineData(
            line_id=safe,
            data=matrix,
            header_info=header,
            trace_metadata=trace_metadata,
        )

    def read_trace_metadata(self, line_id: str) -> Mapping[str, np.ndarray]:
        safe = validate_line_id(line_id)
        with self._lock:
            return _load_trace_metadata(self._store, safe)

    @staticmethod
    def _iter_matrix_blocks(
        matrix: Any,
        *,
        block_rows: int,
        sample_start: int,
        sample_end: int | None,
        trace_start: int,
        trace_end: int | None,
    ):
        iterator = getattr(matrix, "iter_blocks", None)
        if callable(iterator):
            yield from iterator(
                block_rows=block_rows, sample_start=sample_start, sample_end=sample_end,
                trace_start=trace_start, trace_end=trace_end,
            )
            return
        array = np.asarray(matrix)
        row_start = max(0, int(sample_start))
        row_end = array.shape[0] if sample_end is None else min(array.shape[0], int(sample_end))
        col_start = max(0, int(trace_start))
        col_end = array.shape[1] if trace_end is None else min(array.shape[1], int(trace_end))
        step = max(1, int(block_rows))
        for start in range(row_start, row_end, step):
            end = min(start + step, row_end)
            yield start, end, np.asarray(array[start:end, col_start:col_end])

    def iter_dataset_blocks(
        self,
        line_id: str,
        *,
        block_rows: int = 1024,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
    ):
        safe = validate_line_id(line_id)
        with self._lock:
            dataset = self._store.load_gpr_dataset(safe)
            matrix = dataset.matrix
        yield from self._iter_matrix_blocks(
            matrix, block_rows=block_rows, sample_start=sample_start, sample_end=sample_end,
            trace_start=trace_start, trace_end=trace_end,
        )

    def read_window(
        self,
        line_id: str,
        *,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
        max_samples: int = 900,
        max_traces: int = 1800,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        safe = validate_line_id(line_id)
        with self._lock:
            dataset = self._store.load_gpr_dataset(safe)
            return dataset.preview_window(
                sample_start=sample_start,
                sample_end=sample_end,
                trace_start=trace_start,
                trace_end=trace_end,
                max_samples=max_samples,
                max_traces=max_traces,
                normalize=normalize,
            )

    def _artifact_dataset(self, line_id: str, artifact_id: str):
        safe_line = self._validated_line(line_id)
        safe_artifact = str(artifact_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", safe_artifact):
            raise ValueError(f"invalid artifact_id: {artifact_id!r}")
        known = {item.artifact_id for item in index_processing_artifacts(self._store.root, safe_line)}
        if safe_artifact not in known:
            raise KeyError(f"unknown artifact {safe_artifact!r} for line {safe_line!r}")
        return safe_line, safe_artifact, self._store.storage.load_processing_artifact(safe_line, safe_artifact)

    def get_artifact_dataset_info(
        self, line_id: str, artifact_id: str
    ) -> LineDatasetInfo:
        with self._lock:
            safe_line, safe_artifact, dataset = self._artifact_dataset(line_id, artifact_id)
            metadata = dataset.to_metadata()
            metadata["artifact_id"] = safe_artifact
            return LineDatasetInfo(
                line_id=safe_line,
                shape=tuple(int(value) for value in dataset.matrix.shape),
                dtype=str(dataset.matrix.dtype),
                length_m=dataset.length_m,
                time_window_ns=dataset.time_window_ns,
                dielectric_constant=dataset.dielectric_constant,
                source_path=dataset.source_path,
                format_name=dataset.format_name,
                metadata=metadata,
            )

    def read_artifact_dataset(
        self, line_id: str, artifact_id: str
    ) -> ProjectLineData:
        with self._lock:
            safe_line, safe_artifact, dataset = self._artifact_dataset(line_id, artifact_id)
            matrix = np.asarray(dataset.matrix, dtype=np.float32)
            header = dataset.to_metadata()
            header.update(
                artifact_id=safe_artifact,
                distance_axis_m=np.asarray(dataset.distance_axis_m, dtype=np.float32),
                time_axis_ns=np.asarray(dataset.time_axis_ns, dtype=np.float32),
                depth_axis_m=np.asarray(dataset.depth_axis_m, dtype=np.float32),
            )
            trace_metadata = _load_trace_metadata(self._store, safe_line)
            if trace_metadata and any(np.asarray(value).size != dataset.trace_count for value in trace_metadata.values()):
                trace_metadata = {}
        return ProjectLineData(
            line_id=safe_line, data=matrix, header_info=header, trace_metadata=trace_metadata
        )

    def iter_artifact_blocks(
        self,
        line_id: str,
        artifact_id: str,
        *,
        block_rows: int = 1024,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
    ):
        with self._lock:
            _safe_line, _safe_artifact, dataset = self._artifact_dataset(line_id, artifact_id)
            matrix = dataset.matrix
        yield from self._iter_matrix_blocks(
            matrix, block_rows=block_rows, sample_start=sample_start, sample_end=sample_end,
            trace_start=trace_start, trace_end=trace_end,
        )

    def read_artifact_window(
        self,
        line_id: str,
        artifact_id: str,
        *,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
        max_samples: int = 900,
        max_traces: int = 1800,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            _safe_line, _safe_artifact, dataset = self._artifact_dataset(line_id, artifact_id)
            return dataset.preview_window(
                sample_start=sample_start,
                sample_end=sample_end,
                trace_start=trace_start,
                trace_end=trace_end,
                max_samples=max_samples,
                max_traces=max_traces,
                normalize=normalize,
            )

    def save_processing_artifact(
        self,
        line_id: str,
        data: np.ndarray,
        *,
        name: str,
        method_id: str,
        method_name: str,
        params: Mapping[str, Any],
        pipeline: Sequence[Mapping[str, Any]],
        branch_id: str,
        input_dataset: Mapping[str, Any],
        context: ExecutionContext,
    ) -> ProjectArtifact:
        safe = validate_line_id(line_id)
        context.raise_if_cancelled()
        # P1-2：parent_artifact_id 必须出现在 payload 顶层，否则 _base_manifest
        # 只查顶层键 → 恒为 ""，处理谱系（血缘）丢失
        parent_id = str(
            (params or {}).get("parent_artifact_id")
            or (input_dataset or {}).get("parent_artifact_id") or ""
        )
        # P1-1：输出 header（含新时间窗/零点/轴含义）持久化到 manifest，供加载时重建物理轴
        output_header = dict(params.get("output_header") or {})
        payload = {
            "method": method_id,
            "method_name": method_name or name,
            "params": dict(params),
            "branch_id": str(branch_id or f"{safe}:main"),
            "parent_artifact_id": parent_id,
            "output_header": output_header,
            "input_dataset": _json_safe_mapping(input_dataset),
            "manifest": {
                "method_id": method_id,
                "method_name": method_name or name,
                "pipeline": [dict(step) for step in pipeline],
                "artifact_name": name,
                "status": "success",
            },
        }
        with self._lock:
            saved, _ = self._store.save_processed_line(
                safe,
                np.asarray(data),
                payload,
                cancel_requested=context.is_cancelled,
                progress_callback=context.report_progress,
            )
            record = next(
                item
                for item in index_processing_artifacts(self._store.root, safe)
                if item.artifact_id == saved.stem
            )
        return _artifact(record)

    def list_artifacts(self, line_id: str | None = None) -> Sequence[ProjectArtifact]:
        safe = validate_line_id(line_id) if line_id else None
        with self._lock:
            return tuple(_artifact(record) for record in index_processing_artifacts(self._store.root, safe))

    def _validated_line(self, line_id: str) -> str:
        safe = validate_line_id(line_id)
        self._store.get_line(safe)
        return safe

    def list_report_packages(self) -> Sequence[ReportPackage]:
        if not getattr(self._store.storage, "is_hybrid", False):
            return []
        rows = self._store.storage.catalog.list_exports(export_kind="engineering_report")
        result: list[ReportPackage] = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            result.append(ReportPackage(
                package_dir=str(row.get("path") or ""),
                manifest_path=str(metadata.get("manifest_path") or ""),
                generated_at=str(row.get("created_at") or ""),
                file_count=int(metadata.get("file_count") or 0),
                pdf_path=str(metadata.get("pdf_path") or ""),
                html_path=str(metadata.get("html_path") or ""),
                xlsx_path=str(metadata.get("xlsx_path") or ""),
                delivery_zip_path=str(metadata.get("delivery_zip_path") or ""),
                delivery_zip_sha256_path=str(metadata.get("delivery_zip_sha256_path") or ""),
                seal_path=str(metadata.get("seal_path") or ""),
                metadata={**metadata, "project_root": str(self._store.root.resolve())},
            ))
        return tuple(result)

    def audit(
        self,
        *,
        repair_context: bool,
        clean_staging: bool,
        staging_min_age_s: float,
        deep_hash: bool,
    ) -> IntegrityReport:
        with self._lock:
            report = ProjectIntegrityAuditor(self._store).audit(
                repair_context=repair_context,
                clean_staging=clean_staging,
                staging_min_age_s=staging_min_age_s,
                deep_hash=deep_hash,
            )
        issues = tuple(
            IntegrityIssue(
                code=item.code,
                severity=str(item.severity),
                message=item.message,
                module=item.module,
                object_id=item.object_id,
                path=item.path,
                repairable=item.repairable,
                repaired=item.repaired,
                details=item.details,
            )
            for item in report.issues
        )
        return IntegrityReport(
            project_id=self._store.manifest.project_id,
            generated_at=report.generated_at,
            issues=issues,
            repairs=tuple(report.repairs),
            elapsed_ms=report.elapsed_ms,
        )

    def generate_report(
        self,
        *,
        package_name: str | None,
        report_profile: Mapping[str, Any],
        context: ExecutionContext,
    ) -> ReportPackage:
        with self._lock:
            result = generate_project_report_package(
                self._store,
                package_name=package_name,
                report_profile=dict(report_profile),
                cancel_checker=context.is_cancelled,
                progress_callback=context.report_progress,
            )
        metadata = result.to_dict()
        metadata["project_root"] = str(self._store.root.resolve())
        return ReportPackage(
            package_dir=result.package_dir,
            manifest_path=result.manifest_path,
            generated_at=result.generated_at,
            file_count=result.file_count,
            pdf_path=result.pdf_path,
            html_path=result.html_path,
            xlsx_path=result.xlsx_path,
            delivery_zip_path=result.delivery_zip_path,
            delivery_zip_sha256_path=result.delivery_zip_sha256_path,
            seal_path=result.seal_path,
            metadata=metadata,
        )

    def backup(
        self,
        destination_dir: Path | None,
        *,
        require_external_device: bool,
        incremental: bool = False,
        retention_keep: int | None = None,
        context: ExecutionContext,
    ) -> ProjectBackup:
        with self._lock:
            result = backup_project_archive(
                self._store,
                destination_dir,
                cancel_requested=context.is_cancelled,
                progress_callback=context.report_progress,
                require_external_device=require_external_device,
                incremental=incremental,
                retention_keep=retention_keep,
            )
        return ProjectBackup(
            archive_path=result.archive_path,
            file_count=result.file_count,
            size_mb=result.size_mb,
            manifest_sha256=result.manifest_sha256,
            verified=result.verified,
            external_device=result.external_device,
        )


class LegacyFieldProjectRepository(ProjectRepositoryPort):
    """Create/open/restore projects through the established hybrid storage layer."""

    def create(
        self,
        root: Path,
        *,
        name: str,
        location: str,
        operator: str,
        project_no: str,
        device_model: str,
        coordinate_system: str,
        vertical_datum: str,
    ) -> ProjectSessionPort:
        if not str(name).strip():
            raise ValueError("project name must not be empty")
        return LegacyFieldProjectSession(
            FieldProjectStore.create_empty(
                root,
                name=name,
                location=location,
                operator=operator,
                project_no=project_no,
                device_model=device_model,
                coordinate_system=coordinate_system,
                vertical_datum=vertical_datum,
            )
        )

    def open(
        self,
        root: Path,
        *,
        read_only: bool,
        recover_stale_lock: bool,
    ) -> ProjectSessionPort:
        mode = "read_only" if read_only else "auto"
        return LegacyFieldProjectSession(
            FieldProjectStore.open(
                root,
                access_mode=mode,
                recover_stale_lock=recover_stale_lock,
            )
        )

    def restore(
        self,
        archive_path: Path,
        destination_root: Path,
        *,
        project_dir_name: str | None,
    ) -> ProjectRestore:
        result = restore_project_archive(
            archive_path,
            destination_root,
            project_dir_name=project_dir_name,
        )
        return ProjectRestore(
            project_path=result.project_path,
            file_count=result.file_count,
            verified=result.verified,
            source_archive=result.source_archive,
        )


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["LegacyFieldProjectRepository", "LegacyFieldProjectSession"]

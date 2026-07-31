#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project lifecycle, data access, integrity and backup use cases."""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.common.errors import MyGPRError
from mygpr.application.project.ports import ProjectRepositoryPort, ProjectSessionPort
from mygpr.domain.reporting.models import ReportPackage
from mygpr.domain.interpretation.models import (
    BoreholeRecord, InterfaceAnnotation, InterpretationFeature,
)
from mygpr.domain.spatial.models import SpatialResult, SpatialTrack
from mygpr.domain.acquisition.models import ProjectSensorSyncResult, SensorSyncSettings
from mygpr.domain.project.models import (
    IntegrityReport,
    LineDatasetInfo,
    ProjectArtifact,
    ProjectBackup,
    ProjectLine,
    ProjectLineData,
    ProjectRestore,
    ProjectSummary,
)


class ProjectApplicationError(MyGPRError):
    """Stable project application failure exposed through Backend API v1."""

    error_code = "MYGPR_PROJECT_ERROR"
    category = "project"


class ProjectBusyError(ProjectApplicationError):
    """Raised when a project is still leased by queued or running work."""

    error_code = "MYGPR_PROJECT_BUSY"


@dataclass(slots=True)
class ProjectLease:
    """Reference-counted project session lease held across queued/running work."""

    _service: "ProjectService"
    project_id: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._service._release_lease(self.project_id)

    def __enter__(self) -> "ProjectLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.release()


class ProjectService:
    """Own open project sessions without exposing storage implementation objects."""

    def __init__(self, repository: ProjectRepositoryPort) -> None:
        self._repository = repository
        self._sessions: dict[str, ProjectSessionPort] = {}
        self._roots: dict[str, str] = {}
        self._lease_counts: dict[str, int] = {}
        self._lock = RLock()

    def create_project(
        self,
        root: str | Path,
        *,
        name: str,
        location: str = "",
        operator: str = "操作员",
        project_no: str = "",
        device_model: str = "",
        coordinate_system: str = "",
        vertical_datum: str = "",
    ) -> ProjectSummary:
        session = self._repository.create(
            Path(root).expanduser().resolve(),
            name=str(name).strip(),
            location=location,
            operator=operator,
            project_no=project_no,
            device_model=device_model,
            coordinate_system=coordinate_system,
            vertical_datum=vertical_datum,
        )
        return self._register(session)

    def open_project(
        self,
        root: str | Path,
        *,
        read_only: bool = False,
        recover_stale_lock: bool = False,
    ) -> ProjectSummary:
        resolved = str(Path(root).expanduser().resolve())
        with self._lock:
            existing_id = next((pid for pid, value in self._roots.items() if value == resolved), "")
            if existing_id:
                return self._sessions[existing_id].summary
        session = self._repository.open(
            Path(resolved),
            read_only=bool(read_only),
            recover_stale_lock=bool(recover_stale_lock),
        )
        return self._register(session)

    def _register(self, session: ProjectSessionPort) -> ProjectSummary:
        summary = session.summary
        with self._lock:
            existing = self._sessions.get(summary.project_id)
            if existing is not None:
                session.close()
                if self._roots[summary.project_id] != summary.root_path:
                    raise ProjectApplicationError(
                        f"project id already open from another root: {summary.project_id}"
                    )
                return existing.summary
            self._sessions[summary.project_id] = session
            self._roots[summary.project_id] = summary.root_path
            self._lease_counts.setdefault(summary.project_id, 0)
        return summary

    def acquire_lease(self, project_id: str) -> ProjectLease:
        normalized = str(project_id)
        with self._lock:
            if normalized not in self._sessions:
                raise ProjectApplicationError(f"project is not open: {project_id}")
            self._lease_counts[normalized] = self._lease_counts.get(normalized, 0) + 1
        return ProjectLease(self, normalized)

    def active_lease_count(self, project_id: str) -> int:
        with self._lock:
            return int(self._lease_counts.get(str(project_id), 0))

    def _release_lease(self, project_id: str) -> None:
        normalized = str(project_id)
        with self._lock:
            count = self._lease_counts.get(normalized, 0)
            if count <= 1:
                self._lease_counts[normalized] = 0
            else:
                self._lease_counts[normalized] = count - 1

    def close_project(self, project_id: str, *, force: bool = False) -> None:
        normalized = str(project_id)
        with self._lock:
            leases = self._lease_counts.get(normalized, 0)
            if leases and not force:
                raise ProjectBusyError(
                    f"project has {leases} active task lease(s): {normalized}"
                )
            session = self._sessions.pop(normalized, None)
            self._roots.pop(normalized, None)
            self._lease_counts.pop(normalized, None)
        if session is not None:
            session.close()

    def close_all(self, *, force: bool = False) -> None:
        with self._lock:
            busy = {pid: count for pid, count in self._lease_counts.items() if count > 0}
            if busy and not force:
                details = ", ".join(f"{pid}={count}" for pid, count in sorted(busy.items()))
                raise ProjectBusyError(f"projects still have active task leases: {details}")
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._roots.clear()
            self._lease_counts.clear()
        for session in sessions:
            session.close()

    def get_summary(self, project_id: str) -> ProjectSummary:
        return self._session(project_id).summary

    def list_open_projects(self) -> tuple[ProjectSummary, ...]:
        with self._lock:
            return tuple(session.summary for session in self._sessions.values())

    def list_lines(self, project_id: str) -> tuple[ProjectLine, ...]:
        return tuple(self._session(project_id).list_lines())

    def get_line(self, project_id: str, line_id: str) -> ProjectLine:
        return self._session(project_id).get_line(line_id)

    def get_dataset_info(self, project_id: str, line_id: str) -> LineDatasetInfo:
        return self._session(project_id).get_dataset_info(line_id)

    def import_line_source(
        self,
        project_id: str,
        line_id: str,
        source: str | Path,
        *,
        name: str = "",
        copy_into_project: bool = True,
        dielectric_constant: float = 9.0,
        context: ExecutionContext | None = None,
    ) -> ProjectLine:
        return self._session(project_id).import_line_source(
            line_id,
            Path(source).expanduser().resolve(),
            name=name or line_id,
            copy_into_project=bool(copy_into_project),
            dielectric_constant=float(dielectric_constant),
            context=context or ExecutionContext.null(),
        )

    def synchronize_line_sensors(
        self,
        project_id: str,
        line_id: str,
        *,
        rtk_path: str | Path,
        trace_timestamps_path: str | Path | None = None,
        imu_path: str | Path | None = None,
        altimeter_path: str | Path | None = None,
        settings: SensorSyncSettings | None = None,
        context: ExecutionContext | None = None,
    ) -> ProjectSensorSyncResult:
        resolve_optional = lambda value: None if value is None else Path(value).expanduser().resolve()
        return self._session(project_id).synchronize_line_sensors(
            line_id,
            rtk_path=Path(rtk_path).expanduser().resolve(),
            trace_timestamps_path=resolve_optional(trace_timestamps_path),
            imu_path=resolve_optional(imu_path),
            altimeter_path=resolve_optional(altimeter_path),
            settings=settings or SensorSyncSettings(),
            context=context or ExecutionContext.null(),
        )

    def save_line_dataset(
        self,
        project_id: str,
        line_id: str,
        data: np.ndarray,
        *,
        name: str = "",
        length_m: float = 0.0,
        time_window_ns: float = 250.0,
        dielectric_constant: float = 9.0,
        metadata: dict | None = None,
        context: ExecutionContext | None = None,
    ) -> ProjectLine:
        return self._session(project_id).save_line_dataset(
            line_id,
            data,
            name=name or line_id,
            length_m=length_m,
            time_window_ns=time_window_ns,
            dielectric_constant=dielectric_constant,
            metadata=dict(metadata or {}),
            context=context or ExecutionContext.null(),
        )

    def read_dataset(self, project_id: str, line_id: str) -> ProjectLineData:
        return self._session(project_id).read_dataset(line_id)

    def read_trace_metadata(self, project_id: str, line_id: str) -> dict[str, np.ndarray]:
        return {
            str(key): np.array(value, copy=True)
            for key, value in self._session(project_id).read_trace_metadata(line_id).items()
        }

    def iter_dataset_blocks(
        self,
        project_id: str,
        line_id: str,
        *,
        block_rows: int = 1024,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        return self._session(project_id).iter_dataset_blocks(
            line_id,
            block_rows=block_rows,
            sample_start=sample_start,
            sample_end=sample_end,
            trace_start=trace_start,
            trace_end=trace_end,
        )

    def read_window(
        self,
        project_id: str,
        line_id: str,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._session(project_id).read_window(line_id, **kwargs)

    def get_artifact_dataset_info(
        self, project_id: str, line_id: str, artifact_id: str
    ) -> LineDatasetInfo:
        return self._session(project_id).get_artifact_dataset_info(line_id, artifact_id)

    def read_artifact_dataset(
        self, project_id: str, line_id: str, artifact_id: str
    ) -> ProjectLineData:
        return self._session(project_id).read_artifact_dataset(line_id, artifact_id)

    def iter_artifact_blocks(
        self,
        project_id: str,
        line_id: str,
        artifact_id: str,
        *,
        block_rows: int = 1024,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        return self._session(project_id).iter_artifact_blocks(
            line_id, artifact_id, block_rows=block_rows, sample_start=sample_start,
            sample_end=sample_end, trace_start=trace_start, trace_end=trace_end,
        )

    def read_artifact_window(
        self,
        project_id: str,
        line_id: str,
        artifact_id: str,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._session(project_id).read_artifact_window(
            line_id, artifact_id, **kwargs
        )

    def save_processing_artifact(
        self,
        project_id: str,
        line_id: str,
        data: np.ndarray,
        *,
        name: str,
        method_id: str,
        method_name: str,
        params: dict,
        pipeline: Sequence[dict],
        branch_id: str,
        input_dataset: dict,
        context: ExecutionContext | None = None,
    ) -> ProjectArtifact:
        return self._session(project_id).save_processing_artifact(
            line_id,
            data,
            name=name,
            method_id=method_id,
            method_name=method_name,
            params=params,
            pipeline=pipeline,
            branch_id=branch_id,
            input_dataset=input_dataset,
            context=context or ExecutionContext.null(),
        )

    def list_artifacts(
        self,
        project_id: str,
        line_id: str | None = None,
    ) -> tuple[ProjectArtifact, ...]:
        return tuple(self._session(project_id).list_artifacts(line_id))

    def list_interpretation_features(
        self, project_id: str, line_id: str
    ) -> tuple[InterpretationFeature, ...]:
        return tuple(self._session(project_id).list_interpretation_features(line_id))

    def replace_interpretation_features(
        self, project_id: str, line_id: str, features: Sequence[InterpretationFeature]
    ) -> None:
        self._session(project_id).replace_interpretation_features(line_id, tuple(features))

    def load_interface_annotation(
        self, project_id: str, line_id: str, *, create: bool = True
    ) -> InterfaceAnnotation | None:
        return self._session(project_id).load_interface_annotation(line_id, create=create)

    def save_interface_annotation(
        self, project_id: str, annotation: InterfaceAnnotation
    ) -> InterfaceAnnotation:
        return self._session(project_id).save_interface_annotation(annotation)

    def list_boreholes(self, project_id: str) -> tuple[BoreholeRecord, ...]:
        return tuple(self._session(project_id).list_boreholes())

    def save_borehole(self, project_id: str, borehole: BoreholeRecord) -> BoreholeRecord:
        return self._session(project_id).save_borehole(borehole)

    def delete_borehole(self, project_id: str, borehole_id: str) -> bool:
        return self._session(project_id).delete_borehole(borehole_id)

    def depth_at_samples(
        self, project_id: str, line_id: str, samples: np.ndarray
    ) -> np.ndarray:
        return np.asarray(self._session(project_id).depth_at_samples(line_id, samples), dtype=float)

    def load_spatial_tracks(self, project_id: str) -> tuple[SpatialTrack, ...]:
        return tuple(self._session(project_id).load_spatial_tracks())

    def list_spatial_results(self, project_id: str) -> tuple[SpatialResult, ...]:
        return tuple(self._session(project_id).list_spatial_results())

    def spatial_preflight(
        self, project_id: str, *, line_ids: Sequence[str] | None = None,
        generate_surface: bool = True,
    ) -> dict:
        return dict(self._session(project_id).spatial_preflight(
            line_ids=line_ids, generate_surface=generate_surface
        ))

    def create_spatial_result(
        self, project_id: str, *, name: str, line_ids: Sequence[str] | None = None,
        velocity_m_per_ns: float | None = None, generate_surface: bool = True,
    ) -> SpatialResult:
        return self._session(project_id).create_spatial_result(
            name=name, line_ids=line_ids, velocity_m_per_ns=velocity_m_per_ns,
            generate_surface=generate_surface,
        )

    def set_current_spatial_result(self, project_id: str, result_id: str) -> None:
        self._session(project_id).set_current_spatial_result(result_id)

    def list_report_packages(self, project_id: str) -> tuple[ReportPackage, ...]:
        return tuple(self._session(project_id).list_report_packages())

    def audit_project(
        self,
        project_id: str,
        *,
        repair_context: bool = False,
        clean_staging: bool = False,
        staging_min_age_s: float = 3600.0,
        deep_hash: bool = False,
    ) -> IntegrityReport:
        return self._session(project_id).audit(
            repair_context=repair_context,
            clean_staging=clean_staging,
            staging_min_age_s=staging_min_age_s,
            deep_hash=deep_hash,
        )

    def generate_report(
        self,
        project_id: str,
        *,
        package_name: str | None = None,
        report_profile: dict | None = None,
        context: ExecutionContext | None = None,
    ) -> ReportPackage:
        return self._session(project_id).generate_report(
            package_name=package_name,
            report_profile=dict(report_profile or {}),
            context=context or ExecutionContext.null(),
        )

    def backup_project(
        self,
        project_id: str,
        destination_dir: str | Path | None = None,
        *,
        require_external_device: bool = False,
        context: ExecutionContext | None = None,
    ) -> ProjectBackup:
        destination = None if destination_dir is None else Path(destination_dir).expanduser().resolve()
        return self._session(project_id).backup(
            destination,
            require_external_device=require_external_device,
            context=context or ExecutionContext.null(),
        )

    def restore_project(
        self,
        archive_path: str | Path,
        destination_root: str | Path,
        *,
        project_dir_name: str | None = None,
        open_after_restore: bool = True,
    ) -> ProjectRestore:
        restored = self._repository.restore(
            Path(archive_path).expanduser().resolve(),
            Path(destination_root).expanduser().resolve(),
            project_dir_name=project_dir_name,
        )
        if open_after_restore:
            self.open_project(restored.project_path)
        return restored

    def _session(self, project_id: str) -> ProjectSessionPort:
        with self._lock:
            session = self._sessions.get(str(project_id))
        if session is None:
            raise ProjectApplicationError(f"project is not open: {project_id}")
        return session


__all__ = ["ProjectApplicationError", "ProjectBusyError", "ProjectLease", "ProjectService"]

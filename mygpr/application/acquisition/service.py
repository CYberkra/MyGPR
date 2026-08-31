#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless acquisition, sensor synchronization and motion workflow use cases."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mygpr.application.acquisition.ports import (
    AcquisitionReaderPort,
    SensorSidecarParserPort,
    SensorSynchronizerPort,
)
from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.common.errors import MyGPRError
from mygpr.application.project.service import ProjectService
from mygpr.domain.acquisition.models import (
    AcquisitionDataset,
    ImportedLineResult,
    ImportPreflight,
    ProjectSensorSyncResult,
    SensorKind,
    SensorStream,
    SensorSyncSettings,
    SynchronizedSensorData,
)
from mygpr.domain.acquisition.motion import MotionCompensationProfile, build_motion_pipeline
from mygpr.domain.processing.models import PipelineDefinition


class AcquisitionApplicationError(MyGPRError):
    error_code = "MYGPR_ACQUISITION_ERROR"
    category = "acquisition"
    """Raised when an acquisition use case cannot satisfy its contract."""


class AcquisitionService:
    def __init__(
        self,
        reader: AcquisitionReaderPort,
        parser: SensorSidecarParserPort,
        synchronizer: SensorSynchronizerPort,
        projects: ProjectService,
    ) -> None:
        self._reader = reader
        self._parser = parser
        self._synchronizer = synchronizer
        self._projects = projects

    def preflight(
        self,
        source: str | Path,
        *,
        line_id: str = "L01",
        dielectric_constant: float = 9.0,
        context: ExecutionContext | None = None,
    ) -> ImportPreflight:
        execution_context = context or ExecutionContext.null()
        return self._reader.preflight(
            Path(source).expanduser().resolve(),
            line_id=line_id,
            dielectric_constant=dielectric_constant,
            context=execution_context,
        )

    def load_dataset(
        self,
        source: str | Path,
        *,
        line_id: str,
        length_m: float = 0.0,
        dielectric_constant: float = 9.0,
        context: ExecutionContext | None = None,
    ) -> AcquisitionDataset:
        execution_context = context or ExecutionContext.null()
        with self._reader.open_dataset(
            Path(source).expanduser().resolve(),
            line_id=line_id,
            length_m=length_m,
            dielectric_constant=dielectric_constant,
            context=execution_context,
        ) as loaded:
            # Standalone callers own an in-memory copy after the context closes.
            return AcquisitionDataset(
                line_id=loaded.line_id,
                data=np.array(loaded.data, copy=True),
                length_m=loaded.length_m,
                time_window_ns=loaded.time_window_ns,
                dielectric_constant=loaded.dielectric_constant,
                format_name=loaded.format_name,
                source_path=loaded.source_path,
                metadata=dict(loaded.metadata),
                trace_metadata=loaded.trace_metadata,
            )

    def import_line(
        self,
        project_id: str,
        source: str | Path,
        *,
        line_id: str,
        name: str = "",
        copy_into_project: bool = True,
        dielectric_constant: float = 9.0,
        context: ExecutionContext | None = None,
    ) -> ImportedLineResult:
        execution_context = context or ExecutionContext.null()
        source_path = Path(source).expanduser().resolve()
        preflight = self.preflight(
            source_path,
            line_id=line_id,
            dielectric_constant=float(dielectric_constant),
            context=execution_context.child(0, 2),
        )
        if not preflight.can_import:
            raise AcquisitionApplicationError(preflight.message)
        line = self._projects.import_line_source(
            project_id,
            line_id,
            source_path,
            name=name or line_id,
            copy_into_project=copy_into_project,
            dielectric_constant=float(dielectric_constant),
            context=execution_context.child(1, 2),
        )
        info = self._projects.get_dataset_info(project_id, line_id)
        metadata = dict(info.metadata)
        return ImportedLineResult(
            project_id=project_id,
            line_id=line.line_id,
            name=line.name,
            shape=info.shape,
            length_m=info.length_m,
            time_window_ns=info.time_window_ns,
            format_name=info.format_name,
            source_path=info.source_path or str(source_path),
            has_trajectory=bool(preflight.has_trajectory or metadata.get("trajectory_rows") or metadata.get("has_trajectory")),
        )

    def parse_sidecar(self, source: str | Path, *, kind: SensorKind | str) -> SensorStream:
        selected = kind if isinstance(kind, SensorKind) else SensorKind(str(kind))
        return self._parser.parse(Path(source).expanduser().resolve(), kind=selected)

    def synchronize_streams(
        self,
        *,
        trace_timestamps_s: np.ndarray,
        rtk: SensorStream,
        imu: SensorStream | None = None,
        altimeter: SensorStream | None = None,
        settings: SensorSyncSettings | None = None,
        line_id: str = "L01",
        trace_distance_hint_m: np.ndarray | None = None,
        context: ExecutionContext | None = None,
    ) -> SynchronizedSensorData:
        timestamps = np.asarray(trace_timestamps_s, dtype=np.float64).reshape(-1)
        if timestamps.size == 0 or not np.isfinite(timestamps).all():
            raise ValueError("trace_timestamps_s must be a finite non-empty array")
        return self._synchronizer.synchronize(
            trace_timestamps_s=timestamps,
            rtk=rtk,
            imu=imu,
            altimeter=altimeter,
            settings=settings or SensorSyncSettings(),
            line_id=line_id,
            trace_distance_hint_m=trace_distance_hint_m,
            context=context or ExecutionContext.null(),
        )

    def synchronize_project_line(
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
        return self._projects.synchronize_line_sensors(
            project_id,
            line_id,
            rtk_path=rtk_path,
            trace_timestamps_path=trace_timestamps_path,
            imu_path=imu_path,
            altimeter_path=altimeter_path,
            settings=settings or SensorSyncSettings(),
            context=context or ExecutionContext.null(),
        )

    @staticmethod
    def motion_pipeline(profile: MotionCompensationProfile | None = None) -> PipelineDefinition:
        return build_motion_pipeline(profile)


__all__ = ["AcquisitionApplicationError", "AcquisitionService"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence ports consumed by project application services."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
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
    ProjectMetadata, LineQualityReport, SourceFileStatus, LineDeleteResult, BatchImportSummary,
)


class ProjectSessionPort(Protocol):
    @property
    def summary(self) -> ProjectSummary: ...

    def close(self) -> None: ...

    def get_metadata(self) -> ProjectMetadata: ...

    def update_metadata(self, **changes: str | None) -> ProjectMetadata: ...

    def list_quality_reports(self) -> Sequence[LineQualityReport]: ...

    def run_line_quality_check(self, line_id: str) -> LineQualityReport: ...

    def run_project_quality_check(self, *, context: ExecutionContext) -> Sequence[LineQualityReport]: ...

    def check_source_files(self, *, context: ExecutionContext) -> Sequence[SourceFileStatus]: ...

    def relink_line_source(self, line_id: str, new_source: Path, *, allow_mismatch: bool, context: ExecutionContext) -> SourceFileStatus: ...

    def export_source_manifest(self, destination: Path | None = None) -> Path: ...

    def transpose_line_dataset(self, line_id: str, *, context: ExecutionContext) -> LineQualityReport: ...

    def delete_line(self, line_id: str, *, reason: str) -> LineDeleteResult: ...

    def batch_import_lines(self, sources: Sequence[Path], *, context: ExecutionContext) -> BatchImportSummary: ...

    def list_lines(self) -> Sequence[ProjectLine]: ...

    def get_line(self, line_id: str) -> ProjectLine: ...

    def get_dataset_info(self, line_id: str) -> LineDatasetInfo: ...

    def import_line_source(
        self,
        line_id: str,
        source: Path,
        *,
        name: str,
        copy_into_project: bool,
        dielectric_constant: float,
        context: ExecutionContext,
    ) -> ProjectLine: ...

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
    ) -> ProjectSensorSyncResult: ...

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
    ) -> ProjectLine: ...

    def read_dataset(self, line_id: str) -> ProjectLineData: ...

    def read_trace_metadata(self, line_id: str) -> Mapping[str, np.ndarray]: ...

    def iter_dataset_blocks(
        self,
        line_id: str,
        *,
        block_rows: int = 1024,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]: ...

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def get_artifact_dataset_info(
        self, line_id: str, artifact_id: str
    ) -> LineDatasetInfo: ...

    def read_artifact_dataset(
        self, line_id: str, artifact_id: str
    ) -> ProjectLineData: ...

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
    ) -> Iterator[tuple[int, int, np.ndarray]]: ...

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

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
    ) -> ProjectArtifact: ...

    def list_artifacts(self, line_id: str | None = None) -> Sequence[ProjectArtifact]: ...

    def list_interpretation_features(self, line_id: str) -> Sequence[InterpretationFeature]: ...

    def replace_interpretation_features(
        self, line_id: str, features: Sequence[InterpretationFeature]
    ) -> None: ...

    def load_interface_annotation(
        self, line_id: str, *, create: bool
    ) -> InterfaceAnnotation | None: ...

    def save_interface_annotation(self, annotation: InterfaceAnnotation) -> InterfaceAnnotation: ...

    def list_boreholes(self) -> Sequence[BoreholeRecord]: ...

    def save_borehole(self, borehole: BoreholeRecord) -> BoreholeRecord: ...

    def delete_borehole(self, borehole_id: str) -> bool: ...

    def depth_at_samples(self, line_id: str, samples: np.ndarray) -> np.ndarray: ...

    def load_spatial_tracks(self) -> Sequence[SpatialTrack]: ...

    def list_spatial_results(self) -> Sequence[SpatialResult]: ...

    def spatial_preflight(
        self, *, line_ids: Sequence[str] | None, generate_surface: bool
    ) -> Mapping[str, Any]: ...

    def create_spatial_result(
        self, *, name: str, line_ids: Sequence[str] | None,
        velocity_m_per_ns: float | None, generate_surface: bool
    ) -> SpatialResult: ...

    def set_current_spatial_result(self, result_id: str) -> None: ...

    def list_report_packages(self) -> Sequence[ReportPackage]: ...

    def audit(
        self,
        *,
        repair_context: bool,
        clean_staging: bool,
        staging_min_age_s: float,
        deep_hash: bool,
    ) -> IntegrityReport: ...

    def generate_report(
        self,
        *,
        package_name: str | None,
        report_profile: Mapping[str, Any],
        context: ExecutionContext,
    ) -> ReportPackage: ...

    def backup(
        self,
        destination_dir: Path | None,
        *,
        require_external_device: bool,
        context: ExecutionContext,
    ) -> ProjectBackup: ...


class ProjectRepositoryPort(Protocol):
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
    ) -> ProjectSessionPort: ...

    def open(
        self,
        root: Path,
        *,
        read_only: bool,
        recover_stale_lock: bool,
    ) -> ProjectSessionPort: ...

    def restore(
        self,
        archive_path: Path,
        destination_root: Path,
        *,
        project_dir_name: str | None,
    ) -> ProjectRestore: ...


__all__ = ["ProjectRepositoryPort", "ProjectSessionPort"]

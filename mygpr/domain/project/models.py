#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI- and persistence-independent project contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _non_empty(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


@dataclass(frozen=True, slots=True)
class ProjectSummary:
    project_id: str
    name: str
    root_path: str
    schema: str
    storage_backend: str
    revision: int = 0
    status: str = ""
    read_only: bool = False
    line_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _non_empty(self.project_id, "project_id"))
        object.__setattr__(self, "name", _non_empty(self.name, "name"))
        object.__setattr__(self, "root_path", str(Path(self.root_path).resolve()))
        object.__setattr__(self, "revision", max(0, int(self.revision)))
        object.__setattr__(self, "line_count", max(0, int(self.line_count)))


@dataclass(frozen=True, slots=True)
class ProjectLine:
    line_id: str
    name: str
    length_m: float = 0.0
    sample_count: int = 0
    trace_count: int = 0
    data_quality: str = ""
    processing_status: str = ""
    data_format: str = ""
    raw_size_mb: float = 0.0
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "line_id", _non_empty(self.line_id, "line_id"))
        object.__setattr__(self, "name", str(self.name or self.line_id))
        object.__setattr__(self, "length_m", max(0.0, float(self.length_m)))
        object.__setattr__(self, "sample_count", max(0, int(self.sample_count)))
        object.__setattr__(self, "trace_count", max(0, int(self.trace_count)))
        object.__setattr__(self, "raw_size_mb", max(0.0, float(self.raw_size_mb)))


@dataclass(frozen=True, slots=True)
class LineDatasetInfo:
    line_id: str
    shape: tuple[int, int]
    dtype: str
    length_m: float
    time_window_ns: float
    dielectric_constant: float
    source_path: str = ""
    format_name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "line_id", _non_empty(self.line_id, "line_id"))
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"dataset shape must be non-empty 2D, got {shape!r}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "length_m", max(0.0, float(self.length_m)))
        object.__setattr__(self, "time_window_ns", max(0.0, float(self.time_window_ns)))
        object.__setattr__(self, "dielectric_constant", max(0.0, float(self.dielectric_constant)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(slots=True)
class ProjectLineData:
    line_id: str
    data: np.ndarray
    header_info: dict[str, Any] = field(default_factory=dict)
    trace_metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.line_id = _non_empty(self.line_id, "line_id")
        matrix = np.asarray(self.data)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError("project line data must be a non-empty 2D array")
        self.data = matrix
        self.header_info = dict(self.header_info or {})
        self.trace_metadata = {
            str(key): np.array(value, copy=True)
            for key, value in (self.trace_metadata or {}).items()
        }


@dataclass(frozen=True, slots=True)
class ProjectArtifact:
    artifact_id: str
    line_id: str
    name: str
    data_reference: str
    branch_id: str = ""
    parent_artifact_id: str = ""
    method_id: str = ""
    method_name: str = ""
    status: str = "committed"
    shape: tuple[int, ...] = ()
    dtype: str = ""
    sha256: str = ""
    created_at: str = ""
    manifest: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _non_empty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "line_id", _non_empty(self.line_id, "line_id"))
        object.__setattr__(self, "name", str(self.name or self.artifact_id))
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest or {})))


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    code: str
    severity: str
    message: str
    module: str = ""
    object_id: str = ""
    path: str = ""
    repairable: bool = False
    repaired: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _non_empty(self.code, "code"))
        object.__setattr__(self, "severity", str(self.severity or "warning"))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details or {})))


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    project_id: str
    generated_at: str
    issues: tuple[IntegrityIssue, ...]
    repairs: tuple[str, ...] = ()
    elapsed_ms: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def healthy(self) -> bool:
        return self.error_count == 0


@dataclass(frozen=True, slots=True)
class ProjectBackup:
    archive_path: str
    file_count: int
    size_mb: float
    manifest_sha256: str
    verified: bool
    external_device: bool = False


@dataclass(frozen=True, slots=True)
class ProjectRestore:
    project_path: str
    file_count: int
    verified: bool
    source_archive: str


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    project_id: str
    name: str
    location: str = ""
    operator: str = "操作员"
    project_no: str = ""
    device_model: str = ""
    coordinate_system: str = ""
    vertical_datum: str = ""


@dataclass(frozen=True, slots=True)
class LineQualityIssue:
    severity: str
    code: str
    message: str
    suggestion: str = ""


@dataclass(frozen=True, slots=True)
class LineQualityReport:
    line_id: str
    status: str
    status_label: str
    checked_at: str
    sample_count: int
    trace_count: int
    time_window_ns: float
    length_m: float
    amplitude_min: float
    amplitude_max: float
    amplitude_p995: float
    nan_ratio: float
    finite_ratio: float
    trajectory_points: int
    orientation: str
    orientation_message: str
    suggested_action: str
    issues: tuple[LineQualityIssue, ...] = ()
    sampled: bool = False
    evaluated_value_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceFileStatus:
    line_id: str
    role: str
    source_path: str
    source_filename: str
    source_size_bytes: int
    import_mode: str
    project_raw_path: str
    status: str
    status_label: str
    last_checked_at: str = ""
    warning: str = ""
    hash_policy: str = ""
    full_hash_status: str = ""


@dataclass(frozen=True, slots=True)
class LineDeleteResult:
    line_id: str
    line_name: str
    deleted_paths: tuple[str, ...]
    remaining_line_count: int


@dataclass(frozen=True, slots=True)
class BatchImportItemResult:
    source: str
    line_id: str
    name: str
    success: bool
    message: str
    sample_count: int = 0
    trace_count: int = 0
    length_m: float = 0.0
    file_size_mb: float = 0.0
    elapsed_s: float = 0.0
    raw_dir: str = ""
    manifest_path: str = ""
    diagnosis: str = ""


@dataclass(frozen=True, slots=True)
class BatchImportSummary:
    total: int
    succeeded: int
    failed: int
    results: tuple[BatchImportItemResult, ...]

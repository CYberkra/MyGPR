#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI- and persistence-independent acquisition and sensor contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class SensorKind(str, Enum):
    RTK = "rtk"
    IMU = "imu"
    ALTIMETER = "altimeter"


@dataclass(frozen=True, slots=True)
class ImportPreflight:
    path: str
    exists: bool
    is_file: bool
    extension: str
    format_name: str
    support: str
    can_import: bool
    message: str
    suggestions: tuple[str, ...] = ()
    sample_count: int = 0
    trace_count: int = 0
    length_m: float = 0.0
    time_window_ns: float = 0.0
    dielectric_constant: float = 0.0
    data_min: float = 0.0
    data_max: float = 0.0
    source_kind: str = ""
    has_trajectory: bool = False
    column_summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(Path(self.path).expanduser().resolve()))
        object.__setattr__(self, "suggestions", tuple(str(v) for v in self.suggestions))
        object.__setattr__(self, "sample_count", max(0, int(self.sample_count)))
        object.__setattr__(self, "trace_count", max(0, int(self.trace_count)))
        object.__setattr__(self, "length_m", max(0.0, float(self.length_m)))
        object.__setattr__(self, "time_window_ns", max(0.0, float(self.time_window_ns)))
        object.__setattr__(self, "dielectric_constant", max(0.0, float(self.dielectric_constant)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.sample_count, self.trace_count


@dataclass(slots=True)
class AcquisitionDataset:
    line_id: str
    data: np.ndarray
    length_m: float
    time_window_ns: float
    dielectric_constant: float
    format_name: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.line_id = str(self.line_id).strip()
        if not self.line_id:
            raise ValueError("line_id must not be empty")
        matrix = np.asanyarray(self.data)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError("acquisition data must be a non-empty 2D matrix")
        self.data = matrix
        self.length_m = max(0.0, float(self.length_m))
        self.time_window_ns = max(0.0, float(self.time_window_ns))
        self.dielectric_constant = max(0.0, float(self.dielectric_constant))
        self.format_name = str(self.format_name or "unknown")
        self.source_path = str(Path(self.source_path).expanduser().resolve())
        self.metadata = dict(self.metadata or {})
        self.trace_metadata = {
            str(key): np.array(value, copy=True)
            for key, value in (self.trace_metadata or {}).items()
        }

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.data.shape)


@dataclass(frozen=True, slots=True)
class ImportedLineResult:
    project_id: str
    line_id: str
    name: str
    shape: tuple[int, int]
    length_m: float
    time_window_ns: float
    format_name: str
    source_path: str
    has_trajectory: bool = False


@dataclass(frozen=True, slots=True)
class SensorStream:
    kind: SensorKind
    fields: Mapping[str, np.ndarray]
    source_path: str = ""

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, SensorKind) else SensorKind(str(self.kind))
        copied = {str(key): np.array(value, copy=True) for key, value in self.fields.items()}
        timestamps = np.asarray(copied.get("timestamp_s", ()), dtype=np.float64)
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError("sensor stream requires a non-empty timestamp_s array")
        for key, value in copied.items():
            if np.asarray(value).ndim != 1:
                raise ValueError(f"sensor field must be one-dimensional: {key}")
            if np.asarray(value).size != timestamps.size:
                raise ValueError(f"sensor field length mismatch: {key}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fields", MappingProxyType(copied))
        if self.source_path:
            object.__setattr__(self, "source_path", str(Path(self.source_path).expanduser().resolve()))

    @property
    def sample_count(self) -> int:
        return int(np.asarray(self.fields["timestamp_s"]).size)


@dataclass(frozen=True, slots=True)
class SensorSyncSettings:
    rtk_time_offset_s: float = 0.0
    imu_time_offset_s: float = 0.0
    altimeter_time_offset_s: float = 0.0
    maximum_nearest_residual_s: float = 0.25
    gap_warning_s: float = 1.0
    allow_extrapolation: bool = False
    project_crs: str = ""
    lever_arm_x_m: float = 0.0
    lever_arm_y_m: float = 0.0
    lever_arm_z_m: float = 0.0
    radar_trigger_delay_s: float = 0.0
    antenna_height_offset_m: float = 0.0
    radar_clock: Mapping[str, Any] = field(default_factory=dict)
    rtk_clock: Mapping[str, Any] = field(default_factory=dict)
    imu_clock: Mapping[str, Any] = field(default_factory=dict)
    altimeter_clock: Mapping[str, Any] = field(default_factory=dict)
    rtk_alignment: Mapping[str, Any] = field(default_factory=dict)
    imu_alignment: Mapping[str, Any] = field(default_factory=dict)
    altimeter_alignment: Mapping[str, Any] = field(default_factory=dict)
    calibration_profile: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.maximum_nearest_residual_s) < 0:
            raise ValueError("maximum_nearest_residual_s must be non-negative")
        if float(self.gap_warning_s) < 0:
            raise ValueError("gap_warning_s must be non-negative")
        for name in (
            "radar_clock", "rtk_clock", "imu_clock", "altimeter_clock",
            "rtk_alignment", "imu_alignment", "altimeter_alignment", "calibration_profile",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name) or {})))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rtk_time_offset_s": float(self.rtk_time_offset_s),
            "imu_time_offset_s": float(self.imu_time_offset_s),
            "altimeter_time_offset_s": float(self.altimeter_time_offset_s),
            "maximum_nearest_residual_s": float(self.maximum_nearest_residual_s),
            "gap_warning_s": float(self.gap_warning_s),
            "allow_extrapolation": bool(self.allow_extrapolation),
            "project_crs": str(self.project_crs),
            "lever_arm_x_m": float(self.lever_arm_x_m),
            "lever_arm_y_m": float(self.lever_arm_y_m),
            "lever_arm_z_m": float(self.lever_arm_z_m),
            "radar_trigger_delay_s": float(self.radar_trigger_delay_s),
            "antenna_height_offset_m": float(self.antenna_height_offset_m),
            "radar_clock": dict(self.radar_clock) or None,
            "rtk_clock": dict(self.rtk_clock) or None,
            "imu_clock": dict(self.imu_clock) or None,
            "altimeter_clock": dict(self.altimeter_clock) or None,
            "rtk_alignment": dict(self.rtk_alignment) or None,
            "imu_alignment": dict(self.imu_alignment) or None,
            "altimeter_alignment": dict(self.altimeter_alignment) or None,
            "calibration_profile": dict(self.calibration_profile) or None,
        }


@dataclass(slots=True)
class SynchronizedSensorData:
    trace_metadata: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    trajectory: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        self.trace_metadata = {
            str(key): np.array(value, copy=True)
            for key, value in (self.trace_metadata or {}).items()
        }
        self.diagnostics = dict(self.diagnostics or {})
        self.config = dict(self.config or {})
        self.trajectory = tuple(MappingProxyType(dict(row)) for row in self.trajectory)


@dataclass(frozen=True, slots=True)
class ProjectSensorSyncResult:
    project_id: str
    line_id: str
    trajectory_path: str
    manifest_path: str
    trace_metadata_path: str
    diagnostics: Mapping[str, Any]
    config: Mapping[str, Any]
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics or {})))
        object.__setattr__(self, "config", MappingProxyType(dict(self.config or {})))


__all__ = [
    "AcquisitionDataset",
    "ImportedLineResult",
    "ImportPreflight",
    "ProjectSensorSyncResult",
    "SensorKind",
    "SensorStream",
    "SensorSyncSettings",
    "SynchronizedSensorData",
]

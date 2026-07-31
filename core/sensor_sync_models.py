#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable data contracts for radar and navigation sensor synchronization."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from core.trajectory_model import TrajectoryModel

SYNC_SCHEMA = "mygpr.sensor_sync.v2"

@dataclass
class ClockDomain:
    name: str = "relative_seconds"
    epoch: str = "relative_start"
    unit: str = "s"
    time_scale: str = "device"

    def factor_to_seconds(self) -> float:
        factors = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
        try:
            return factors[self.unit]
        except KeyError as exc:
            raise ValueError(f"不支持的时间单位：{self.unit}") from exc

    @classmethod
    def from_value(cls, value: "ClockDomain | dict[str, Any] | None", *, default_name: str) -> "ClockDomain":
        if isinstance(value, ClockDomain):
            return value
        if isinstance(value, dict):
            return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})
        return cls(name=default_name)

@dataclass
class TimeAlignmentModel:
    mode: Literal["constant", "affine", "piecewise"] = "constant"
    offset_s: float = 0.0
    scale: float = 1.0
    reference_s: float = 0.0
    anchors: list[tuple[float, float]] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: "TimeAlignmentModel | dict[str, Any] | None", *, offset_s: float = 0.0) -> "TimeAlignmentModel":
        if isinstance(value, TimeAlignmentModel):
            return value
        if isinstance(value, dict):
            payload = dict(value)
            if "anchors" in payload:
                payload["anchors"] = [tuple(map(float, pair)) for pair in payload["anchors"]]
            return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})
        return cls(offset_s=float(offset_s))

    def apply(self, timestamps_s: np.ndarray) -> np.ndarray:
        values = np.asarray(timestamps_s, dtype=np.float64)
        if self.mode == "constant":
            return values + float(self.offset_s)
        if self.mode == "affine":
            return float(self.reference_s) + (values - float(self.reference_s)) * float(self.scale) + float(self.offset_s)
        if self.mode == "piecewise":
            if len(self.anchors) < 2:
                raise ValueError("分段时钟模型至少需要两个锚点。")
            source: np.ndarray = np.asarray([pair[0] for pair in self.anchors], dtype=np.float64)
            target: np.ndarray = np.asarray([pair[1] for pair in self.anchors], dtype=np.float64)
            if np.any(np.diff(source) <= 0):
                raise ValueError("分段时钟模型的源锚点必须严格递增。")
            return np.interp(values, source, target, left=np.nan, right=np.nan)
        raise ValueError(f"未知时钟模型：{self.mode}")

@dataclass
class SensorCalibrationProfile:
    profile_id: str = ""
    radar_device_id: str = ""
    rtk_device_id: str = ""
    imu_device_id: str = ""
    lever_arm_x_m: float = 0.0
    lever_arm_y_m: float = 0.0
    lever_arm_z_m: float = 0.0
    trigger_delay_s: float = 0.0
    position_sigma_m: float = 0.05
    attitude_sigma_deg: float = 0.5
    height_sigma_m: float = 0.10
    timing_sigma_s: float = 0.01
    valid_from: str = ""
    valid_to: str = ""

    @classmethod
    def from_value(cls, value: "SensorCalibrationProfile | dict[str, Any] | None") -> "SensorCalibrationProfile | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})
        raise TypeError("calibration_profile must be a mapping or SensorCalibrationProfile")

@dataclass
class SensorSyncConfig:
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
    radar_clock: ClockDomain | dict[str, Any] | None = None
    rtk_clock: ClockDomain | dict[str, Any] | None = None
    imu_clock: ClockDomain | dict[str, Any] | None = None
    altimeter_clock: ClockDomain | dict[str, Any] | None = None
    rtk_alignment: TimeAlignmentModel | dict[str, Any] | None = None
    imu_alignment: TimeAlignmentModel | dict[str, Any] | None = None
    altimeter_alignment: TimeAlignmentModel | dict[str, Any] | None = None
    calibration_profile: SensorCalibrationProfile | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        profile = SensorCalibrationProfile.from_value(self.calibration_profile)
        self.calibration_profile = profile
        if profile is not None:
            if not any((self.lever_arm_x_m, self.lever_arm_y_m, self.lever_arm_z_m)):
                self.lever_arm_x_m, self.lever_arm_y_m, self.lever_arm_z_m = (
                    profile.lever_arm_x_m, profile.lever_arm_y_m, profile.lever_arm_z_m
                )
            if self.radar_trigger_delay_s == 0.0:
                self.radar_trigger_delay_s = profile.trigger_delay_s
        self.radar_clock = ClockDomain.from_value(self.radar_clock, default_name="radar")
        self.rtk_clock = ClockDomain.from_value(self.rtk_clock, default_name="rtk")
        self.imu_clock = ClockDomain.from_value(self.imu_clock, default_name="imu")
        self.altimeter_clock = ClockDomain.from_value(self.altimeter_clock, default_name="altimeter")
        self.rtk_alignment = TimeAlignmentModel.from_value(self.rtk_alignment, offset_s=self.rtk_time_offset_s)
        self.imu_alignment = TimeAlignmentModel.from_value(self.imu_alignment, offset_s=self.imu_time_offset_s)
        self.altimeter_alignment = TimeAlignmentModel.from_value(self.altimeter_alignment, offset_s=self.altimeter_time_offset_s)

@dataclass
class StreamDiagnostics:
    name: str
    source_count: int = 0
    coverage_ratio: float = 0.0
    accepted_ratio: float = 0.0
    invalid_trace_count: int = 0
    duplicate_timestamp_count: int = 0
    nonmonotonic_step_count: int = 0
    raw_nonmonotonic_step_count: int = 0
    raw_duplicate_timestamp_count: int = 0
    gap_count: int = 0
    median_nearest_residual_s: float = 0.0
    max_nearest_residual_s: float = 0.0

@dataclass
class SensorSyncDiagnostics:
    schema: str = SYNC_SCHEMA
    trace_count: int = 0
    trace_start_s: float = 0.0
    trace_end_s: float = 0.0
    trace_nonmonotonic_step_count: int = 0
    trace_duplicate_timestamp_count: int = 0
    rtk: StreamDiagnostics = field(default_factory=lambda: StreamDiagnostics("rtk"))
    imu: StreamDiagnostics = field(default_factory=lambda: StreamDiagnostics("imu"))
    altimeter: StreamDiagnostics = field(default_factory=lambda: StreamDiagnostics("altimeter"))
    fixed_solution_ratio: float = 0.0
    float_solution_ratio: float = 0.0
    unknown_solution_ratio: float = 0.0
    distance_reverse_count: int = 0
    position_jump_count: int = 0
    maximum_position_step_m: float = 0.0
    median_position_step_m: float = 0.0
    lever_arm_applied: bool = False
    clock_models: dict[str, Any] = field(default_factory=dict)
    calibration_profile_id: str = ""
    uncertainty_summary: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class SensorSyncResult:
    trajectory: TrajectoryModel
    trace_metadata: dict[str, np.ndarray]
    diagnostics: SensorSyncDiagnostics
    config: SensorSyncConfig

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": SYNC_SCHEMA,
            "config": asdict(self.config),
            "diagnostics": self.diagnostics.to_dict(),
            "trace_fields": {
                key: {"dtype": str(np.asarray(value).dtype), "count": int(np.asarray(value).size)}
                for key, value in self.trace_metadata.items()
            },
        }

__all__ = [
    "SYNC_SCHEMA", "ClockDomain", "TimeAlignmentModel", "SensorCalibrationProfile",
    "SensorSyncConfig", "StreamDiagnostics", "SensorSyncDiagnostics", "SensorSyncResult",
]

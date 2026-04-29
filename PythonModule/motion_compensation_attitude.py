#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR 姿态/APC 足迹修正模块（V1 元数据修正）。

V1 仅修正轨迹元数据，不修改 B-scan 振幅值：
- 输入 local_x_m / local_y_m 视为平台参考点轨迹
- 根据 roll/pitch/yaw 与可选 flight_height_m 计算地面足迹偏移
- 根据 APC lever arm 参数估计天线相位中心的水平偏移
- 通过 meta["trace_metadata_updates"] 返回修正后的 local_x_m / local_y_m /
  trace_distance_m / footprint_x_m / footprint_y_m

该阶段不会原地修改传入的 trace_metadata。
"""

from __future__ import annotations

import numpy as np


REQUIRED_FIELDS = ("roll_deg", "pitch_deg", "yaw_deg", "local_x_m", "local_y_m")


def _extract_numeric_field(
    trace_metadata: dict,
    key: str,
    trace_count: int,
) -> np.ndarray:
    """Extract a 1D numeric field trimmed to trace_count."""
    values = np.asarray(trace_metadata[key], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"trace_metadata['{key}'] 必须为一维数组")
    if values.size < trace_count:
        raise ValueError(f"trace_metadata['{key}'] 长度不足，无法覆盖全部道")
    trimmed = values[:trace_count].astype(np.float64, copy=True)
    if not np.all(np.isfinite(trimmed)):
        raise ValueError(f"trace_metadata['{key}'] 包含 NaN 或无穷值")
    return trimmed


def _compute_trace_distance(local_x_m: np.ndarray, local_y_m: np.ndarray) -> np.ndarray:
    """Recompute cumulative trace distance from local XY."""
    if local_x_m.size == 0:
        return np.array([], dtype=np.float64)
    step = np.sqrt(np.diff(local_x_m) ** 2 + np.diff(local_y_m) ** 2)
    return np.concatenate(([0.0], np.cumsum(step))).astype(np.float64)


def _apply_yaw_rotation(x_body_m: np.ndarray, y_body_m: np.ndarray, yaw_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate body-frame XY offsets into the local XY plane using yaw only."""
    local_x = x_body_m * np.cos(yaw_rad) - y_body_m * np.sin(yaw_rad)
    local_y = x_body_m * np.sin(yaw_rad) + y_body_m * np.cos(yaw_rad)
    return local_x, local_y


def method_motion_compensation_attitude(
    data: np.ndarray,
    apc_offset_x_m: float = 0.0,
    apc_offset_y_m: float = 0.0,
    apc_offset_z_m: float = 0.0,
    max_abs_tilt_deg: float = 20.0,
    trace_metadata: dict | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """姿态/APC 足迹修正（V1 仅输出元数据更新）。"""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("姿态补偿需要二维 B-scan 数据")

    amplitude_out = np.array(arr, copy=True)
    trace_count = int(arr.shape[1])
    meta: dict[str, object] = {
        "method": "motion_compensation_attitude",
        "apc_offset_x_m": float(apc_offset_x_m),
        "apc_offset_y_m": float(apc_offset_y_m),
        "apc_offset_z_m": float(apc_offset_z_m),
        "max_abs_tilt_deg": float(max_abs_tilt_deg),
        "trace_count": trace_count,
        "provenance": {
            "geometry_model": "yaw_rotated_tilt_plus_apc_offset_v1",
            "attitude_handling": "clamp",
            "required_fields": list(REQUIRED_FIELDS),
        },
    }

    if max_abs_tilt_deg <= 0:
        raise ValueError("max_abs_tilt_deg 必须为正数")

    if trace_metadata is None:
        meta["skipped"] = True
        meta["reason"] = "缺少 trace_metadata，无法进行姿态/APC 足迹修正"
        return amplitude_out, meta

    missing_fields = [field for field in REQUIRED_FIELDS if field not in trace_metadata]
    if missing_fields:
        meta["skipped"] = True
        meta["missing_fields"] = missing_fields
        meta["reason"] = f"缺少姿态/APC 修正所需字段: {', '.join(missing_fields)}"
        return amplitude_out, meta

    try:
        local_x_m = _extract_numeric_field(trace_metadata, "local_x_m", trace_count)
        local_y_m = _extract_numeric_field(trace_metadata, "local_y_m", trace_count)
        roll_deg = _extract_numeric_field(trace_metadata, "roll_deg", trace_count)
        pitch_deg = _extract_numeric_field(trace_metadata, "pitch_deg", trace_count)
        yaw_deg = _extract_numeric_field(trace_metadata, "yaw_deg", trace_count)
    except ValueError as exc:
        meta["skipped"] = True
        meta["reason"] = str(exc)
        return amplitude_out, meta

    flight_height_m = None
    if "flight_height_m" in trace_metadata:
        try:
            flight_height_m = _extract_numeric_field(trace_metadata, "flight_height_m", trace_count)
        except ValueError as exc:
            meta["skipped"] = True
            meta["reason"] = str(exc)
            return amplitude_out, meta

        if np.any(flight_height_m <= 0.0):
            meta["skipped"] = True
            meta["reason"] = "trace_metadata['flight_height_m'] 包含零或负值，无法构造有效足迹几何"
            return amplitude_out, meta

    tilt_limit_deg = float(max_abs_tilt_deg)
    roll_used_deg = np.clip(roll_deg, -tilt_limit_deg, tilt_limit_deg)
    pitch_used_deg = np.clip(pitch_deg, -tilt_limit_deg, tilt_limit_deg)
    clamped_mask = (np.abs(roll_deg) > tilt_limit_deg) | (np.abs(pitch_deg) > tilt_limit_deg)

    warnings: list[str] = []
    if np.any(clamped_mask):
        warnings.append(
            f"检测到 {int(np.count_nonzero(clamped_mask))} 道姿态超限；roll/pitch 已按 ±{tilt_limit_deg:.1f}° 钳制"
        )
    meta["warnings"] = warnings
    meta["attitude_clamped"] = bool(np.any(clamped_mask))
    meta["clamped_trace_count"] = int(np.count_nonzero(clamped_mask))
    meta["used_flight_height_m"] = flight_height_m is not None

    yaw_rad = np.deg2rad(yaw_deg)
    roll_used_rad = np.deg2rad(roll_used_deg)
    pitch_used_rad = np.deg2rad(pitch_used_deg)

    apc_body_x = np.full(trace_count, float(apc_offset_x_m), dtype=np.float64)
    apc_body_y = np.full(trace_count, float(apc_offset_y_m), dtype=np.float64)
    apc_local_x_m, apc_local_y_m = _apply_yaw_rotation(apc_body_x, apc_body_y, yaw_rad)

    if flight_height_m is None:
        projection_height_m = np.zeros(trace_count, dtype=np.float64)
        projection_height_source = "lever_arm_only"
    else:
        projection_height_m = flight_height_m + float(apc_offset_z_m)
        if np.any(projection_height_m <= 0.0):
            meta["skipped"] = True
            meta["reason"] = "flight_height_m 与 apc_offset_z_m 组合后非正，无法构造有效投影高度"
            return amplitude_out, meta
        projection_height_source = "flight_height_m"

    pitch_body_m = projection_height_m * np.tan(pitch_used_rad)
    roll_body_m = projection_height_m * np.tan(roll_used_rad)
    footprint_dx_m, footprint_dy_m = _apply_yaw_rotation(pitch_body_m, roll_body_m, yaw_rad)

    corrected_local_x_m = local_x_m + apc_local_x_m + footprint_dx_m
    corrected_local_y_m = local_y_m + apc_local_y_m + footprint_dy_m
    trace_distance_m = _compute_trace_distance(corrected_local_x_m, corrected_local_y_m)
    corrected_footprint_x_m = corrected_local_x_m.copy()
    corrected_footprint_y_m = corrected_local_y_m.copy()

    meta["projection_height_source"] = projection_height_source
    meta["projection_height_min_m"] = float(np.min(projection_height_m)) if projection_height_m.size else 0.0
    meta["projection_height_max_m"] = float(np.max(projection_height_m)) if projection_height_m.size else 0.0
    meta["trace_metadata_updates"] = {
        "local_x_m": corrected_local_x_m,
        "local_y_m": corrected_local_y_m,
        "trace_distance_m": trace_distance_m,
        "footprint_x_m": corrected_footprint_x_m,
        "footprint_y_m": corrected_footprint_y_m,
    }
    return amplitude_out, meta

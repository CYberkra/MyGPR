#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry-depth context validation for UAV-GPR workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.runtime_warnings import build_runtime_warning


def method_geometry_depth_context(
    data: np.ndarray,
    require_velocity_model: bool = True,
    require_trace_spacing: bool = True,
    require_time_window: bool = True,
    require_agl: bool = False,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate geometry context and pass migration/depth hints downstream."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    info = dict(header_info or {})
    metadata = dict(trace_metadata or {})
    warnings: list[dict[str, Any]] = []

    velocity_model = info.get("velocity_model")
    velocity = info.get("velocity_m_per_ns")
    if isinstance(velocity_model, dict):
        velocity = velocity_model.get("velocity_m_per_ns", velocity)
    velocity_value = _positive_float(velocity)
    if require_velocity_model and velocity_value is None:
        warnings.append(
            build_runtime_warning(
                "missing_velocity_model",
                "缺少速度模型，迁移/时深转换只能使用后续算法默认速度。",
                method_id="geometry_depth_context",
                severity="warning",
            )
        )

    trace_interval = _resolve_trace_interval(info, metadata)
    if require_trace_spacing and trace_interval is None:
        warnings.append(
            build_runtime_warning(
                "missing_trace_spacing",
                "缺少道距或 trace_distance_m，几何深度校正无法确认横向尺度。",
                method_id="geometry_depth_context",
                severity="warning",
            )
        )

    time_window_ns = _positive_float(info.get("total_time_ns"))
    if require_time_window and time_window_ns is None:
        warnings.append(
            build_runtime_warning(
                "missing_time_window",
                "缺少 total_time_ns，迁移和时深转换无法确认纵向时间尺度。",
                method_id="geometry_depth_context",
                severity="warning",
            )
        )

    has_agl = any(
        key in metadata for key in ("height_agl_m", "flight_height_m", "altitude_agl_m")
    )
    if require_agl and not has_agl:
        warnings.append(
            build_runtime_warning(
                "missing_agl_height",
                "缺少 AGL/飞行高度元数据，无法执行严格 UAV 几何校正。",
                method_id="geometry_depth_context",
                severity="warning",
            )
        )

    migration_context = {
        "velocity_m_per_ns": velocity_value,
        "trace_interval_m": trace_interval,
        "time_window_ns": time_window_ns,
        "has_agl_height": bool(has_agl),
        "samples": int(arr.shape[0]),
        "traces": int(arr.shape[1]),
    }
    updates = {
        "geometry_depth_context": migration_context,
    }
    if trace_interval is not None:
        updates["trace_interval_m"] = float(trace_interval)
    if time_window_ns is not None:
        updates["total_time_ns"] = float(time_window_ns)

    return np.array(arr, copy=True), {
        "method": "geometry_depth_context",
        "geometry_depth_context": migration_context,
        "header_info_updates": updates,
        "runtime_warnings": warnings,
    }


def _resolve_trace_interval(
    header_info: dict[str, Any], trace_metadata: dict[str, np.ndarray]
) -> float | None:
    direct = _safe_float(header_info.get("trace_interval_m"))
    if direct is not None and direct > 0:
        return direct
    distance = trace_metadata.get("trace_distance_m")
    if distance is None:
        return None
    arr = np.asarray(distance, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        return None
    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _positive_float(value: Any) -> float | None:
    result = _safe_float(value)
    return result if result is not None and result > 0.0 else None

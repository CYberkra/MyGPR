#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-visible UAV-GPR attitude/APC footprint atomic node."""

from __future__ import annotations

from typing import Any

import numpy as np

from PythonModule.motion_compensation_core import (
    build_attitude_updates,
    clone_metadata,
    motion_warning,
    select_height,
)
from core.scalar_utils import to_float


METHOD_ID = "motion_compensation_attitude"
REQUIRED_FIELDS = ("roll_deg", "pitch_deg", "yaw_deg", "local_x_m", "local_y_m")


def _skip(
    data: np.ndarray,
    *,
    reason: str,
    code: str,
    trace_count: int,
    warnings: list[dict[str, Any]] | None = None,
    quality_flags: list[str] | None = None,
    **extra: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime_warnings = list(warnings or [])
    flags = list(quality_flags or [])
    if code not in flags:
        flags.append(code)
    runtime_warnings.append(motion_warning(METHOD_ID, code, reason, **extra))
    return np.array(data, copy=True), {
        "method": METHOD_ID,
        "skipped": True,
        "reason": reason,
        "trace_count": int(trace_count),
        "runtime_warnings": runtime_warnings,
        "quality_flags": sorted(set(flags)),
        **extra,
    }


def method_motion_compensation_attitude(
    data: np.ndarray,
    apc_offset_x_m: float = 0.0,
    apc_offset_y_m: float = 0.0,
    apc_offset_z_m: float = 0.0,
    max_abs_tilt_deg: float = 20.0,
    height_source: str = "auto",
    trace_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Update footprint metadata using shared UAV motion V2 geometry assumptions."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("姿态补偿需要二维 B-scan 数据")
    amplitude_out = np.array(arr, copy=True)
    trace_count = int(arr.shape[1])
    apc_offset_x_value = to_float(apc_offset_x_m, default=0.0)
    apc_offset_y_value = to_float(apc_offset_y_m, default=0.0)
    apc_offset_z_value = to_float(apc_offset_z_m, default=0.0)
    max_abs_tilt_value = to_float(max_abs_tilt_deg, default=20.0)
    if max_abs_tilt_value <= 0:
        raise ValueError("max_abs_tilt_deg 必须为正数")
    if trace_metadata is None:
        return _skip(
            amplitude_out,
            reason="缺少 trace_metadata，无法进行姿态/APC 足迹修正",
            code="missing_trace_metadata",
            trace_count=trace_count,
        )

    metadata = clone_metadata(trace_metadata)
    missing_fields = [field for field in REQUIRED_FIELDS if field not in metadata]
    if missing_fields:
        return _skip(
            amplitude_out,
            reason=f"缺少姿态/APC 修正所需字段: {', '.join(missing_fields)}",
            code="missing_attitude_fields",
            trace_count=trace_count,
            missing_fields=missing_fields,
        )

    warnings: list[dict[str, Any]] = []
    quality_flags: list[str] = []
    try:
        height_m, height_source_used = select_height(
            metadata,
            trace_count,
            height_source=height_source,
            method_id=METHOD_ID,
            warnings=warnings,
            quality_flags=quality_flags,
        )
    except ValueError as exc:
        return _skip(
            amplitude_out,
            reason=str(exc),
            code="invalid_height_agl",
            trace_count=trace_count,
            warnings=warnings,
            quality_flags=quality_flags,
        )

    if height_m is not None and (
        height_m.size != trace_count
        or not np.isfinite(height_m).all()
        or np.any(height_m <= 0.0)
    ):
        return _skip(
            amplitude_out,
            reason="投影高度包含非有限、零或负值，无法构造有效足迹几何",
            code="invalid_height_agl",
            trace_count=trace_count,
            warnings=warnings,
            quality_flags=quality_flags,
        )

    try:
        updates = build_attitude_updates(
            metadata,
            trace_count,
            height_m=height_m,
            apc_offset_x_m=apc_offset_x_value,
            apc_offset_y_m=apc_offset_y_value,
            apc_offset_z_m=apc_offset_z_value,
            max_abs_tilt_deg=max_abs_tilt_value,
            method_id=METHOD_ID,
            quality_flags=quality_flags,
            warnings=warnings,
        )
    except ValueError as exc:
        return _skip(
            amplitude_out,
            reason=str(exc),
            code="invalid_attitude_or_position",
            trace_count=trace_count,
            warnings=warnings,
            quality_flags=quality_flags,
        )

    if not updates:
        warning_codes = {item.get("code") for item in warnings}
        if "invalid_projection_height" in warning_codes:
            return _skip(
                amplitude_out,
                reason="投影高度包含非有限、零或负值，无法构造有效足迹几何",
                code="invalid_projection_height",
                trace_count=trace_count,
                warnings=warnings,
                quality_flags=quality_flags,
            )
        return _skip(
            amplitude_out,
            reason="缺少足够的高度或 APC 参数，姿态/APC 足迹修正未产生更新",
            code="no_attitude_updates",
            trace_count=trace_count,
            warnings=warnings,
            quality_flags=quality_flags,
        )

    warning_codes = {item.get("code") for item in warnings}
    clamped_count = 0
    for item in warnings:
        if item.get("code") == "attitude_clamped":
            clamped_count = int(item.get("details", {}).get("clamped_trace_count", 0))
            break
    projection_height_source = str(height_source_used) if height_source_used else "lever_arm_only"
    projection_height = None
    if height_m is not None:
        projection_height = height_m + apc_offset_z_value
    meta: dict[str, Any] = {
        "method": METHOD_ID,
        "skipped": False,
        "apc_offset_x_m": apc_offset_x_value,
        "apc_offset_y_m": apc_offset_y_value,
        "apc_offset_z_m": apc_offset_z_value,
        "max_abs_tilt_deg": max_abs_tilt_value,
        "trace_count": trace_count,
        "height_source_requested": str(height_source),
        "height_source_used": str(height_source_used) if height_source_used else None,
        "projection_height_source": projection_height_source,
        "projection_height_min_m": (
            float(np.min(projection_height)) if projection_height is not None else 0.0
        ),
        "projection_height_max_m": (
            float(np.max(projection_height)) if projection_height is not None else 0.0
        ),
        "trace_metadata_updates": updates,
        "trace_metadata_out": {
            **{
                key: np.array(value, copy=True)
                for key, value in metadata.items()
                if np.asarray(value).ndim == 1 and np.asarray(value).size >= trace_count
            },
            **updates,
        },
        "attitude_clamped": "attitude_clamped" in warning_codes,
        "clamped_trace_count": clamped_count,
        "warnings": [
            f"检测到 {clamped_count} 道姿态超限；roll/pitch 已钳制"
        ]
        if clamped_count
        else [],
        "runtime_warnings": warnings,
        "quality_flags": sorted(set(quality_flags)),
        "provenance": {
            "schema": "motion_compensation_atomic_v2",
            "shared_core": "PythonModule.motion_compensation_core",
            "geometry_model": "yaw_rotated_tilt_plus_apc_offset_v2",
            "attitude_handling": "clamp",
            "required_fields": list(REQUIRED_FIELDS),
            "height_priority": ["height_agl_m", "flight_height_m"],
        },
    }
    return amplitude_out, meta

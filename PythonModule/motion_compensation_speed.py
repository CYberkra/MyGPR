#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-visible UAV-GPR equal-distance resampling atomic node."""

from __future__ import annotations

from typing import Any

import numpy as np

from PythonModule.motion_compensation_core import (
    clone_metadata,
    compute_trace_distance,
    metadata_for_output,
    motion_warning,
    resample_equal_distance,
)
from core.scalar_utils import to_float
from core.trace_metadata_utils import build_uniform_trace_distance_m


METHOD_ID = "motion_compensation_speed"


def _skip(
    data: np.ndarray,
    *,
    reason: str,
    code: str,
    trace_count: int,
    **extra: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    return np.array(data, copy=True), {
        "method": METHOD_ID,
        "skipped": True,
        "reason": reason,
        "source_traces": int(trace_count),
        "runtime_warnings": [motion_warning(METHOD_ID, code, reason, **extra)],
        "quality_flags": [code],
        **extra,
    }


def _derive_distance_from_xy(metadata: dict[str, np.ndarray], trace_count: int) -> np.ndarray:
    if "local_x_m" not in metadata or "local_y_m" not in metadata:
        raise ValueError("缺少 trace_distance_m，且无法从 local_x_m / local_y_m 推导")
    local_x = np.asarray(metadata["local_x_m"], dtype=np.float64)
    local_y = np.asarray(metadata["local_y_m"], dtype=np.float64)
    if local_x.ndim != 1 or local_y.ndim != 1:
        raise ValueError("local_x_m / local_y_m 必须为一维数组")
    if local_x.size < trace_count or local_y.size < trace_count:
        raise ValueError("local_x_m / local_y_m 长度不足，无法覆盖全部道")
    return compute_trace_distance(local_x[:trace_count], local_y[:trace_count])


def method_motion_compensation_speed(
    data: np.ndarray,
    trace_metadata: dict[str, Any] | None = None,
    spacing_m: float | None = None,
    interpolation_mode: str = "linear",
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample the B-scan and trace metadata onto an equal-distance axis."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("速度误差补偿需要二维 B-scan 数据")
    trace_count = int(arr.shape[1])
    if interpolation_mode != "linear":
        raise ValueError(f"interpolation_mode '{interpolation_mode}' 不受支持；当前仅支持 'linear'")
    if trace_metadata is None:
        return _skip(
            arr,
            reason="缺少 trace_metadata，无法进行等距重采样",
            code="missing_trace_metadata",
            trace_count=trace_count,
        )

    metadata = clone_metadata(trace_metadata)
    try:
        if "trace_distance_m" in metadata:
            source_distance = np.asarray(metadata["trace_distance_m"], dtype=np.float64)
            if source_distance.ndim != 1 or source_distance.size < trace_count:
                raise ValueError("trace_metadata['trace_distance_m'] 长度不足或不是一维数组")
            source_distance = source_distance[:trace_count]
            distance_source = "trace_distance_m"
        else:
            source_distance = _derive_distance_from_xy(metadata, trace_count)
            distance_source = "local_xy"
        if not np.isfinite(source_distance).all():
            raise ValueError("trace_distance_m 包含非有限值")
        if np.any(np.diff(source_distance) < 0.0):
            raise ValueError("trace_distance_m 必须单调非递减；当前轨迹存在非单调距离")
    except ValueError as exc:
        return _skip(
            arr,
            reason=str(exc),
            code="invalid_trace_distance_m",
            trace_count=trace_count,
        )

    requested_spacing = to_float(spacing_m, default=0.0) if spacing_m is not None else 0.0
    target_distance = build_uniform_trace_distance_m(
        source_distance,
        spacing_m=requested_spacing if requested_spacing > 0.0 else None,
    )
    effective_spacing = 0.0
    positive_spacing = np.diff(np.asarray(target_distance, dtype=np.float64))
    positive_spacing = positive_spacing[positive_spacing > 0.0]
    if positive_spacing.size:
        effective_spacing = float(positive_spacing[0])

    prepared_metadata = metadata_for_output(
        metadata,
        {"trace_distance_m": source_distance.astype(np.float32)},
        trace_count,
    )
    corrected, trace_metadata_out, resample_meta = resample_equal_distance(
        arr,
        prepared_metadata,
        spacing_m=effective_spacing if effective_spacing > 0.0 else requested_spacing,
    )
    resample_meta["spacing_m"] = effective_spacing
    resample_meta["distance_source"] = distance_source
    return corrected.astype(np.float32, copy=False), {
        "method": METHOD_ID,
        "skipped": False,
        "interpolation_mode": str(interpolation_mode),
        "trace_metadata_out": trace_metadata_out,
        "runtime_warnings": [],
        "quality_flags": [],
        "provenance": {
            "schema": "motion_compensation_atomic_v2",
            "shared_core": "PythonModule.motion_compensation_core",
        },
        **resample_meta,
    }

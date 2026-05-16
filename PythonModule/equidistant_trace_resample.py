#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Equal-distance trace resampling for GPR B-scan data."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.trace_metadata_utils import (
    build_uniform_trace_distance_m,
    resample_bscan_columns_linear,
    resample_trace_metadata,
)


def method_equidistant_trace_resample(
    data: np.ndarray,
    spacing_m: float = 0.0,
    trace_metadata: dict[str, np.ndarray] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample B-scan columns onto a uniform trace-distance axis."""

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.shape[1] == 0:
        raise ValueError("输入数据没有有效道")

    metadata = trace_metadata or {}
    if "trace_distance_m" not in metadata:
        raise ValueError("equidistant_trace_resample requires trace_distance_m")

    source_distance = np.asarray(metadata["trace_distance_m"], dtype=np.float64)
    if source_distance.ndim != 1 or source_distance.size != arr.shape[1]:
        raise ValueError("trace_distance_m 必须是一维数组且长度等于 B-scan 道数")
    if np.any(~np.isfinite(source_distance)):
        raise ValueError("trace_distance_m 包含非有限值")
    if np.any(np.diff(source_distance) < 0.0):
        raise ValueError("trace_distance_m 必须单调非递减")

    requested_spacing = _as_float(spacing_m, default=0.0)
    target_distance = build_uniform_trace_distance_m(
        source_distance,
        spacing_m=requested_spacing if requested_spacing > 0.0 else None,
    )
    result = resample_bscan_columns_linear(arr, source_distance, target_distance)
    metadata_out = resample_trace_metadata(
        metadata,
        target_trace_distance_m=target_distance,
    )

    effective_spacing = (
        float(np.median(np.diff(target_distance)))
        if target_distance.size > 1
        else 0.0
    )
    return result, {
        "method": "equidistant_trace_resample",
        "spacing_m": requested_spacing,
        "effective_spacing_m": effective_spacing,
        "input_traces": int(arr.shape[1]),
        "output_traces": int(result.shape[1]),
        "trace_metadata_out": metadata_out,
    }


def _as_float(value: Any, *, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str) and value.strip() == "":
        return float(default)
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        arr = None
    try:
        if arr is not None:
            if arr.size == 0:
                return float(default)
            return float(arr.reshape(-1)[0])
        return float(value)
    except (TypeError, ValueError):
        return float(default)

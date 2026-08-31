#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace-axis median smoothing for GPR B-scan data."""

from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.domain.processing.warnings import build_runtime_warning


def method_trace_median_filter(
    data: np.ndarray,
    window_traces: int = 5,
    preserve_mean: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a median filter along the trace axis only.

    This is a denoising/smoothing operator, not background subtraction. It keeps
    the B-scan shape unchanged and replaces each sample row by a local median
    across neighboring traces.
    """
    from scipy.ndimage import median_filter

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    n_samples, n_traces = arr.shape
    requested_window = _safe_int(window_traces, 5)
    window = _odd_window(requested_window, n_traces)
    warnings: list[dict[str, Any]] = []
    if window < 3 or n_traces < 3:
        warnings.append(
            build_runtime_warning(
                "trace_median_window_too_small",
                "道数或窗口过小，道向中值滤波已跳过。",
                method_id="trace_median_filter",
            )
        )
        result = np.array(arr, copy=True)
    else:
        result = median_filter(arr, size=(1, window), mode="nearest")
        if preserve_mean:
            # Keep the global DC level stable for visual comparison workflows.
            result = result + (float(np.mean(arr)) - float(np.mean(result)))

    meta: dict[str, Any] = {
        "method": "trace_median_filter",
        "axis": "trace",
        "requested_window_traces": int(requested_window),
        "effective_window_traces": int(window),
        "preserve_mean": bool(preserve_mean),
        "input_shape": [int(n_samples), int(n_traces)],
    }
    if warnings:
        meta["runtime_warnings"] = warnings
    return result.astype(np.float32, copy=False), meta


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _odd_window(value: int | float | str, upper_bound: int) -> int:
    window = _safe_int(value, 5)
    window = max(1, window)
    window = min(window, max(1, int(upper_bound)))
    if window % 2 == 0:
        window = max(1, window - 1)
    return window

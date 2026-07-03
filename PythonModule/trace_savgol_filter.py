#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace-axis Savitzky-Golay smoothing for GPR B-scan data."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.runtime_warnings import build_runtime_warning


_VALID_MODES = {"interp", "nearest", "mirror", "constant", "wrap"}


def method_trace_savgol_filter(
    data: np.ndarray,
    window_traces: int = 7,
    polyorder: int = 2,
    derivative: int = 0,
    mode: str = "interp",
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Savitzky-Golay smoothing along the trace axis only.

    The method is intentionally constrained to smoothing (derivative=0) for the
    daily-processing UI. Trace-axis derivative diagnostics should be implemented
    as a separate method if needed.
    """
    from scipy.signal import savgol_filter

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    n_samples, n_traces = arr.shape
    warnings: list[dict[str, Any]] = []

    requested_derivative = _safe_int(derivative, 0)
    derivative_i = 0
    if requested_derivative != 0:
        warnings.append(
            build_runtime_warning(
                "trace_savgol_derivative_forced_to_zero",
                "道向 Savitzky-Golay 平滑固定使用 derivative=0；导数诊断未在该方法中开放。",
                method_id="trace_savgol_filter",
            )
        )

    window = _odd_window(window_traces, n_traces)
    order = max(0, _safe_int(polyorder, 2))
    if order >= window:
        adjusted = max(0, window - 1)
        warnings.append(
            build_runtime_warning(
                "trace_savgol_polyorder_adjusted",
                f"polyorder={order} 不小于窗口长度，已自动调整为 {adjusted}。",
                method_id="trace_savgol_filter",
            )
        )
        order = adjusted

    mode_key = str(mode or "interp").strip().lower()
    if mode_key not in _VALID_MODES:
        warnings.append(
            build_runtime_warning(
                "trace_savgol_mode_adjusted",
                f"不支持的 mode={mode!r}，已改用 interp。",
                method_id="trace_savgol_filter",
            )
        )
        mode_key = "interp"

    if window < 3 or n_traces < 3 or order < 1:
        warnings.append(
            build_runtime_warning(
                "trace_savgol_window_too_small",
                "道数、窗口或阶数不足，道向 Savitzky-Golay 平滑已跳过。",
                method_id="trace_savgol_filter",
            )
        )
        result = np.array(arr, copy=True)
    else:
        if mode_key == "interp" and window > n_traces:
            mode_key = "nearest"
        result = savgol_filter(
            arr,
            window_length=window,
            polyorder=order,
            deriv=derivative_i,
            axis=1,
            mode=mode_key,
        )

    meta: dict[str, Any] = {
        "method": "trace_savgol_filter",
        "axis": "trace",
        "requested_window_traces": _safe_int(window_traces, 7),
        "effective_window_traces": int(window),
        "polyorder": int(order),
        "requested_derivative": int(requested_derivative),
        "derivative": int(derivative_i),
        "mode": str(mode_key),
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
    window = _safe_int(value, 7)
    window = max(1, window)
    window = min(window, max(1, int(upper_bound)))
    if window % 2 == 0:
        window = max(1, window - 1)
    return window

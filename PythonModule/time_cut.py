#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time-window cropping for GPR B-scan data."""

from __future__ import annotations

from math import ceil, floor
from typing import Any

import numpy as np


def method_time_cut(
    data: np.ndarray,
    mode: str = "remove_below",
    time_start_ns: float = 0.0,
    time_end_ns: float = 0.0,
    time_window_ns: float | None = None,
    time_step_s: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop a B-scan by two-way travel time.

    Modes:
        remove_below: keep samples shallower than ``time_end_ns``.
        remove_above: remove samples before ``time_start_ns``.
        keep_range: keep samples from ``time_start_ns`` to ``time_end_ns``.

    ``time_end_ns=0`` means "bottom of profile", so the default is a no-op.
    """

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("输入数据为空")

    samples = int(arr.shape[0])
    resolved_mode = str(mode or "remove_below").strip().lower()
    if resolved_mode not in {"remove_below", "remove_above", "keep_range"}:
        raise ValueError(
            "time_cut mode 必须是 remove_below、remove_above 或 keep_range"
        )

    total_ns = _resolve_total_time_ns(
        samples,
        time_window_ns=time_window_ns,
        time_step_s=time_step_s,
    )
    start_idx = _time_to_index(
        time_start_ns,
        samples=samples,
        total_time_ns=total_ns,
        rounding="floor",
    )
    end_ns = _safe_float(time_end_ns, default=0.0)
    end_idx = (
        samples
        if end_ns <= 0.0
        else _time_to_index(
            end_ns,
            samples=samples,
            total_time_ns=total_ns,
            rounding="ceil",
        )
    )

    if resolved_mode == "remove_below":
        cut_start, cut_end = 0, max(1, min(end_idx, samples))
    elif resolved_mode == "remove_above":
        cut_start, cut_end = min(start_idx, samples - 1), samples
    else:
        cut_start = min(start_idx, samples - 1)
        cut_end = max(cut_start + 1, min(end_idx, samples))

    result = arr[cut_start:cut_end, :].astype(np.float32, copy=True)
    kept_duration_ns = total_ns * float(cut_end - cut_start) / float(samples)
    start_time_ns = total_ns * float(cut_start) / float(samples)
    end_time_ns = total_ns * float(cut_end) / float(samples)

    return result, {
        "method": "time_cut",
        "mode": resolved_mode,
        "time_start_ns": _safe_float(time_start_ns, default=0.0),
        "time_end_ns": end_ns,
        "time_start_idx": int(cut_start),
        "time_end_idx": int(cut_end),
        "input_samples": int(samples),
        "output_samples": int(result.shape[0]),
        "effective_time_start_ns": float(start_time_ns),
        "effective_time_end_ns": float(end_time_ns),
        "header_info_updates": {
            "total_time_ns": float(kept_duration_ns),
            "a_scan_length": int(result.shape[0]),
            "time_cut_offset_ns": float(start_time_ns),
        },
    }


def _resolve_total_time_ns(
    samples: int,
    *,
    time_window_ns: float | None,
    time_step_s: float | None,
) -> float:
    total_ns = _safe_float(time_window_ns, default=0.0)
    if total_ns > 0.0:
        return total_ns
    step_s = _safe_float(time_step_s, default=0.0)
    if step_s > 0.0:
        return step_s * 1.0e9 * float(samples)
    return float(samples)


def _time_to_index(
    value_ns: float,
    *,
    samples: int,
    total_time_ns: float,
    rounding: str,
) -> int:
    value = max(0.0, _safe_float(value_ns, default=0.0))
    ratio = value / max(total_time_ns, 1.0e-12)
    raw = ratio * float(samples)
    idx = floor(raw) if rounding == "floor" else ceil(raw)
    return max(0, min(int(idx), samples))


def _safe_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)

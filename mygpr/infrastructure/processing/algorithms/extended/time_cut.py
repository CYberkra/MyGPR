#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time-window cropping for GPR B-scan data."""

from __future__ import annotations

from math import ceil, floor
from typing import Any

import numpy as np

from mygpr.domain.common.scalars import to_float
from mygpr.domain.processing.timebase import snap_sample_index


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
    A no-op requires no time basis and leaves the chained header untouched;
    a real cut without any time basis raises ``ValueError`` (the former
    samples-as-ns silent fallback is removed).
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
    time_start_value = to_float(time_start_ns, default=0.0)
    time_end_value = to_float(time_end_ns, default=0.0)
    if resolved_mode == "remove_below":
        requests_cut = time_end_value > 0.0
    elif resolved_mode == "remove_above":
        requests_cut = time_start_value > 0.0
    else:  # keep_range
        requests_cut = time_start_value > 0.0 or time_end_value > 0.0
    if not requests_cut:
        # Documented no-op: nothing to cut, so no time basis is required.
        # The header must NOT be updated here — the old samples-as-ns
        # total_time_ns leaked into the chained header and poisoned
        # downstream time windows (audit report 2026-09-05 §3.3).
        result = arr.astype(np.float32, copy=True)
        return result, {
            "method": "time_cut",
            "mode": resolved_mode,
            "time_start_ns": float(time_start_value),
            "time_end_ns": float(time_end_value),
            "time_start_idx": 0,
            "time_end_idx": int(samples),
            "input_samples": int(samples),
            "output_samples": int(result.shape[0]),
        }
    header = kwargs.get("header_info") or {}
    source_axis = header.get("time_axis_ns")
    total_ns = 0.0
    if source_axis is None:
        total_ns = _resolve_total_time_ns(
            samples, time_window_ns=time_window_ns, time_step_s=time_step_s,
            header_info=header,
        )
    offset_ns = to_float(header.get("time_cut_offset_ns"), default=0.0)
    if source_axis is not None:
        axis, axis_step = _validate_time_axis(source_axis, samples)
        offset_ns = float(axis[0])
    endpoint = bool(header.get("time_axis_endpoint", False))
    step_ns = total_ns / max(1, samples - 1 if endpoint else samples)
    if source_axis is not None:
        step_ns = axis_step
    elif endpoint and to_float(time_step_s, default=0.0) > 0:
        step_ns = float(time_step_s) * 1e9
    # Index conversion uses an exclusive boundary one interval after the last
    # sample, even when the source window explicitly includes its endpoint.
    total_ns = step_ns * samples
    start_idx = _time_to_index(
        max(0.0, time_start_value - offset_ns) if time_start_value > 0 else 0.0,
        samples=samples,
        total_time_ns=total_ns,
        rounding="ceil",
    )
    end_idx = (
        samples
        if time_end_value <= 0.0
        else _time_to_index(
            max(0.0, time_end_value - offset_ns),
            samples=samples,
            total_time_ns=total_ns,
            rounding="ceil",
        )
    )
    if source_axis is not None:
        start_idx = 0 if time_start_value <= 0 else int(np.searchsorted(axis, time_start_value, side="left"))
        end_idx = samples if time_end_value <= 0 else int(np.searchsorted(axis, time_end_value, side="left"))
    if resolved_mode == "remove_below":
        cut_start, cut_end = 0, end_idx
    elif resolved_mode == "remove_above":
        cut_start, cut_end = start_idx, samples
    else:
        cut_start, cut_end = start_idx, end_idx
    if cut_start >= cut_end:
        raise ValueError("time_cut 选择范围没有采样点")
    result = arr[cut_start:cut_end, :].astype(np.float32, copy=True)
    kept_duration_ns = step_ns * max(0, cut_end - cut_start - int(endpoint))
    start_time_ns = float(axis[cut_start]) if source_axis is not None else offset_ns + step_ns * cut_start
    end_time_ns = (float(axis[cut_end]) if cut_end < samples else float(axis[-1]) + step_ns) if source_axis is not None else offset_ns + step_ns * cut_end
    axis_updates = {}
    for key in ("time_axis_ns", "depth_axis_m"):
        values = header.get(key)
        if values is not None:
            vector = np.asarray(values)
            if vector.shape == (samples,):
                axis_updates[key] = vector[cut_start:cut_end].copy()
    return result, {
        "method": "time_cut",
        "mode": resolved_mode,
        "time_start_ns": float(time_start_value),
        "time_end_ns": float(time_end_value),
        "time_start_idx": int(cut_start),
        "time_end_idx": int(cut_end),
        "input_samples": int(samples),
        "output_samples": int(result.shape[0]),
        "effective_time_start_ns": float(start_time_ns),
        "effective_time_end_ns": float(end_time_ns),
        "header_info_updates": {
            **axis_updates,
            "total_time_ns": float(kept_duration_ns),
            "time_window_ns": float(kept_duration_ns),
            "time_axis_endpoint": endpoint,
            "time_step_s": step_ns * 1e-9,
            "a_scan_length": int(result.shape[0]),
            "time_cut_offset_ns": float(start_time_ns),
        },
    }


def _validate_time_axis(source_axis: Any, samples: int) -> tuple[np.ndarray, float]:
    source = np.asarray(source_axis)
    axis = np.asarray(source, dtype=np.float64)
    if axis.shape != (samples,) or not np.all(np.isfinite(axis)) or samples < 2:
        raise ValueError("time_cut 时间轴无效")
    step = (axis[-1] - axis[0]) / (samples - 1)
    expected = axis[0] + np.arange(samples) * step
    tolerance = (np.abs(np.spacing(source)).astype(np.float64) * 1.5
                 if source.dtype.kind == "f" else np.zeros(samples))
    tolerance += np.finfo(np.float64).eps * np.maximum(np.abs(expected), 1) * 4
    if step <= 0 or np.any(np.abs(axis - expected) > tolerance):
        raise ValueError("time_cut 不支持非均匀时间轴")
    return axis, float(step)


def _resolve_total_time_ns(
    samples: int,
    *,
    time_window_ns: float | None,
    time_step_s: float | None,
    header_info: dict[str, Any] | None = None,
) -> float:
    """Resolve the total two-way travel time basis in nanoseconds.

    Resolution order:
        1. explicit ``time_window_ns`` parameter
        2. explicit ``time_step_s`` parameter (``step * 1e9 * samples``)
        3. ``header_info["total_time_ns"]`` / ``header_info["time_window_ns"]``
           (injected by ``extended/_runtime_kwargs`` on the native path)

    Missing a time basis is a hard error: silently treating the sample
    count as nanoseconds produced dimensionally wrong crops ("samples-as-ns"
    fallback, audit report 2026-09-05 §3.3).
    """
    try:
        total_ns = to_float(time_window_ns, default=0.0)
    except (TypeError, ValueError):
        total_ns = 0.0
    if total_ns > 0.0:
        return total_ns
    try:
        step_s = to_float(time_step_s, default=0.0)
    except (TypeError, ValueError):
        step_s = 0.0
    if step_s > 0.0:
        return step_s * 1.0e9 * float(samples)
    if isinstance(header_info, dict):
        for key in ("total_time_ns", "time_window_ns"):
            try:
                header_ns = to_float(header_info.get(key), default=0.0)
            except (TypeError, ValueError):
                header_ns = 0.0
            if header_ns > 0.0:
                return header_ns
    raise ValueError(
        "time_cut 缺少时间基准：无法把 time_start_ns/time_end_ns 映射到采样"
        "点。请提供 time_window_ns 或 time_step_s 参数，或传入含 "
        "total_time_ns/time_window_ns 的 header_info（旧版把采样点数当 ns "
        "使用的静默回退已移除）。"
    )


def _time_to_index(
    value_ns: float,
    *,
    samples: int,
    total_time_ns: float,
    rounding: str,
) -> int:
    try:
        value = max(0.0, to_float(value_ns, default=0.0))
    except (TypeError, ValueError):
        value = 0.0
    ratio = value / max(total_time_ns, 1.0e-12)
    raw = ratio * float(samples)
    raw = snap_sample_index(raw)
    idx = floor(raw) if rounding == "floor" else ceil(raw)
    return max(0, min(int(idx), samples))

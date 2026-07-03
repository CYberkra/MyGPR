#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-time correction for ndarray-based GPR processing.

Round-2 drop-in version:
- keeps the ndarray API unchanged
- removes unreachable dead code
- uses direct slicing instead of per-trace np.roll loops
- preserves current semantics: shift upward in time and zero-fill the tail
"""

from __future__ import annotations

import numpy as np

from core.scalar_utils import to_float, to_float_or_none


def _resolve_time_step_s(ny: int, time_step_s: float | None) -> float:
    if time_step_s is not None:
        value = to_float(time_step_s, default=0.0)
        if value > 0:
            return value
    return 48e-9 / max(1, int(ny))


def _apply_zero_time_shift(
    data: np.ndarray,
    new_zero_time: float = 5.0,
    time_step_s: float | None = None,
) -> tuple[np.ndarray, int, float]:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"输入数据必须是2维数组，当前维度: {arr.ndim}, shape: {arr.shape}"
        )

    ny, nx = arr.shape
    if ny == 0 or nx == 0:
        raise ValueError(f"输入数据维度为0: shape={arr.shape}")

    step_s = _resolve_time_step_s(ny, time_step_s)
    step_ns = step_s * 1e9

    zero_time_value = to_float_or_none(new_zero_time)
    if zero_time_value is None:
        raise ValueError("new_zero_time must be numeric")
    shift_samples = int(max(0.0, zero_time_value) / max(step_ns, 1.0e-12))
    shift_samples = max(0, min(shift_samples, ny - 1))

    if shift_samples == 0:
        result = arr.astype(np.float32, copy=True)
    else:
        result = np.zeros((ny, nx), dtype=np.float32)
        result[:-shift_samples, :] = arr[shift_samples:, :]

    return result.astype(np.float32, copy=False), int(shift_samples), float(step_s)


def set_zero_time(
    infilename="",
    outfilename="",
    outimagename="",
    length_trace=48,
    Start_position=0,
    Scans_per_meter=50,
    newZeroTime=5.7,
):
    """
    Legacy CSV-I/O wrapper kept for compatibility.

    Note:
    This wrapper now follows the ndarray path used by the GUI:
    it keeps the original matrix size, shifts data upward by the
    zero-time offset, and zero-fills the tail.
    """
    try:
        from read_file_data import readcsv, savecsv, save_image
    except ImportError:
        import sys
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from read_file_data import readcsv, savecsv, save_image

    data = np.asarray(readcsv(infilename), dtype=np.float64)
    ny, nx = data.shape
    length_trace_value = to_float(length_trace, default=48.0)
    scans_per_meter_value = to_float(Scans_per_meter, default=50.0)
    scans_per_meter = scans_per_meter_value if scans_per_meter_value > 0 else 1.0
    start_position = to_float(Start_position, default=0.0)
    new_zero_time = to_float_or_none(newZeroTime)
    if new_zero_time is None:
        new_zero_time = 5.7
    twtt = np.linspace(0, length_trace_value, ny)
    x = np.linspace(
        start_position,
        start_position + nx / scans_per_meter,
        nx,
    )

    if ny <= 0 or nx <= 0:
        return {
            "data": [],
            "x": x.tolist(),
            "twtt": twtt.tolist(),
            "error_sign": 1,
            "error_feedback": "输入数据为空",
        }

    if new_zero_time >= float(twtt[-1]):
        return {
            "data": [],
            "x": x.tolist(),
            "twtt": twtt.tolist(),
            "error_sign": 2,
            "error_feedback": "The newZeroTime absolute value must <= The maximum value of the timeline",
        }

    time_step_s = (length_trace_value * 1e-9) / max(1, ny)
    result, shift_samples, _ = _apply_zero_time_shift(
        data,
        new_zero_time=new_zero_time,
        time_step_s=time_step_s,
    )

    if outfilename:
        savecsv(result, outfilename)
    if outimagename:
        save_image(
            result,
            outimagename,
            "Data[set_zero_time]",
            time_range=(0, length_trace_value),
            distance_range=(float(x[0]), float(x[-1])) if len(x) else (0.0, 0.0),
        )

    return {
        "data": result.tolist(),
        "x": x.tolist(),
        "twtt": twtt.tolist(),
        "error_sign": 0,
        "error_feedback": "",
        "shift_samples": int(shift_samples),
    }


def method_set_zero_time(data, new_zero_time=5.0, time_step_s=None, **kwargs):
    """零时间校正 - GUI / auto-tune ndarray 接口。"""
    zero_time_value = to_float_or_none(new_zero_time)
    if zero_time_value is None:
        raise ValueError("new_zero_time must be numeric")
    result, shift_samples, step_s = _apply_zero_time_shift(
        data,
        new_zero_time=zero_time_value,
        time_step_s=time_step_s,
    )
    return result, {
        "method": "set_zero_time",
        "new_zero_time": zero_time_value,
        "shift_samples": int(shift_samples),
        "time_step_s": float(step_s),
    }


if __name__ == "__main__":
    print("This module is intended to be imported by the UAV-GPR processing engine.")

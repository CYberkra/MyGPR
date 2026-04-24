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


def _resolve_time_step_s(ny: int, time_step_s: float | None) -> float:
    if time_step_s is not None:
        try:
            value = float(time_step_s)
            if value > 0:
                return value
        except Exception:
            pass
    return 48e-9 / max(1, int(ny))


def _apply_zero_time_shift(
    data: np.ndarray,
    new_zero_time: float = 5.0,
    time_step_s: float | None = None,
) -> tuple[np.ndarray, int, float]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"输入数据必须是2维数组，当前维度: {arr.ndim}, shape: {arr.shape}"
        )

    ny, nx = arr.shape
    if ny == 0 or nx == 0:
        raise ValueError(f"输入数据维度为0: shape={arr.shape}")

    step_s = _resolve_time_step_s(ny, time_step_s)
    step_ns = step_s * 1e9

    shift_samples = int(max(0.0, float(new_zero_time)) / max(step_ns, 1.0e-12))
    shift_samples = max(0, min(shift_samples, ny - 1))

    result = np.zeros((ny, nx), dtype=np.float64)
    if shift_samples == 0:
        result[:] = arr
    else:
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
    twtt = np.linspace(0, float(length_trace), ny)
    scans_per_meter = float(Scans_per_meter) if float(Scans_per_meter) > 0 else 1.0
    x = np.linspace(
        float(Start_position),
        float(Start_position) + nx / scans_per_meter,
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

    if float(newZeroTime) >= float(twtt[-1]):
        return {
            "data": [],
            "x": x.tolist(),
            "twtt": twtt.tolist(),
            "error_sign": 2,
            "error_feedback": "The newZeroTime absolute value must <= The maximum value of the timeline",
        }

    time_step_s = (float(length_trace) * 1e-9) / max(1, ny)
    result, shift_samples, _ = _apply_zero_time_shift(
        data,
        new_zero_time=float(newZeroTime),
        time_step_s=time_step_s,
    )

    if outfilename:
        savecsv(result, outfilename)
    if outimagename:
        save_image(
            result,
            outimagename,
            "Data[set_zero_time]",
            time_range=(0, float(length_trace)),
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
    result, shift_samples, step_s = _apply_zero_time_shift(
        data,
        new_zero_time=float(new_zero_time),
        time_step_s=time_step_s,
    )
    return result, {
        "method": "set_zero_time",
        "new_zero_time": float(new_zero_time),
        "shift_samples": int(shift_samples),
        "time_step_s": float(step_s),
    }


if __name__ == "__main__":
    print("This module is intended to be imported by the UAV-GPR processing engine.")

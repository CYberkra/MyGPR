#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeWOW / low-frequency drift correction for ndarray-based GPR processing.

Round-2 drop-in version:
- keeps the public ndarray API unchanged
- preserves current edge behavior exactly (valid moving average + edge padding)
- replaces per-trace Python loops with a vectorized cumulative-sum implementation
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401

from core.gprpy_compat import apply_gprpy_dewow


def _apply_dewow_exact(data: np.ndarray, window: int) -> np.ndarray:
    return apply_gprpy_dewow(data, window)


def dewow(
    infilename="",
    outfilename="",
    outimagename="",
    length_trace=48,
    Start_position=0,
    Scans_per_meter=50,
    window=23,
):
    """
    Legacy CSV-I/O wrapper kept for compatibility with older scripts.
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

    raw = np.asarray(readcsv(infilename), dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={raw.shape}")

    ny, nx = raw.shape
    twtt = np.linspace(0, length_trace, ny)
    scans_per_meter = float(Scans_per_meter) if float(Scans_per_meter) > 0 else 1.0
    x = np.linspace(0, nx / scans_per_meter, nx)

    result = _apply_dewow_exact(raw, int(window))
    if outfilename:
        savecsv(result, outfilename)
    if outimagename:
        save_image(
            result,
            outimagename,
            "dewow",
            time_range=(0, float(length_trace)),
            distance_range=(float(Start_position), float(Start_position) + float(x[-1]))
            if len(x)
            else (0.0, 0.0),
        )

    return twtt, x, result


def method_dewow(data, window=23, **kwargs):
    """去直流（DeWOW）- GUI / auto-tune ndarray 接口。"""
    result = _apply_dewow_exact(data, int(window))
    return result, {"method": "dewow", "window": int(round(float(window)))}


if __name__ == "__main__":
    print("This module is intended to be imported by the UAV-GPR processing engine.")

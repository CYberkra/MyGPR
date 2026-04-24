#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Median-based background suppression for GPR B-scan data."""

from __future__ import annotations

import numpy as np


def method_median_background_2d(
    data: np.ndarray,
    ntraces: int = 51,
    **kwargs,
):
    """Suppress horizontal background using median trace estimation.

    Args:
        data: Input array with shape (samples, traces).
        ntraces: Window width along trace axis. If >= trace count, use full-width median.

    Returns:
        tuple: (result_array, metadata_dict)
    """
    from scipy.ndimage import median_filter

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")

    ntraces = max(1, int(ntraces))
    if ntraces >= arr.shape[1]:
        background = np.median(arr, axis=1, keepdims=True)
    else:
        if ntraces % 2 == 0:
            ntraces += 1
        background = median_filter(arr, size=(1, ntraces), mode="nearest")

    result = arr - background
    return result.astype(np.float32), {
        "method": "median_background_2D",
        "ntraces": ntraces,
    }

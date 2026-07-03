#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Median-based background suppression for GPR B-scan data."""

from __future__ import annotations

import numpy as np

from core.background_time_range import (
    apply_time_range_to_result,
    resolve_time_range_selection,
)


def method_median_background_2d(
    data: np.ndarray,
    ntraces: int = 51,
    time_start_idx=None,
    time_end_idx=None,
    time_start_ns=None,
    time_end_ns=None,
    time_window_ns=None,
    edge_taper_samples: int = 0,
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

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")

    ntraces = max(1, int(ntraces))
    if ntraces >= arr.shape[1]:
        background = np.median(arr, axis=1, keepdims=True)
    else:
        if ntraces % 2 == 0:
            ntraces += 1
        background = median_filter(arr, size=(1, ntraces), mode="nearest")

    full_result = arr - background
    selection = resolve_time_range_selection(
        arr.shape,
        time_start_idx=time_start_idx,
        time_end_idx=time_end_idx,
        time_start_ns=time_start_ns,
        time_end_ns=time_end_ns,
        time_window_ns=time_window_ns,
    )
    result = apply_time_range_to_result(
        arr,
        full_result,
        selection,
        edge_taper_samples=edge_taper_samples,
    )
    return result.astype(np.float32, copy=False), {
        "method": "median_background_2D",
        "ntraces": ntraces,
        "time_start_idx": int(selection.start_idx),
        "time_end_idx": int(selection.end_idx),
        "time_range_source": selection.source,
    }

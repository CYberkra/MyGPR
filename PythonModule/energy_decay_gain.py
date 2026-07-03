#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust energy-decay gain for GPR B-scan data."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.scalar_utils import to_float, to_int


def method_energy_decay_gain(
    data: np.ndarray,
    strength: float = 1.0,
    smoothing_samples: int = 31,
    min_gain: float = 0.5,
    max_gain: float = 8.0,
    floor_ratio: float = 0.05,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply gain from a robust per-time energy decay estimate.

    The decay curve uses median absolute amplitude across traces. This keeps a
    compact strong reflector from dominating the gain estimate.
    """

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    resolved_strength = max(0.0, to_float(strength, default=1.0))
    smooth_window = max(1, to_int(smoothing_samples, default=31))
    gain_min = max(0.0, to_float(min_gain, default=0.5))
    gain_max = max(gain_min, to_float(max_gain, default=8.0))
    floor = max(0.0, to_float(floor_ratio, default=0.05))

    decay = np.median(np.abs(arr), axis=1).astype(np.float64, copy=False)
    decay_smooth = _moving_average(decay, smooth_window)
    positive = decay_smooth[np.isfinite(decay_smooth) & (decay_smooth > 0.0)]
    if positive.size == 0:
        gain_curve = np.ones(arr.shape[0], dtype=np.float32)
        return arr.copy(), {
            "method": "energy_decay_gain",
            "strength": resolved_strength,
            "smoothing_samples": smooth_window,
            "min_gain": gain_min,
            "max_gain": gain_max,
            "floor_ratio": floor,
            "gain_curve": gain_curve,
            "decay_curve": decay_smooth.astype(np.float32),
            "reference_level": 0.0,
        }

    reference = float(np.quantile(positive, 0.85))
    floor_value = max(float(np.median(positive)) * floor, 1.0e-12)
    safe_decay = np.maximum(decay_smooth, floor_value)
    raw_gain = np.power(reference / safe_decay, resolved_strength)
    gain_curve = np.clip(raw_gain, gain_min, gain_max).astype(np.float32)
    result = arr * gain_curve[:, np.newaxis]

    return result.astype(np.float32, copy=False), {
        "method": "energy_decay_gain",
        "strength": resolved_strength,
        "smoothing_samples": smooth_window,
        "min_gain": gain_min,
        "max_gain": gain_max,
        "floor_ratio": floor,
        "gain_curve": gain_curve,
        "decay_curve": decay_smooth.astype(np.float32),
        "reference_level": reference,
    }


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or arr.size <= 1:
        return arr.copy()
    window = min(int(window), arr.size)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    prefix = np.empty(padded.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(padded, out=prefix[1:])
    return (prefix[window:] - prefix[:-window]) / float(window)

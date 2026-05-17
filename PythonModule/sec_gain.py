#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEC增益（深度补偿）- round-2 drop-in version."""

from __future__ import annotations

import numpy as np


def _finite_float(value, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} 必须是有限数值")
    return resolved


def method_sec_gain(data, gain_min=1.0, gain_max=6.0, power=1.0, **kwargs):
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    gain_min = _finite_float(gain_min, "gain_min")
    gain_max = _finite_float(gain_max, "gain_max")
    power = max(_finite_float(power, "power"), 1.0e-6)

    n_samples = int(arr.shape[0])
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64) ** power
    gain_curve = gain_min + (gain_max - gain_min) * t
    result = arr * gain_curve[:, None]

    return result.astype(np.float32, copy=False), {
        "method": "sec_gain",
        "gain_min": gain_min,
        "gain_max": gain_max,
        "power": power,
        "gain_curve": gain_curve.astype(np.float32, copy=False),
    }

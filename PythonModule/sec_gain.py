#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEC增益（深度补偿）- round-2 drop-in version."""

from __future__ import annotations

import numpy as np


def method_sec_gain(
    data: np.ndarray,
    gain_min: float = 1.0,
    gain_max: float = 6.0,
    power: float = 1.0,
    **kwargs: object,
) -> tuple[np.ndarray, dict[str, object]]:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    gain_min = float(gain_min)
    gain_max = float(gain_max)
    power = max(float(power), 1.0e-6)

    n_samples = int(arr.shape[0])
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32) ** np.float32(power)
    gain_curve = (gain_min + (gain_max - gain_min) * t).astype(np.float32)
    result = arr * gain_curve[:, None]

    return result.astype(np.float32, copy=False), {
        "method": "sec_gain",
        "gain_min": gain_min,
        "gain_max": gain_max,
        "power": power,
        "gain_curve": gain_curve,
    }

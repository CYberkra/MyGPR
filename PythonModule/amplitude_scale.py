#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constant scaling and simple amplitude normalization for GPR B-scans."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_float(value: Any, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} 必须是有限数值")
    return resolved


def method_amplitude_scale(
    data: np.ndarray,
    mode: str = "constant",
    scale: float = 1.0,
    target: float = 1.0,
    eps: float = 1.0e-8,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply explicit amplitude scale or global normalization."""

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    resolved_mode = str(mode or "constant").strip().lower()
    target_value = _finite_float(target, "target")
    safe_eps = max(_finite_float(eps, "eps"), 1.0e-12)

    if resolved_mode == "constant":
        requested_scale = _finite_float(scale, "scale")
        effective_scale = requested_scale
    elif resolved_mode == "peak":
        peak = float(np.max(np.abs(arr)))
        effective_scale = target_value / max(peak, safe_eps)
    elif resolved_mode == "rms":
        rms = float(np.sqrt(np.mean(np.asarray(arr, dtype=np.float64) ** 2)))
        effective_scale = target_value / max(rms, safe_eps)
    else:
        raise ValueError("amplitude_scale mode 必须是 constant、peak 或 rms")

    result = arr * np.float32(effective_scale)
    return result.astype(np.float32, copy=False), {
        "method": "amplitude_scale",
        "mode": resolved_mode,
        "scale": _finite_float(scale, "scale"),
        "target": target_value,
        "effective_scale": float(effective_scale),
    }

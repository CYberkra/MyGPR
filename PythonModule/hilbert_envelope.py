#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hilbert-envelope attribute extraction for GPR B-scan data."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import hilbert


def _finite_float(value: Any, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} 必须是有限数值")
    return resolved


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def method_hilbert_envelope(
    data: np.ndarray,
    normalize: bool = False,
    log_compress: bool = False,
    eps: float = 1.0e-8,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute trace-wise analytic-signal envelope along the time/sample axis."""

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    normalize_enabled = _as_bool(normalize)
    log_compress_enabled = _as_bool(log_compress)
    safe_eps = max(_finite_float(eps, "eps"), 1.0e-12)
    envelope = np.abs(hilbert(arr.astype(np.float64), axis=0))
    peak_before_norm = float(np.max(envelope)) if envelope.size else 0.0

    if normalize_enabled:
        envelope = envelope / max(peak_before_norm, safe_eps)
    if log_compress_enabled:
        envelope = np.log1p(envelope)

    return envelope.astype(np.float32, copy=False), {
        "method": "hilbert_envelope",
        "normalize": normalize_enabled,
        "log_compress": log_compress_enabled,
        "peak_before_norm": peak_before_norm,
    }

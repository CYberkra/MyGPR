#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hilbert-envelope attribute extraction for GPR B-scan data."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import hilbert


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

    safe_eps = max(float(eps), 1.0e-12)
    envelope = np.abs(hilbert(arr.astype(np.float64), axis=0))
    peak_before_norm = float(np.max(envelope)) if envelope.size else 0.0

    if bool(normalize):
        envelope = envelope / max(peak_before_norm, safe_eps)
    if bool(log_compress):
        envelope = np.log1p(envelope)

    return envelope.astype(np.float32, copy=False), {
        "method": "hilbert_envelope",
        "normalize": bool(normalize),
        "log_compress": bool(log_compress),
        "peak_before_norm": peak_before_norm,
    }

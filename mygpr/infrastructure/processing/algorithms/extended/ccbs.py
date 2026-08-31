#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-correlation-based background subtraction for GPR B-scan data."""

from __future__ import annotations

import numpy as np

_EPSILON = 1.0e-10


def _resolve_reference_wave(
    b_scan: np.ndarray,
    reference_wave: np.ndarray | None,
) -> np.ndarray:
    if b_scan.ndim != 2:
        raise ValueError(
            f"b_scan must be 2D array with shape (M, N), got {b_scan.ndim}D"
        )
    if reference_wave is None:
        return np.mean(b_scan, axis=1)
    reference = np.asarray(reference_wave)
    if reference.ndim == 0 or reference.shape[0] != b_scan.shape[0]:
        length = 0 if reference.ndim == 0 else reference.shape[0]
        raise ValueError(
            f"reference_wave length ({length}) must match "
            f"b_scan rows ({b_scan.shape[0]})"
        )
    return reference.reshape(-1)


def _normalized_cross_correlation(
    b_scan: np.ndarray,
    reference_wave: np.ndarray,
) -> np.ndarray | None:
    ref_norm = np.linalg.norm(reference_wave)
    if ref_norm < _EPSILON:
        return None
    trace_norms = np.linalg.norm(b_scan, axis=0)
    trace_norms_safe = np.where(trace_norms < _EPSILON, 1.0, trace_norms)
    dot_products = np.dot(reference_wave, b_scan)
    ncc_values = dot_products / (ref_norm * trace_norms_safe)
    return np.where(trace_norms < _EPSILON, 0.0, ncc_values)


def _subtract_weighted_background(
    b_scan: np.ndarray,
    reference_wave: np.ndarray,
    ncc_values: np.ndarray,
) -> np.ndarray:
    b_mean = np.mean(b_scan, axis=1)
    weights = np.clip((ncc_values + 1.0) / 2.0, 0.0, 1.0)
    b_mean_2d = b_mean[:, np.newaxis]
    reference_2d = reference_wave[:, np.newaxis]
    weights_2d = weights[np.newaxis, :]
    processed = np.array(
        b_scan,
        dtype=np.result_type(b_scan.dtype, np.float32),
        copy=True,
    )
    processed -= b_mean_2d
    processed -= weights_2d * (reference_2d - b_mean_2d)
    return processed


def apply_ccbs_filter(
    b_scan: np.ndarray,
    reference_wave: np.ndarray | None = None,
) -> np.ndarray:
    """Remove correlated GPR background using an adaptive reference/mean mix."""

    scan = np.asarray(b_scan)
    reference = _resolve_reference_wave(scan, reference_wave)
    ncc_values = _normalized_cross_correlation(scan, reference)
    if ncc_values is None:
        return scan.copy()
    return _subtract_weighted_background(scan, reference, ncc_values)


def method_ccbs(
    data: np.ndarray,
    reference_wave: np.ndarray | None = None,
    **kwargs: object,
) -> tuple[np.ndarray, dict[str, object]]:
    """Registry-compatible CCBS wrapper."""

    del kwargs
    result = apply_ccbs_filter(data, reference_wave=reference_wave)
    metadata = {
        "method": "CCBS",
        "description": "Cross-Correlation-Based Background Subtraction",
        "reference_used": reference_wave is not None,
    }
    return result, metadata

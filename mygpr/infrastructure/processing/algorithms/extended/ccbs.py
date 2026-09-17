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
    use_custom_ref: bool = False,
    reference_trace_index: int | None = None,
    trace_metadata: dict[str, Any] | None = None,
    **kwargs: object,
) -> tuple[np.ndarray, dict[str, object]]:
    """Registry-compatible CCBS wrapper.

    Reference-wave resolution order (Wu 2022: measured reference echo of the
    target-free zone):
    1. explicit ``reference_wave`` array (Python API / tests);
    2. ``trace_metadata["reference_wave"]`` array injected by the pipeline;
    3. ``reference_trace_index`` parameter (JSON-safe scalar, GUI-usable);
    4. fall back to the per-row mean with a warning when
       ``use_custom_ref`` is requested but no source resolves.
    """

    del kwargs
    resolved = reference_wave
    source = "explicit" if resolved is not None else None
    warnings: list[str] = []
    if resolved is None and use_custom_ref:
        metadata_arrays = trace_metadata or {}
        candidate = metadata_arrays.get("reference_wave")
        if candidate is not None:
            resolved = np.asarray(candidate).reshape(-1)
            source = "trace_metadata"
    if resolved is None and use_custom_ref and reference_trace_index is not None:
        index = int(reference_trace_index)
        scan = np.asarray(data)
        trace_count = int(scan.shape[1]) if scan.ndim == 2 else 0
        if 0 <= index < trace_count:
            resolved = scan[:, index].reshape(-1)
            source = "reference_trace_index"
        else:
            warnings.append(
                f"reference_trace_index {index} 超出范围 [0, {trace_count})"
            )
    if resolved is None and use_custom_ref and source is None:
        warnings.append(
            "use_custom_ref 已启用但未提供参考波"
            "（trace_metadata['reference_wave'] 或 reference_trace_index），"
            "回退为均值参考"
        )
    result = apply_ccbs_filter(data, reference_wave=resolved)
    metadata = {
        "method": "CCBS",
        "description": "Cross-Correlation-Based Background Subtraction",
        "reference_used": resolved is not None,
        "reference_source": source,
        "runtime_warnings": warnings,
    }
    return result, metadata

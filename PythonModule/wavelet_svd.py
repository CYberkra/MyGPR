#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wavelet + SVD denoising."""

from __future__ import annotations

import numpy as np
import pywt

from PythonModule.svd_subspace import method_svd_subspace


MAD_NORMALIZER = 0.6745
DEFAULT_THRESHOLD_STRATEGY = "mad_universal"
LEGACY_THRESHOLD_STRATEGY = "global_fraction"


def _resolve_threshold_strategy(strategy: str | None) -> str:
    normalized = str(strategy or DEFAULT_THRESHOLD_STRATEGY).strip().lower()
    if normalized in {DEFAULT_THRESHOLD_STRATEGY, LEGACY_THRESHOLD_STRATEGY}:
        return normalized
    return DEFAULT_THRESHOLD_STRATEGY


def _build_mad_universal_detail_thresholds(
    coeffs: list,
    threshold_scale: float,
) -> tuple[float, list[dict[str, float | int]]]:
    if len(coeffs) < 2:
        return 0.0, []

    finest_detail = np.asarray(coeffs[-1][2], dtype=np.float64)
    if finest_detail.size == 0:
        sigma = 0.0
    else:
        median = float(np.median(finest_detail))
        mad = float(np.median(np.abs(finest_detail - median)))
        sigma = mad / MAD_NORMALIZER if mad > 0.0 else 0.0

    detail_thresholds: list[dict[str, float | int]] = []
    actual_levels = len(coeffs) - 1
    for idx in range(1, len(coeffs)):
        _, _, cD = coeffs[idx]
        coefficient_count = max(int(np.asarray(cD).size), 2)
        universal_threshold = sigma * float(np.sqrt(2.0 * np.log(coefficient_count)))
        detail_thresholds.append(
            {
                "level": actual_levels - idx + 1,
                "coefficient_count": coefficient_count,
                "abs_threshold": float(threshold_scale * universal_threshold),
            }
        )

    return sigma, detail_thresholds


def method_wavelet_svd(
    data: np.ndarray,
    wavelet: str = "db4",
    levels: int = 3,
    threshold: float = 0.1,
    rank_start: int = 2,
    rank_end: int = 40,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
    threshold_mode: str = "soft",
    **kwargs,
):
    """Apply wavelet decomposition, SVD on approximation, and threshold details."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")

    if threshold_mode not in {"soft", "hard"}:
        threshold_mode = "soft"

    max_level = pywt.dwtn_max_level(arr.shape, wavelet)
    actual_levels = max(1, min(int(levels), max_level if max_level > 0 else 1))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    threshold_strategy = _resolve_threshold_strategy(threshold_strategy)

    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=actual_levels)

    filtered_coeffs: list = [None] * len(coeffs)
    approx = coeffs[0]
    approx_recon, _ = method_svd_subspace(
        approx,
        rank_start=rank_start,
        rank_end=min(rank_end, min(approx.shape)),
    )
    filtered_coeffs[0] = approx_recon.astype(np.float64)

    meta: dict[str, object] = {
        "method": "wavelet_svd",
        "wavelet": wavelet,
        "levels": actual_levels,
        "threshold": threshold,
        "rank_start": rank_start,
        "rank_end": rank_end,
        "threshold_mode": threshold_mode,
        "threshold_strategy": threshold_strategy,
    }

    if threshold_strategy == LEGACY_THRESHOLD_STRATEGY:
        max_value = float(np.max(np.abs(arr))) if arr.size else 0.0
        abs_threshold = threshold * max_value
        meta["global_abs_threshold"] = abs_threshold

        for idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[idx]
            filtered_coeffs[idx] = (
                pywt.threshold(cH, abs_threshold, mode=threshold_mode),
                pywt.threshold(cV, abs_threshold, mode=threshold_mode),
                pywt.threshold(cD, abs_threshold, mode=threshold_mode),
            )
    else:
        estimated_sigma, detail_thresholds = _build_mad_universal_detail_thresholds(
            coeffs,
            threshold,
        )
        meta["estimated_sigma"] = estimated_sigma
        meta["detail_thresholds"] = detail_thresholds

        for idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[idx]
            abs_threshold = float(detail_thresholds[idx - 1]["abs_threshold"])
            filtered_coeffs[idx] = (
                pywt.threshold(cH, abs_threshold, mode=threshold_mode),
                pywt.threshold(cV, abs_threshold, mode=threshold_mode),
                pywt.threshold(cD, abs_threshold, mode=threshold_mode),
            )

    reconstructed = pywt.waverec2(filtered_coeffs, wavelet)
    reconstructed = reconstructed[: arr.shape[0], : arr.shape[1]]

    return reconstructed.astype(np.float32), meta

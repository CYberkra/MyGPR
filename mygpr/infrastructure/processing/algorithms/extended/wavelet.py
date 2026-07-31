"""Optional native wavelet and wavelet-SVD denoising.

PyWavelets is imported only when one of these methods executes so the complete
backend remains importable on CPU-only/minimal installations.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.infrastructure.processing.algorithms.global_spectral import (
    method_svd_subspace_native,
)

MAD_NORMALIZER = 0.6745
DEFAULT_THRESHOLD_STRATEGY = "mad_universal"
LEGACY_THRESHOLD_STRATEGY = "global_fraction"


def _require_pywt():
    try:
        import pywt
    except ModuleNotFoundError as exc:
        if exc.name != "pywt":
            raise
        raise ImportError(
            "Wavelet processing requires PyWavelets. Install with: pip install PyWavelets"
        ) from exc
    return pywt


def _resolve_threshold_strategy(strategy: str | None) -> str:
    normalized = str(strategy or DEFAULT_THRESHOLD_STRATEGY).strip().lower()
    if normalized in {DEFAULT_THRESHOLD_STRATEGY, LEGACY_THRESHOLD_STRATEGY}:
        return normalized
    return DEFAULT_THRESHOLD_STRATEGY


def _build_mad_universal_detail_thresholds(
    coeffs: list[Any], threshold_scale: float
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
        _, _, c_d = coeffs[idx]
        coefficient_count = max(int(np.asarray(c_d).size), 2)
        universal_threshold = sigma * float(np.sqrt(2.0 * np.log(coefficient_count)))
        detail_thresholds.append(
            {
                "level": actual_levels - idx + 1,
                "coefficient_count": coefficient_count,
                "abs_threshold": float(threshold_scale * universal_threshold),
            }
        )
    return sigma, detail_thresholds


def _threshold_details(
    coeffs: list[Any],
    *,
    arr: np.ndarray,
    threshold: float,
    threshold_strategy: str,
    threshold_mode: str,
    pywt: Any,
) -> tuple[list[Any], dict[str, Any]]:
    filtered: list[Any] = [coeffs[0]]
    metadata: dict[str, Any] = {}
    if threshold_strategy == LEGACY_THRESHOLD_STRATEGY:
        maximum = float(np.max(np.abs(arr))) if arr.size else 0.0
        absolute = threshold * maximum
        metadata["global_abs_threshold"] = absolute
        for c_h, c_v, c_d in coeffs[1:]:
            filtered.append(
                (
                    pywt.threshold(c_h, absolute, mode=threshold_mode),
                    pywt.threshold(c_v, absolute, mode=threshold_mode),
                    pywt.threshold(c_d, absolute, mode=threshold_mode),
                )
            )
        return filtered, metadata
    sigma, thresholds = _build_mad_universal_detail_thresholds(coeffs, threshold)
    metadata["estimated_sigma"] = sigma
    metadata["detail_thresholds"] = thresholds
    for index, (c_h, c_v, c_d) in enumerate(coeffs[1:]):
        absolute = float(thresholds[index]["abs_threshold"])
        filtered.append(
            (
                pywt.threshold(c_h, absolute, mode=threshold_mode),
                pywt.threshold(c_v, absolute, mode=threshold_mode),
                pywt.threshold(c_d, absolute, mode=threshold_mode),
            )
        )
    return filtered, metadata


def method_wavelet_2d(
    data: np.ndarray,
    wavelet: str = "db4",
    levels: int = 2,
    threshold: float = 0.1,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
    threshold_mode: str = "soft",
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    del kwargs
    pywt = _require_pywt()
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if threshold_mode not in {"soft", "hard"}:
        threshold_mode = "soft"
    max_level = pywt.dwtn_max_level(arr.shape, wavelet)
    actual_levels = max(1, min(int(levels), max_level if max_level > 0 else 1))
    resolved_threshold = float(np.clip(threshold, 0.0, 1.0))
    strategy = _resolve_threshold_strategy(threshold_strategy)
    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=actual_levels)
    filtered, threshold_metadata = _threshold_details(
        coeffs,
        arr=arr,
        threshold=resolved_threshold,
        threshold_strategy=strategy,
        threshold_mode=threshold_mode,
        pywt=pywt,
    )
    reconstructed = pywt.waverec2(filtered, wavelet)[: arr.shape[0], : arr.shape[1]]
    return reconstructed.astype(np.float32, copy=False), {
        "method": "wavelet_2d",
        "wavelet": wavelet,
        "levels": actual_levels,
        "threshold": resolved_threshold,
        "threshold_mode": threshold_mode,
        "threshold_strategy": strategy,
        **threshold_metadata,
    }


def method_wavelet_svd(
    data: np.ndarray,
    wavelet: str = "db4",
    levels: int = 3,
    threshold: float = 0.1,
    rank_start: int = 2,
    rank_end: int = 40,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
    threshold_mode: str = "soft",
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    del kwargs
    pywt = _require_pywt()
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if threshold_mode not in {"soft", "hard"}:
        threshold_mode = "soft"
    max_level = pywt.dwtn_max_level(arr.shape, wavelet)
    actual_levels = max(1, min(int(levels), max_level if max_level > 0 else 1))
    resolved_threshold = float(np.clip(threshold, 0.0, 1.0))
    strategy = _resolve_threshold_strategy(threshold_strategy)
    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=actual_levels)
    approximation, svd_meta = method_svd_subspace_native(
        coeffs[0],
        {
            "rank_start": int(rank_start),
            "rank_end": min(int(rank_end), min(coeffs[0].shape)),
            "solver": "exact",
        },
    )
    detail_only = [np.asarray(approximation, dtype=np.float64), *coeffs[1:]]
    filtered, threshold_metadata = _threshold_details(
        detail_only,
        arr=arr,
        threshold=resolved_threshold,
        threshold_strategy=strategy,
        threshold_mode=threshold_mode,
        pywt=pywt,
    )
    reconstructed = pywt.waverec2(filtered, wavelet)[: arr.shape[0], : arr.shape[1]]
    return reconstructed.astype(np.float32, copy=False), {
        "method": "wavelet_svd",
        "wavelet": wavelet,
        "levels": actual_levels,
        "threshold": resolved_threshold,
        "rank_start": int(rank_start),
        "rank_end": int(rank_end),
        "threshold_mode": threshold_mode,
        "threshold_strategy": strategy,
        "svd_solver": svd_meta.get("solver", "exact"),
        **threshold_metadata,
    }


__all__ = [
    "DEFAULT_THRESHOLD_STRATEGY",
    "LEGACY_THRESHOLD_STRATEGY",
    "method_wavelet_2d",
    "method_wavelet_svd",
]

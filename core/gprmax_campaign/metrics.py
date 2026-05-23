#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard metrics for synthetic paired gprMax arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_paired_metrics(
    raw: np.ndarray,
    background: np.ndarray,
    target_response: np.ndarray,
    roi: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute standardized paired metrics with safe zero-denominator handling."""
    warnings: list[dict[str, str]] = []
    _validate_2d_shape_compatibility(raw, background, target_response)

    raw = np.asarray(raw, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    target_response = np.asarray(target_response, dtype=np.float64)

    raw_energy = float(np.sum(np.square(raw)))
    background_energy = float(np.sum(np.square(background)))
    target_energy = float(np.sum(np.square(target_response)))
    mean_abs_target = float(np.mean(np.abs(target_response)))
    max_abs_target = float(np.max(np.abs(target_response)))

    mae = float(np.mean(np.abs(raw - background)))
    mse = float(np.mean(np.square(raw - background)))
    rmse = float(np.sqrt(mse))
    psnr = _compute_psnr(raw, background, mse, warnings)

    metrics: dict[str, Any] = {
        "raw_shape": list(raw.shape),
        "background_shape": list(background.shape),
        "target_response_shape": list(target_response.shape),
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "background_min": float(np.min(background)),
        "background_max": float(np.max(background)),
        "background_mean": float(np.mean(background)),
        "background_std": float(np.std(background)),
        "target_response_min": float(np.min(target_response)),
        "target_response_max": float(np.max(target_response)),
        "target_response_mean": float(np.mean(target_response)),
        "target_response_std": float(np.std(target_response)),
        "raw_energy": raw_energy,
        "background_energy": background_energy,
        "target_response_energy": target_energy,
        "target_to_background_energy_ratio": _safe_ratio(
            target_energy,
            background_energy,
            "target_to_background_energy_ratio",
            warnings,
        ),
        "target_to_raw_energy_ratio": _safe_ratio(
            target_energy,
            raw_energy,
            "target_to_raw_energy_ratio",
            warnings,
        ),
        "mean_abs_target_response": mean_abs_target,
        "max_abs_target_response": max_abs_target,
        "raw_background_mae": mae,
        "raw_background_mse": mse,
        "raw_background_rmse": rmse,
        "raw_background_psnr": psnr,
        # Higher value suggests response is more concentrated/spiky.
        "sparsity_or_concentration_proxy": _safe_ratio(
            max_abs_target,
            mean_abs_target,
            "sparsity_or_concentration_proxy",
            warnings,
        ),
        # Backward-compatible keys used by prior reports/tests.
        "abs_difference_mean": mae,
        "abs_difference_max": max_abs_target,
    }

    roi_ratio = _compute_roi_energy_ratio(target_response, roi, warnings)
    metrics["roi_energy_ratio"] = roi_ratio
    metrics["warnings"] = warnings
    return metrics


def _validate_2d_shape_compatibility(
    raw: np.ndarray,
    background: np.ndarray,
    target_response: np.ndarray,
) -> None:
    if raw.ndim != 2 or background.ndim != 2 or target_response.ndim != 2:
        raise ValueError("raw/background/target_response must all be 2D arrays")
    if raw.shape != background.shape or raw.shape != target_response.shape:
        raise ValueError(
            "raw/background/target_response must share the same shape: "
            f"raw={raw.shape}, background={background.shape}, "
            f"target_response={target_response.shape}"
        )


def _compute_psnr(
    raw: np.ndarray,
    background: np.ndarray,
    mse: float,
    warnings: list[dict[str, str]],
) -> float | None:
    if mse == 0.0:
        warnings.append(
            {
                "code": "raw_background_psnr_mse_zero",
                "message": "raw/background MSE is zero; PSNR is mathematically infinite.",
            }
        )
        return None
    peak = float(max(np.max(np.abs(raw)), np.max(np.abs(background))))
    if peak == 0.0:
        warnings.append(
            {
                "code": "raw_background_psnr_peak_zero",
                "message": "raw/background peak is zero; PSNR is undefined.",
            }
        )
        return None
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))


def _safe_ratio(
    numerator: float,
    denominator: float,
    code: str,
    warnings: list[dict[str, str]],
) -> float | None:
    if denominator == 0.0:
        warnings.append(
            {
                "code": f"{code}_denominator_zero",
                "message": f"{code} denominator is zero; returning null.",
            }
        )
        return None
    return float(numerator / denominator)


def _compute_roi_energy_ratio(
    target_response: np.ndarray,
    roi: dict[str, Any] | None,
    warnings: list[dict[str, str]],
) -> float | None:
    if roi is None:
        return None
    if not isinstance(roi, dict):
        warnings.append(
            {
                "code": "roi_invalid_type",
                "message": "ROI is not a dict; roi_energy_ratio set to null.",
            }
        )
        return None

    sample_range = roi.get("sample_range")
    trace_range = roi.get("trace_range")
    if not (
        isinstance(sample_range, list)
        and len(sample_range) == 2
        and isinstance(trace_range, list)
        and len(trace_range) == 2
    ):
        warnings.append(
            {
                "code": "roi_missing_ranges",
                "message": "ROI must contain sample_range and trace_range [start, end].",
            }
        )
        return None

    try:
        s0, s1 = int(sample_range[0]), int(sample_range[1])
        t0, t1 = int(trace_range[0]), int(trace_range[1])
    except Exception:
        warnings.append(
            {
                "code": "roi_non_integer_range",
                "message": "ROI ranges must be integer-compatible.",
            }
        )
        return None

    n_samples, n_traces = target_response.shape
    if not (0 <= s0 < s1 <= n_samples and 0 <= t0 < t1 <= n_traces):
        warnings.append(
            {
                "code": "roi_out_of_bounds",
                "message": (
                    "ROI range is out of bounds or empty: "
                    f"sample_range={sample_range}, trace_range={trace_range}, "
                    f"shape={target_response.shape}"
                ),
            }
        )
        return None

    roi_energy = float(np.sum(np.square(target_response[s0:s1, t0:t1])))
    total_energy = float(np.sum(np.square(target_response)))
    return _safe_ratio(roi_energy, total_energy, "roi_energy_ratio", warnings)

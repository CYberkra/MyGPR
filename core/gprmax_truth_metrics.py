#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground-truth-aware metrics for gprMax auto-tune validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.quality_metrics import ratio_fidelity, relative_reduction


EPS = 1.0e-9


def compute_ground_truth_metrics(
    reference: np.ndarray,
    processed: np.ndarray,
    ground_truth: dict[str, Any],
    *,
    reference_roi: dict[str, int] | None = None,
    processed_roi: dict[str, int] | None = None,
) -> dict[str, float]:
    """Compute scalar target/background metrics from a gprMax ground truth manifest.

    `reference_roi` and `processed_roi` describe the same physical analysis zone
    before and after a method. Their offset is used to shift target ROIs for
    methods such as zero-time correction that change row indices.
    """
    ref = _as_clean_2d(reference)
    proc = _as_clean_2d(processed)
    row_shift, col_shift = _roi_shift(reference_roi, processed_roi)
    ref_analysis_roi = _resolve_analysis_roi(ground_truth, ref.shape, reference_roi)
    proc_analysis_roi = _shift_roi(ref_analysis_roi, row_shift, col_shift)

    ref_target_rois = _target_rois(ground_truth, ref.shape)
    if not ref_target_rois:
        return _compute_no_target_metrics(
            ref,
            proc,
            ref_analysis_roi=ref_analysis_roi,
            proc_analysis_roi=proc_analysis_roi,
        )
    proc_target_rois = [
        _shift_roi(roi, row_shift, col_shift) for roi in ref_target_rois
    ]

    ref_target_mask = _mask_from_rois(ref.shape, ref_target_rois)
    proc_target_mask = _mask_from_rois(proc.shape, proc_target_rois)
    ref_analysis_mask = _mask_from_rois(ref.shape, [ref_analysis_roi])
    proc_analysis_mask = _mask_from_rois(proc.shape, [proc_analysis_roi])
    ref_background_mask = ref_analysis_mask & ~ref_target_mask
    proc_background_mask = proc_analysis_mask & ~proc_target_mask
    if not np.any(ref_background_mask):
        ref_background_mask = ~ref_target_mask
    if not np.any(proc_background_mask):
        proc_background_mask = ~proc_target_mask

    ref_target_energy = _masked_mean_square(ref, ref_target_mask)
    proc_target_energy = _masked_mean_square(proc, proc_target_mask)
    ref_background_energy = _masked_mean_square(ref, ref_background_mask)
    proc_background_energy = _masked_mean_square(proc, proc_background_mask)

    target_energy_preservation = _safe_ratio(proc_target_energy, ref_target_energy)
    background_reduction = relative_reduction(
        ref_background_energy,
        proc_background_energy,
    )
    target_saliency_before = _safe_ratio(
        _masked_mean_abs(ref, ref_target_mask),
        _masked_mean_abs(ref, ref_background_mask),
    )
    target_saliency_after = _safe_ratio(
        _masked_mean_abs(proc, proc_target_mask),
        _masked_mean_abs(proc, proc_background_mask),
    )
    target_saliency_gain = _safe_ratio(
        target_saliency_after,
        target_saliency_before,
    )
    false_positive_ratio = _safe_ratio(
        _masked_percentile_abs(proc, proc_background_mask, 99.0),
        _masked_percentile_abs(proc, proc_target_mask, 99.0),
    )
    target_contrast_after = target_saliency_after
    truth_score = _truth_score(
        target_energy_preservation=target_energy_preservation,
        target_saliency_gain=target_saliency_gain,
        background_reduction=background_reduction,
        false_positive_ratio=false_positive_ratio,
        target_contrast_after=target_contrast_after,
    )

    return {
        "truth_target_energy_preservation": float(target_energy_preservation),
        "truth_target_saliency_before": float(target_saliency_before),
        "truth_target_saliency_after": float(target_saliency_after),
        "truth_target_saliency_gain": float(target_saliency_gain),
        "truth_background_energy_reduction": float(background_reduction),
        "truth_false_positive_ratio": float(false_positive_ratio),
        "truth_target_contrast_after": float(target_contrast_after),
        "truth_score": float(truth_score),
        "truth_target_count": float(len(ref_target_rois)),
    }


def _compute_no_target_metrics(
    reference: np.ndarray,
    processed: np.ndarray,
    *,
    ref_analysis_roi: dict[str, int],
    proc_analysis_roi: dict[str, int],
) -> dict[str, float]:
    ref_mask = _mask_from_rois(reference.shape, [ref_analysis_roi])
    proc_mask = _mask_from_rois(processed.shape, [proc_analysis_roi])
    ref_energy = _masked_mean_square(reference, ref_mask)
    proc_energy = _masked_mean_square(processed, proc_mask)
    background_reduction = relative_reduction(ref_energy, proc_energy)
    false_positive_ratio = _safe_ratio(
        _masked_percentile_abs(processed, proc_mask, 99.0),
        _masked_percentile_abs(reference, ref_mask, 99.0),
    )
    truth_score = _no_target_truth_score(
        background_reduction=background_reduction,
        false_positive_ratio=false_positive_ratio,
    )
    return {
        "truth_target_energy_preservation": 1.0,
        "truth_target_saliency_before": 0.0,
        "truth_target_saliency_after": 0.0,
        "truth_target_saliency_gain": 1.0,
        "truth_background_energy_reduction": float(background_reduction),
        "truth_false_positive_ratio": float(false_positive_ratio),
        "truth_target_contrast_after": 0.0,
        "truth_score": float(truth_score),
        "truth_target_count": 0.0,
    }


def _as_clean_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"ground-truth metrics require 2D data, got shape={arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_analysis_roi(
    ground_truth: dict[str, Any],
    shape: tuple[int, int],
    fallback_roi: dict[str, int] | None,
) -> dict[str, int]:
    source = ground_truth.get("analysis_roi")
    if not isinstance(source, dict):
        source = fallback_roi
    if not isinstance(source, dict):
        source = {
            "time_start_idx": 0,
            "time_end_idx": int(shape[0]),
            "dist_start_idx": 0,
            "dist_end_idx": int(shape[1]),
        }
    return _clamp_roi(source, shape)


def _target_rois(
    ground_truth: dict[str, Any],
    shape: tuple[int, int],
) -> list[dict[str, int]]:
    rois: list[dict[str, int]] = []
    for target in ground_truth.get("targets", []) or []:
        if not isinstance(target, dict):
            continue
        if target.get("must_preserve") is False:
            continue
        roi = target.get("roi")
        if isinstance(roi, dict):
            rois.append(_clamp_roi(roi, shape))
    return rois


def _roi_shift(
    reference_roi: dict[str, int] | None,
    processed_roi: dict[str, int] | None,
) -> tuple[int, int]:
    if not isinstance(reference_roi, dict) or not isinstance(processed_roi, dict):
        return 0, 0
    return (
        int(processed_roi.get("time_start_idx", 0))
        - int(reference_roi.get("time_start_idx", 0)),
        int(processed_roi.get("dist_start_idx", 0))
        - int(reference_roi.get("dist_start_idx", 0)),
    )


def _shift_roi(roi: dict[str, int], row_shift: int, col_shift: int) -> dict[str, int]:
    return {
        "time_start_idx": int(roi.get("time_start_idx", 0)) + int(row_shift),
        "time_end_idx": int(roi.get("time_end_idx", 0)) + int(row_shift),
        "dist_start_idx": int(roi.get("dist_start_idx", 0)) + int(col_shift),
        "dist_end_idx": int(roi.get("dist_end_idx", 0)) + int(col_shift),
    }


def _clamp_roi(roi: dict[str, Any], shape: tuple[int, int]) -> dict[str, int]:
    samples, traces = int(shape[0]), int(shape[1])
    t0 = max(0, min(int(roi.get("time_start_idx", 0)), max(samples - 1, 0)))
    t1 = max(t0 + 1, min(int(roi.get("time_end_idx", samples)), samples))
    d0 = max(0, min(int(roi.get("dist_start_idx", 0)), max(traces - 1, 0)))
    d1 = max(d0 + 1, min(int(roi.get("dist_end_idx", traces)), traces))
    return {
        "time_start_idx": int(t0),
        "time_end_idx": int(t1),
        "dist_start_idx": int(d0),
        "dist_end_idx": int(d1),
    }


def _mask_from_rois(shape: tuple[int, int], rois: list[dict[str, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for roi in rois:
        clamped = _clamp_roi(roi, shape)
        mask[
            clamped["time_start_idx"] : clamped["time_end_idx"],
            clamped["dist_start_idx"] : clamped["dist_end_idx"],
        ] = True
    return mask


def _masked_values(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != data.shape or not np.any(mask):
        return np.asarray([], dtype=np.float64)
    values = np.asarray(data, dtype=np.float64)[mask]
    return values[np.isfinite(values)]


def _masked_mean_square(data: np.ndarray, mask: np.ndarray) -> float:
    values = _masked_values(data, mask)
    if values.size == 0:
        return 0.0
    return float(np.mean(values**2))


def _masked_mean_abs(data: np.ndarray, mask: np.ndarray) -> float:
    values = _masked_values(data, mask)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values)))


def _masked_percentile_abs(data: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    values = _masked_values(data, mask)
    if values.size == 0:
        return 0.0
    return float(np.percentile(np.abs(values), percentile))


def _safe_ratio(numerator: float, denominator: float) -> float:
    den = float(denominator)
    num = float(numerator)
    if abs(den) <= EPS:
        return 1.0 if abs(num) <= EPS else float(num / EPS)
    return float(num / den)


def _truth_score(
    *,
    target_energy_preservation: float,
    target_saliency_gain: float,
    background_reduction: float,
    false_positive_ratio: float,
    target_contrast_after: float,
) -> float:
    target_fidelity = ratio_fidelity(
        float(target_energy_preservation),
        target=1.0,
        tol=0.6,
    )
    saliency_term = float(np.clip(np.log1p(max(0.0, target_saliency_gain)) / np.log(3.0), 0.0, 1.5))
    background_term = float(np.clip(background_reduction, -1.0, 1.0))
    contrast_term = float(np.clip(np.log1p(max(0.0, target_contrast_after)) / np.log(6.0), 0.0, 1.5))
    false_positive_penalty = float(np.clip(false_positive_ratio, 0.0, 3.0))
    return float(
        1.8 * target_fidelity
        + 1.1 * saliency_term
        + 1.2 * background_term
        + 0.8 * contrast_term
        - 1.0 * false_positive_penalty
    )


def _no_target_truth_score(
    *,
    background_reduction: float,
    false_positive_ratio: float,
) -> float:
    background_term = float(np.clip(background_reduction, -1.0, 1.0))
    false_positive_penalty = float(np.clip(false_positive_ratio - 1.0, 0.0, 8.0))
    calm_background_bonus = ratio_fidelity(
        max(float(false_positive_ratio), EPS),
        target=0.25,
        tol=1.0,
    )
    return float(
        1.8 * background_term
        + 1.0 * calm_background_bonus
        - 1.4 * false_positive_penalty
    )


__all__ = ["compute_ground_truth_metrics"]

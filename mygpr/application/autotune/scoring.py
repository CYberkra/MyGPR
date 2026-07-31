#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Method-family quality scorers for auto-tune trials."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mygpr.domain.autotune.quality_metrics import (
    baseline_bias,
    clipping_ratio,
    deep_zone_contrast,
    depth_rms_cv,
    edge_preservation,
    first_break_sharpness,
    first_break_std,
    horizontal_coherence,
    hot_pixel_ratio,
    kurtosis_or_spikiness,
    local_saliency_preservation,
    low_freq_energy_ratio,
    pre_zero_energy_ratio,
    ratio_fidelity,
    relative_reduction,
    target_band_energy_ratio,
)
from mygpr.domain.common.scalars import to_int
from mygpr.application.autotune.candidate_generators import _agc_window_min, _resolve_time_step_ns
from mygpr.application.autotune.utils import _safe_ratio
from mygpr.domain.autotune.models import TrialScore


def _score_zero_time(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    zero_idx = int(params.get("_zero_idx", params.get("_backup_samples", 0)))
    detector = str(params.get("_detector", "threshold"))
    threshold = float(params.get("_threshold", 0.05) or 0.05)
    before_pre = pre_zero_energy_ratio(before, zero_idx)
    after_pre = pre_zero_energy_ratio(after, zero_idx)
    after_std = first_break_std(
        after,
        method=detector if detector != "manual" else "threshold",
        threshold=max(threshold, 0.03),
    )
    sharpness = first_break_sharpness(after, max(1, zero_idx))
    sharp_norm = sharpness / max(float(np.mean(np.abs(after))), 1.0e-6)
    std_norm = after_std / max(float(before.shape[0]), 1.0)

    penalties = {
        "pre_zero_regression": max(0.0, after_pre - before_pre) * 4.0,
        "large_shift": max(
            0.0,
            params.get("new_zero_time", 0.0)
            - _resolve_time_step_ns(before.shape[0], header_info)
            * before.shape[0]
            * 0.2,
        )
        / max(
            _resolve_time_step_ns(before.shape[0], header_info) * before.shape[0], 1.0
        ),
    }
    score = (
        -3.2 * after_pre - 1.8 * std_norm + 1.6 * sharp_norm - sum(penalties.values())
    )
    metrics = {
        "pre_zero_energy_ratio": float(after_pre),
        "first_break_std": float(after_std),
        "first_break_sharpness": float(sharp_norm),
    }
    reason = (
        f"零时前能量={after_pre:.4f}，首波离散度={after_std:.2f}，锐度={sharp_norm:.3f}；"
        f"检测={detector}，回退样本={zero_idx}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_drift(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    baseline_before = baseline_bias(before)
    baseline_after = baseline_bias(after)
    baseline_drop = relative_reduction(baseline_before, baseline_after)

    low_freq_before = low_freq_energy_ratio(before)
    low_freq_after = low_freq_energy_ratio(after)
    low_freq_drop = relative_reduction(low_freq_before, low_freq_after)

    band_ratio_raw = target_band_energy_ratio(before, after)
    band_keep = float(np.clip(band_ratio_raw, 0.0, 1.25))
    band_fidelity = ratio_fidelity(band_ratio_raw, target=1.0, tol=0.18)
    peak_ratio_raw = _safe_ratio(
        float(np.percentile(np.abs(after), 99.0)),
        float(np.percentile(np.abs(before), 99.0)),
    )
    peak_ratio = float(np.clip(peak_ratio_raw, 0.0, 1.35))
    peak_fidelity = ratio_fidelity(peak_ratio_raw, target=1.0, tol=0.22)
    penalties = {
        "baseline_regression": max(0.0, -baseline_drop) * 2.5,
        "low_freq_regression": max(0.0, -low_freq_drop) * 3.0,
        "band_distortion": max(0.0, 0.72 - band_fidelity) * 2.5,
        "peak_distortion": max(0.0, 0.72 - peak_fidelity) * 1.5,
    }
    score = (
        2.4 * baseline_drop
        + 2.8 * low_freq_drop
        + 1.6 * band_fidelity
        + 0.6 * peak_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "baseline_bias_before": float(baseline_before),
        "baseline_bias_after": float(baseline_after),
        "baseline_drop": float(baseline_drop),
        "low_freq_energy_ratio_before": float(low_freq_before),
        "low_freq_energy_ratio_after": float(low_freq_after),
        "low_freq_drop": float(low_freq_drop),
        "target_band_energy_ratio": float(band_ratio_raw),
        "target_band_keep": float(band_keep),
        "target_band_fidelity": float(band_fidelity),
        "peak_ratio": float(peak_ratio),
        "peak_ratio_raw": float(peak_ratio_raw),
        "peak_fidelity": float(peak_fidelity),
    }
    reason = (
        f"基线改善={baseline_drop:.3f}，低频改善={low_freq_drop:.3f}，"
        f"目标频带保真={band_fidelity:.3f}，峰值保真={peak_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_background(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    coherence = horizontal_coherence(after)
    saliency = local_saliency_preservation(before, after)
    edge = edge_preservation(before, after)
    peak_ratio = float(
        np.percentile(np.abs(after), 99.0)
        / max(np.percentile(np.abs(before), 99.0), 1.0e-6)
    )
    penalties = {
        "edge_loss": max(0.0, 0.72 - edge) * 3.0,
        "target_drop": max(0.0, 0.60 - peak_ratio) * 2.5,
    }
    score = -3.0 * coherence + 2.2 * saliency + 1.2 * edge - sum(penalties.values())
    metrics = {
        "horizontal_coherence": float(coherence),
        "local_saliency_preservation": float(saliency),
        "edge_preservation": float(edge),
        "peak_ratio": float(peak_ratio),
    }
    reason = f"背景一致性={coherence:.4f}，显著结构保留={saliency:.3f}，边缘保留={edge:.3f}。"
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_fk_filter(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    coherence_before = horizontal_coherence(before)
    coherence_after = horizontal_coherence(after)
    coherence_drop = relative_reduction(coherence_before, coherence_after)
    saliency_ratio = local_saliency_preservation(before, after)
    saliency_fidelity = ratio_fidelity(saliency_ratio, target=1.0, tol=0.18)
    edge_ratio = edge_preservation(before, after)
    edge_fidelity = ratio_fidelity(edge_ratio, target=1.0, tol=0.18)
    band_ratio_raw = target_band_energy_ratio(before, after)
    band_keep = float(np.clip(band_ratio_raw, 0.0, 1.25))
    band_fidelity = ratio_fidelity(band_ratio_raw, target=1.0, tol=0.20)
    peak_ratio_raw = float(
        np.percentile(np.abs(after), 99.0)
        / max(np.percentile(np.abs(before), 99.0), 1.0e-6)
    )
    peak_fidelity = ratio_fidelity(peak_ratio_raw, target=1.0, tol=0.25)
    penalties = {
        "coherence_regression": max(0.0, -coherence_drop) * 2.6,
        "saliency_distortion": max(0.0, 0.72 - saliency_fidelity) * 2.2,
        "edge_distortion": max(0.0, 0.75 - edge_fidelity) * 2.2,
        "band_distortion": max(0.0, 0.72 - band_fidelity) * 2.8,
        "peak_distortion": max(0.0, 0.70 - peak_fidelity) * 1.8,
    }
    score = (
        2.5 * coherence_drop
        + 1.4 * saliency_fidelity
        + 1.3 * edge_fidelity
        + 1.8 * band_fidelity
        + 0.5 * peak_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "horizontal_coherence_before": float(coherence_before),
        "horizontal_coherence_after": float(coherence_after),
        "horizontal_coherence_drop": float(coherence_drop),
        "local_saliency_preservation": float(saliency_ratio),
        "local_saliency_fidelity": float(saliency_fidelity),
        "edge_preservation": float(edge_ratio),
        "edge_fidelity": float(edge_fidelity),
        "target_band_energy_ratio": float(band_ratio_raw),
        "target_band_keep": float(band_keep),
        "target_band_fidelity": float(band_fidelity),
        "peak_ratio": float(peak_ratio_raw),
        "peak_fidelity": float(peak_fidelity),
    }
    reason = (
        f"背景改善={coherence_drop:.3f}，显著结构保真={saliency_fidelity:.3f}，"
        f"边缘保真={edge_fidelity:.3f}，目标频带保真={band_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_frequency_filter(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    low_freq_before = low_freq_energy_ratio(before)
    low_freq_after = low_freq_energy_ratio(after)
    low_freq_drop = relative_reduction(low_freq_before, low_freq_after)
    hot_drop = relative_reduction(hot_pixel_ratio(before), hot_pixel_ratio(after))
    saliency_ratio = local_saliency_preservation(before, after)
    saliency_fidelity = ratio_fidelity(saliency_ratio, target=1.0, tol=0.18)
    edge_ratio = edge_preservation(before, after)
    edge_fidelity = ratio_fidelity(edge_ratio, target=1.0, tol=0.18)
    band_ratio_raw = target_band_energy_ratio(before, after)
    band_fidelity = ratio_fidelity(band_ratio_raw, target=1.0, tol=0.22)
    peak_ratio_raw = _safe_ratio(
        float(np.percentile(np.abs(after), 99.0)),
        float(np.percentile(np.abs(before), 99.0)),
    )
    peak_fidelity = ratio_fidelity(peak_ratio_raw, target=1.0, tol=0.28)
    penalties = {
        "low_freq_regression": max(0.0, -low_freq_drop) * 2.0,
        "saliency_distortion": max(0.0, 0.72 - saliency_fidelity) * 2.4,
        "edge_distortion": max(0.0, 0.72 - edge_fidelity) * 2.2,
        "band_distortion": max(0.0, 0.70 - band_fidelity) * 2.8,
        "peak_distortion": max(0.0, 0.68 - peak_fidelity) * 1.8,
    }
    score = (
        1.7 * max(0.0, low_freq_drop)
        + 0.8 * max(0.0, hot_drop)
        + 1.5 * saliency_fidelity
        + 1.2 * edge_fidelity
        + 1.8 * band_fidelity
        + 0.4 * peak_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "low_freq_energy_ratio_before": float(low_freq_before),
        "low_freq_energy_ratio_after": float(low_freq_after),
        "low_freq_drop": float(low_freq_drop),
        "hot_pixel_drop": float(hot_drop),
        "local_saliency_preservation": float(saliency_ratio),
        "local_saliency_fidelity": float(saliency_fidelity),
        "edge_preservation": float(edge_ratio),
        "edge_fidelity": float(edge_fidelity),
        "target_band_energy_ratio": float(band_ratio_raw),
        "target_band_fidelity": float(band_fidelity),
        "peak_ratio": float(peak_ratio_raw),
        "peak_fidelity": float(peak_fidelity),
    }
    reason = (
        f"低频残留改善={low_freq_drop:.3f}，目标频带保真={band_fidelity:.3f}，"
        f"显著结构保真={saliency_fidelity:.3f}，边缘保真={edge_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_gain(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    window = int(params.get("window", 0) or 0)
    sample_count = int(header_info.get("a_scan_length") or before.shape[0])
    recommended_min_window = _agc_window_min(sample_count, header_info)
    short_window_ratio = (
        max(0.0, recommended_min_window / max(float(window), 1.0) - 1.0)
        if window > 0
        else 0.0
    )
    rms_cv = depth_rms_cv(after)
    deep_before = deep_zone_contrast(before)
    deep_after = deep_zone_contrast(after)
    deep_gain_raw = _safe_ratio(deep_after, deep_before)
    deep_gain_effective = float(np.log1p(np.clip(deep_gain_raw, 0.0, 12.0)))
    clip = clipping_ratio(after)
    hot = hot_pixel_ratio(after)
    shallow_before = float(np.std(_zone_slice(before, 0.0, 0.2)))
    shallow_after = float(np.std(_zone_slice(after, 0.0, 0.2)))
    shallow_blow_raw = _safe_ratio(shallow_after, shallow_before)
    shallow_blow = float(np.clip(shallow_blow_raw, 0.0, 4.0))
    penalties = {
        "clipping": clip * 10.0,
        "hot_pixels": hot * 6.0,
        "shallow_blowup": max(0.0, shallow_blow - 2.3) * 1.0,
        "short_window": short_window_ratio * 2.5,
    }
    score = -2.0 * rms_cv + 2.6 * deep_gain_effective - sum(penalties.values())
    metrics = {
        "depth_rms_cv": float(rms_cv),
        "deep_zone_contrast": float(deep_after),
        "deep_gain_ratio": float(deep_gain_raw),
        "deep_gain_effective": float(deep_gain_effective),
        "clipping_ratio": float(clip),
        "hot_pixel_ratio": float(hot),
        "shallow_blow_ratio": float(shallow_blow_raw),
        "window": float(window),
        "recommended_min_window": float(recommended_min_window),
        "short_window_ratio": float(short_window_ratio),
    }
    reason = (
        f"深部对比提升={deep_gain_raw:.3f}，有效提升={deep_gain_effective:.3f}，"
        f"深浅均衡CV={rms_cv:.4f}，过曝比={clip:.4f}，"
        f"窗口下限={recommended_min_window}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_impulse(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    hot_before = hot_pixel_ratio(before)
    hot_after = hot_pixel_ratio(after)
    hot_drop = relative_reduction(hot_before, hot_after)
    spiky_before = kurtosis_or_spikiness(before)
    spiky_after = kurtosis_or_spikiness(after)
    spiky_drop = relative_reduction(spiky_before, spiky_after)
    edge_ratio = edge_preservation(before, after)
    edge_fidelity = ratio_fidelity(edge_ratio, target=1.0, tol=0.18)
    penalties = {
        "hot_regression": max(0.0, -hot_drop) * 3.0,
        "spiky_regression": max(0.0, -spiky_drop) * 2.5,
        "edge_distortion": max(0.0, 0.75 - edge_fidelity) * 2.4,
    }
    score = (
        2.6 * hot_drop
        + 1.9 * spiky_drop
        + 1.5 * edge_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "hot_pixel_ratio_before": float(hot_before),
        "hot_pixel_ratio_after": float(hot_after),
        "hot_pixel_drop": float(hot_drop),
        "spikiness_before": float(spiky_before),
        "spikiness_after": float(spiky_after),
        "spikiness_drop": float(spiky_drop),
        "edge_preservation": float(edge_ratio),
        "edge_fidelity": float(edge_fidelity),
    }
    reason = (
        f"热点改善={hot_drop:.3f}，尖峰改善={spiky_drop:.3f}，"
        f"边缘保真={edge_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_denoise(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    hot_before = hot_pixel_ratio(before)
    hot_after = hot_pixel_ratio(after)
    hot_drop = relative_reduction(hot_before, hot_after)
    spiky_before = kurtosis_or_spikiness(before)
    spiky_after = kurtosis_or_spikiness(after)
    spiky_drop = relative_reduction(spiky_before, spiky_after)
    edge_ratio = edge_preservation(before, after)
    edge_fidelity = ratio_fidelity(edge_ratio, target=1.0, tol=0.18)
    saliency_ratio = local_saliency_preservation(before, after)
    saliency_fidelity = ratio_fidelity(saliency_ratio, target=1.0, tol=0.18)
    band_keep_raw = target_band_energy_ratio(before, after)
    band_keep = float(np.clip(band_keep_raw, 0.0, 1.25))
    band_fidelity = ratio_fidelity(band_keep_raw, target=1.0, tol=0.20)
    penalties = {
        "hot_regression": max(0.0, -hot_drop) * 2.5,
        "spiky_regression": max(0.0, -spiky_drop) * 2.5,
        "edge_distortion": max(0.0, 0.72 - edge_fidelity) * 2.4,
        "saliency_distortion": max(0.0, 0.72 - saliency_fidelity) * 2.4,
        "band_distortion": max(0.0, 0.72 - band_fidelity) * 2.4,
    }
    score = (
        2.2 * hot_drop
        + 1.8 * spiky_drop
        + 1.5 * saliency_fidelity
        + 1.2 * edge_fidelity
        + 1.1 * band_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "hot_pixel_ratio_before": float(hot_before),
        "hot_pixel_ratio_after": float(hot_after),
        "hot_pixel_drop": float(hot_drop),
        "spikiness_before": float(spiky_before),
        "spikiness_after": float(spiky_after),
        "spikiness_drop": float(spiky_drop),
        "edge_preservation": float(edge_ratio),
        "edge_fidelity": float(edge_fidelity),
        "local_saliency_preservation": float(saliency_ratio),
        "local_saliency_fidelity": float(saliency_fidelity),
        "target_band_energy_ratio": float(band_keep_raw),
        "target_band_keep": float(band_keep),
        "target_band_fidelity": float(band_fidelity),
    }
    reason = (
        f"热点改善={hot_drop:.3f}，尖峰改善={spiky_drop:.3f}，边缘保真={edge_fidelity:.3f}，"
        f"显著结构保真={saliency_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _zone_slice(data: np.ndarray, start_ratio: float, end_ratio: float) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    start = int(np.floor(arr.shape[0] * start_ratio))
    end = int(np.ceil(arr.shape[0] * end_ratio))
    return arr[max(0, start) : max(start + 1, min(arr.shape[0], end)), :]


def _slice_depth_band(
    data: np.ndarray, start_ratio: float, end_ratio: float
) -> np.ndarray:
    """Slice a depth band using normalized start/end ratios."""
    return _zone_slice(data, start_ratio, end_ratio)


def _slice_bounds(data: np.ndarray, bounds: dict[str, int]) -> np.ndarray:
    """Slice a 2D array with validated ROI/context bounds."""
    arr = np.asarray(data, dtype=np.float64)
    t0 = max(
        0,
        min(to_int(bounds.get("time_start_idx"), default=0), arr.shape[0] - 1),
    )
    t1 = max(
        t0 + 1,
        min(to_int(bounds.get("time_end_idx"), default=arr.shape[0]), arr.shape[0]),
    )
    d0 = max(
        0,
        min(to_int(bounds.get("dist_start_idx"), default=0), arr.shape[1] - 1),
    )
    d1 = max(
        d0 + 1,
        min(to_int(bounds.get("dist_end_idx"), default=arr.shape[1]), arr.shape[1]),
    )
    return arr[t0:t1, d0:d1]


def _score_motion_comp(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    """Score motion compensation height normalization.

    Rewards:
    - Reduced depth RMS CV (more consistent trace amplitudes)
    - Improved horizontal coherence
    Penalizes:
    - Clipping artifacts
    - Regression in amplitude consistency
    """
    from mygpr.domain.autotune.quality_metrics import depth_rms_cv, horizontal_coherence, clipping_ratio

    rms_cv_before = depth_rms_cv(before)
    rms_cv_after = depth_rms_cv(after)
    rms_cv_drop = relative_reduction(rms_cv_before, rms_cv_after)

    coh_before = horizontal_coherence(before)
    coh_after = horizontal_coherence(after)
    coh_gain = _safe_ratio(coh_after, coh_before) - 1.0

    clip = clipping_ratio(after)

    penalties = {
        "clipping": clip * 8.0,
        "rms_cv_regression": max(0.0, -rms_cv_drop) * 4.0,
    }

    score = 2.5 * rms_cv_drop + 1.8 * max(0.0, coh_gain) - sum(penalties.values())

    metrics = {
        "depth_rms_cv_before": float(rms_cv_before),
        "depth_rms_cv_after": float(rms_cv_after),
        "depth_rms_cv_drop": float(rms_cv_drop),
        "horizontal_coherence_before": float(coh_before),
        "horizontal_coherence_after": float(coh_after),
        "coherence_gain": float(coh_gain),
        "clipping_ratio": float(clip),
    }

    reason = (
        f"深浅均衡CV改善={rms_cv_drop:.3f}，横向相干增益={coh_gain:.3f}，"
        f"过曝比={clip:.4f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


_SCORE_FUNCTIONS: dict[
    str, Callable[[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]], TrialScore]
] = {
    "zero_time": _score_zero_time,
    "drift": _score_drift,
    "background": _score_background,
    "fk": _score_fk_filter,
    "frequency": _score_frequency_filter,
    "denoise": _score_denoise,
    "gain": _score_gain,
    "impulse": _score_impulse,
    "motion_comp": _score_motion_comp,
}

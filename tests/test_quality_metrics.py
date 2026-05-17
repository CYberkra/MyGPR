#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""质量指标方向性测试。"""

from __future__ import annotations

import numpy as np

from PythonModule.dewow import method_dewow
from core.processing_engine import _apply_subtracting_average_2d
from core.quality_metrics import (
    baseline_bias,
    compute_benchmark_metrics,
    deep_zone_contrast,
    detect_first_break_indices,
    estimate_lateral_correlation_length,
    estimate_singular_elbow_rank,
    first_break_std,
    clipping_ratio,
    horizontal_coherence,
    hot_pixel_ratio,
    local_saliency_preservation,
    low_freq_energy_ratio,
    pre_zero_energy_ratio,
    ratio_fidelity,
    relative_reduction,
    target_band_energy_ratio,
    detect_ridge_indices,
    extract_roi_and_context,
    periodic_banding_ratio,
)


def _build_test_profile(samples: int = 128, traces: int = 32) -> np.ndarray:
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    data = 0.35 * np.sin(2.0 * np.pi * 2.0 * t)
    data = np.repeat(data, traces, axis=1)
    data += 0.04 * rng.normal(size=(samples, traces))

    first_break = 20 + (np.sin(np.linspace(0.0, 2.0 * np.pi, traces)) * 2.0).astype(int)
    for col, idx in enumerate(first_break):
        data[idx : idx + 3, col] += np.array([1.0, 2.3, 1.2])

    data[58:61, :] += 0.55
    data[88:91, 10:22] += np.array([[0.15], [0.35], [0.18]])
    return data.astype(np.float32)


def test_dewow_reduces_baseline_and_low_frequency_energy():
    raw = _build_test_profile()
    filtered, _ = method_dewow(raw, window=32)

    assert baseline_bias(filtered) < baseline_bias(raw)
    assert low_freq_energy_ratio(filtered) < low_freq_energy_ratio(raw)
    assert target_band_energy_ratio(raw, filtered) > 0.35


def test_background_suppression_reduces_horizontal_coherence_but_keeps_saliency():
    raw = _build_test_profile()
    filtered, _ = _apply_subtracting_average_2d(raw, ntraces=31)

    assert horizontal_coherence(filtered) < horizontal_coherence(raw)
    assert local_saliency_preservation(raw, filtered) > 0.2


def test_gain_related_metrics_detect_deep_contrast_improvement():
    raw = _build_test_profile()
    gain_curve = np.linspace(1.0, 6.0, raw.shape[0], dtype=np.float32)[:, None]
    gained = raw * gain_curve

    assert deep_zone_contrast(gained) > deep_zone_contrast(raw)
    assert pre_zero_energy_ratio(gained, 5) >= 0.0


def test_zero_index_metrics_handle_non_finite_zero_idx():
    raw = _build_test_profile()

    assert pre_zero_energy_ratio(raw, np.nan) == 0.0
    assert np.isfinite(compute_benchmark_metrics(raw, raw, zero_idx=np.inf)["first_break_sharpness_after"])


def test_target_band_energy_ratio_handles_non_finite_band_values():
    raw = _build_test_profile()

    ratio = target_band_energy_ratio(raw, raw, band=(np.nan, np.inf))

    assert np.isfinite(ratio)
    assert ratio > 0.0
    assert np.isfinite(target_band_energy_ratio(raw, raw, band=(0.2,)))


def test_low_freq_energy_ratio_handles_non_finite_cutoff_ratio():
    raw = _build_test_profile()

    assert np.isfinite(low_freq_energy_ratio(raw, cutoff_ratio=np.nan))
    assert np.isfinite(low_freq_energy_ratio(raw, cutoff_ratio=np.inf))


def test_relative_reduction_rewards_lower_after_value():
    assert relative_reduction(10.0, 6.0) > 0.0
    assert relative_reduction(10.0, 10.0) == 0.0
    assert relative_reduction(10.0, 12.0) < 0.0


def test_relative_reduction_and_ratio_fidelity_do_not_return_nan_for_bad_scalars():
    assert np.isfinite(relative_reduction(np.nan, 1.0))
    assert np.isfinite(relative_reduction(10.0, np.inf))
    assert np.isfinite(ratio_fidelity(np.nan))
    assert np.isfinite(ratio_fidelity(1.0, target=np.inf, tol=np.nan))


def test_ratio_fidelity_peaks_near_unity_and_penalizes_overshoot():
    center = ratio_fidelity(1.0)
    mild = ratio_fidelity(1.1)
    overshoot = ratio_fidelity(1.4)

    assert center >= mild >= overshoot


def test_compute_benchmark_metrics_exposes_stable_metric_schema():
    raw = _build_test_profile()
    filtered, _ = method_dewow(raw, window=32)

    metrics = compute_benchmark_metrics(raw, filtered, zero_idx=20)

    assert metrics["baseline_bias_after"] <= metrics["baseline_bias_before"]
    assert (
        metrics["low_freq_energy_ratio_after"]
        <= metrics["low_freq_energy_ratio_before"]
    )
    assert "target_band_energy_ratio" in metrics
    assert "local_saliency_preservation" in metrics
    assert "clipping_ratio_after" in metrics
    assert "first_break_sharpness_after" in metrics


def test_first_break_metrics_handle_non_finite_controls():
    raw = _build_test_profile()

    indices = detect_first_break_indices(
        raw,
        threshold=np.nan,
        search_ratio=np.inf,
    )

    assert indices.shape == (raw.shape[1],)
    assert np.isfinite(first_break_std(raw, threshold=np.nan))


def test_lateral_correlation_length_handles_non_finite_max_lag():
    raw = _build_test_profile()

    assert estimate_lateral_correlation_length(raw, max_lag=np.nan) >= 1
    assert estimate_lateral_correlation_length(raw, max_lag=np.inf) >= 1


def test_singular_elbow_rank_handles_non_finite_max_rank():
    raw = _build_test_profile()

    assert estimate_singular_elbow_rank(raw, max_rank=np.nan) >= 1
    assert estimate_singular_elbow_rank(raw, max_rank=np.inf) >= 1


def test_ridge_detection_handles_non_finite_row_range():
    raw = _build_test_profile()

    detected = detect_ridge_indices(raw, row_range=(np.nan, np.inf))

    assert detected.shape == (raw.shape[1],)
    assert np.isfinite(detected).all()


def test_extract_roi_and_context_handles_non_finite_bounds():
    raw = _build_test_profile()

    result = extract_roi_and_context(
        raw,
        roi_bounds={
            "time_start_idx": np.nan,
            "time_end_idx": np.inf,
            "dist_start_idx": np.nan,
            "dist_end_idx": np.inf,
        },
    )

    assert result["roi_data"].shape == raw.shape


def test_gain_quality_metrics_handle_non_finite_control_values():
    raw = _build_test_profile()

    assert np.isfinite(deep_zone_contrast(raw, deep_ratio=np.nan))
    assert np.isfinite(clipping_ratio(raw, high_quantile=np.nan))
    assert np.isfinite(hot_pixel_ratio(raw, z=np.nan))


def test_periodic_banding_ratio_handles_non_finite_trace_band():
    raw = _build_test_profile()

    ratio = periodic_banding_ratio(raw, trace_band=(np.nan, np.inf))

    assert np.isfinite(ratio)
    assert ratio >= 0.0
    assert np.isfinite(periodic_banding_ratio(raw, trace_band=(0.08,)))

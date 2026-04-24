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
    horizontal_coherence,
    local_saliency_preservation,
    low_freq_energy_ratio,
    pre_zero_energy_ratio,
    ratio_fidelity,
    relative_reduction,
    target_band_energy_ratio,
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


def test_relative_reduction_rewards_lower_after_value():
    assert relative_reduction(10.0, 6.0) > 0.0
    assert relative_reduction(10.0, 10.0) == 0.0
    assert relative_reduction(10.0, 12.0) < 0.0


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

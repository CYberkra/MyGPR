#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round-2 ndarray processing kernel regression tests."""

from __future__ import annotations

import numpy as np
import pywt

from PythonModule.dewow import method_dewow
from PythonModule.hankel_svd import method_hankel_svd
from PythonModule.rpca_background import method_rpca_background
from PythonModule.sec_gain import method_sec_gain
from PythonModule.set_zero_time import method_set_zero_time
from PythonModule.wavelet_2d import method_wavelet_2d
from PythonModule.wavelet_svd import method_wavelet_svd
from PythonModule.wnnm_placeholder import method_wnnm_placeholder


def test_method_dewow_window_one_returns_zero_like_current_behavior():
    raw = np.arange(24, dtype=np.float32).reshape(6, 4)

    result, meta = method_dewow(raw, window=1)

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)
    assert meta["window"] == 1


def test_method_set_zero_time_shifts_up_and_zero_fills_tail():
    raw = np.arange(20, dtype=np.float32).reshape(5, 4)

    result, meta = method_set_zero_time(raw, new_zero_time=20.0, time_step_s=10e-9)

    expected = np.zeros_like(raw)
    expected[:-2, :] = raw[2:, :]
    assert np.array_equal(result, expected)
    assert meta["shift_samples"] == 2
    assert meta["new_zero_time"] == 20.0
    assert meta["time_step_s"] == 10e-9


def test_method_sec_gain_returns_metadata_dict_and_curve():
    raw = np.ones((4, 3), dtype=np.float32)

    result, meta = method_sec_gain(raw, gain_min=1.0, gain_max=4.0, power=1.0)

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert isinstance(meta, dict)
    assert meta["method"] == "sec_gain"
    assert meta["gain_min"] == 1.0
    assert meta["gain_max"] == 4.0
    assert meta["power"] == 1.0
    assert meta["gain_curve"].shape == (4,)
    assert np.allclose(result[:, 0], meta["gain_curve"])


def test_method_hankel_svd_keeps_contract_and_ignores_legacy_batch_kwarg():
    raw = np.arange(30, dtype=np.float32).reshape(10, 3)

    result, meta = method_hankel_svd(
        raw,
        window_length=4,
        rank=2,
        batch_size=8,
    )

    assert result.shape == raw.shape
    assert isinstance(meta, dict)
    assert meta["method"] == "hankel_svd"
    assert meta["window_length"] == 4
    assert meta["rank_requested"] == 2
    assert meta["rank_mode"] == "fixed"
    assert meta["effective_rank_min"] == 2
    assert meta["effective_rank_max"] == 2
    assert meta["svd_backend"] in {"truncated", "full", "mixed"}
    assert meta["fallback_columns"] >= 0


def test_method_rpca_background_separates_low_rank_component_contract():
    rows, cols = 18, 12
    low_rank = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None] @ np.ones(
        (1, cols), dtype=np.float32
    )
    sparse = np.zeros((rows, cols), dtype=np.float32)
    sparse[4, 3] = 2.5
    sparse[11, 8] = -1.7
    raw = low_rank + sparse

    result, meta = method_rpca_background(raw, lam=0.2, mu=0.8, max_iter=80, tol=1e-5)

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert isinstance(meta, dict)
    assert meta["method"] == "rpca_background"
    assert meta["iterations"] >= 1
    assert meta["sparse_ratio"] > 0.0
    assert abs(float(result[4, 3])) > 0.5
    assert abs(float(result[11, 8])) > 0.5


def test_method_rpca_background_treats_zero_mu_as_auto_init():
    rows, cols = 16, 10
    low_rank = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None] @ np.ones(
        (1, cols), dtype=np.float32
    )
    sparse = np.zeros((rows, cols), dtype=np.float32)
    sparse[6, 4] = 1.8
    raw = low_rank + sparse

    _, meta = method_rpca_background(raw, lam=0.15, mu=0.0, max_iter=60, tol=1e-5)

    assert meta["mu"] > 1e-6


def test_method_wavelet_2d_keeps_contract_and_reduces_impulse_noise_energy():
    rng = np.random.default_rng(10)
    rows, cols = 32, 24
    base = np.sin(np.linspace(0.0, 4.0 * np.pi, rows, dtype=np.float32))[:, None]
    raw = np.repeat(base, cols, axis=1)
    raw = raw + 0.05 * rng.standard_normal(size=raw.shape).astype(np.float32)
    raw[8, 4] += 3.0
    raw[21, 17] -= 2.5
    expected_levels = max(1, min(2, pywt.dwtn_max_level(raw.shape, "db4")))

    result, meta = method_wavelet_2d(raw, levels=2, threshold=0.12)

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert isinstance(meta, dict)
    assert meta["method"] == "wavelet_2d"
    assert meta["wavelet"] == "db4"
    assert meta["levels"] == expected_levels
    assert meta["threshold"] == 0.12
    assert abs(float(result[8, 4])) < abs(float(raw[8, 4]))
    assert abs(float(result[21, 17])) < abs(float(raw[21, 17]))


def test_method_wavelet_svd_keeps_contract_and_reduces_impulse_noise_energy():
    rng = np.random.default_rng(11)
    rows, cols = 32, 24
    base = np.sin(np.linspace(0.0, 4.0 * np.pi, rows, dtype=np.float32))[:, None]
    raw = np.repeat(base, cols, axis=1)
    raw = raw + 0.05 * rng.standard_normal(size=raw.shape).astype(np.float32)
    raw[10, 6] += 3.0
    raw[25, 19] -= 2.2
    expected_levels = max(1, min(2, pywt.dwtn_max_level(raw.shape, "db4")))

    result, meta = method_wavelet_svd(
        raw,
        levels=2,
        threshold=0.08,
        rank_start=1,
        rank_end=6,
    )

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert isinstance(meta, dict)
    assert meta["method"] == "wavelet_svd"
    assert meta["wavelet"] == "db4"
    assert meta["levels"] == expected_levels
    assert meta["threshold"] == 0.08
    assert meta["rank_start"] == 1
    assert meta["rank_end"] == 6
    assert abs(float(result[10, 6])) < abs(float(raw[10, 6]))
    assert abs(float(result[25, 19])) < abs(float(raw[25, 19]))


def test_method_wavelet_2d_uses_mad_universal_strategy_by_default():
    rng = np.random.default_rng(0)
    raw = rng.normal(0.0, 1.0, size=(64, 48)).astype(np.float32)

    _, meta = method_wavelet_2d(raw, levels=2, threshold=0.12)

    assert meta["threshold_strategy"] == "mad_universal"
    assert isinstance(meta["estimated_sigma"], (int, float))
    estimated_sigma = float(meta["estimated_sigma"])
    detail_thresholds = meta["detail_thresholds"]
    assert estimated_sigma > 0.0
    assert isinstance(detail_thresholds, list)
    assert len(detail_thresholds) == meta["levels"]
    assert all(float(item["abs_threshold"]) > 0.0 for item in detail_thresholds)


def test_method_wavelet_2d_supports_legacy_global_threshold_fallback():
    rng = np.random.default_rng(1)
    raw = rng.normal(0.0, 1.0, size=(64, 48)).astype(np.float32)

    _, meta = method_wavelet_2d(
        raw,
        levels=2,
        threshold=0.12,
        threshold_strategy="global_fraction",
    )

    assert meta["threshold_strategy"] == "global_fraction"
    assert isinstance(meta["global_abs_threshold"], (int, float))
    assert float(meta["global_abs_threshold"]) > 0.0
    assert "detail_thresholds" not in meta


def test_method_wavelet_svd_uses_mad_universal_strategy_by_default():
    rng = np.random.default_rng(2)
    raw = rng.normal(0.0, 1.0, size=(64, 48)).astype(np.float32)

    _, meta = method_wavelet_svd(raw, levels=2, threshold=0.08, rank_start=1, rank_end=6)

    assert meta["threshold_strategy"] == "mad_universal"
    assert isinstance(meta["estimated_sigma"], (int, float))
    estimated_sigma = float(meta["estimated_sigma"])
    detail_thresholds = meta["detail_thresholds"]
    assert estimated_sigma > 0.0
    assert isinstance(detail_thresholds, list)
    assert len(detail_thresholds) == meta["levels"]
    assert all(float(item["abs_threshold"]) > 0.0 for item in detail_thresholds)


def test_method_wavelet_svd_supports_legacy_global_threshold_fallback():
    rng = np.random.default_rng(3)
    raw = rng.normal(0.0, 1.0, size=(64, 48)).astype(np.float32)

    _, meta = method_wavelet_svd(
        raw,
        levels=2,
        threshold=0.08,
        rank_start=1,
        rank_end=6,
        threshold_strategy="global_fraction",
    )

    assert meta["threshold_strategy"] == "global_fraction"
    assert isinstance(meta["global_abs_threshold"], (int, float))
    assert float(meta["global_abs_threshold"]) > 0.0
    assert "detail_thresholds" not in meta


def test_method_wnnm_placeholder_stays_identity_while_deferred():
    raw = np.arange(30, dtype=np.float32).reshape(10, 3)

    result = method_wnnm_placeholder(raw, weight=0.25)

    assert np.array_equal(result, raw)

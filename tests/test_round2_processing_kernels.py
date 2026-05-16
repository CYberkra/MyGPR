#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round-2 ndarray processing kernel regression tests."""

from __future__ import annotations

import numpy as np
import pywt

from core.gprpy_compat import (
    apply_gprpy_agc_gain,
    apply_gprpy_dewow,
    apply_gprpy_rem_mean_trace,
)
from core.processing_engine import (
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from PythonModule.dewow import method_dewow
from PythonModule.amplitude_scale import method_amplitude_scale
from PythonModule.equidistant_trace_resample import method_equidistant_trace_resample
from PythonModule.energy_decay_gain import method_energy_decay_gain
from PythonModule.hankel_svd import method_hankel_svd
from PythonModule.hilbert_envelope import method_hilbert_envelope
from PythonModule.rpca_background import method_rpca_background
from PythonModule.sec_gain import method_sec_gain
from PythonModule.set_zero_time import method_set_zero_time
from PythonModule.svd_background import method_svd_background
from PythonModule.time_cut import method_time_cut
from PythonModule.trace_qc import method_trace_qc
from PythonModule.wavelet_2d import method_wavelet_2d
from PythonModule.wavelet_svd import method_wavelet_svd
from PythonModule.wnnm_placeholder import method_wnnm_placeholder


def test_method_dewow_window_one_matches_gprpy_baseline():
    raw = np.arange(24, dtype=np.float32).reshape(6, 4)

    result, meta = method_dewow(raw, window=1)
    expected = apply_gprpy_dewow(raw, 1)

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert np.allclose(result, expected)
    assert meta["window"] == 1


def test_gprpy_baseline_helpers_match_processing_engine_kernels():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(10, 6)).astype(np.float32)

    dewow_expected = apply_gprpy_dewow(raw, 5)
    mean_trace_expected = apply_gprpy_rem_mean_trace(raw, 3)
    agc_expected = apply_gprpy_agc_gain(raw, 5)

    assert dewow_expected.shape == raw.shape
    assert mean_trace_expected.shape == raw.shape
    assert agc_expected.shape == raw.shape
    assert np.isfinite(dewow_expected).all()
    assert np.isfinite(mean_trace_expected).all()
    assert np.isfinite(agc_expected).all()


def test_method_set_zero_time_shifts_up_and_zero_fills_tail():
    raw = np.arange(20, dtype=np.float32).reshape(5, 4)

    result, meta = method_set_zero_time(raw, new_zero_time=20.0, time_step_s=10e-9)

    expected = np.zeros_like(raw)
    expected[:-2, :] = raw[2:, :]
    assert np.array_equal(result, expected)
    assert meta["shift_samples"] == 2
    assert meta["new_zero_time"] == 20.0
    assert meta["time_step_s"] == 10e-9


def test_method_time_cut_removes_below_selected_time():
    raw = np.arange(40, dtype=np.float32).reshape(10, 4)

    result, meta = method_time_cut(
        raw,
        mode="remove_below",
        time_end_ns=40.0,
        time_window_ns=100.0,
    )

    assert np.array_equal(result, raw[:4, :])
    assert meta["time_start_idx"] == 0
    assert meta["time_end_idx"] == 4
    assert meta["output_samples"] == 4
    assert meta["header_info_updates"]["total_time_ns"] == 40.0


def test_method_time_cut_keeps_middle_range():
    raw = np.arange(40, dtype=np.float32).reshape(10, 4)

    result, meta = method_time_cut(
        raw,
        mode="keep_range",
        time_start_ns=20.0,
        time_end_ns=60.0,
        time_window_ns=100.0,
    )

    assert np.array_equal(result, raw[2:6, :])
    assert meta["time_start_idx"] == 2
    assert meta["time_end_idx"] == 6
    assert meta["header_info_updates"]["time_cut_offset_ns"] == 20.0


def test_method_time_cut_accepts_numpy_scalar_time_params():
    raw = np.arange(40, dtype=np.float32).reshape(10, 4)

    result, meta = method_time_cut(
        raw,
        mode="keep_range",
        time_start_ns=np.array([20.0]),
        time_end_ns=np.array([60.0]),
        time_window_ns=np.array([100.0]),
    )

    assert np.array_equal(result, raw[2:6, :])
    assert meta["time_start_idx"] == 2
    assert meta["time_end_idx"] == 6
    assert meta["time_start_ns"] == 20.0
    assert meta["time_end_ns"] == 60.0


def test_time_cut_runtime_params_use_header_total_time_ns():
    raw = np.arange(40, dtype=np.float32).reshape(10, 4)
    params = prepare_runtime_params(
        "time_cut",
        {"mode": "remove_below", "time_end_ns": 40.0},
        {"total_time_ns": 100.0},
        None,
        raw.shape,
    )

    result, meta = run_processing_method(raw, "time_cut", params)

    assert np.array_equal(result, raw[:4, :])
    assert meta["time_end_idx"] == 4
    assert meta["header_info_updates"]["total_time_ns"] == 40.0


def test_trace_qc_default_marks_without_changing_data():
    raw = np.ones((6, 5), dtype=np.float32)
    raw[:, 2] = 0.0

    result, meta = method_trace_qc(raw, empty_rms_threshold=0.1)

    assert np.array_equal(result, raw)
    assert meta["mode"] == "mark"
    assert meta["bad_trace_indices"].tolist() == [2]
    assert meta["trace_metadata_updates"]["trace_qc_bad_mask"].tolist() == [
        0,
        0,
        1,
        0,
        0,
    ]


def test_trace_qc_accepts_numpy_scalar_thresholds():
    raw = np.ones((6, 4), dtype=np.float32)
    raw[:, 1] = 0.0

    result, meta = method_trace_qc(
        raw,
        empty_rms_threshold=np.array([0.1]),
        spike_zscore=np.array([3.0]),
    )

    assert np.array_equal(result, raw)
    assert meta["empty_rms_threshold"] == 0.1
    assert meta["spike_zscore"] == 3.0
    assert meta["bad_trace_indices"].tolist() == [1]


def test_trace_qc_mute_and_remove_modes():
    raw = np.ones((4, 5), dtype=np.float32)
    raw[:, 3] = 100.0

    muted, muted_meta = method_trace_qc(raw, mode="mute", spike_zscore=2.0)
    assert np.allclose(muted[:, 3], 0.0)
    assert muted_meta["bad_trace_indices"].tolist() == [3]

    removed, removed_meta = method_trace_qc(
        raw,
        mode="remove",
        manual_trace_indices="1,3",
        trace_metadata={"trace_index": np.arange(5, dtype=np.int32)},
    )
    assert removed.shape == (4, 3)
    assert removed_meta["bad_trace_indices"].tolist() == [1, 3]
    assert removed_meta["trace_metadata_out"]["trace_index"].tolist() == [0, 2, 4]


def test_trace_qc_remove_filters_runtime_trace_metadata():
    raw = np.ones((4, 5), dtype=np.float32)
    trace_metadata = {"trace_index": np.arange(5, dtype=np.int32)}
    params = prepare_runtime_params(
        "trace_qc",
        {"mode": "remove", "manual_trace_indices": "0,4"},
        None,
        trace_metadata,
        raw.shape,
    )

    result, meta = run_processing_method(raw, "trace_qc", params)
    merged = merge_result_trace_metadata(trace_metadata, meta)

    assert result.shape == (4, 3)
    assert merged["trace_index"].tolist() == [1, 2, 3]


def test_equidistant_trace_resample_resamples_data_and_metadata():
    raw = np.vstack(
        [
            np.array([0.0, 2.0, 4.0], dtype=np.float32),
            np.array([10.0, 20.0, 40.0], dtype=np.float32),
        ]
    )
    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "trace_distance_m": np.array([0.0, 2.0, 4.0], dtype=np.float32),
        "local_x_m": np.array([0.0, 2.0, 4.0], dtype=np.float32),
    }

    result, meta = method_equidistant_trace_resample(
        raw,
        spacing_m=1.0,
        trace_metadata=trace_metadata,
    )

    assert result.shape == (2, 5)
    assert np.allclose(result[0], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(result[1], [10.0, 15.0, 20.0, 30.0, 40.0])
    assert np.allclose(
        meta["trace_metadata_out"]["trace_distance_m"],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    )
    assert meta["trace_metadata_out"]["trace_index"].tolist() == [0, 1, 2, 3, 4]


def test_equidistant_trace_resample_accepts_numpy_scalar_spacing():
    raw = np.array([[0.0, 2.0, 4.0]], dtype=np.float32)
    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "trace_distance_m": np.array([0.0, 2.0, 4.0], dtype=np.float32),
    }

    result, meta = method_equidistant_trace_resample(
        raw,
        spacing_m=np.array([1.0]),
        trace_metadata=trace_metadata,
    )

    assert result.shape == (1, 5)
    assert np.allclose(result[0], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert meta["spacing_m"] == 1.0


def test_equidistant_trace_resample_runtime_metadata_roundtrip():
    raw = np.ones((3, 3), dtype=np.float32)
    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "trace_distance_m": np.array([0.0, 1.5, 3.0], dtype=np.float32),
    }
    params = prepare_runtime_params(
        "equidistant_trace_resample",
        {"spacing_m": 1.0},
        None,
        trace_metadata,
        raw.shape,
    )

    result, meta = run_processing_method(raw, "equidistant_trace_resample", params)
    merged = merge_result_trace_metadata(trace_metadata, meta)

    assert result.shape == (3, 4)
    assert np.allclose(merged["trace_distance_m"], [0.0, 1.0, 2.0, 3.0])


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


def test_method_energy_decay_gain_amplifies_late_low_energy_rows():
    row_scale = np.linspace(1.0, 0.1, 12, dtype=np.float32)
    raw = np.repeat(row_scale[:, np.newaxis], 6, axis=1)

    result, meta = method_energy_decay_gain(
        raw,
        strength=1.0,
        smoothing_samples=1,
        max_gain=10.0,
    )

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert meta["method"] == "energy_decay_gain"
    assert meta["gain_curve"].shape == (12,)
    assert float(meta["gain_curve"][-1]) > float(meta["gain_curve"][0])
    assert float(result[-1, 0]) > float(raw[-1, 0])


def test_method_energy_decay_gain_uses_robust_trace_statistic():
    raw = np.ones((8, 5), dtype=np.float32)
    raw[4, 0] = 1000.0

    result, meta = method_energy_decay_gain(
        raw,
        strength=1.0,
        smoothing_samples=1,
        max_gain=10.0,
    )

    assert np.isfinite(result).all()
    assert float(meta["decay_curve"][4]) == 1.0
    assert float(meta["gain_curve"][4]) <= 1.1


def test_gain_methods_accept_numpy_scalar_parameters():
    raw = np.ones((8, 3), dtype=np.float32)

    sec_result, sec_meta = method_sec_gain(
        raw,
        gain_min=np.array([1.0]),
        gain_max=np.array([4.0]),
        power=np.array([1.0]),
    )
    decay_result, decay_meta = method_energy_decay_gain(
        raw,
        strength=np.array([1.0]),
        smoothing_samples=np.array([3]),
        min_gain=np.array([0.5]),
        max_gain=np.array([4.0]),
        floor_ratio=np.array([0.05]),
    )
    scale_result, scale_meta = method_amplitude_scale(
        raw,
        mode="constant",
        scale=np.array([2.0]),
    )

    assert sec_result.shape == raw.shape
    assert sec_meta["gain_max"] == 4.0
    assert decay_result.shape == raw.shape
    assert decay_meta["smoothing_samples"] == 3
    assert np.allclose(scale_result, raw * 2.0)
    assert scale_meta["effective_scale"] == 2.0


def test_method_amplitude_scale_constant_mode():
    raw = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)

    result, meta = method_amplitude_scale(raw, mode="constant", scale=2.5)

    assert np.allclose(result, raw * 2.5)
    assert meta["mode"] == "constant"
    assert meta["effective_scale"] == 2.5


def test_method_amplitude_scale_peak_and_rms_normalization():
    raw = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    peak_result, peak_meta = method_amplitude_scale(raw, mode="peak", target=1.0)
    rms_result, rms_meta = method_amplitude_scale(raw, mode="rms", target=1.0)

    assert np.isclose(np.max(np.abs(peak_result)), 1.0)
    assert np.isclose(np.sqrt(np.mean(rms_result.astype(np.float64) ** 2)), 1.0)
    assert peak_meta["effective_scale"] == 0.25
    assert rms_meta["mode"] == "rms"


def test_method_hilbert_envelope_returns_trace_envelope():
    samples = 128
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float64)
    raw = np.column_stack([np.cos(t), 2.0 * np.cos(t)]).astype(np.float32)

    result, meta = method_hilbert_envelope(raw)

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert np.allclose(result[:, 0], 1.0, atol=1e-5)
    assert np.allclose(result[:, 1], 2.0, atol=1e-5)
    assert meta["method"] == "hilbert_envelope"


def test_method_hilbert_envelope_normalize_and_log_compress():
    raw = np.array([[1.0, 2.0], [-1.0, -2.0]], dtype=np.float32)

    result, meta = method_hilbert_envelope(raw, normalize=True, log_compress=True)

    assert np.isfinite(result).all()
    assert float(np.max(result)) <= float(np.log1p(1.0) + 1e-6)
    assert meta["normalize"] is True
    assert meta["log_compress"] is True


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


def test_method_svd_background_removes_selected_low_rank_component():
    rows, cols = 18, 10
    background = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None] @ np.ones(
        (1, cols), dtype=np.float32
    )
    anomaly = np.zeros((rows, cols), dtype=np.float32)
    anomaly[8, 4] = 2.0
    raw = background + anomaly

    result, estimated_background = method_svd_background(raw, rank=1)

    assert result.shape == raw.shape
    assert estimated_background.shape == raw.shape
    assert np.linalg.norm(estimated_background) > 0.0
    assert np.linalg.norm(result) < np.linalg.norm(raw)


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

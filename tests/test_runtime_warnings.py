#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""处理引擎结构化告警测试。"""

from __future__ import annotations

import numpy as np

from core.gprpy_compat import apply_gprpy_agc_gain
from core.processing_engine import run_processing_method
from core.runtime_warnings import merge_runtime_warnings


def test_merge_runtime_warnings_accepts_generator_without_materializing_group():
    warnings = (
        {"code": "demo", "message": "warning", "details": {"idx": idx}}
        for idx in range(2)
    )

    merged = merge_runtime_warnings(warnings)

    assert [item["details"]["idx"] for item in merged] == [0, 1]


def test_agc_gain_emits_clamp_and_fallback_warning():
    data = np.ones((16, 4), dtype=np.float32)
    result, meta = run_processing_method(data, "agcGain", {"window": 99})

    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}
    assert result.shape == data.shape
    assert "parameter_clamped" in codes
    assert "global_gain_fallback" in codes


def test_agc_gain_default_matches_gprpy_window_norm():
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 1.0, size=(40, 6)).astype(np.float32)

    result, meta = run_processing_method(data, "agcGain", {"window": 5})

    expected = apply_gprpy_agc_gain(data, 5)
    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}

    assert np.allclose(result, expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    assert "agc_low_energy_gain_guard" not in codes


def test_agc_gain_limits_low_energy_noise_amplification():
    data = np.zeros((160, 12), dtype=np.float32)
    wave = np.sin(np.linspace(0.0, np.pi * 4.0, 20, dtype=np.float32))
    data[20:40, :] = wave[:, None]
    data[120:, ::2] = 1.0e-3
    data[120:, 1::2] = -1.0e-3

    result, meta = run_processing_method(
        data, "agcGain", {"window": 5, "_low_energy_guard": True}
    )

    signal_rms = float(np.sqrt(np.mean(result[20:40, :] ** 2)))
    tail_rms = float(np.sqrt(np.mean(result[120:, :] ** 2)))
    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}

    assert result.shape == data.shape
    assert tail_rms < signal_rms * 0.2
    assert "agc_low_energy_gain_guard" in codes


def test_agc_gain_warning_path_accepts_numpy_scalar_runtime_params():
    data = np.ones((16, 4), dtype=np.float32)

    result, meta = run_processing_method(
        data,
        "agcGain",
        {
            "window": np.array([1]),
            "_low_energy_guard": True,
            "time_step_s": np.array([1.0e-12]),
        },
    )

    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}
    assert result.shape == data.shape
    assert "agc_window_too_short" in codes


def test_compensating_gain_accepts_numpy_scalar_runtime_params():
    data = np.ones((4, 2), dtype=np.float32)

    result, meta = run_processing_method(
        data,
        "compensatingGain",
        {"gain_min": np.array([0.0]), "gain_max": np.array([6.0])},
    )

    assert result.shape == data.shape
    assert np.allclose(result[0], 1.0)
    assert result[-1, 0] > result[0, 0]
    assert meta["method_id"] == "compensatingGain"


def test_running_average_preserves_shape_and_emits_warning_when_window_too_large():
    data = np.arange(60, dtype=np.float32).reshape(10, 6)
    result, meta = run_processing_method(data, "running_average_2D", {"ntraces": 999})

    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}
    assert result.shape == data.shape
    assert "window_clamped" in codes


def test_normalize_result_emits_data_sanitized_warning():
    data = np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=np.float32)
    result, meta = run_processing_method(data, "dewow", {"window": 2})

    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}
    assert np.isfinite(result).all()
    assert "data_sanitized" in codes

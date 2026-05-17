#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the time-axis frequency filter."""

from __future__ import annotations

import numpy as np

from core.processing_engine import prepare_runtime_params, run_processing_method
from PythonModule.frequency_filter_1d import method_frequency_filter_1d


def _tone_profile() -> tuple[np.ndarray, float]:
    sample_rate_hz = 2.0e9
    samples = 512
    traces = 6
    t = np.arange(samples, dtype=np.float64) / sample_rate_hz
    low = np.sin(2.0 * np.pi * 50.0e6 * t)
    target = 0.7 * np.sin(2.0 * np.pi * 250.0e6 * t)
    profile = (low + target)[:, None] * np.linspace(0.9, 1.1, traces)[None, :]
    return profile.astype(np.float32), sample_rate_hz


def _fft_bin_energy(data: np.ndarray, sample_rate_hz: float, freq_hz: float) -> float:
    spec = np.abs(np.fft.rfft(data[:, 0])) ** 2
    freqs = np.fft.rfftfreq(data.shape[0], d=1.0 / sample_rate_hz)
    idx = int(np.argmin(np.abs(freqs - freq_hz)))
    return float(spec[idx])


def test_bandpass_suppresses_out_of_band_tone_and_preserves_target_band():
    raw, sample_rate_hz = _tone_profile()

    filtered, meta = method_frequency_filter_1d(
        raw,
        filter_type="bandpass",
        low_freq_mhz=180.0,
        high_freq_mhz=320.0,
        taper_ratio=0.04,
        sample_rate_hz=sample_rate_hz,
    )

    low_before = _fft_bin_energy(raw, sample_rate_hz, 50.0e6)
    low_after = _fft_bin_energy(filtered, sample_rate_hz, 50.0e6)
    target_before = _fft_bin_energy(raw, sample_rate_hz, 250.0e6)
    target_after = _fft_bin_energy(filtered, sample_rate_hz, 250.0e6)

    assert low_after < low_before * 0.05
    assert target_after > target_before * 0.80
    assert meta["effective_params"]["low_freq_mhz"] == 180.0
    assert meta["effective_params"]["high_freq_mhz"] == 320.0


def test_notch_suppresses_center_frequency_without_shape_change():
    raw, sample_rate_hz = _tone_profile()

    filtered, meta = method_frequency_filter_1d(
        raw,
        filter_type="notch",
        notch_freq_mhz=50.0,
        notch_width_mhz=30.0,
        notch_depth=1.0,
        sample_rate_hz=sample_rate_hz,
    )

    assert filtered.shape == raw.shape
    assert _fft_bin_energy(filtered, sample_rate_hz, 50.0e6) < (
        _fft_bin_energy(raw, sample_rate_hz, 50.0e6) * 0.10
    )
    assert meta["effective_params"]["notch_freq_mhz"] == 50.0


def test_processing_engine_injects_sampling_rate_from_header():
    raw, _sample_rate_hz = _tone_profile()
    header_info = {"total_time_ns": 256.0}
    runtime_params = prepare_runtime_params(
        "frequency_filter_1d",
        {"filter_type": "highpass", "low_freq_mhz": 120.0},
        header_info,
        {},
        raw.shape,
    )

    filtered, meta = run_processing_method(raw, "frequency_filter_1d", runtime_params)

    assert filtered.shape == raw.shape
    assert runtime_params["sample_rate_hz"] > 0.0
    assert meta["nyquist_mhz"] > 0.0


def test_missing_sampling_info_skips_filter_instead_of_guessing_frequency():
    raw, _sample_rate_hz = _tone_profile()

    filtered, meta = method_frequency_filter_1d(raw, filter_type="bandpass")

    assert np.array_equal(filtered, raw)
    assert meta["skipped"] is True
    assert meta["runtime_warnings"][0]["code"] == "frequency_sampling_missing"


def test_non_finite_sampling_info_skips_filter():
    raw, _sample_rate_hz = _tone_profile()

    filtered, meta = method_frequency_filter_1d(
        raw,
        filter_type="bandpass",
        sample_rate_hz=np.inf,
        time_step_s=np.nan,
        sample_interval_ns=np.inf,
    )

    assert np.array_equal(filtered, raw)
    assert meta["skipped"] is True
    assert meta["runtime_warnings"][0]["code"] == "frequency_sampling_missing"


def test_non_finite_frequency_params_are_sanitized():
    raw, sample_rate_hz = _tone_profile()

    filtered, meta = method_frequency_filter_1d(
        raw,
        filter_type="bandpass",
        low_freq_mhz=np.inf,
        high_freq_mhz=np.nan,
        taper_ratio=np.inf,
        sample_rate_hz=sample_rate_hz,
    )

    assert filtered.shape == raw.shape
    assert np.isfinite(filtered).all()
    assert meta["effective_params"]["skipped"] is True

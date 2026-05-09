#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for data-aware auto-tune parameter constraints."""

from __future__ import annotations

import numpy as np

from core.auto_tune import auto_tune_method


def _small_profile(samples: int = 96, traces: int = 36) -> np.ndarray:
    rng = np.random.default_rng(20260507)
    time = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, traces, dtype=np.float32)[None, :]
    direct = np.exp(-((time - 0.12) ** 2) / 0.002)
    target = 0.8 * np.exp(-((time - (0.38 + 0.12 * x**2)) ** 2) / 0.004)
    horizontal = 0.18 * np.sin(2 * np.pi * time * 4.0)
    noise = 0.03 * rng.standard_normal((samples, traces)).astype(np.float32)
    return (direct + target + horizontal + noise).astype(np.float32)


def _constraint_codes(result: dict) -> set[str]:
    return {str(item.get("code")) for item in result.get("constraint_warnings", [])}


def test_auto_tune_constrains_explicit_background_window_to_trace_count():
    raw = _small_profile(traces=36)

    result = auto_tune_method(
        raw,
        "subtracting_average_2D",
        candidate_params=[{"ntraces": 501}, {"ntraces": 21}],
        search_mode="fast",
    )

    assert result["best_params"]["ntraces"] <= raw.shape[1]
    assert result["execution_stats"]["constraint_adjustment_count"] >= 1
    assert "auto_tune_parameter_clamped" in _constraint_codes(result)
    assert any(
        trial.get("requested_params", {}).get("ntraces") == 501
        and trial.get("params", {}).get("ntraces") <= raw.shape[1]
        and trial.get("effective_params", {}).get("ntraces") <= raw.shape[1]
        and trial.get("constraint_warnings")
        for trial in result["all_trials"]
    )


def test_auto_tune_constrains_svd_rank_end_to_matrix_rank_limit():
    raw = _small_profile(samples=64, traces=12)

    result = auto_tune_method(
        raw,
        "svd_subspace",
        candidate_params=[{"rank_start": 1, "rank_end": 40}],
        search_mode="fast",
    )

    rank_limit = min(raw.shape)
    assert result["best_params"]["rank_end"] <= rank_limit
    assert result["best_params"]["rank_start"] <= result["best_params"]["rank_end"]
    assert result["execution_stats"]["constraint_adjustment_count"] >= 1
    assert result["all_trials"][0]["requested_params"]["rank_end"] == 40
    assert result["all_trials"][0]["effective_params"]["rank_end"] <= rank_limit


def test_auto_tune_constrains_zero_time_to_safe_search_window():
    raw = _small_profile(samples=100, traces=36)

    result = auto_tune_method(
        raw,
        "set_zero_time",
        candidate_params=[{"new_zero_time": 200.0}, {"new_zero_time": 4.0}],
        header_info={"total_time_ns": 50.0},
        search_mode="fast",
    )

    safe_max_ns = 50.0 * 0.35
    assert result["best_params"]["new_zero_time"] <= safe_max_ns
    assert result["execution_stats"]["constraint_adjustment_count"] >= 1
    assert any(
        trial.get("requested_params", {}).get("new_zero_time") == 200.0
        and trial.get("effective_params", {}).get("new_zero_time") <= safe_max_ns
        and trial.get("constraint_warnings")
        for trial in result["all_trials"]
    )


def test_auto_tune_constrains_agc_window_to_time_aware_minimum():
    raw = _small_profile(samples=200, traces=24)

    result = auto_tune_method(
        raw,
        "agcGain",
        candidate_params=[{"window": 7}],
        header_info={"total_time_ns": 1.0},
        search_mode="fast",
    )

    assert result["best_params"]["window"] >= 100
    assert result["execution_stats"]["constraint_adjustment_count"] >= 1
    assert any(
        trial.get("requested_params", {}).get("window") == 7
        and trial.get("effective_params", {}).get("window") >= 100
        and trial.get("constraint_warnings")
        for trial in result["all_trials"]
    )


def test_auto_tune_agc_generated_candidates_respect_time_aware_minimum():
    raw = _small_profile(samples=200, traces=24)

    result = auto_tune_method(
        raw,
        "agcGain",
        header_info={"total_time_ns": 1.0},
        search_mode="fast",
    )

    windows = [trial["params"]["window"] for trial in result["all_trials"]]
    assert min(windows) >= 100


def test_auto_tune_constrains_frequency_filter_to_nyquist_and_valid_band():
    raw = _small_profile(samples=128, traces=24)

    result = auto_tune_method(
        raw,
        "frequency_filter_1d",
        candidate_params=[
            {
                "filter_type": "bandpass",
                "low_freq_mhz": 900.0,
                "high_freq_mhz": 1200.0,
                "taper_ratio": 0.8,
            }
        ],
        header_info={"total_time_ns": 128.0},
        search_mode="fast",
    )

    best = result["best_params"]
    assert best["high_freq_mhz"] <= 500.0
    assert best["low_freq_mhz"] < best["high_freq_mhz"]
    assert best["taper_ratio"] <= 0.5
    assert result["execution_stats"]["constraint_adjustment_count"] >= 1
    assert result["all_trials"][0]["requested_params"]["high_freq_mhz"] == 1200.0
    assert result["all_trials"][0]["effective_params"]["high_freq_mhz"] <= 500.0

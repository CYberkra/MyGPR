#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contracts for AutoTune V1 bounded candidate generator."""

from __future__ import annotations

import numpy as np

from core.autotune_candidate_generator import (
    generate_autotune_v1_candidates,
    summarize_candidate_features,
)


def _demo_bscan(samples: int = 160, traces: int = 72) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    x = np.linspace(0.0, 1.0, traces)[None, :]
    layer = np.sin(2.0 * np.pi * (9.0 * t + 0.15 * x)) * np.exp(-2.0 * t)
    reflector = np.exp(-((t - (0.55 + 0.08 * np.sin(2 * np.pi * x))) ** 2) / 0.0015)
    anomaly = np.exp(-((t - 0.35) ** 2) / 0.002 - ((x - 0.55) ** 2) / 0.006)
    return layer + 0.45 * reflector + 0.8 * anomaly


def test_candidate_generator_builds_stable_bounded_space_with_hash():
    data = _demo_bscan()
    metadata = {
        "dt_ns": 0.08,
        "center_frequency": 400,  # MHz-style header value
        "trace_spacing_m": 0.05,
        "target_lateral_scale_m": 0.6,
    }
    result = generate_autotune_v1_candidates(data, metadata=metadata, target_goal="balanced")
    again = generate_autotune_v1_candidates(data, metadata=metadata, target_goal="balanced")

    assert result.profile_id == "balanced"
    assert result.candidate_space_hash == again.candidate_space_hash
    assert len(result.candidates) >= 20
    assert {candidate.category for candidate in result.candidates} >= {
        "background_suppression",
        "dewow",
        "bandpass",
        "gain",
        "denoise",
    }
    assert result.features.n_samples == data.shape[0]
    assert result.features.n_traces == data.shape[1]
    assert result.features.center_frequency_hz == 400_000_000.0


def test_landslide_profile_caps_svd_and_marks_agc_display_only():
    data = _demo_bscan()
    result = generate_autotune_v1_candidates(
        data,
        metadata={"dt_ns": 0.1, "center_frequency_hz": 250_000_000},
        target_goal="landslide_interface",
    )
    svd = [candidate for candidate in result.candidates if candidate.method == "svd_rank_sweep"]
    assert svd
    assert max(candidate.parameters["remove_rank"] for candidate in svd) <= 1
    assert any("interface_profiles_use_svd_rank_cap" == warning for warning in result.warnings)

    agc = [candidate for candidate in result.candidates if candidate.method == "agc"]
    assert agc
    assert all(candidate.display_only and not candidate.metric_safe for candidate in agc)
    assert any("agc_excluded_from_scoring" in candidate.warnings for candidate in agc)


def test_candidate_generator_can_exclude_display_only_for_synthetic_scoring():
    data = _demo_bscan()
    with_display = generate_autotune_v1_candidates(data, metadata={"dt_ns": 0.1}, target_goal="object_like_anomaly")
    no_display = generate_autotune_v1_candidates(
        data,
        metadata={"dt_ns": 0.1},
        target_goal="object_like_anomaly",
        include_display_only=False,
    )

    assert any(candidate.display_only for candidate in with_display.candidates)
    assert not any(candidate.display_only for candidate in no_display.candidates)
    assert with_display.candidate_space_hash != no_display.candidate_space_hash


def test_feature_summary_uses_metadata_when_no_data_is_available():
    features = summarize_candidate_features(
        None,
        metadata={
            "n_samples": 256,
            "n_traces": 96,
            "total_time_ns": 80.0,
            "center_frequency": 500,
        },
    )
    assert features.n_samples == 256
    assert features.n_traces == 96
    assert features.dt_seconds is not None
    assert features.center_frequency_hz == 500_000_000.0

    result = generate_autotune_v1_candidates(None, metadata=features.to_dict(), target_goal="deep_weak")
    assert result.profile_id == "deep_weak_reflector"
    assert any(candidate.category == "dewow" for candidate in result.candidates)

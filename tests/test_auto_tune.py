#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单步 auto-tune 回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from core.auto_tune import AutoTuneError, auto_select_method_group, auto_tune_method
from core.methods_registry import METHOD_METADATA, get_auto_tune_stage
from core.preset_profiles import RECOMMENDED_RUN_PROFILES, WORKFLOW_STAGES
from core.workflow_data import METHOD_CATEGORIES


EXPECTED_PUBLIC_DENOISE_METHODS = {
    "hankel_svd",
    "svd_subspace",
    "wavelet_2d",
    "wavelet_svd",
}


def _build_test_profile(samples: int = 128, traces: int = 32) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    data = 0.3 * np.sin(2.0 * np.pi * 1.8 * t)
    data = np.repeat(data, traces, axis=1)
    data += 0.03 * rng.normal(size=(samples, traces))

    first_break = 18 + (np.cos(np.linspace(0.0, 2.0 * np.pi, traces)) * 2.0).astype(int)
    for col, idx in enumerate(first_break):
        data[idx : idx + 3, col] += np.array([0.9, 2.2, 1.1])

    data[55:58, :] += 0.45
    data[90:93, 12:22] += np.array([[0.12], [0.28], [0.16]])
    return data.astype(np.float32)


def test_auto_tune_zero_time_returns_shift_parameter():
    raw = _build_test_profile()
    result = auto_tune_method(
        raw,
        "set_zero_time",
        header_info={"total_time_ns": float(raw.shape[0])},
        base_params={"new_zero_time": 5.0},
        search_mode="standard",
    )

    assert result["family"] == "zero_time"
    assert "new_zero_time" in result["best_params"]
    assert result["best_params"]["new_zero_time"] >= 0.0
    assert len(result["all_trials"]) >= 4
    assert result["recommended_profile"] == "balanced"
    assert set(result["profiles"].keys()) >= {
        "conservative",
        "balanced",
        "aggressive",
    }
    assert len(result["coarse_trials"]) >= 1
    assert all(trial["stage"] == "coarse" for trial in result["coarse_trials"])
    assert result["search_plan"]["coarse_budget"] >= 1
    assert result["execution_stats"]["coarse_trial_count"] == len(
        result["coarse_trials"]
    )
    assert result["execution_stats"]["fine_trial_count"] == len(result["fine_trials"])
    assert result["execution_stats"]["total_trial_count"] == len(result["all_trials"])


def test_auto_tune_dewow_returns_window_candidate():
    raw = _build_test_profile()
    result = auto_tune_method(raw, "dewow", base_params={"window": 23})

    assert result["family"] == "drift"
    assert 8 <= int(result["best_params"]["window"]) <= raw.shape[0] // 2
    assert len(result["all_trials"]) >= 3
    assert np.isfinite(result["best_score"])
    assert len(result["coarse_trials"]) >= 1
    assert len(result["fine_trials"]) >= 1


def test_auto_tune_sec_gain_returns_gain_parameters():
    raw = _build_test_profile()
    result = auto_tune_method(
        raw,
        "sec_gain",
        base_params={"gain_min": 1.0, "gain_max": 4.5, "power": 1.1},
    )

    assert result["family"] == "gain"
    assert "gain_max" in result["best_params"]
    assert "power" in result["best_params"]
    assert len(result["all_trials"]) >= 40
    assert result["profiles"]["balanced"]["params"]
    assert result["profiles"]["conservative"]["params"]
    assert result["profiles"]["aggressive"]["params"]


def test_auto_tune_background_scans_more_candidates():
    raw = _build_test_profile(traces=256)
    result = auto_tune_method(
        raw,
        "subtracting_average_2D",
        base_params={"ntraces": 51},
        roi_spec={
            "mode": "crop",
            "bounds": {
                "time_start_idx": 40,
                "time_end_idx": 110,
                "dist_start_idx": 24,
                "dist_end_idx": 220,
            },
            "label": "当前裁剪区",
        },
        search_mode="standard",
    )

    assert result["family"] == "background"
    assert len(result["all_trials"]) >= 10
    assert "ntraces" in result["best_params"]
    assert len(result["pareto_trials"]) >= 1
    assert len(result["coarse_trials"]) >= 1
    assert len(result["fine_trials"]) >= 1
    assert result["roi_info"]["source"] in {"crop", "auto", "full"}
    assert any(trial["stage"] == "fine" for trial in result["all_trials"])


def test_auto_tune_fk_filter_returns_angle_band_candidate():
    raw = _build_test_profile(traces=96)
    result = auto_tune_method(
        raw,
        "fk_filter",
        base_params={"angle_low": 12, "angle_high": 55, "taper_width": 4},
        search_mode="standard",
    )

    assert result["family"] == "fk"
    assert (
        0
        <= int(result["best_params"]["angle_low"])
        < int(result["best_params"]["angle_high"])
        <= 90
    )
    assert (
        int(result["best_params"]["angle_high"])
        - int(result["best_params"]["angle_low"])
        >= 8
    )
    assert result["best_params"]["taper_width"] >= 0
    assert len(result["coarse_trials"]) >= 1
    assert len(result["fine_trials"]) >= 1


def test_auto_tune_svd_subspace_returns_rank_interval_candidate():
    raw = _build_test_profile(traces=96)
    result = auto_tune_method(
        raw,
        "svd_subspace",
        base_params={"rank_start": 1, "rank_end": 20},
        search_mode="standard",
    )

    assert result["family"] == "denoise"
    assert result["best_params"]["rank_start"] >= 1
    assert result["best_params"]["rank_end"] >= result["best_params"]["rank_start"]
    assert len(result["coarse_trials"]) >= 1
    assert len(result["fine_trials"]) >= 1
    assert result["profiles"]["balanced"]["params"]


def test_auto_tune_wavelet_svd_returns_threshold_and_rank_candidate():
    raw = _build_test_profile(traces=96)
    result = auto_tune_method(
        raw,
        "wavelet_svd",
        base_params={
            "wavelet": "db4",
            "levels": 2,
            "threshold": 0.05,
            "rank_start": 1,
            "rank_end": 20,
        },
        search_mode="fast",
    )

    assert result["family"] == "denoise"
    assert result["best_params"]["wavelet"] == "db4"
    assert result["best_params"]["levels"] >= 1
    assert result["best_params"]["threshold"] > 0.0
    assert result["best_params"]["rank_end"] >= result["best_params"]["rank_start"]
    assert len(result["coarse_trials"]) >= 1


def test_public_denoise_scope_is_frozen_to_current_public_methods():
    public_denoise_methods = {
        method_key
        for method_key, meta in METHOD_METADATA.items()
        if meta.get("category") == "denoising" and meta.get("visibility") == "public"
    }

    assert public_denoise_methods == EXPECTED_PUBLIC_DENOISE_METHODS


def test_auto_tune_hankel_svd_uses_small_bounded_candidate_set():
    raw = _build_test_profile(traces=96)
    result = auto_tune_method(
        raw,
        "hankel_svd",
        base_params={"window_length": 48, "rank": 4},
        search_mode="fast",
    )

    assert result["family"] == "denoise"
    assert result["best_params"]["window_length"] >= 0
    assert result["best_params"]["rank"] >= 0
    assert len(result["coarse_trials"]) == 4
    assert result["fine_trials"] == []
    assert result["execution_stats"]["coarse_trial_count"] == 4
    assert result["execution_stats"]["fine_trial_count"] == 0
    observed = {
        (
            int(trial["params"]["window_length"]),
            int(trial["params"]["rank"]),
        )
        for trial in result["coarse_trials"]
    }
    assert observed == {(0, 0), (48, 0), (0, 4), (48, 4)}
    assert all("recovery_mode" not in trial["params"] for trial in result["all_trials"])


def test_auto_tune_wavelet_2d_returns_threshold_candidate():
    raw = _build_test_profile(traces=96)
    result = auto_tune_method(
        raw,
        "wavelet_2d",
        base_params={"wavelet": "db4", "levels": 2, "threshold": 0.1},
        search_mode="fast",
    )

    assert result["family"] == "denoise"
    assert result["best_params"]["wavelet"] == "db4"
    assert result["best_params"]["levels"] >= 1
    assert result["best_params"]["threshold"] > 0.0
    assert len(result["coarse_trials"]) >= 1


def test_auto_tune_manual_roi_uses_explicit_bounds():
    raw = _build_test_profile(traces=128)
    result = auto_tune_method(
        raw,
        "subtracting_average_2D",
        base_params={"ntraces": 51},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": 30,
                "time_end_idx": 96,
                "dist_start_idx": 8,
                "dist_end_idx": 90,
            },
            "label": "手动框选 ROI",
        },
    )

    assert result["roi_info"]["source"] == "manual"
    assert result["roi_info"]["label"] == "手动框选 ROI"
    assert any(trial.get("roi_used") for trial in result["all_trials"])


def test_auto_tune_zero_time_uses_family_specific_shallow_region_even_with_manual_roi():
    raw = _build_test_profile(traces=128)
    result = auto_tune_method(
        raw,
        "set_zero_time",
        header_info={"total_time_ns": float(raw.shape[0])},
        base_params={"new_zero_time": 5.0},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": 70,
                "time_end_idx": 110,
                "dist_start_idx": 10,
                "dist_end_idx": 90,
            },
            "label": "深部手动 ROI",
        },
    )

    assert result["roi_info"]["source"] == "manual"
    assert any(
        trial.get("roi_source") == "shallow_first_break"
        for trial in result["all_trials"]
    )


def test_auto_tune_gain_uses_deep_zone_scoring_even_with_manual_roi():
    raw = _build_test_profile(traces=96)
    result = auto_tune_method(
        raw,
        "sec_gain",
        base_params={"gain_min": 1.0, "gain_max": 4.5, "power": 1.1},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": 10,
                "time_end_idx": 30,
                "dist_start_idx": 8,
                "dist_end_idx": 40,
            },
            "label": "浅层手动 ROI",
        },
    )

    assert any(trial.get("roi_source") == "deep_zone" for trial in result["all_trials"])


def test_auto_tune_background_returns_to_single_roi_scoring_when_roi_exists():
    raw = _build_test_profile(traces=128)
    result = auto_tune_method(
        raw,
        "subtracting_average_2D",
        base_params={"ntraces": 51},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": 30,
                "time_end_idx": 80,
                "dist_start_idx": 12,
                "dist_end_idx": 90,
            },
            "label": "目标 ROI",
        },
    )

    assert any(trial.get("roi_used") for trial in result["all_trials"])
    assert all(
        trial.get("roi_source") == "manual"
        for trial in result["all_trials"]
        if trial.get("roi_used")
    )
    assert all(
        not trial.get("context_metrics")
        for trial in result["all_trials"]
        if trial.get("roi_used")
    )


def test_auto_tune_keeps_failed_trials_outside_best_path():
    raw = _build_test_profile()
    result = auto_tune_method(
        raw,
        "dewow",
        candidate_params=[
            {"window": "bad"},
            {"window": 23},
            {"window": 31},
        ],
    )

    assert result["failed_trials"]
    assert any(
        trial.get("params", {}).get("window") == "bad"
        for trial in result["failed_trials"]
    )
    assert all(not trial.get("valid", True) for trial in result["failed_trials"])
    assert all(trial.get("valid", True) for trial in result["pareto_trials"])
    assert result["best_params"]["window"] != "bad"
    assert all(
        trial.get("params", {}).get("window") != "bad"
        for trial in result["pareto_trials"]
    )
    assert all(
        profile.get("params", {}).get("window") != "bad"
        for profile in result["profiles"].values()
    )
    assert result["recommended_profile"] == "balanced"
    assert set(result["profiles"].keys()) >= {
        "conservative",
        "balanced",
        "aggressive",
    }
    assert result["selection_margin"] >= 0.0
    assert 0.0 <= result["selection_confidence"] <= 1.0
    assert result["execution_stats"]["failed_trial_count"] == len(
        result["failed_trials"]
    )
    assert result["execution_stats"]["valid_trial_count"] >= len(
        result["pareto_trials"]
    )


def test_auto_tune_raises_when_all_candidates_fail():
    raw = _build_test_profile()

    with pytest.raises(AutoTuneError, match="所有候选均未成功执行"):
        auto_tune_method(
            raw,
            "dewow",
            candidate_params=[{"window": "bad"}],
        )


def test_auto_tune_stage_metadata_is_available_for_comparable_methods():
    assert get_auto_tune_stage("subtracting_average_2D") == "background"
    assert get_auto_tune_stage("fk_filter") == "background"
    assert get_auto_tune_stage("hankel_svd") == "denoise"
    assert get_auto_tune_stage("svd_subspace") == "denoise"
    assert get_auto_tune_stage("wavelet_2d") == "denoise"
    assert get_auto_tune_stage("wavelet_svd") == "denoise"


def test_auto_select_method_group_returns_best_background_method():
    raw = _build_test_profile(traces=96)
    result = auto_select_method_group(
        raw,
        ["subtracting_average_2D", "median_background_2D", "fk_filter"],
        base_params_map={
            "subtracting_average_2D": {"ntraces": 51},
            "median_background_2D": {"ntraces": 51},
            "fk_filter": {"angle_low": 12, "angle_high": 55, "taper_width": 4},
        },
        roi_spec={
            "mode": "crop",
            "bounds": {
                "time_start_idx": 35,
                "time_end_idx": 95,
                "dist_start_idx": 10,
                "dist_end_idx": 80,
            },
            "label": "当前裁剪区",
        },
        search_mode="fast",
    )

    assert result["stage"] == "background"
    assert result["best_method_key"] in {
        "subtracting_average_2D",
        "median_background_2D",
        "fk_filter",
    }
    assert len(result["candidates"]) == 3
    assert result["best_params"]
    assert np.isfinite(result["outer_score"])


def test_auto_select_method_group_returns_best_public_denoise_method():
    raw = _build_test_profile(traces=96)
    result = auto_select_method_group(
        raw,
        sorted(EXPECTED_PUBLIC_DENOISE_METHODS),
        base_params_map={
            "hankel_svd": {"window_length": 48, "rank": 4},
            "svd_subspace": {"rank_start": 1, "rank_end": 20},
            "wavelet_2d": {"wavelet": "db4", "levels": 2, "threshold": 0.1},
            "wavelet_svd": {
                "wavelet": "db4",
                "levels": 2,
                "threshold": 0.05,
                "rank_start": 1,
                "rank_end": 20,
            },
        },
        search_mode="fast",
    )

    assert result["stage"] == "denoise"
    assert result["best_method_key"] in EXPECTED_PUBLIC_DENOISE_METHODS
    assert len(result["candidates"]) == len(EXPECTED_PUBLIC_DENOISE_METHODS)
    assert result["best_params"]
    assert np.isfinite(result["outer_score"])


def test_auto_select_method_group_rejects_mixed_stage_methods():
    raw = _build_test_profile(traces=64)
    with pytest.raises(AutoTuneError, match="同组方法必须属于同一 auto-tune stage"):
        auto_select_method_group(
            raw,
            ["subtracting_average_2D", "sec_gain"],
            base_params_map={
                "subtracting_average_2D": {"ntraces": 51},
                "sec_gain": {"gain_min": 1.0, "gain_max": 4.5, "power": 1.1},
            },
            search_mode="fast",
        )


def test_workflow_denoising_category_exposes_exact_public_denoise_methods():
    assert set(METHOD_CATEGORIES["denoising"]["methods"]) == EXPECTED_PUBLIC_DENOISE_METHODS


def test_stage3_workflow_methods_expose_exact_public_denoise_methods():
    stage3_methods = {
        method_key
        for method_key in WORKFLOW_STAGES["stage3"]["methods"].keys()
        if METHOD_METADATA.get(method_key, {}).get("category") == "denoising"
        and METHOD_METADATA.get(method_key, {}).get("visibility") == "public"
    }

    assert stage3_methods == EXPECTED_PUBLIC_DENOISE_METHODS


def test_recommended_profiles_cover_all_public_denoise_methods():
    exposed = set()
    for profile in RECOMMENDED_RUN_PROFILES.values():
        exposed.update(
            method_key
            for method_key in profile.get("order", [])
            if method_key in EXPECTED_PUBLIC_DENOISE_METHODS
        )

    assert exposed == EXPECTED_PUBLIC_DENOISE_METHODS

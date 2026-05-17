#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual-baseline vs auto-tune comparison backend tests."""

from __future__ import annotations

import numpy as np

import core.auto_tune_comparison as auto_tune_comparison
from core.auto_tune_comparison import (
    run_auto_tune_comparison,
    to_summary_dict,
)


def _build_drift_profile(samples: int = 96, traces: int = 28) -> np.ndarray:
    rng = np.random.default_rng(17)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    drift = 0.42 * np.sin(2.0 * np.pi * 0.65 * t)
    signal = 0.16 * np.sin(2.0 * np.pi * 10.0 * t)
    data = np.repeat(drift + signal, traces, axis=1)
    data += 0.035 * rng.normal(size=(samples, traces))

    ridge = 32 + (np.sin(np.linspace(0.0, 2.0 * np.pi, traces)) * 3.0).astype(int)
    for trace_idx, row in enumerate(ridge):
        data[row : row + 3, trace_idx] += np.array([0.35, 0.9, 0.35])
    data[62:66, 8:18] += 0.22 * np.hanning(4)[:, None]
    data += 0.06 * np.sin(2.0 * np.pi * 3.0 * x)
    return data.astype(np.float32)


def _truth_manifest() -> dict:
    return {
        "schema": "mygpr_gprmax_ground_truth_v1",
        "scenario_id": "comparison_truth_demo",
        "analysis_roi": {
            "time_start_idx": 22,
            "time_end_idx": 76,
            "dist_start_idx": 4,
            "dist_end_idx": 23,
        },
        "targets": [
            {
                "target_id": "ridge_01",
                "type": "hyperbola",
                "material": "metal",
                "depth_m": 0.42,
                "must_preserve": True,
                "roi": {
                    "time_start_idx": 29,
                    "time_end_idx": 40,
                    "dist_start_idx": 6,
                    "dist_end_idx": 22,
                },
            }
        ],
        "background_rois": [
            {
                "time_start_idx": 10,
                "time_end_idx": 20,
                "dist_start_idx": 3,
                "dist_end_idx": 9,
            }
        ],
        "source_paths": {
            "manifest_file": "synthetic_manifest.json",
            "ground_truth_file": "ground_truth.yaml",
        },
        "conversion_warnings": ["synthetic warning"],
    }


def test_comparison_uses_current_ui_params_as_manual_baseline():
    raw = _build_drift_profile()

    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": 24,
                "time_end_idx": 76,
                "dist_start_idx": 4,
                "dist_end_idx": 23,
            },
            "label": "目标区",
        },
        search_mode="fast",
    )

    assert result.manual.source == "current_ui_params"
    assert result.manual.pipeline == ["dewow"]
    assert result.automatic.pipeline == ["dewow"]
    assert result.roi_info["source"] == "manual"
    assert result.roi_info["label"] == "目标区"
    assert result.manual.params_by_method["dewow"] == {"window": 1}
    assert result.automatic.params_by_method["dewow"]["window"] != 1
    assert np.isfinite(result.manual.metrics["comparison_score"])
    assert np.isfinite(result.automatic.metrics["comparison_score"])
    assert result.metric_delta["comparison_score"] > 0.0
    assert result.verdict == "auto_better"


def test_comparison_manual_roi_accepts_none_and_numpy_scalar_bounds():
    raw = _build_drift_profile(samples=72, traces=18)

    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": None,
                "time_end_idx": np.array([60]),
                "dist_start_idx": np.int64(2),
                "dist_end_idx": np.array([16]),
            },
        },
        search_mode="fast",
    )

    assert result.roi_info["bounds"] == {
        "time_start_idx": 0,
        "time_end_idx": 60,
        "dist_start_idx": 2,
        "dist_end_idx": 16,
    }


def test_comparison_profile_fallback_uses_experience_baseline():
    raw = _build_drift_profile(samples=80, traces=24)

    result = run_auto_tune_comparison(
        raw,
        baseline_profile_key="uav_gpr_experience_baseline_v1",
        search_mode="fast",
    )

    assert result.manual.source == "experience_profile"
    assert result.baseline_profile_key == "uav_gpr_experience_baseline_v1"
    assert result.manual.pipeline == result.automatic.pipeline
    assert result.manual.pipeline[0] == "set_zero_time"
    assert "dewow" in result.manual.params_by_method
    assert set(result.automatic.auto_tune_results) >= {"set_zero_time", "dewow"}


def test_comparison_summary_is_json_safe_and_excludes_arrays():
    raw = _build_drift_profile(samples=72, traces=18)
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )

    summary = to_summary_dict(result)

    assert summary["manual"]["shape"] == list(result.manual.result.shape)
    assert summary["automatic"]["shape"] == list(result.automatic.result.shape)
    assert "result" not in summary["manual"]
    assert "result" not in summary["automatic"]
    assert isinstance(summary["automatic"]["params_by_method"]["dewow"]["window"], int)
    assert summary["metric_delta"]["comparison_score"] == result.metric_delta[
        "comparison_score"
    ]
    assert summary["ground_truth_info"]["enabled"] is False
    assert "parameter_domain" in summary["automatic"]["auto_tune_results"]["dewow"]
    assert "risk_flags" in summary["automatic"]["auto_tune_results"]["dewow"]


def test_comparison_uses_header_ground_truth_metrics_and_summary_info():
    raw = _build_drift_profile()
    result = run_auto_tune_comparison(
        raw,
        header_info={"ground_truth": _truth_manifest()},
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )

    assert "truth_score" in result.manual.metrics
    assert "truth_score" in result.automatic.metrics
    assert "truth_score" in result.metric_delta

    summary = to_summary_dict(result)
    info = summary["ground_truth_info"]
    assert info["enabled"] is True
    assert info["scenario_id"] == "comparison_truth_demo"
    assert info["target_count"] == 1
    assert info["has_background_rois"] is True
    assert info["targets"][0]["material"] == "metal"
    assert info["source_paths"]["ground_truth_file"] == "ground_truth.yaml"
    assert info["conversion_warnings"] == ["synthetic warning"]


def test_comparison_does_not_pass_ground_truth_into_auto_tune_search(monkeypatch):
    raw = _build_drift_profile(samples=72, traces=18)

    def fake_auto_tune_method(*args, **kwargs):
        header = kwargs.get("header_info") or {}
        assert "ground_truth" not in header
        return {
            "method_key": "dewow",
            "recommended_params": {"window": 5},
            "best_params": {"window": 5},
            "best_score": 1.0,
            "all_trials": [{"params": {"window": 5}, "score": 1.0}],
        }

    monkeypatch.setattr(
        auto_tune_comparison,
        "auto_tune_method",
        fake_auto_tune_method,
    )
    result = run_auto_tune_comparison(
        raw,
        header_info={"ground_truth": _truth_manifest()},
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )

    assert "truth_score" in result.automatic.metrics
    assert result.ground_truth_info["enabled"] is True


def test_comparison_summary_serializes_nonfinite_metrics_as_null():
    raw = _build_drift_profile(samples=72, traces=18)
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )
    result.manual.metrics["nan_metric"] = np.float64(np.nan)
    result.automatic.metrics["inf_metric"] = np.array([np.inf])

    summary = to_summary_dict(result)

    assert summary["manual"]["metrics"]["nan_metric"] is None
    assert summary["automatic"]["metrics"]["inf_metric"] == [None]

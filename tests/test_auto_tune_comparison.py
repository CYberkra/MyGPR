#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual-baseline vs auto-tune comparison backend tests."""

from __future__ import annotations

import json

import numpy as np

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
    assert "parameter_domain" in summary["automatic"]["auto_tune_results"]["dewow"]
    assert "risk_flags" in summary["automatic"]["auto_tune_results"]["dewow"]


def test_comparison_summary_sanitizes_nonfinite_scalar_values():
    raw = _build_drift_profile(samples=72, traces=18)
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )
    result.metric_delta["comparison_score"] = np.inf
    result.manual.metrics["comparison_score"] = np.nan
    result.automatic.params_by_method["dewow"]["bad"] = np.array([1.0, np.inf])

    summary = to_summary_dict(result)

    assert summary["metric_delta"]["comparison_score"] is None
    assert summary["manual"]["metrics"]["comparison_score"] is None
    assert summary["automatic"]["params_by_method"]["dewow"]["bad"] == [1.0, None]
    json.dumps(summary, allow_nan=False)

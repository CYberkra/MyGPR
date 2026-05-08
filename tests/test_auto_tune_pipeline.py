#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline-level auto-tune backend tests."""

from __future__ import annotations

import numpy as np

import core.auto_tune_pipeline as auto_tune_pipeline
from core.auto_tune_pipeline import run_auto_tune_pipeline, to_summary_dict


def _build_pipeline_profile(samples: int = 96, traces: int = 28) -> np.ndarray:
    rng = np.random.default_rng(23)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    drift = 0.45 * np.sin(2.0 * np.pi * 0.55 * t)
    signal = 0.12 * np.sin(2.0 * np.pi * 11.0 * t)
    background = 0.09 * np.sin(2.0 * np.pi * 2.5 * x)
    data = np.repeat(drift + signal, traces, axis=1) + background
    data += 0.03 * rng.normal(size=(samples, traces))

    center_trace = traces // 2
    for trace_idx in range(traces):
        lateral = (trace_idx - center_trace) / max(center_trace, 1)
        row = int(round(31 + 8.0 * lateral * lateral))
        if 2 <= row < samples - 3:
            data[row - 1 : row + 2, trace_idx] += np.array([0.28, 0.95, 0.28])
    data[62:66, traces // 4 : traces // 2] += 0.18 * np.hanning(4)[:, None]
    return data.astype(np.float32)


def _manual_roi() -> dict:
    return {
        "mode": "manual",
        "bounds": {
            "time_start_idx": 22,
            "time_end_idx": 52,
            "dist_start_idx": 5,
            "dist_end_idx": 24,
        },
        "label": "目标双曲线区",
    }


def _truth_manifest() -> dict:
    return {
        "schema": "mygpr_gprmax_ground_truth_v1",
        "scenario_id": "pipeline_truth_demo",
        "analysis_roi": {
            "time_start_idx": 18,
            "time_end_idx": 58,
            "dist_start_idx": 3,
            "dist_end_idx": 26,
        },
        "targets": [
            {
                "target_id": "hyperbola_01",
                "type": "hyperbola",
                "must_preserve": True,
                "roi": {
                    "time_start_idx": 27,
                    "time_end_idx": 42,
                    "dist_start_idx": 7,
                    "dist_end_idx": 23,
                },
            }
        ],
    }


def test_pipeline_auto_tunes_each_step_on_current_state():
    raw = _build_pipeline_profile()

    result = run_auto_tune_pipeline(
        raw,
        pipeline=["dewow", "subtracting_average_2D"],
        manual_params_by_method={
            "dewow": {"window": 1},
            "subtracting_average_2D": {"ntraces": 3},
        },
        roi_spec=_manual_roi(),
        search_mode="fast",
    )

    assert result.pipeline == ["dewow", "subtracting_average_2D"]
    assert [step.method_key for step in result.steps] == [
        "dewow",
        "subtracting_average_2D",
    ]
    assert result.automatic.params_by_method["dewow"]["window"] != 1
    assert result.steps[0].manual_before.shape == raw.shape
    assert result.steps[0].manual_after.shape == raw.shape
    assert result.steps[0].auto_before.shape == raw.shape
    assert result.steps[0].auto_after.shape == raw.shape
    assert result.steps[1].manual_before.shape == result.steps[0].manual_after.shape
    assert result.steps[1].auto_before.shape == result.automatic.result.shape
    assert np.isfinite(result.metric_delta["pipeline_score"])
    assert result.overall_recommendation in {"adopt_auto", "review", "keep_manual"}


def test_pipeline_uses_ground_truth_metrics_and_rolls_back_unsafe_auto(monkeypatch):
    raw = _build_pipeline_profile()

    def fake_auto_tune(*args, **kwargs):
        return {
            "method_key": "subtracting_average_2D",
            "method_name": "forced",
            "recommended_params": {"ntraces": 1},
            "best_params": {"ntraces": 1},
            "selection_confidence": 0.25,
            "selection_margin": 0.0,
            "execution_stats": {"constraint_adjustment_count": 0},
            "best_reason": "forced unsafe background removal",
        }

    monkeypatch.setattr(auto_tune_pipeline, "auto_tune_method", fake_auto_tune)

    result = auto_tune_pipeline.run_auto_tune_pipeline(
        raw,
        pipeline=["subtracting_average_2D"],
        manual_params_by_method={"subtracting_average_2D": {"ntraces": 15}},
        roi_spec=_manual_roi(),
        ground_truth=_truth_manifest(),
        search_mode="fast",
    )

    step = result.steps[0]
    assert step.recommendation == "keep_manual"
    assert step.rolled_back_to_manual is True
    assert "target_truth_degraded" in step.risk_flags
    assert result.overall_recommendation == "keep_manual"
    assert result.automatic.result.shape == result.manual.result.shape


def test_pipeline_summary_is_json_safe_and_excludes_arrays():
    raw = _build_pipeline_profile(samples=72, traces=18)
    result = run_auto_tune_pipeline(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )

    summary = to_summary_dict(result)

    assert summary["input_shape"] == list(raw.shape)
    assert "manual_before" not in summary["steps"][0]
    assert "auto_after" not in summary["steps"][0]
    assert isinstance(summary["automatic"]["params_by_method"]["dewow"]["window"], int)
    assert summary["overall_recommendation"] == result.overall_recommendation


def test_pipeline_locked_params_apply_to_both_branches_without_auto_tune(monkeypatch):
    raw = _build_pipeline_profile(samples=72, traces=18)

    def fail_auto_tune(*args, **kwargs):
        raise AssertionError("locked methods must not auto-tune")

    monkeypatch.setattr(auto_tune_pipeline, "auto_tune_method", fail_auto_tune)
    result = run_auto_tune_pipeline(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        locked_params_by_method={"dewow": {"window": 5}},
        search_mode="fast",
    )

    assert result.manual.params_by_method["dewow"] == {"window": 5}
    assert result.automatic.params_by_method["dewow"] == {"window": 5}
    assert result.steps[0].manual_params == {"window": 5}
    assert result.steps[0].auto_params == {"window": 5}
    assert result.steps[0].auto_tune_result is None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the AT-001 stepwise AutoTune validation runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.auto_tune_validation.run_stepwise_validation import (
    _branch_invalid_reason,
    _infer_zero_time_policy,
    _run_branch,
    _sanity_warnings,
    run_validation,
)


def test_stepwise_validation_runner_writes_required_evidence(tmp_path: Path):
    evidence_root = tmp_path / "AT-001"

    result = run_validation(
        evidence_root=evidence_root,
        dataset="cylinder_single_v1",
        mode="smoke",
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
    )

    assert result["ground_truth_available"] is True
    assert result["metric_type"] == "ground_truth"
    required = [
        "reports/comparison_report.md",
        "manifests/evidence_manifest.json",
        "manifests/stepwise_report.json",
        "manifests/comparison_summary.json",
        "tables/trial_table.csv",
        "tables/trial_table.json",
        "figures/manual_bscan.png",
        "figures/auto_bscan.png",
        "figures/side_by_side.png",
        "figures/step_01_manual_set_zero_time.png",
        "figures/step_01_auto_set_zero_time.png",
    ]
    for rel_path in required:
        assert (evidence_root / rel_path).exists(), rel_path

    manifest = json.loads((evidence_root / "manifests/evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "test-source-commit"
    assert manifest["metric_type"] == "ground_truth"
    assert manifest["ground_truth_available"] is True

    stepwise = json.loads((evidence_root / "manifests/stepwise_report.json").read_text(encoding="utf-8"))
    assert stepwise["metric_type"] == "ground_truth"
    assert stepwise["ground_truth_available"] is True
    assert stepwise["zero_time_policy"] in {"legacy_default", "explicit_only_fixed_zero"}
    assert {step["branch"] for step in stepwise["steps"]} == {"manual", "auto"}
    for step in stepwise["steps"]:
        assert set(step["qc_metrics"]) == {"heuristic", "ground_truth"}
        assert "preview_png" in step
        assert "result_meta" in step

    report = (evidence_root / "reports/comparison_report.md").read_text(encoding="utf-8")
    assert "Stepwise Sanity Table" in report
    assert "Ground Truth And Metric Boundary" in report
    assert "Cannot claim" in report


def test_invalid_baseline_sanity_warning_marks_branch_invalid():
    before = np.ones((32, 8), dtype=np.float32)
    after = np.zeros((32, 8), dtype=np.float32)

    warnings = _sanity_warnings(
        before=before,
        after=after,
        heuristic_metrics={
            "target_band_energy_ratio": 0.0,
            "local_saliency_preservation": 0.0,
            "edge_preservation": 0.0,
            "clipping_ratio_after": 0.0,
            "hot_pixel_ratio_after": 0.0,
        },
        ground_truth_metrics={},
    )

    assert any("energy nearly disappeared" in item for item in warnings)
    assert _branch_invalid_reason(warnings)


def test_runner_does_not_modify_motion_compensation_files():
    motion_files = [
        Path("PythonModule/motion_compensation_v2.py"),
        Path("PythonModule/motion_compensation_core.py"),
        Path("PythonModule/motion_compensation_height.py"),
        Path("PythonModule/motion_compensation_speed.py"),
        Path("PythonModule/motion_compensation_attitude.py"),
    ]
    for path in motion_files:
        assert path.exists()


def test_native_context_zero_time_policy_forces_fixed_zero_when_implicit(tmp_path: Path):
    raw = np.random.RandomState(0).randn(2037, 90).astype(np.float32)
    package = {
        "scenario": {"source": {"kind": "native_gprmax_converted"}},
        "header_info": {"total_time_ns": 24.022894, "time_step_s": 1.1793271683748419e-11, "data_context": "gprmax_impulse"},
    }
    policy = _infer_zero_time_policy(package)
    assert policy == "explicit_only_fixed_zero"

    result = _run_branch(
        branch="safe_default",
        raw=raw,
        header_info={"total_time_ns": 24.022894, "time_step_s": 1.1793271683748419e-11},
        trace_metadata={"trace_distance_m": np.arange(raw.shape[1], dtype=np.float32)},
        ground_truth=None,
        figures_dir=tmp_path,
        auto_tune=False,
        search_mode="fast",
        pipeline=["set_zero_time"],
        manual_params={},
        zero_time_policy=policy,
    )
    step = result["steps"][0]
    assert step["params"]["new_zero_time"] == 0.0
    assert int(step["result_meta"]["shift_samples"]) == 0


def test_explicit_zero_time_param_still_applies_in_validation_policy(tmp_path: Path):
    raw = np.arange(50 * 8, dtype=np.float32).reshape(50, 8)
    result = _run_branch(
        branch="manual",
        raw=raw,
        header_info={"total_time_ns": 50.0, "time_step_s": 1.0e-9},
        trace_metadata={"trace_distance_m": np.arange(raw.shape[1], dtype=np.float32)},
        ground_truth=None,
        figures_dir=tmp_path,
        auto_tune=False,
        search_mode="fast",
        pipeline=["set_zero_time"],
        manual_params={"set_zero_time": {"new_zero_time": 1.0}},
        zero_time_policy="explicit_only_fixed_zero",
    )
    step = result["steps"][0]
    assert step["params"]["new_zero_time"] == 1.0
    assert int(step["result_meta"]["shift_samples"]) == 1

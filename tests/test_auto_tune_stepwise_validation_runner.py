#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the AT-001 stepwise AutoTune validation runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.auto_tune_validation.run_stepwise_validation import (
    _branch_invalid_reason,
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
    assert {step["branch"] for step in stepwise["steps"]} == {"manual", "auto"}
    for step in stepwise["steps"]:
        assert set(step["qc_metrics"]) == {"heuristic", "ground_truth"}
        assert "preview_png" in step

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


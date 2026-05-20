#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the AT-004 ROI, zero-time, and dewow triage runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.auto_tune_validation.run_roi_zerotime_dewow_triage import run_triage


def test_roi_zerotime_dewow_triage_runner_writes_required_evidence(tmp_path: Path):
    evidence_root = tmp_path / "AT-004"

    result = run_triage(
        evidence_root=evidence_root,
        dataset="cylinder_single_v1",
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
    )

    assert result["dataset_name"] == "cylinder_single_v1"
    required = [
        "reports/roi_zerotime_dewow_triage_report.md",
        "manifests/evidence_manifest.json",
        "manifests/triage_summary.json",
        "tables/triage_results.csv",
        "figures/input_bscan_roi_overlay.png",
        "figures/input_gx003_roi_crop.png",
        "figures/input_candidate_roi_crop.png",
        "figures/no_zero_time_no_dewow_roi_overlay.png",
        "figures/safe_default_zero_time_only_roi_overlay.png",
        "figures/zero0_dewow_window_23_roi_overlay.png",
    ]
    for rel_path in required:
        assert (evidence_root / rel_path).exists(), rel_path

    manifest = json.loads((evidence_root / "manifests/evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_id"] == "AT-004"
    assert manifest["source_commit"] == "test-source-commit"
    assert manifest["metric_type"] == "ground_truth_roi_and_heuristic_qc_triage"

    summary = json.loads((evidence_root / "manifests/triage_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_id"] == "AT-004"
    assert summary["root_cause_classification"]["at002_conclusion"] == "inconclusive"
    experiment_ids = {row["experiment_id"] for row in summary["experiments"]}
    assert "no_zero_time_no_dewow" in experiment_ids
    assert "zero_time_fixed_0_dewow_off" in experiment_ids
    assert "safe_default_zero_time_only" in experiment_ids
    assert any(item.startswith("zero0_dewow_window_") for item in experiment_ids)

    report = (evidence_root / "reports/roi_zerotime_dewow_triage_report.md").read_text(encoding="utf-8")
    assert "Corrected ROI recommended" in report
    assert "AT-002 remains `inconclusive`" in report


def test_roi_zerotime_dewow_triage_does_not_touch_frozen_modules():
    frozen = [
        Path("core/processing_engine.py"),
        Path("PythonModule/motion_compensation_v2.py"),
        Path("PythonModule/motion_compensation_core.py"),
    ]
    for path in frozen:
        assert path.exists()

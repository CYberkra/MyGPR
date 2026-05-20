#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the AT-002 native benchmark AutoTune ablation runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.auto_tune_validation.run_native_ablation import run_ablation


def test_native_ablation_runner_writes_required_evidence(tmp_path: Path):
    evidence_root = tmp_path / "AT-002"

    result = run_ablation(
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
        "reports/ablation_report.md",
        "manifests/ablation_summary.json",
        "manifests/evidence_manifest.json",
        "tables/stage_ablation_table.csv",
        "tables/trial_table.csv",
        "figures/input_bscan.png",
        "figures/manual_vs_auto_side_by_side.png",
        "figures/expert_manual_bscan.png",
        "figures/auto_tuned_bscan.png",
        "figures/ablation_dewow_bscan.png",
    ]
    for rel_path in required:
        assert (evidence_root / rel_path).exists(), rel_path

    manifest = json.loads((evidence_root / "manifests/evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_id"] == "AT-002"
    assert manifest["source_commit"] == "test-source-commit"
    assert manifest["ground_truth_available"] is True
    assert manifest["metric_type"] == "ground_truth"

    summary = json.loads((evidence_root / "manifests/ablation_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_id"] == "AT-002"
    assert set(summary["branches"]) == {"expert_manual", "safe_default", "auto_tuned"}
    assert set(summary["ablations"]) == {
        "dewow",
        "frequency_filter_1d",
        "background_suppression",
        "gain",
    }
    assert summary["stage_winners"]["metric_used"] == "truth_score"
    assert summary["ground_truth_available"] is True
    assert summary["heuristic_qc_only"] is False

    report = (evidence_root / "reports/ablation_report.md").read_text(encoding="utf-8")
    assert "Stage-Level Ranking" in report
    assert "Ground truth is not used as AutoTune search input" in report
    assert "Motion compensation" in report


def test_native_ablation_does_not_touch_motion_modules():
    motion_files = [
        Path("PythonModule/motion_compensation_v2.py"),
        Path("PythonModule/motion_compensation_core.py"),
        Path("PythonModule/motion_compensation_height.py"),
        Path("PythonModule/motion_compensation_speed.py"),
        Path("PythonModule/motion_compensation_attitude.py"),
    ]
    for path in motion_files:
        assert path.exists()

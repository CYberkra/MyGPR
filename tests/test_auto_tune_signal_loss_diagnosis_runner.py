#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the AT-003 signal-loss diagnosis runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.auto_tune_validation.run_signal_loss_diagnosis import run_diagnosis


def test_signal_loss_diagnosis_runner_writes_required_evidence(tmp_path: Path):
    evidence_root = tmp_path / "AT-003"

    result = run_diagnosis(
        evidence_root=evidence_root,
        dataset="cylinder_single_v1",
        mode="smoke",
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
    )

    assert result["dataset_name"] == "cylinder_single_v1"
    required = [
        "reports/signal_loss_diagnosis_report.md",
        "manifests/evidence_manifest.json",
        "manifests/step_diagnostics.json",
        "tables/step_diagnostics.csv",
        "figures/input_bscan_roi_overlay.png",
        "figures/expert_manual_step_01_set_zero_time_roi_overlay.png",
        "figures/expert_manual_step_01_set_zero_time_roi_crop.png",
        "figures/expert_manual_stepwise_energy_curve.png",
    ]
    for rel_path in required:
        assert (evidence_root / rel_path).exists(), rel_path

    manifest = json.loads((evidence_root / "manifests/evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_id"] == "AT-003"
    assert manifest["source_commit"] == "test-source-commit"
    assert manifest["metric_type"] == "ground_truth_and_heuristic_diagnostics"

    diagnostics = json.loads((evidence_root / "manifests/step_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["artifact_id"] == "AT-003"
    assert diagnostics["first_failing_step"]
    assert diagnostics["likely_root_cause"]["at002_conclusion"] == "inconclusive"
    first_step = diagnostics["diagnostics"][0]
    assert "roi_before_energy" in first_step
    assert "roi_after_energy" in first_step
    assert "roi_overlay_png" in first_step

    report = (evidence_root / "reports/signal_loss_diagnosis_report.md").read_text(encoding="utf-8")
    assert "First Failing Step" in report
    assert "AT-002 conclusion remains `inconclusive`" in report

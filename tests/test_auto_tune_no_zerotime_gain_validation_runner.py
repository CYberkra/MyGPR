#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the AT-005A no-zero-time gain validation runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.auto_tune_validation.run_no_zerotime_gain_validation import run_validation


def test_no_zerotime_gain_validation_writes_reports_and_loop(tmp_path: Path):
    evidence_root = tmp_path / "AT-005A"

    result = run_validation(
        evidence_root=evidence_root,
        dataset="cylinder_single_v1",
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
        include_field=False,
    )

    assert result["hundred_round_loop_completed"] is True
    required = [
        "reports/no_zerotime_gain_validation_report.md",
        "reports/no_zerotime_gain_validation_report.html",
        "reports/hundred_round_iteration_summary.md",
        "manifests/evidence_manifest.json",
        "manifests/validation_summary.json",
        "manifests/hundred_round_iteration_log.json",
        "tables/lane_metrics.csv",
        "tables/gain_variant_table.csv",
        "tables/trial_table.csv",
        "tables/hundred_round_iteration_log.csv",
        "figures/input_bscan_roi_overlay.png",
        "figures/background_suppression_only.png",
        "figures/sec_gain_comparison.png",
        "figures/time_power_gain_comparison.png",
        "figures/agc_gain_comparison.png",
        "figures/gain_variant_summary.png",
    ]
    for rel_path in required:
        assert (evidence_root / rel_path).exists(), rel_path

    manifest = json.loads((evidence_root / "manifests/evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_id"] == "AT-005A"
    assert manifest["source_commit"] == "test-source-commit"
    assert manifest["metric_type"] == "ground_truth_gain_validation_and_field_heuristic_qc"

    summary = json.loads((evidence_root / "manifests/validation_summary.json").read_text(encoding="utf-8"))
    assert summary["zero_time_policy"] == "excluded"
    assert summary["field_lane"]["status"] == "skipped"
    assert summary["hundred_round_loop"]["total_iterations"] == 100
    assert any(row["gain_method"] == "agcGain" for row in summary["lane_rows"])
    assert any(
        "agc_non_amplitude_preserving_display_gain" in row["sanity_warnings"]
        for row in summary["lane_rows"]
        if row["gain_method"] == "agcGain"
    )

    html = (evidence_root / "reports/no_zerotime_gain_validation_report.html").read_text(encoding="utf-8")
    assert "Zero-time" in html
    assert "AGC is display-oriented" in html

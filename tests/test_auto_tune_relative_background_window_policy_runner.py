#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-011 relative background-window policy runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.auto_tune_validation.run_relative_background_window_policy import _parse_ratio_candidates, run_validation


def test_at011_runner_writes_required_outputs(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data = np.random.RandomState(29).randn(300, 60).astype(np.float32)
    np.savetxt(dataset_dir / "mygpr_bscan.csv", data, delimiter=",")
    (dataset_dir / "scenario.json").write_text(
        json.dumps(
            {
                "schema": "mygpr_gprmax_scenario_v1",
                "scenario_id": "native_small",
                "source": {"kind": "native_gprmax_converted"},
                "simulation": {
                    "sample_count": 300,
                    "trace_count": 60,
                    "time_step_s": 1.2e-11,
                    "total_time_ns": 3.6,
                    "trace_step_m": 0.02,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (dataset_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "analysis_roi": {"time_start_idx": 30, "time_end_idx": 220, "dist_start_idx": 8, "dist_end_idx": 50},
                "targets": [
                    {
                        "id": "target_0",
                        "roi": {"time_start_idx": 90, "time_end_idx": 150, "dist_start_idx": 20, "dist_end_idx": 30},
                    }
                ],
                "background_rois": [
                    {"time_start_idx": 10, "time_end_idx": 60, "dist_start_idx": 0, "dist_end_idx": 8}
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    evidence_root = tmp_path / "AT-011"
    result = run_validation(
        evidence_root=evidence_root,
        dataset=str(dataset_dir),
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
        ratio_candidates=[0.05, 0.10, 0.20, 0.40, 0.70, 1.0],
    )

    assert result["generated_candidate_ratios"]
    assert result["generated_ntraces"]
    summary = json.loads(
        (evidence_root / "manifests/relative_background_window_policy_summary.json").read_text(encoding="utf-8")
    )
    assert summary["source_commit"] == "test-source-commit"
    assert summary["zero_time_policy"] == "excluded"
    assert summary["dewow_policy"] == "excluded_primary"
    assert summary["generated_candidate_ratios"]
    assert summary["generated_ntraces"]
    assert summary["best_policy_label"] in {"local", "medium", "large", "near_full_line", "full_line"}

    required = [
        "reports/relative_background_window_policy_report.md",
        "reports/relative_background_window_policy_report.html",
        "manifests/evidence_manifest.json",
        "manifests/relative_background_window_policy_summary.json",
        "tables/generated_candidate_policy.csv",
        "tables/gx003_relative_candidate_metrics.csv",
        "tables/policy_label_summary.csv",
        "figures/input_bscan_roi_overlay.png",
        "figures/relative_candidate_sweep_summary.png",
        "figures/policy_label_metric_summary.png",
    ]
    for rel in required:
        assert (evidence_root / rel).exists(), rel


def test_parse_ratio_candidates_rejects_invalid_values():
    assert _parse_ratio_candidates("0.05, 0.1,1") == [0.05, 0.1, 1.0]
    with pytest.raises(ValueError, match="positive finite"):
        _parse_ratio_candidates("0.1,0")
    with pytest.raises(ValueError, match="invalid ratio"):
        _parse_ratio_candidates("0.1,bad")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-007 post zero-time policy rerun runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.auto_tune_validation.run_post_zero_time_policy_rerun import run_post_rerun


def test_post_zero_time_policy_rerun_uses_guarded_policy_and_writes_outputs(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data = np.random.RandomState(7).randn(256, 32).astype(np.float32)
    np.savetxt(dataset_dir / "mygpr_bscan.csv", data, delimiter=",")
    (dataset_dir / "scenario.json").write_text(
        json.dumps(
            {
                "schema": "mygpr_gprmax_scenario_v1",
                "scenario_id": "native_small",
                "source": {"kind": "native_gprmax_converted"},
                "simulation": {
                    "sample_count": 256,
                    "trace_count": 32,
                    "time_step_s": 1.2e-11,
                    "total_time_ns": 3.072,
                    "trace_step_m": 0.01,
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
                "analysis_roi": {"time_start_idx": 40, "time_end_idx": 160, "dist_start_idx": 8, "dist_end_idx": 24},
                "targets": [
                    {
                        "id": "target_0",
                        "roi": {"time_start_idx": 80, "time_end_idx": 120, "dist_start_idx": 12, "dist_end_idx": 18},
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

    historical_at002 = tmp_path / "at002.json"
    historical_at003 = tmp_path / "at003.json"
    historical_at002.write_text(
        json.dumps(
            {
                "stage_ablation_table": [
                    {"branch": "safe_default", "valid": False, "branch_invalid_reason": "zero-time/effective signal energy nearly disappeared"}
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    historical_at003.write_text(
        json.dumps(
            {
                "source_commit": "old",
                "first_failing_step": {
                    "branch": "safe_default",
                    "step_index": 1,
                    "method_key": "set_zero_time",
                    "global_energy_ratio": 0.0022,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    evidence_root = tmp_path / "AT-007"
    result = run_post_rerun(
        evidence_root=evidence_root,
        dataset=str(dataset_dir),
        mode="smoke",
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
        historical_at002_summary=historical_at002,
        historical_at003_summary=historical_at003,
    )

    assert result["zero_time_policy"] == "explicit_only_fixed_zero"
    assert result["zero_time_shift_eliminated"] is True
    required = [
        "reports/post_zero_time_policy_rerun_report.md",
        "reports/post_zero_time_policy_rerun_report.html",
        "manifests/evidence_manifest.json",
        "manifests/post_fix_ablation_summary.json",
        "manifests/post_fix_signal_loss_summary.json",
        "tables/post_fix_ablation_table.csv",
        "tables/post_fix_step_diagnostics.csv",
        "tables/before_after_validity_comparison.csv",
        "tables/before_after_metric_comparison.csv",
    ]
    for rel in required:
        assert (evidence_root / rel).exists(), rel

    step_csv = (evidence_root / "tables" / "post_fix_step_diagnostics.csv").read_text(encoding="utf-8")
    assert "zero_time_policy" in step_csv

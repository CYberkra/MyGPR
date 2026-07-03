#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-008A no-dewow post-fix validation runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.auto_tune_validation.run_no_dewow_post_fix_validation import run_validation


def test_at008a_primary_lane_excludes_zero_time_and_dewow(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data = np.random.RandomState(11).randn(256, 32).astype(np.float32)
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

    at007 = tmp_path / "at007.json"
    at005a = tmp_path / "at005a.json"
    at007.write_text(
        json.dumps(
            {"post_fix": {"first_failing_step": {"method_key": "dewow"}}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    at005a.write_text(
        json.dumps(
            {"best_visual_gain_variant": "energy_decay_gain"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    evidence_root = tmp_path / "AT-008A"
    result = run_validation(
        evidence_root=evidence_root,
        dataset=str(dataset_dir),
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
        historical_at007=at007,
        historical_at005a=at005a,
        include_dewow_side_lanes=True,
    )

    assert result["zero_time_policy"] == "excluded"
    assert "excluded_primary" in result["dewow_policy"]

    summary = json.loads((evidence_root / "manifests/no_dewow_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["source_commit"] == "test-source-commit"
    assert summary["zero_time_policy"] == "excluded"
    assert "dewow_policy" in summary
    assert any(row["dewow_policy"] == "excluded_primary" for row in summary["lane_rows"])
    assert any("dewow" in row["pipeline"] for row in summary["lane_rows"] if row["lane_id"].startswith("lane_6"))

    primary_rows = [row for row in summary["lane_rows"] if row.get("primary_lane")]
    assert all("set_zero_time" not in row["pipeline"] for row in primary_rows)
    assert all("dewow" not in row["pipeline"] for row in primary_rows)

    required = [
        "reports/no_dewow_post_fix_validation_report.md",
        "reports/no_dewow_post_fix_validation_report.html",
        "manifests/evidence_manifest.json",
        "manifests/no_dewow_validation_summary.json",
        "tables/lane_metrics.csv",
        "tables/gain_variant_table.csv",
        "tables/before_after_dewow_comparison.csv",
        "figures/input_bscan_roi_overlay.png",
        "figures/no_dewow_lane_summary.png",
    ]
    for rel in required:
        assert (evidence_root / rel).exists(), rel

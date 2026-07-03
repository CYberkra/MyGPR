#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-009 background/gain policy refinement runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.auto_tune_validation.run_background_gain_policy_refinement import run_validation


def test_at009_runner_writes_required_outputs(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data = np.random.RandomState(13).randn(256, 32).astype(np.float32)
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

    at008a = tmp_path / "at008a.json"
    at008a.write_text(
        json.dumps({"best_gain_variant": "energy_decay_gain"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    evidence_root = tmp_path / "AT-009"
    result = run_validation(
        evidence_root=evidence_root,
        dataset=str(dataset_dir),
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
        historical_at008a=at008a,
    )

    assert result["recommended_ntraces_range"]
    summary = json.loads((evidence_root / "manifests/background_gain_policy_summary.json").read_text(encoding="utf-8"))
    assert summary["source_commit"] == "test-source-commit"
    assert summary["zero_time_policy"] == "excluded"
    assert summary["dewow_policy"] == "excluded_primary"
    assert summary["background_domain_candidates"] == [9, 17, 25, 33, 41, 49, 57, 65, 73]
    assert summary["best_gain_variant"] in {"energy_decay_gain", "sec_gain", "time_power_gain_local", "agcGain"}
    assert "autotune_status" in summary

    required = [
        "reports/background_gain_policy_refinement_report.md",
        "reports/background_gain_policy_refinement_report.html",
        "manifests/evidence_manifest.json",
        "manifests/background_gain_policy_summary.json",
        "tables/background_sweep_metrics.csv",
        "tables/gain_policy_metrics.csv",
        "tables/constrained_autotune_comparison.csv",
        "figures/background_ntraces_sweep_summary.png",
        "figures/gain_policy_summary.png",
        "figures/manual_vs_constrained_auto_summary.png",
    ]
    for rel in required:
        assert (evidence_root / rel).exists(), rel

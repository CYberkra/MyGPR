#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-014 multi-scene metric-fidelity validation runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.auto_tune_validation.run_multi_scene_metric_fidelity_validation import (
    run_metric_fidelity_validation,
)


def _roi(
    roi_id: str,
    roi_type: str,
    s0: int,
    s1: int,
    t0: int,
    t1: int,
    *,
    obj: str = "",
) -> dict[str, object]:
    return {
        "roi_id": roi_id,
        "roi_type": roi_type,
        "source": "model_ground_truth",
        "claim_level": "ground_truth_metric",
        "sample_start_idx": s0,
        "sample_end_idx": s1,
        "trace_start_idx": t0,
        "trace_end_idx": t1,
        "associated_object_id": obj,
        "notes": roi_type,
    }


def _write_complete_scene(base: Path, *, scene_id: str, shape: tuple[int, int], rois: list[dict[str, object]]) -> None:
    (base / "converted").mkdir(parents=True, exist_ok=True)
    (base / "manifests").mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(abs(hash(scene_id)) % (2**32))
    data = rs.randn(*shape).astype(np.float32)
    np.savetxt(base / "converted" / "data.csv", data, delimiter=",")
    (base / "manifests" / "ground_truth.json").write_text(
        json.dumps(
            {
                "schema": "mygpr_ground_truth_v2",
                "scenario_id": scene_id.lower(),
                "rois": rois,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_at014_runner_writes_metric_fidelity_artifacts(tmp_path: Path) -> None:
    evidence_repo = tmp_path / "evidence_repo"
    gprmax = evidence_repo / "gprmax"

    gx003 = gprmax / "GX-003_audited_native_gprmax_benchmark"
    (gx003 / "tables").mkdir(parents=True, exist_ok=True)
    (gx003 / "manifests").mkdir(parents=True, exist_ok=True)
    np.savetxt(gx003 / "tables" / "mygpr_bscan.csv", np.random.RandomState(7).randn(300, 45), delimiter=",")
    (gx003 / "manifests" / "gprmax_package_audit.json").write_text("{}", encoding="utf-8")

    _write_complete_scene(
        gprmax / "GX-004_no_target_false_positive_control",
        scene_id="GX-004",
        shape=(320, 45),
        rois=[
            _roi("gx004_nt", "no_target_region", 40, 110, 6, 22),
            _roi("gx004_neg", "negative_control", 140, 200, 10, 28),
            _roi("gx004_bg", "local_background", 210, 260, 15, 35),
        ],
    )
    _write_complete_scene(
        gprmax / "GX-005_multi_target_varying_depth",
        scene_id="GX-005",
        shape=(320, 45),
        rois=[
            _roi("gx005_ta", "target", 70, 120, 8, 16, obj="target_A"),
            _roi("gx005_tb", "target", 150, 210, 25, 34, obj="target_B"),
            _roi("gx005_bg_a", "local_background", 70, 120, 2, 7, obj="target_A"),
            _roi("gx005_bg_b", "local_background", 150, 210, 36, 43, obj="target_B"),
            _roi("gx005_neg", "negative_control", 240, 300, 6, 20),
        ],
    )
    _write_complete_scene(
        gprmax / "GX-006_layered_complex_background",
        scene_id="GX-006",
        shape=(320, 45),
        rois=[
            _roi("gx006_layer", "layer_interface", 100, 140, 4, 40),
            _roi("gx006_target", "target", 170, 230, 20, 30, obj="target_C"),
            _roi("gx006_bg", "local_background", 170, 230, 2, 12, obj="target_C"),
            _roi("gx006_neg", "negative_control", 230, 300, 30, 43),
        ],
    )

    out = tmp_path / "AT-014"
    result = run_metric_fidelity_validation(
        evidence_root=out,
        evidence_repo_root=evidence_repo,
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="at014-test-commit",
        ratio_candidates=[0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
    )
    assert result["included_scenes"] == ["GX-003", "GX-004", "GX-005", "GX-006"]
    assert result["gate_status"] == "blocked"

    manifest = json.loads((out / "manifests" / "evidence_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "manifests" / "metric_fidelity_summary.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "at014-test-commit"
    assert summary["source_commit"] == "at014-test-commit"
    assert summary["at011_policy_unchanged"] is True
    assert summary["lane_policy"]["zero_time"] == "excluded_or_fixed_zero"
    assert summary["lane_policy"]["dewow"] == "excluded_primary"
    assert summary["at013_proxy_limitation_addressed"] is True

    required = [
        "reports/multi_scene_metric_fidelity_validation_report.md",
        "reports/multi_scene_metric_fidelity_validation_report.html",
        "tables/scene_candidate_metrics.csv",
        "tables/gx004_false_positive_fidelity_metrics.csv",
        "tables/gx005_per_target_processed_metrics.csv",
        "tables/gx006_layer_interface_metrics.csv",
        "tables/gate_reassessment.csv",
        "tables/warnings_and_risk_flags.csv",
        "figures/scene_candidate_metric_fidelity_overview.png",
    ]
    for rel in required:
        assert (out / rel).exists(), rel

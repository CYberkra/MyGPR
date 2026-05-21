#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-013 multi-scene relative policy validation runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.auto_tune_validation.run_multi_scene_relative_policy_validation import run_multi_scene_validation


def _write_scene(base: Path, *, scene_id: str, shape: tuple[int, int], with_target: bool) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "converted").mkdir(exist_ok=True)
    (base / "manifests").mkdir(exist_ok=True)
    data = np.random.RandomState(abs(hash(scene_id)) % (2**32)).randn(*shape).astype(np.float32)
    np.savetxt(base / "converted" / "data.csv", data, delimiter=",")
    rois = [
        {
            "roi_id": f"{scene_id.lower()}_neg",
            "roi_type": "negative_control",
            "source": "model_ground_truth",
            "claim_level": "ground_truth_metric",
            "sample_start_idx": 10,
            "sample_end_idx": min(shape[0], 80),
            "trace_start_idx": 2,
            "trace_end_idx": min(shape[1], 16),
            "associated_object_id": "",
            "notes": "neg",
        }
    ]
    if with_target:
        rois.append(
            {
                "roi_id": f"{scene_id.lower()}_target",
                "roi_type": "target",
                "source": "model_ground_truth",
                "claim_level": "ground_truth_metric",
                "sample_start_idx": min(shape[0] // 3, shape[0] - 2),
                "sample_end_idx": min(shape[0] // 2, shape[0] - 1),
                "trace_start_idx": min(shape[1] // 3, shape[1] - 2),
                "trace_end_idx": min(shape[1] // 2, shape[1] - 1),
                "associated_object_id": "target_A",
                "notes": "target",
            }
        )
    gt = {
        "schema": "mygpr_ground_truth_v2",
        "scenario_id": scene_id.lower(),
        "has_buried_target": with_target,
        "rois": rois,
    }
    (base / "manifests" / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    bm = {
        "schema": "mygpr_native_benchmark_manifest_v1",
        "artifact_id": scene_id,
        "scenario_id": scene_id.lower(),
        "generation_status": "complete",
        "trace_spacing_m": 0.01,
        "trace_count": shape[1],
        "sample_count": shape[0],
    }
    (base / "manifests" / "benchmark_manifest.json").write_text(json.dumps(bm, ensure_ascii=False, indent=2), encoding="utf-8")


def test_at013_runner_writes_expected_artifacts(tmp_path: Path):
    ev = tmp_path / "evidence_repo"
    gprmax = ev / "gprmax"
    _write_scene(gprmax / "GX-004_no_target_false_positive_control", scene_id="GX-004", shape=(280, 45), with_target=False)
    _write_scene(gprmax / "GX-005_multi_target_varying_depth", scene_id="GX-005", shape=(300, 45), with_target=True)
    _write_scene(gprmax / "GX-006_layered_complex_background", scene_id="GX-006", shape=(300, 45), with_target=True)

    gx003 = gprmax / "GX-003_audited_native_gprmax_benchmark"
    (gx003 / "tables").mkdir(parents=True, exist_ok=True)
    (gx003 / "manifests").mkdir(parents=True, exist_ok=True)
    data = np.random.RandomState(17).randn(280, 45).astype(np.float32)
    np.savetxt(gx003 / "tables" / "mygpr_bscan.csv", data, delimiter=",")
    (gx003 / "manifests" / "gprmax_package_audit.json").write_text("{}", encoding="utf-8")

    out = tmp_path / "AT-013"
    result = run_multi_scene_validation(
        evidence_root=out,
        evidence_repo_root=ev,
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="at013-test-commit",
        ratio_candidates=[0.1, 0.2, 0.4, 0.7, 1.0],
    )
    assert result["included_scenes"] == ["GX-003", "GX-004", "GX-005", "GX-006"]
    assert (out / "reports/multi_scene_relative_policy_validation_report.md").exists()
    assert (out / "reports/multi_scene_relative_policy_validation_report.html").exists()
    assert (out / "manifests/evidence_manifest.json").exists()
    summary = json.loads((out / "manifests/multi_scene_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["source_commit"] == "at013-test-commit"
    assert summary["gate_status"]["gate_status"] in {"pass", "partial_pass", "blocked", "inconclusive"}
    for rel in [
        "tables/scene_candidate_metrics.csv",
        "tables/scene_gate_status.csv",
        "tables/per_target_metrics.csv",
        "tables/false_positive_metrics.csv",
        "tables/warnings_and_risk_flags.csv",
        "figures/scene_candidate_overview.png",
    ]:
        assert (out / rel).exists(), rel

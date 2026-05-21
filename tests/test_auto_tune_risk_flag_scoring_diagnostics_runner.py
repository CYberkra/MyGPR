#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-016 risk-flag and scoring diagnostics runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.auto_tune_validation.run_risk_flag_scoring_diagnostics import (
    run_risk_flag_scoring_diagnostics,
)


def _write_json_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) for k in fields})


def _prepare_fake_at014(root: Path) -> None:
    tables = root / "tables"
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    scene_rows: list[dict[str, object]] = []
    for scene in ("GX-003", "GX-004", "GX-005", "GX-006"):
        # local n=9 as best
        scene_rows.append(
            {
                "scene_id": scene,
                "candidate_label": "local",
                "generated_ntraces": 9,
                "candidate_score": 0.95,
                "clipping_ratio": 0.01,
                "false_positive_proxy": 0.1,
                "roi_contrast": 1.2,
                "target_preservation": 1.1,
                "max_target_preservation_gap": 0.28 if scene == "GX-005" else None,
            }
        )
        scene_rows.append(
            {
                "scene_id": scene,
                "candidate_label": "medium",
                "generated_ntraces": 21,
                "candidate_score": 0.90,
                "clipping_ratio": 0.02,
                "false_positive_proxy": 0.2,
                "roi_contrast": 0.9,
                "target_preservation": 0.8,
                "max_target_preservation_gap": 0.31 if scene == "GX-005" else None,
            }
        )
    _write_json_csv(tables / "scene_candidate_metrics.csv", scene_rows)

    _write_json_csv(
        tables / "gx004_false_positive_fidelity_metrics.csv",
        [
            {
                "scene_id": "GX-004",
                "candidate_label": "local",
                "generated_ntraces": 9,
                "false_positive_proxy": 0.08,
                "artifact_risk": False,
                "clipping_ratio": 0.01,
                "global_energy_ratio": 0.5,
            },
            {
                "scene_id": "GX-004",
                "candidate_label": "medium",
                "generated_ntraces": 21,
                "false_positive_proxy": 0.12,
                "artifact_risk": False,
                "clipping_ratio": 0.02,
                "global_energy_ratio": 0.4,
            },
        ],
    )
    _write_json_csv(
        tables / "gx005_per_target_processed_metrics.csv",
        [
            {
                "scene_id": "GX-005",
                "candidate_label": "local",
                "generated_ntraces": 9,
                "target_id": "target_A",
                "target_preservation_ratio": 0.10,
                "contrast_ratio": 1.40,
            },
            {
                "scene_id": "GX-005",
                "candidate_label": "local",
                "generated_ntraces": 9,
                "target_id": "target_B",
                "target_preservation_ratio": 0.39,
                "contrast_ratio": 0.80,
            },
            {
                "scene_id": "GX-005",
                "candidate_label": "medium",
                "generated_ntraces": 21,
                "target_id": "target_A",
                "target_preservation_ratio": 0.08,
                "contrast_ratio": 0.95,
            },
            {
                "scene_id": "GX-005",
                "candidate_label": "medium",
                "generated_ntraces": 21,
                "target_id": "target_B",
                "target_preservation_ratio": 0.35,
                "contrast_ratio": 1.05,
            },
        ],
    )
    _write_json_csv(
        tables / "gx006_layer_interface_metrics.csv",
        [
            {
                "scene_id": "GX-006",
                "candidate_label": "local",
                "generated_ntraces": 9,
                "layer_interface_energy_ratio": 0.62,
                "clutter_false_positive_proxy": 0.09,
                "interface_suppression_risk": True,
                "clutter_false_positive_risk": False,
            },
            {
                "scene_id": "GX-006",
                "candidate_label": "medium",
                "generated_ntraces": 21,
                "layer_interface_energy_ratio": 0.55,
                "clutter_false_positive_proxy": 0.08,
                "interface_suppression_risk": True,
                "clutter_false_positive_risk": False,
            },
        ],
    )
    _write_json_csv(
        tables / "gate_reassessment.csv",
        [
            {
                "gate_status": "blocked",
                "risk_flags": [
                    "gx005_target_imbalance",
                    "gx006_interface_suppression",
                    "synthetic_thin2d_scene_limit",
                ],
                "scene_best_labels": {
                    "GX-003": "local",
                    "GX-004": "local",
                    "GX-005": "local",
                    "GX-006": "local",
                },
            }
        ],
    )
    _write_json_csv(
        tables / "warnings_and_risk_flags.csv",
        [
            {
                "scene_id": "GX-005",
                "candidate_label": "local",
                "generated_ntraces": 9,
                "risk_flags": ["gx005_target_imbalance_risk"],
            },
            {
                "scene_id": "GX-006",
                "candidate_label": "local",
                "generated_ntraces": 9,
                "risk_flags": ["gx006_interface_suppression_risk"],
            },
        ],
    )

    (manifests / "metric_fidelity_summary.json").write_text(
        json.dumps(
            {
                "source_commit": "fake-at014-source",
                "included_scenes": ["GX-003", "GX-004", "GX-005", "GX-006"],
                "gate_reassessment": {
                    "gate_status": "blocked",
                    "risk_flags": [
                        "gx005_target_imbalance",
                        "gx006_interface_suppression",
                        "synthetic_thin2d_scene_limit",
                    ],
                },
                "scene_best": {
                    "GX-003": {"best_candidate_label": "local", "best_candidate_ntraces": 9},
                    "GX-004": {"best_candidate_label": "local", "best_candidate_ntraces": 9},
                    "GX-005": {"best_candidate_label": "local", "best_candidate_ntraces": 9},
                    "GX-006": {"best_candidate_label": "local", "best_candidate_ntraces": 9},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_at016_runner_generates_expected_artifacts(tmp_path: Path) -> None:
    at014_root = tmp_path / "AT-014"
    out_root = tmp_path / "AT-016"
    _prepare_fake_at014(at014_root)
    result = run_risk_flag_scoring_diagnostics(
        at014_root=at014_root,
        evidence_root=out_root,
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="at016-test-commit",
    )
    assert result["gate_status"] == "blocked"
    assert result["at014_rerun"] is False
    assert result["scoring_changed"] is False
    assert result["algorithms_changed"] is False

    summary = json.loads((out_root / "manifests" / "risk_flag_scoring_diagnostics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_root / "manifests" / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert summary["source_commit"] == "at016-test-commit"
    assert manifest["source_commit"] == "at016-test-commit"
    assert summary["gate_status"] == "blocked"
    assert summary["risk_flags"] == [
        "gx005_target_imbalance",
        "gx006_interface_suppression",
        "synthetic_thin2d_scene_limit",
    ]

    required = [
        "reports/risk_flag_scoring_diagnostics_report.md",
        "reports/risk_flag_scoring_diagnostics_report.html",
        "tables/ranking_breakdown.csv",
        "tables/risk_vs_score_matrix.csv",
        "tables/gx004_no_target_safety_diagnostics.csv",
        "tables/gx005_target_balance_diagnostics.csv",
        "tables/gx006_interface_suppression_diagnostics.csv",
        "tables/gate_blocker_explanation.csv",
        "tables/recommended_next_actions.csv",
    ]
    for rel in required:
        assert (out_root / rel).exists(), rel


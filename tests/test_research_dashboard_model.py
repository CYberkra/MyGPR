#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for read-only research dashboard state loading."""

from __future__ import annotations

import json
from pathlib import Path

from core.research_dashboard import (
    load_at_bg_artifact,
    load_dashboard_state,
    load_gprmax_artifact,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_missing_evidence_root_does_not_crash(tmp_path: Path):
    config = tmp_path / "dashboard.json"
    _write_json(
        config,
        {
            "evidence_root_candidates": [str(tmp_path / "missing")],
            "gprmax_artifacts": [],
            "at_bg_artifacts": [],
            "draft_scenes": ["fixture_scene_pending"],
        },
    )

    state = load_dashboard_state(config)

    assert state["evidence_root"] == ""
    assert state["scene_status"][0]["paired_evidence"] == "draft"
    assert state["warnings"]


def test_gprmax_artifact_reads_manifest_metrics_and_claim_boundary(tmp_path: Path):
    artifact = tmp_path / "gprmax" / "GX-008_scene037"
    _write_json(
        artifact / "manifests" / "evidence_manifest.json",
        {
            "artifact_id": "GX-008_scene037",
            "artifact_role": "synthetic_complete_2d_paired_diagnostic",
            "scene_id": "scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate",
            "raw_shape": [936, 41],
            "background_shape": [936, 41],
            "target_response_shape": [936, 41],
            "run_backend_final": "GPU via wrapper",
            "source_commit": "abc123",
            "claim_boundary": ["not field validation"],
        },
    )
    _write_json(artifact / "tables" / "standard_paired_metrics.json", {"raw_energy": 1.0})
    (artifact / "figures").mkdir()
    (artifact / "figures" / "raw_preview.png").write_bytes(b"png")

    summary = load_gprmax_artifact(artifact).to_dict()

    assert summary["status"] == "complete"
    assert summary["scene_id"] == "scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate"
    assert summary["raw_shape"] == [936, 41]
    assert summary["metrics_path"].endswith("standard_paired_metrics.json")
    assert summary["claim_boundary"] == ["not field validation"]


def test_gprmax_artifact_warns_on_malformed_json(tmp_path: Path):
    artifact = tmp_path / "bad_artifact"
    (artifact / "manifests").mkdir(parents=True)
    (artifact / "manifests" / "evidence_manifest.json").write_text("{bad", encoding="utf-8")

    summary = load_gprmax_artifact(artifact)

    assert summary.status == "partial"
    assert any("malformed json" in warning for warning in summary.warnings)


def test_at_bg_artifact_reads_selected_parameters(tmp_path: Path):
    artifact = tmp_path / "autotune" / "AT-BG"
    _write_json(
        artifact / "manifests" / "evidence_manifest.json",
        {
            "artifact_id": "AT-BG",
            "artifact_role": "synthetic_background_suppression_autotune_diagnostic",
            "scene_id": "scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate",
            "trial_count": 17,
            "selected_trial_id": "trial_003",
            "claim_boundary": ["not production scoring"],
        },
    )
    _write_json(
        artifact / "tables" / "selected_parameters.json",
        {
            "selected": {
                "trial_id": "trial_003",
                "method": "mean_background_subtraction",
                "parameter_set": {"mode": "moving_window_mean", "window_size": 5},
            }
        },
    )
    _write_json(artifact / "tables" / "trial_table.json", [{"trial_id": "trial_001"}])

    summary = load_at_bg_artifact(artifact).to_dict()

    assert summary["status"] == "complete"
    assert summary["selected_trial_id"] == "trial_003"
    assert summary["selected_method"] == "mean_background_subtraction"
    assert summary["selected_parameters"]["window_size"] == 5
    assert summary["trial_count"] == 17


def test_dashboard_reads_at_bg_multi_scene_comparison(tmp_path: Path):
    evidence = tmp_path / "evidence"
    artifact = evidence / "autotune" / "AT-BG-004B"
    _write_json(artifact / "manifests" / "evidence_manifest.json", {"artifact_id": "AT-BG-004B"})
    _write_json(
        artifact / "tables" / "method_rank_summary.json",
        [{"method": "mean_background_subtraction", "mean_rank": 1}],
    )
    config = tmp_path / "dashboard.json"
    _write_json(
        config,
        {
            "evidence_root_candidates": [str(evidence)],
            "gprmax_artifacts": [],
            "at_bg_artifacts": [{"id": "AT-BG_multi_scene", "path": "autotune/AT-BG-004B"}],
            "draft_scenes": [],
        },
    )

    state = load_dashboard_state(config)

    assert state["method_rank_summary"][0]["mean_rank"] == 1
    assert state["at_bg_artifacts"][0]["status"] == "complete"

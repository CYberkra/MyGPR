#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the read-only GX-008 gprMax model inspector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.gprmax_model_inspector import (
    clone_scene_as_draft,
    load_scene_model,
    save_scene_draft,
    validate_scene_draft,
)


ROOT = Path(__file__).resolve().parents[1]
GX008_MODELS = ROOT / "experiments" / "gprmax" / "GX-008" / "models"


SCENE_037 = "scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate"
SCENE_038 = "scene_038_gssi_ey_depth07_radius03_air_sand_interface_n80_pair_gate"


def test_parses_gx008_scene037_pair_contract():
    model = load_scene_model(SCENE_037)

    assert model.scene_id == SCENE_037
    assert model.domain == "1.0 0.15 0.40"
    assert model.expected_num_runs == 80
    assert model.pair_contract_status == "pairable"
    assert model.pair_contract_checks["raw_has_target"] is True
    assert model.pair_contract_checks["background_has_no_target"] is True
    assert "scripts\\run_gprmax_gpu_env.bat" in model.generated_gpu_command


def test_parses_scene038_depth_supplement_material_and_roi():
    model = load_scene_model(SCENE_038)

    assert model.target_material == "pec"
    assert model.expected_num_runs == 80
    assert model.roi["trace_range"] == [20, 65]
    assert "dry_sand_tableii" in model.materials_text


def test_missing_scene_files_return_warnings(tmp_path: Path):
    model = load_scene_model("scene_missing", models_root=tmp_path)

    assert model.scene_id == "scene_missing"
    assert model.pair_contract_status == "warning"
    assert any("missing file" in warning for warning in model.warnings)
    assert any("missing json" in warning for warning in model.warnings)


def test_validate_scene_draft_returns_checks():
    result = validate_scene_draft(SCENE_037)

    assert result["status"] == "pairable"
    assert result["checks"]["domain_same"] is True
    assert result["checks"]["background_has_no_target"] is True


def test_model_editor_v0_does_not_write_files():
    with pytest.raises(PermissionError):
        clone_scene_as_draft(SCENE_037)
    with pytest.raises(PermissionError):
        save_scene_draft(SCENE_037)


def test_invalid_roi_json_warns_without_crash(tmp_path: Path):
    scene = tmp_path / "scene_bad"
    scene.mkdir()
    raw = GX008_MODELS / SCENE_037 / "raw_with_target.in"
    background = GX008_MODELS / SCENE_037 / "background_only.in"
    (scene / "raw_with_target.in").write_text(raw.read_text(encoding="utf-8"), encoding="utf-8")
    (scene / "background_only.in").write_text(background.read_text(encoding="utf-8"), encoding="utf-8")
    (scene / "materials.txt").write_text("material_name,eps_r\npvc_like,3.5\n", encoding="utf-8")
    (scene / "roi_draft.json").write_text("{bad", encoding="utf-8")
    (scene / "scene_manifest_draft.json").write_text(
        json.dumps({"expected_num_runs": 80, "target_material": "pec"}),
        encoding="utf-8",
    )

    model = load_scene_model("scene_bad", models_root=tmp_path)

    assert any("malformed json" in warning for warning in model.warnings)
    assert model.pair_contract_status == "pairable"

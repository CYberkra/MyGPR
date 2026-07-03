#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax campaign dry-run validator."""

from __future__ import annotations

from pathlib import Path

from core.gprmax_campaign.campaign_loader import load_campaign_yaml
from core.gprmax_campaign.validator import validate_campaign


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "gprmax_campaign"


def _issues_by_code(scene_result):
    return {item.code for item in scene_result.issues}


def test_validate_valid_campaign_ready():
    campaign = load_campaign_yaml(FIXTURE_DIR / "campaign_valid.yaml")
    result = validate_campaign(campaign)
    assert result.status == "ready"
    assert result.total_scenes == 1
    assert result.ready_count == 1
    assert result.invalid_count == 0


def test_validate_missing_background_is_invalid():
    campaign = load_campaign_yaml(FIXTURE_DIR / "campaign_missing_background.yaml")
    result = validate_campaign(campaign)
    assert result.status == "invalid"
    assert result.invalid_count == 1
    scene = result.scenes[0]
    assert scene.status == "invalid"
    assert "background_model_missing" in _issues_by_code(scene)


def test_validate_duplicate_scene_ids_detected():
    campaign = load_campaign_yaml(FIXTURE_DIR / "campaign_duplicate_scene.yaml")
    result = validate_campaign(campaign)
    assert result.status == "invalid"
    assert result.total_scenes == 2
    assert result.invalid_count == 2
    assert all("scene_id_duplicate" in _issues_by_code(scene) for scene in result.scenes)


def test_validate_missing_expected_outputs_detected():
    campaign = load_campaign_yaml(FIXTURE_DIR / "campaign_missing_expected_outputs.yaml")
    result = validate_campaign(campaign)
    assert result.status == "invalid"
    assert "expected_outputs_missing" in _issues_by_code(result.scenes[0])


def test_output_root_validation_does_not_create_directory(tmp_path):
    campaign_yaml = tmp_path / "campaign.yaml"
    models = tmp_path / "models"
    annotations = tmp_path / "annotations"
    models.mkdir()
    annotations.mkdir()
    (models / "raw_with_target.in").write_text("# raw\n", encoding="utf-8")
    (models / "background_only.in").write_text("# bg\n", encoding="utf-8")
    (models / "materials.txt").write_text("soil 6 0 1 0\n", encoding="utf-8")
    (annotations / "target_roi.json").write_text("{}", encoding="utf-8")
    output_root = tmp_path / "new_output_root" / "scene_pack"
    campaign_yaml.write_text(
        "\n".join(
            [
                "campaign_id: GX-TMP",
                f"output_root: {output_root.as_posix()}",
                "gprmax_executable: gprMax",
                "scenes:",
                "  - scene_id: s1",
                "    raw_model: models/raw_with_target.in",
                "    background_model: models/background_only.in",
                "    materials: models/materials.txt",
                "    target_roi: annotations/target_roi.json",
                "    expected_outputs:",
                "      - raw_with_target",
                "      - background_only",
                "      - target_response",
                "    tags: [tmp]",
            ]
        ),
        encoding="utf-8",
    )
    assert not output_root.exists()
    campaign = load_campaign_yaml(campaign_yaml)
    result = validate_campaign(campaign)
    assert result.status == "ready"
    assert not output_root.exists()

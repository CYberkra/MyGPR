#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax campaign YAML loader."""

from __future__ import annotations

from pathlib import Path

from core.gprmax_campaign.campaign_loader import load_campaign_yaml


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "gprmax_campaign"


def test_load_valid_campaign_yaml():
    campaign = load_campaign_yaml(FIXTURE_DIR / "campaign_valid.yaml")
    assert campaign.campaign_id == "GX-RUN-001_valid"
    assert campaign.gprmax_executable == "gprMax"
    assert len(campaign.scenes) == 1
    assert campaign.scenes[0].scene_id == "scene_valid_01"


def test_loader_resolves_relative_paths_from_yaml_directory():
    campaign = load_campaign_yaml(FIXTURE_DIR / "campaign_valid.yaml")
    scene = campaign.scenes[0]
    assert scene.raw_model == (FIXTURE_DIR / "models" / "raw_with_target.in").resolve()
    assert scene.background_model == (
        FIXTURE_DIR / "models" / "background_only.in"
    ).resolve()
    assert scene.materials == (FIXTURE_DIR / "models" / "materials.txt").resolve()
    assert scene.target_roi == (
        FIXTURE_DIR / "annotations" / "target_roi.json"
    ).resolve()


def test_loader_allows_missing_expected_outputs_for_validator_stage():
    campaign = load_campaign_yaml(FIXTURE_DIR / "campaign_missing_expected_outputs.yaml")
    assert len(campaign.scenes) == 1
    assert campaign.scenes[0].scene_id == "scene_missing_expected_outputs"
    assert campaign.scenes[0].expected_outputs is None

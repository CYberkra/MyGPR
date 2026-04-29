#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motion compensation registry, preset, and config validation tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.methods_registry import (
    PROCESSING_METHODS,
    METHOD_METADATA,
    METHOD_CATEGORY_LABELS,
    AUTO_TUNE_STAGE_BY_METHOD,
    get_public_method_keys,
    get_method_category,
    is_public_method,
)
from core.workflow_data import METHOD_CATEGORIES, QUICK_PRESETS, WorkflowConfig
from core.preset_profiles import RECOMMENDED_RUN_PROFILES

# All five V1 motion stages
V1_MOTION_METHODS = [
    "trajectory_smoothing",
    "motion_compensation_speed",
    "motion_compensation_attitude",
    "motion_compensation_height",
    "motion_compensation_vibration",
]

BASE_DIR = Path(__file__).parent.parent


def test_motion_methods_registered_public_and_have_params():
    """All five V1 stages are visible, categorized, and have parameter definitions."""
    public_keys = get_public_method_keys()
    for key in V1_MOTION_METHODS:
        assert key in PROCESSING_METHODS, f"{key} not in PROCESSING_METHODS"
        assert is_public_method(key), f"{key} is not public"
        assert key in public_keys, f"{key} not in public method keys"

        params = PROCESSING_METHODS[key].get("params", [])
        assert len(params) > 0, f"{key} has no parameter definitions"

        category = get_method_category(key)
        assert category == "motion_compensation", f"{key} category is {category}, expected motion_compensation"

        assert key in METHOD_METADATA, f"{key} not in METHOD_METADATA"
        assert METHOD_METADATA[key]["visibility"] == "public"
        assert METHOD_METADATA[key]["maturity"] == "experimental"


def test_motion_compensation_category_exists():
    """The motion_compensation category is defined with all five methods."""
    assert "motion_compensation" in METHOD_CATEGORIES
    cat = METHOD_CATEGORIES["motion_compensation"]
    assert cat["name"] == "运动补偿"
    for key in V1_MOTION_METHODS:
        assert key in cat["methods"], f"{key} not in motion_compensation category methods"

    assert "motion_compensation" in METHOD_CATEGORY_LABELS
    assert METHOD_CATEGORY_LABELS["motion_compensation"] == "运动补偿"


def test_auto_tune_stage_assigned_for_all_motion_methods():
    """All motion methods map to the motion_comp auto-tune stage."""
    for key in V1_MOTION_METHODS:
        assert AUTO_TUNE_STAGE_BY_METHOD.get(key) == "motion_comp", f"{key} auto_tune_stage mismatch"
        assert PROCESSING_METHODS[key].get("auto_tune_family") == "motion_comp"
        assert PROCESSING_METHODS[key].get("auto_tune_enabled") is True


def test_motion_compensation_v1_quick_preset_exists():
    """The motion_compensation_v1 quick preset is defined with correct sequencing."""
    assert "motion_compensation_v1" in QUICK_PRESETS
    preset = QUICK_PRESETS["motion_compensation_v1"]
    assert preset["name"] == "运动补偿 V1"

    method_ids = [m["method_id"] for m in preset["methods"]]
    assert method_ids == V1_MOTION_METHODS

    for m in preset["methods"]:
        assert m["category"] == "motion_compensation"
        assert m["enabled"] is True


def test_motion_compensation_v1_recommended_profile_exists():
    """The recommended run profile exists and sequences the deterministic V1 stages only."""
    assert "motion_compensation_v1" in RECOMMENDED_RUN_PROFILES
    profile = RECOMMENDED_RUN_PROFILES["motion_compensation_v1"]
    assert profile["label"] == "运动补偿 V1"
    assert profile["order"] == V1_MOTION_METHODS

    # Ensure no experimental/non-V1 methods sneak into the default preset
    forbidden = {"autofocus", "dem_coupling", "antenna_pattern_inversion", "rpm_notch"}
    order_lower = " ".join(profile["order"]).lower()
    for f in forbidden:
        assert f not in order_lower, f"forbidden keyword {f} found in profile order"


def test_motion_compensation_v1_preset_applies_to_workflow_config():
    """Applying the preset to a WorkflowConfig produces the expected methods."""
    cfg = WorkflowConfig()
    ok = cfg.apply_preset("motion_compensation_v1")
    assert ok is True
    enabled = cfg.get_enabled_methods()
    assert len(enabled) == 5
    assert [m.method_id for m in enabled] == V1_MOTION_METHODS


def test_cli_config_validates(tmp_path: Path):
    """CLI config file validates against the new benchmark preset."""
    import cli_batch

    config_path = BASE_DIR / "config" / "motion_compensation_v1_benchmark.json"
    assert config_path.exists(), f"config file not found: {config_path}"

    cfg = cli_batch.load_config(str(config_path))
    result = cli_batch.validate_config(cfg, repo_root=str(tmp_path))
    assert result.ok is True, f"validation failed: {result.errors}"
    assert result.errors == []


def test_motion_methods_have_reasonable_defaults():
    """Parameter defaults fall within advertised min/max ranges."""
    for key in V1_MOTION_METHODS:
        for p in PROCESSING_METHODS[key].get("params", []):
            name = p["name"]
            default = p.get("default")
            if default is None:
                continue
            if "min" in p and default < p["min"]:
                pytest.fail(f"{key}.{name} default {default} < min {p['min']}")
            if "max" in p and default > p["max"]:
                pytest.fail(f"{key}.{name} default {default} > max {p['max']}")

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
from PythonModule.motion_compensation_core import AIR_WAVE_SPEED_M_PER_NS

CORE_MOTION_METHODS = [
    "trajectory_smoothing",
    "motion_compensation_speed",
    "motion_compensation_attitude",
    "motion_compensation_height",
]
VIBRATION_METHOD = "motion_compensation_vibration"

BASE_DIR = Path(__file__).parent.parent


def test_motion_methods_registered_public_and_have_params():
    """The four public atomic nodes are visible and use the motion category."""
    public_keys = get_public_method_keys()
    for key in CORE_MOTION_METHODS:
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
    """The motion_compensation category is defined with core methods only."""
    assert "motion_compensation" in METHOD_CATEGORIES
    cat = METHOD_CATEGORIES["motion_compensation"]
    assert cat["name"] == "运动补偿"
    for key in CORE_MOTION_METHODS:
        assert key in cat["methods"], f"{key} not in motion_compensation category methods"
    assert VIBRATION_METHOD not in cat["methods"]

    assert "motion_compensation" in METHOD_CATEGORY_LABELS
    assert METHOD_CATEGORY_LABELS["motion_compensation"] == "运动补偿"
    assert METHOD_METADATA[VIBRATION_METHOD]["category"] == "artifact_suppression"
    assert METHOD_METADATA[VIBRATION_METHOD]["display_name"] == "周期条带伪影抑制（实验）"


def test_auto_tune_stage_assigned_for_all_motion_methods():
    """Core motion methods map to motion_comp; vibration is artifact suppression."""
    for key in CORE_MOTION_METHODS:
        assert AUTO_TUNE_STAGE_BY_METHOD.get(key) == "motion_comp", f"{key} auto_tune_stage mismatch"
        assert PROCESSING_METHODS[key].get("auto_tune_family") == "motion_comp"
        assert PROCESSING_METHODS[key].get("auto_tune_enabled") is True
    assert AUTO_TUNE_STAGE_BY_METHOD.get(VIBRATION_METHOD) == "artifact"
    assert PROCESSING_METHODS[VIBRATION_METHOD].get("auto_tune_family") == "denoise"


def test_motion_compensation_v1_quick_preset_exists():
    """The legacy preset key now routes to the four V2-core atomic nodes."""
    assert "motion_compensation_v1" in QUICK_PRESETS
    preset = QUICK_PRESETS["motion_compensation_v1"]
    assert "V1" not in preset["name"]
    assert "Legacy" not in preset["name"]

    method_ids = [m["method_id"] for m in preset["methods"]]
    assert method_ids == CORE_MOTION_METHODS

    for m in preset["methods"]:
        assert m["category"] == "motion_compensation"
        assert m["enabled"] is True


def test_motion_compensation_v1_recommended_profile_exists():
    """Compatibility profiles sequence only the four V2-core atomic nodes."""
    assert "motion_compensation_v1" in RECOMMENDED_RUN_PROFILES
    profile = RECOMMENDED_RUN_PROFILES["motion_compensation_v1"]
    assert "V1" not in profile["label"]
    assert "Legacy" not in profile["label"]
    assert profile["order"] == CORE_MOTION_METHODS

    assert "motion_compensation_core_v1" in RECOMMENDED_RUN_PROFILES
    core_profile = RECOMMENDED_RUN_PROFILES["motion_compensation_core_v1"]
    assert "V1" not in core_profile["label"]
    assert "Legacy" not in core_profile["label"]
    assert core_profile["order"] == CORE_MOTION_METHODS
    assert VIBRATION_METHOD not in core_profile["order"]

    # Ensure no experimental/non-V1 methods sneak into the default preset
    forbidden = {"autofocus", "dem_coupling", "antenna_pattern_inversion", "rpm_notch"}
    order_lower = " ".join(profile["order"]).lower()
    for f in forbidden:
        assert f not in order_lower, f"forbidden keyword {f} found in profile order"


def test_motion_compensation_v1_preset_applies_to_workflow_config():
    """Applying compatibility presets yields only the four atomic motion nodes."""
    cfg = WorkflowConfig()
    ok = cfg.apply_preset("motion_compensation_v1")
    assert ok is True
    enabled = cfg.get_enabled_methods()
    assert len(enabled) == 4
    assert [m.method_id for m in enabled] == CORE_MOTION_METHODS

    cfg = WorkflowConfig()
    ok = cfg.apply_preset("motion_compensation_core_v1")
    assert ok is True
    enabled = cfg.get_enabled_methods()
    assert len(enabled) == 4
    assert [m.method_id for m in enabled] == CORE_MOTION_METHODS


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
    for key in CORE_MOTION_METHODS + [VIBRATION_METHOD]:
        for p in PROCESSING_METHODS[key].get("params", []):
            name = p["name"]
            default = p.get("default")
            if default is None:
                continue
            if "min" in p and default < p["min"]:
                pytest.fail(f"{key}.{name} default {default} < min {p['min']}")
            if "max" in p and default > p["max"]:
                pytest.fail(f"{key}.{name} default {default} > max {p['max']}")


def test_atomic_motion_nodes_use_v2_core_defaults_and_no_legacy_ui():
    """Atomic motion nodes remain public but no user-facing Legacy/V1 node is exposed."""
    public_keys = get_public_method_keys()
    for key in CORE_MOTION_METHODS:
        func = PROCESSING_METHODS[key]["func"]
        assert "legacy" not in getattr(func, "__module__", "").lower()

    height_params = {
        p["name"]: p for p in PROCESSING_METHODS["motion_compensation_height"]["params"]
    }
    assert height_params["wave_speed_m_per_ns"]["default"] == pytest.approx(
        AIR_WAVE_SPEED_M_PER_NS
    )
    assert height_params["height_source"]["default"] == "auto"
    assert set(height_params["height_source"]["choices"]) == {
        "auto",
        "height_agl_m",
        "flight_height_m",
    }

    user_visible_names = " ".join(
        str(PROCESSING_METHODS[key].get("name", "")) for key in public_keys
    )
    assert "Legacy" not in user_visible_names
    assert "兼容旧基准" not in user_visible_names


def test_motion_v2_frontend_params_are_strategy_not_sensor_arrays():
    """V2 UI exposes strategy controls; per-trace sensor values must come from trace_metadata."""
    method = PROCESSING_METHODS["motion_compensation_v2"]
    param_names = {p["name"] for p in method.get("params", [])}
    expected_strategy_params = {
        "height_reference_mode",
        "manual_height_m",
        "height_source",
        "compensate_time_shift",
        "compensate_amplitude",
        "max_shift_samples",
        "max_shift_ns",
        "max_amplitude_scale",
        "resample_spacing_m",
        "apc_offset_x_m",
        "apc_offset_y_m",
        "apc_offset_z_m",
        "max_abs_tilt_deg",
    }
    forbidden_manual_sensor_params = {
        "height_agl_m",
        "flight_height_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "local_x_m",
        "local_y_m",
        "trace_timestamp_s",
        "timestamp_s",
        "trace_distance_m",
    }

    assert param_names == expected_strategy_params
    assert param_names.isdisjoint(forbidden_manual_sensor_params)
    assert "trace_metadata" in method.get("description", "")

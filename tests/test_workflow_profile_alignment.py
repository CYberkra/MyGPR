#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow/profile alignment regression tests."""

from __future__ import annotations

from core.preset_profiles import (
    RECOMMENDED_RUN_PROFILES,
    WORKFLOW_PRESETS,
    build_profile_workflow_summary,
)


def test_high_quality_workflow_stage_grouping_matches_run_order():
    summary = build_profile_workflow_summary("high_quality_uav_gpr")

    flattened = [
        method_key
        for stage in summary["stages"]
        for method_key in stage["method_keys"]
    ]
    stage2 = next(
        stage for stage in summary["stages"] if stage["stage_key"] == "stage2"
    )
    stage4 = next(
        stage for stage in summary["stages"] if stage["stage_key"] == "stage4"
    )

    assert flattened == RECOMMENDED_RUN_PROFILES["high_quality_uav_gpr"]["order"]
    assert summary["unassigned_methods"] == []
    assert stage2["method_keys"] == [
        "frequency_filter_1d",
        "motion_compensation_v2",
        "subtracting_average_2D",
    ]
    assert stage4["method_keys"] == [
        "manual_velocity_model",
        "geometry_depth_context",
        "sec_gain",
    ]
    assert "fk_filter" not in flattened
    assert (
        WORKFLOW_PRESETS["high_quality_uav_gpr"]["stages"]["stage2"][
            "motion_compensation_v2"
        ]
        is True
    )
    assert (
        WORKFLOW_PRESETS["high_quality_uav_gpr"]["stages"]["stage4"][
            "geometry_depth_context"
        ]
        is True
    )


def test_high_quality_workflow_summary_records_motion_sensor_dependency():
    summary = build_profile_workflow_summary("high_quality_uav_gpr")

    warning = next(
        item
        for item in summary["sensor_dependency_warnings"]
        if item["method_key"] == "motion_compensation_v2"
    )

    assert warning["code"] == "motion_compensation_v2_sensor_dependency"
    assert "height_agl_m" in warning["required_any"]
    assert "flight_height_m" in warning["required_any"]
    assert "roll_deg" in warning["optional"]

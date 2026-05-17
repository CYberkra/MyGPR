#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow registry contract tests."""

from __future__ import annotations

from core.workflow_data import WorkflowMethod
from core.workflow_registry import (
    MYGPR_STANDARD_METHOD_ORDER,
    OUTPUT_EFFECT_EMPTY_MESSAGES,
    OUTPUT_EFFECT_LABELS,
    OUTPUT_EFFECT_TITLES,
    candidate_methods_for_workflow_method,
    default_params_for_method,
    quick_preset_spec,
    recommended_next_methods_for,
    validate_workflow_registry,
    workflow_port_labels,
    workflow_registry_snapshot,
)


def test_workflow_registry_has_no_contract_errors():
    issues = validate_workflow_registry()
    assert [issue for issue in issues if issue.severity == "error"] == []

    snapshot = workflow_registry_snapshot()
    assert snapshot.stage_count >= 8
    assert snapshot.quick_preset_count >= 3
    assert "bscan" in snapshot.output_effect_kinds


def test_mygpr_standard_registry_contract_is_original_five_step_chain():
    spec = quick_preset_spec("mygpr_standard")

    assert spec.name == "MyGPR 标准流程"
    assert list(spec.method_ids) == MYGPR_STANDARD_METHOD_ORDER
    assert list(spec.stage_ids) == [
        "zero_time",
        "trace_correction",
        "background_clutter",
        "gain",
        "spatial_denoise",
    ]


def test_gain_stage_candidates_and_defaults_are_registry_visible():
    method = WorkflowMethod(
        category="gain",
        stage_id="gain",
        method_id="sec_gain",
        params={"gain_min": 1.0},
    )

    candidates = candidate_methods_for_workflow_method(method)
    assert {"sec_gain", "agcGain", "energy_decay_gain", "compensatingGain"}.issubset(
        set(candidates)
    )
    assert "window" in default_params_for_method("agcGain")


def test_output_drop_recommendations_are_core_contract_not_canvas_local():
    assert recommended_next_methods_for(
        WorkflowMethod("preprocessing", "set_zero_time", stage_id="zero_time")
    ) == ["dewow", "frequency_filter_1d"]
    assert "sec_gain" in recommended_next_methods_for(
        WorkflowMethod(
            "background_removal",
            "subtracting_average_2D",
            stage_id="background_clutter",
        )
    )
    assert "svd_subspace" in recommended_next_methods_for(
        WorkflowMethod("gain", "agcGain", stage_id="gain")
    )


def test_output_effect_metadata_is_complete_and_user_visible():
    expected = {"bscan", "compare", "qc", "spectrum", "evidence"}

    assert set(OUTPUT_EFFECT_LABELS) == expected
    assert set(OUTPUT_EFFECT_TITLES) == expected
    assert set(OUTPUT_EFFECT_EMPTY_MESSAGES) == expected
    assert OUTPUT_EFFECT_LABELS["bscan"] == "查看此步 B-scan"
    assert OUTPUT_EFFECT_TITLES["compare"] == "前后对比"
    assert OUTPUT_EFFECT_TITLES["spectrum"] == "频谱视图"


def test_port_label_mapping_distinguishes_velocity_preview_and_evidence():
    assert workflow_port_labels(
        WorkflowMethod("migration", "manual_velocity_model", stage_id="velocity_model")
    ) == ("data", "velocity")
    assert workflow_port_labels(
        WorkflowMethod("preview", "bscan_preview", stage_id="preview")
    ) == ("data", "preview")
    assert workflow_port_labels(
        WorkflowMethod("evidence", "export_package", stage_id="export")
    ) == ("data", "evidence")

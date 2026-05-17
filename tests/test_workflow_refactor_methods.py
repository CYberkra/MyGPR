#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow refactor method and config tests."""

from __future__ import annotations

import numpy as np
import pytest

from core.processing_engine import (
    merge_result_header_info,
    prepare_runtime_params,
    run_processing_method,
)
from core.workflow_data import build_default_workflow_config
from core.workflow_data import WorkflowConfig, WorkflowMethod


def test_dc_shift_mean_and_median_keep_shape_and_remove_offsets():
    data = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 12.0, 100.0],
            [3.0, 14.0, 400.0],
        ],
        dtype=np.float32,
    )

    mean_result, mean_meta = run_processing_method(
        data, "dc_shift", {"estimator": "mean", "scope": "per_trace"}
    )
    median_result, median_meta = run_processing_method(
        data, "dc_shift", {"estimator": "median", "scope": "per_trace"}
    )

    assert mean_result.shape == data.shape
    assert median_result.shape == data.shape
    assert np.allclose(np.mean(mean_result, axis=0), 0.0)
    assert np.allclose(np.median(median_result, axis=0), 0.0)
    assert mean_meta["estimator"] == "mean"
    assert median_meta["estimator"] == "median"


def test_dc_shift_non_finite_input_reports_warning():
    data = np.array([[1.0, np.nan], [2.0, np.inf]], dtype=np.float32)

    result, meta = run_processing_method(
        data, "dc_shift", {"estimator": "mean", "scope": "global"}
    )

    assert np.isfinite(result).all()
    assert any(
        warning.get("code") == "data_sanitized"
        for warning in meta.get("runtime_warnings", [])
    )


def test_manual_velocity_model_writes_header_updates():
    data = np.ones((8, 4), dtype=np.float32)

    result, meta = run_processing_method(
        data,
        "manual_velocity_model",
        {
            "mode": "dielectric",
            "epsilon_r": 9.0,
            "velocity_m_per_ns": 0.10,
            "uncertainty_fraction": 0.2,
        },
    )
    merged = merge_result_header_info({}, meta, result.shape)

    assert np.array_equal(result, data)
    assert np.isclose(merged["velocity_m_per_ns"], 0.299792458 / 3.0)
    assert merged["velocity_model"]["epsilon_r"] == 9.0
    assert merged["velocity_model"]["uncertainty_fraction"] == 0.2


def test_manual_velocity_model_rejects_non_finite_parameters():
    data = np.ones((8, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="有限值"):
        run_processing_method(
            data,
            "manual_velocity_model",
            {"mode": "dielectric", "epsilon_r": np.inf},
        )

    with pytest.raises(ValueError, match="有限值"):
        run_processing_method(
            data,
            "manual_velocity_model",
            {"mode": "velocity", "velocity_m_per_ns": np.nan},
        )

    with pytest.raises(ValueError, match="非负有限值"):
        run_processing_method(
            data,
            "manual_velocity_model",
            {"uncertainty_fraction": np.inf},
        )


def test_geometry_depth_context_warns_for_missing_inputs_and_resolves_spacing():
    data = np.ones((8, 4), dtype=np.float32)
    params = prepare_runtime_params(
        "geometry_depth_context",
        {
            "require_velocity_model": True,
            "require_trace_spacing": True,
            "require_time_window": True,
            "require_agl": False,
        },
        {"total_time_ns": 80.0},
        {"trace_distance_m": np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)},
        data.shape,
    )

    result, meta = run_processing_method(data, "geometry_depth_context", params)
    context = meta["geometry_depth_context"]

    assert np.array_equal(result, data)
    assert context["trace_interval_m"] == 0.5
    assert context["time_window_ns"] == 80.0
    assert any(
        warning.get("code") == "missing_velocity_model"
        for warning in meta.get("runtime_warnings", [])
    )


def test_kirchhoff_runtime_length_ignores_invalid_track_length():
    params = prepare_runtime_params(
        "kirchhoff_migration",
        {},
        {"track_length_m": "bad", "trace_interval_m": 0.5},
        None,
        (8, 4),
    )

    assert params["length_m"] == 1.5


def test_default_workflow_contains_gain_candidates_and_hidden_migration():
    config = build_default_workflow_config("high_quality_uav_gpr")
    method_ids = [method.method_id for method in config.methods]
    enabled_ids = [method.method_id for method in config.get_enabled_methods()]
    migration = next(method for method in config.methods if method.method_id == "kirchhoff_migration")

    assert method_ids[:3] == ["set_zero_time", "dc_shift", "dewow"]
    assert "manual_velocity_model" in enabled_ids
    assert "geometry_depth_context" in enabled_ids
    assert "sec_gain" in enabled_ids
    assert migration.hidden is True
    assert migration.method_id not in enabled_ids


def test_workflow_config_roundtrip_preserves_realtime_stage_and_hidden_flags():
    config = WorkflowConfig(
        name="实时实验模板",
        template_type="user",
        realtime_enabled=True,
        methods=[
            WorkflowMethod(
                category="preprocessing",
                stage_id="trace_correction",
                method_id="dc_shift",
                enabled=True,
                order=0,
                params={"estimator": "median", "scope": "per_trace"},
            ),
            WorkflowMethod(
                category="migration",
                stage_id="migration",
                method_id="kirchhoff_migration",
                enabled=True,
                hidden=True,
                order=1,
                params={"velocity": 0.1},
            ),
        ],
    )

    restored = WorkflowConfig.from_dict(config.to_dict())

    assert restored.name == "实时实验模板"
    assert restored.template_type == "user"
    assert restored.realtime_enabled is True
    assert restored.methods[0].stage_id == "trace_correction"
    assert restored.methods[0].node_id
    assert restored.canvas_links
    assert restored.canvas_links[0].from_node == restored.methods[0].node_id
    assert restored.canvas_links[0].to_node == restored.methods[1].node_id
    assert restored.methods[0].params["estimator"] == "median"
    assert restored.methods[1].stage_id == "migration"
    assert restored.methods[1].hidden is True
    assert [method.method_id for method in restored.get_enabled_methods()] == [
        "dc_shift"
    ]

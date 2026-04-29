#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic five-error motion benchmark registry tests."""

from __future__ import annotations

import numpy as np

from core.benchmark_registry import BENCHMARK_SCENARIOS, generate_benchmark_sample
from core.quality_metrics import periodic_banding_ratio


def test_motion_v1_benchmark_returns_required_context_and_is_deterministic():
    data_a, context_a = generate_benchmark_sample("motion_compensation_v1", seed=42)
    data_b, context_b = generate_benchmark_sample("motion_compensation_v1", seed=42)

    assert np.array_equal(data_a, data_b)
    assert context_a["scenario"] == "motion_compensation"
    assert context_a["scenario"] in BENCHMARK_SCENARIOS
    assert context_a["sample_id"] == "motion_compensation_v1"
    assert set(("header_info", "trace_metadata", "ground_truth_trace_metadata", "expected_metrics")).issubset(
        context_a.keys()
    )
    assert context_a["header_info"]["a_scan_length"] == data_a.shape[0]
    assert context_a["header_info"]["num_traces"] == data_a.shape[1]
    assert context_a["default_methods"] == [
        "trajectory_smoothing",
        "motion_compensation_height",
    ]
    assert np.array_equal(
        context_a["trace_metadata"]["trace_distance_m"],
        context_b["trace_metadata"]["trace_distance_m"],
    )
    assert np.array_equal(
        context_a["ground_truth_trace_metadata"]["reflector_ridge_idx"],
        context_b["ground_truth_trace_metadata"]["reflector_ridge_idx"],
    )


def test_motion_v1_benchmark_injects_all_five_errors():
    data, context = generate_benchmark_sample("motion_compensation_v1", seed=42)
    trace_metadata = context["trace_metadata"]
    ground_truth = context["ground_truth_trace_metadata"]
    injected = context["injected_errors"]
    metric_config = context["expected_metrics"]["metric_config"]

    height_delta = trace_metadata["flight_height_m"] - ground_truth["flight_height_m"]
    lateral_delta = trace_metadata["local_y_m"] - ground_truth["local_y_m"]
    spacing_delta = np.diff(trace_metadata["trace_distance_m"] - ground_truth["trace_distance_m"])
    footprint_delta = np.sqrt(
        (trace_metadata["footprint_x_m"] - ground_truth["footprint_x_m"]) ** 2
        + (trace_metadata["footprint_y_m"] - ground_truth["footprint_y_m"]) ** 2
    )

    assert np.max(np.abs(height_delta)) > 0.12
    assert np.max(np.abs(lateral_delta)) > 0.35
    assert np.std(spacing_delta) > 0.05
    assert np.max(np.abs(trace_metadata["roll_deg"] - ground_truth["roll_deg"])) > 4.0
    assert np.max(np.abs(trace_metadata["pitch_deg"] - ground_truth["pitch_deg"])) > 3.0
    assert np.max(np.abs(trace_metadata["yaw_deg"] - ground_truth["yaw_deg"])) > 5.0
    assert np.mean(footprint_delta) > 1.0
    assert np.max(np.abs(injected["height_shift_samples"])) > 2.0
    assert np.mean(injected["footprint_error_m"]) > 1.0
    assert periodic_banding_ratio(
        data,
        trace_band=tuple(metric_config["banding_trace_band"]),
        row_range=tuple(metric_config["banding_row_range"]),
    ) > context["expected_metrics"]["minimum_uncompensated"]["periodic_banding_ratio"]

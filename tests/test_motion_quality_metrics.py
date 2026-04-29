#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Objective motion quality metric regression tests."""

from __future__ import annotations

import pytest

from core.benchmark_registry import generate_benchmark_sample
from core.quality_metrics import compute_motion_quality_metrics


def _benchmark_metrics_payload(seed: int = 42):
    data, context = generate_benchmark_sample("motion_compensation_v1", seed=seed)
    config = context["expected_metrics"]["metric_config"]
    metrics = compute_motion_quality_metrics(
        data,
        context["trace_metadata"],
        context["ground_truth_trace_metadata"],
        ground_truth_data=context["ground_truth_data"],
        ridge_row_range=tuple(config["ridge_row_range"]),
        target_row_range=tuple(config["target_row_range"]),
        banding_trace_band=tuple(config["banding_trace_band"]),
        banding_row_range=tuple(config["banding_row_range"]),
    )
    return data, context, metrics


def test_motion_metrics_detect_raw_errors():
    _, context, metrics = _benchmark_metrics_payload(seed=42)
    expected = context["expected_metrics"]

    for key, value in expected["raw"].items():
        assert metrics[key] == pytest.approx(value, rel=1e-9, abs=1e-9)

    assert metrics["raw_ridge_rmse_samples"] > expected["minimum_uncompensated"]["raw_ridge_rmse_samples"]
    assert metrics["trace_spacing_std_m"] > expected["minimum_uncompensated"]["trace_spacing_std_m"]
    assert metrics["path_rmse_m"] > expected["minimum_uncompensated"]["path_rmse_m"]
    assert metrics["footprint_rmse_m"] > expected["minimum_uncompensated"]["footprint_rmse_m"]
    assert metrics["periodic_banding_ratio"] > expected["minimum_uncompensated"]["periodic_banding_ratio"]
    assert metrics["target_preservation_ratio"] > expected["minimum_uncompensated"]["target_preservation_ratio"]


def test_motion_metrics_score_ground_truth_as_better_than_raw():
    _, context, raw_metrics = _benchmark_metrics_payload(seed=42)
    config = context["expected_metrics"]["metric_config"]
    reference_metrics = compute_motion_quality_metrics(
        context["ground_truth_data"],
        context["ground_truth_trace_metadata"],
        context["ground_truth_trace_metadata"],
        ground_truth_data=context["ground_truth_data"],
        ridge_row_range=tuple(config["ridge_row_range"]),
        target_row_range=tuple(config["target_row_range"]),
        banding_trace_band=tuple(config["banding_trace_band"]),
        banding_row_range=tuple(config["banding_row_range"]),
    )

    assert reference_metrics["raw_ridge_rmse_samples"] == pytest.approx(0.0)
    assert reference_metrics["path_rmse_m"] == pytest.approx(0.0)
    assert reference_metrics["footprint_rmse_m"] == pytest.approx(0.0)
    assert reference_metrics["target_preservation_ratio"] == pytest.approx(1.0)
    assert reference_metrics["raw_ridge_rmse_samples"] < raw_metrics["raw_ridge_rmse_samples"]
    assert reference_metrics["periodic_banding_ratio"] < raw_metrics["periodic_banding_ratio"]

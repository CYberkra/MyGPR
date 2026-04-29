#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for deterministic vibration-safe motion compensation."""

from __future__ import annotations

import copy

import numpy as np

from PythonModule.motion_compensation_vibration import (  # type: ignore[import]
    method_motion_compensation_vibration,
)
from core.benchmark_registry import generate_benchmark_sample
from core.quality_metrics import periodic_banding_ratio, target_preservation_ratio


def _relative_reduction(before: float, after: float) -> float:
    """Return normalized improvement for lower-is-better metrics."""
    if before <= 0.0:
        return 0.0
    return float((before - after) / before)


def test_vibration_compensation_reduces_periodic_banding_without_rpm():
    """Benchmark striping should drop >= 30% while target energy stays >= 85%."""
    raw, context = generate_benchmark_sample("motion_compensation_v1", seed=42)
    trace_metadata = copy.deepcopy(context["trace_metadata"])
    metric_config = context["expected_metrics"]["metric_config"]

    corrected, meta = method_motion_compensation_vibration(
        raw,
        trace_metadata=trace_metadata,
    )

    raw_banding = periodic_banding_ratio(
        raw,
        trace_band=tuple(metric_config["banding_trace_band"]),
        row_range=tuple(metric_config["banding_row_range"]),
    )
    corrected_banding = periodic_banding_ratio(
        corrected,
        trace_band=tuple(metric_config["banding_trace_band"]),
        row_range=tuple(metric_config["banding_row_range"]),
    )
    target_ratio = target_preservation_ratio(
        corrected,
        context["ground_truth_data"],
        row_range=tuple(metric_config["target_row_range"]),
    )

    assert corrected.shape == raw.shape
    assert corrected.dtype == np.float32
    assert corrected is not raw
    assert meta["rpm_required"] is False
    assert meta["guidance_source"] == "radar_only_fallback"
    assert meta["fallback_used"] is True
    assert _relative_reduction(raw_banding, corrected_banding) >= 0.30
    assert target_ratio >= 0.85


def test_vibration_compensation_radar_only_fallback_is_explicit_and_deterministic():
    """Missing guidance metadata should still produce stable radar-only output."""
    raw, context = generate_benchmark_sample("motion_compensation_v1", seed=42)
    trace_metadata = {
        "trace_index": np.array(context["trace_metadata"]["trace_index"], copy=True),
        "time_window_ns": float(context["trace_metadata"]["time_window_ns"]),
    }
    trace_metadata_before = copy.deepcopy(trace_metadata)

    out_a, meta_a = method_motion_compensation_vibration(raw, trace_metadata=trace_metadata)
    out_b, meta_b = method_motion_compensation_vibration(raw, trace_metadata=trace_metadata)

    assert np.array_equal(out_a, out_b)
    assert meta_a["rpm_required"] is False
    assert meta_a["guidance_source"] == "radar_only_fallback"
    assert meta_a["fallback_used"] is True
    assert "trace_metadata_updates" not in meta_a
    assert meta_a["spectral_strength"] == meta_b["spectral_strength"]
    assert trace_metadata.keys() == trace_metadata_before.keys()
    assert np.array_equal(trace_metadata["trace_index"], trace_metadata_before["trace_index"])
    assert trace_metadata["time_window_ns"] == trace_metadata_before["time_window_ns"]


def test_vibration_compensation_optionally_uses_angular_rate_metadata_without_mutation():
    """Available angular rates should be smoothed in returned metadata copies only."""
    raw, context = generate_benchmark_sample("motion_compensation_v1", seed=42)
    traces = raw.shape[1]
    phase = np.linspace(0.0, 1.0, traces, dtype=np.float64)
    trace_metadata = copy.deepcopy(context["trace_metadata"])
    trace_metadata["angular_rate_x"] = 2.0 * np.sin(2.0 * np.pi * 8.0 * phase) + 0.3 * np.cos(
        2.0 * np.pi * 1.5 * phase
    )
    trace_metadata["angular_rate_y"] = 1.5 * np.cos(2.0 * np.pi * 7.0 * phase + 0.1)
    trace_metadata["angular_rate_z"] = 1.2 * np.sin(2.0 * np.pi * 6.5 * phase - 0.2)
    original_x = np.array(trace_metadata["angular_rate_x"], copy=True)

    corrected, meta = method_motion_compensation_vibration(raw, trace_metadata=trace_metadata)

    assert corrected.shape == raw.shape
    assert meta["guidance_source"] == "angular_rate_guided"
    assert meta["fallback_used"] is False
    assert meta["rpm_required"] is False
    assert set(("angular_rate_x", "angular_rate_y", "angular_rate_z")).issubset(
        meta["trace_metadata_updates"].keys()
    )
    assert np.array_equal(trace_metadata["angular_rate_x"], original_x)
    assert not np.array_equal(meta["trace_metadata_updates"]["angular_rate_x"], original_x)

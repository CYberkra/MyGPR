#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for motion_compensation_speed equal-distance resampling."""

from __future__ import annotations

import numpy as np

from PythonModule.motion_compensation_speed import method_motion_compensation_speed  # type: ignore[import]


def _copy_metadata(trace_metadata: dict) -> dict:
    return {key: np.array(value, copy=True) for key, value in trace_metadata.items()}


def _synthetic_bscan(samples: int, distance_m: np.ndarray) -> np.ndarray:
    base = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, np.newaxis]
    amplitude = np.sin(distance_m[np.newaxis, :] * np.pi / 3.0).astype(np.float32)
    return base + amplitude


def test_motion_compensation_speed_equalizes_spacing_to_threshold():
    distance = np.array([0.0, 0.5, 1.8, 2.4, 3.7, 4.2, 5.6, 6.1, 7.8, 9.0], dtype=np.float64)
    data = _synthetic_bscan(32, distance)
    trace_metadata = {
        "trace_index": np.arange(distance.size, dtype=np.int32),
        "trace_distance_m": distance.copy(),
        "alignment_status": np.full(distance.size, "aligned", dtype="<U16"),
    }

    corrected, meta = method_motion_compensation_speed(
        data,
        trace_metadata=trace_metadata,
        spacing_m=1.0,
    )

    assert meta.get("skipped") is not True
    output_distance = np.asarray(meta["trace_metadata_out"]["trace_distance_m"], dtype=np.float64)
    spacing_std = float(np.std(np.diff(output_distance)))
    assert spacing_std <= 0.02
    assert corrected.shape[1] == output_distance.size


def test_motion_compensation_speed_output_count_matches_metadata_length():
    distance = np.array([0.0, 1.0, 2.0, 4.0, 5.0], dtype=np.float64)
    data = _synthetic_bscan(16, distance)
    trace_metadata = {
        "trace_index": np.arange(distance.size, dtype=np.int32),
        "trace_distance_m": distance.copy(),
        "local_x_m": distance.copy(),
        "local_y_m": np.zeros(distance.size, dtype=np.float64),
    }

    corrected, meta = method_motion_compensation_speed(
        data,
        trace_metadata=trace_metadata,
        spacing_m=1.0,
    )

    trace_metadata_out = meta["trace_metadata_out"]
    assert corrected.shape[1] == len(trace_metadata_out["trace_distance_m"])
    assert corrected.shape[1] == len(trace_metadata_out["trace_index"])
    assert meta["source_traces"] == distance.size
    assert meta["target_traces"] == corrected.shape[1]
    assert meta["spacing_m"] == 1.0


def test_motion_compensation_speed_skips_nonmonotonic_distance():
    distance = np.array([0.0, 1.0, 0.8, 2.0], dtype=np.float64)
    data = _synthetic_bscan(8, distance)
    trace_metadata = {
        "trace_distance_m": distance.copy(),
    }

    corrected, meta = method_motion_compensation_speed(data, trace_metadata=trace_metadata)

    assert meta["skipped"] is True
    assert "单调" in meta["reason"]
    assert np.array_equal(corrected, data)


def test_motion_compensation_speed_derives_distance_from_xy_fallback():
    local_x = np.array([0.0, 0.6, 1.7, 2.6, 4.0], dtype=np.float64)
    local_y = np.array([0.0, 0.1, 0.1, 0.2, 0.2], dtype=np.float64)
    data = _synthetic_bscan(20, np.array([0.0, 0.6, 1.7, 2.6, 4.0], dtype=np.float64))
    trace_metadata = {
        "trace_index": np.arange(local_x.size, dtype=np.int32),
        "local_x_m": local_x.copy(),
        "local_y_m": local_y.copy(),
    }

    corrected, meta = method_motion_compensation_speed(
        data,
        trace_metadata=trace_metadata,
        spacing_m=1.0,
    )

    assert meta.get("skipped") is not True
    assert meta["distance_source"] == "local_xy"
    output_distance = np.asarray(meta["trace_metadata_out"]["trace_distance_m"], dtype=np.float64)
    assert corrected.shape[1] == output_distance.size
    assert np.all(np.diff(output_distance) >= -1e-12)


def test_motion_compensation_speed_does_not_mutate_inputs():
    distance = np.array([0.0, 0.7, 1.9, 3.0, 4.0], dtype=np.float64)
    data = _synthetic_bscan(12, distance)
    trace_metadata = {
        "trace_index": np.arange(distance.size, dtype=np.int32),
        "trace_distance_m": distance.copy(),
        "local_x_m": distance.copy(),
        "local_y_m": np.zeros(distance.size, dtype=np.float64),
    }
    original_data = data.copy()
    original_metadata = _copy_metadata(trace_metadata)

    corrected, meta = method_motion_compensation_speed(
        data,
        trace_metadata=trace_metadata,
        spacing_m=1.0,
    )

    assert corrected is not data
    assert meta.get("skipped") is not True
    assert np.array_equal(data, original_data)
    for key, original in original_metadata.items():
        assert np.array_equal(trace_metadata[key], original), f"字段 {key} 被原地修改"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RED tests for motion Phase-1 trace metadata foundations."""

from __future__ import annotations

import importlib
import numpy as np


def test_align_sidecar_records_adds_trace_timestamps_and_local_xy_without_breaking_legacy_fields():
    module = importlib.import_module("core.trace_metadata_utils")
    align_sidecar_records = module.align_sidecar_records

    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "longitude": np.array([100.0, 100.0005, 100.0010], dtype=np.float64),
        "latitude": np.array([30.0, 30.0, 30.0], dtype=np.float64),
        "trace_distance_m": np.array([0.0, 55.6, 111.2], dtype=np.float32),
        "flight_height_m": np.array([5.0, 5.1, 5.2], dtype=np.float32),
    }
    trace_timestamps_s = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    sidecar_records = {
        "timestamp_s": np.array([9.5, 10.5, 11.5, 12.5], dtype=np.float64),
        "longitude": np.array([100.0, 100.0005, 100.0010, 100.0015], dtype=np.float64),
        "latitude": np.array([30.0, 30.0, 30.0, 30.0], dtype=np.float64),
        "ground_elevation_m": np.array([10.0, 10.1, 10.2, 10.3], dtype=np.float32),
        "flight_height_m": np.array([5.0, 5.1, 5.2, 5.3], dtype=np.float32),
    }

    enriched = align_sidecar_records(
        trace_metadata,
        sidecar_records,
        trace_timestamps_s=trace_timestamps_s,
    )

    assert np.array_equal(enriched["trace_index"], trace_metadata["trace_index"])
    assert np.array_equal(enriched["longitude"], trace_metadata["longitude"])
    assert np.array_equal(enriched["latitude"], trace_metadata["latitude"])
    assert np.array_equal(enriched["trace_distance_m"], trace_metadata["trace_distance_m"])
    assert np.array_equal(enriched["trace_timestamp_s"], trace_timestamps_s)
    assert enriched["local_x_m"].shape == (3,)
    assert enriched["local_y_m"].shape == (3,)
    assert enriched["alignment_status"].shape == (3,)
    assert float(enriched["local_x_m"][0]) == 0.0
    assert float(enriched["local_x_m"][2]) > float(enriched["local_x_m"][1])
    assert abs(float(enriched["local_y_m"][2])) < 1.0


def test_build_uniform_trace_distance_m_returns_equal_spacing_axis():
    module = importlib.import_module("core.trace_metadata_utils")
    build_uniform_trace_distance_m = module.build_uniform_trace_distance_m

    trace_distance_m = np.array([0.0, 1.2, 2.8, 4.5], dtype=np.float32)

    uniform_distance = build_uniform_trace_distance_m(trace_distance_m, spacing_m=1.5)

    assert np.array_equal(
        uniform_distance,
        np.array([0.0, 1.5, 3.0, 4.5], dtype=np.float32),
    )


def test_resample_trace_metadata_interpolates_numeric_fields_and_marks_resampled_status():
    module = importlib.import_module("core.trace_metadata_utils")
    resample_trace_metadata = module.resample_trace_metadata

    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "trace_distance_m": np.array([0.0, 2.0, 4.0], dtype=np.float32),
        "trace_timestamp_s": np.array([10.0, 11.0, 12.0], dtype=np.float64),
        "local_x_m": np.array([0.0, 2.0, 4.0], dtype=np.float32),
        "local_y_m": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "longitude": np.array([100.0, 100.0002, 100.0004], dtype=np.float64),
        "latitude": np.array([30.0, 30.0, 30.0], dtype=np.float64),
        "alignment_status": np.array(["aligned", "aligned", "aligned"], dtype="<U16"),
    }
    target_trace_distance_m = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    resampled = resample_trace_metadata(
        trace_metadata,
        target_trace_distance_m=target_trace_distance_m,
    )

    assert np.array_equal(resampled["trace_distance_m"], target_trace_distance_m)
    assert np.array_equal(resampled["trace_index"], np.arange(5, dtype=np.int32))
    assert resampled["trace_timestamp_s"].shape == (5,)
    assert float(resampled["trace_timestamp_s"][0]) == 10.0
    assert float(resampled["trace_timestamp_s"][-1]) == 12.0
    assert np.allclose(resampled["local_x_m"], target_trace_distance_m)
    assert np.allclose(resampled["local_y_m"], 0.0)
    assert resampled["alignment_status"].shape == (5,)
    assert set(resampled["alignment_status"].tolist()) == {"resampled"}

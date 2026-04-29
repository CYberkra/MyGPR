#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RED tests for optional backend sidecar integration helpers."""

from __future__ import annotations

import importlib

import numpy as np


def test_integrate_optional_sidecars_without_payloads_keeps_legacy_metadata_unchanged():
    module = importlib.import_module("core.trace_metadata_utils")
    integrate_optional_sidecars = module.integrate_optional_sidecars

    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "longitude": np.array([100.0, 100.0005, 100.0010], dtype=np.float64),
        "latitude": np.array([30.0, 30.0, 30.0], dtype=np.float64),
        "trace_distance_m": np.array([0.0, 55.6, 111.2], dtype=np.float32),
        "flight_height_m": np.array([5.0, 5.1, 5.2], dtype=np.float32),
    }

    integrated = integrate_optional_sidecars(trace_metadata)

    assert set(integrated.keys()) == set(trace_metadata.keys())
    for key, values in trace_metadata.items():
        assert np.array_equal(integrated[key], values)


def test_integrate_optional_sidecars_with_rtk_and_imu_payloads_adds_motion_ready_fields():
    module = importlib.import_module("core.trace_metadata_utils")
    integrate_optional_sidecars = module.integrate_optional_sidecars

    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "longitude": np.array([100.0, 100.0005, 100.0010], dtype=np.float64),
        "latitude": np.array([30.0, 30.0, 30.0], dtype=np.float64),
        "trace_distance_m": np.array([0.0, 55.6, 111.2], dtype=np.float32),
        "flight_height_m": np.array([5.0, 5.1, 5.2], dtype=np.float32),
    }
    trace_timestamps_s = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    rtk_payload = {
        "source_kind": "rtk",
        "timestamp_s": np.array([9.5, 10.5, 11.5, 12.5], dtype=np.float64),
        "longitude": np.array([100.0, 100.0005, 100.0010, 100.0015], dtype=np.float64),
        "latitude": np.array([30.0, 30.0, 30.0, 30.0], dtype=np.float64),
        "ground_elevation_m": np.array([10.0, 10.1, 10.2, 10.3], dtype=np.float32),
        "flight_height_m": np.array([5.0, 5.1, 5.2, 5.3], dtype=np.float32),
    }
    imu_payload = {
        "source_kind": "imu",
        "timestamp_s": np.array([9.5, 10.5, 11.5, 12.5], dtype=np.float64),
        "roll_deg": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "pitch_deg": np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        "yaw_deg": np.array([180.0, 181.0, 182.0, 183.0], dtype=np.float32),
    }

    integrated = integrate_optional_sidecars(
        trace_metadata,
        trace_timestamps_s=trace_timestamps_s,
        rtk_payload=rtk_payload,
        imu_payload=imu_payload,
    )

    assert np.array_equal(integrated["trace_index"], trace_metadata["trace_index"])
    assert np.array_equal(integrated["trace_timestamp_s"], trace_timestamps_s)
    assert integrated["local_x_m"].shape == (3,)
    assert integrated["local_y_m"].shape == (3,)
    assert np.allclose(integrated["local_y_m"], 0.0, atol=1.0)
    assert set(integrated["alignment_status"].tolist()) == {"aligned"}
    assert np.allclose(integrated["roll_deg"], np.array([0.5, 1.5, 2.5], dtype=np.float32))
    assert np.allclose(integrated["pitch_deg"], np.array([0.25, 0.75, 1.25], dtype=np.float32))
    assert np.allclose(integrated["yaw_deg"], np.array([180.5, 181.5, 182.5], dtype=np.float32))

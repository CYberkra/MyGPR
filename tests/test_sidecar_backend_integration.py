#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RED tests for backend-only sidecar file integration helpers."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import numpy as np
import pytest


def _write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_integrate_optional_sidecars_without_files_keeps_legacy_metadata(tmp_path: Path):
    module = importlib.import_module("core.sidecar_integration")
    load_and_integrate_optional_sidecars = module.load_and_integrate_optional_sidecars

    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "longitude": np.array([100.0, 100.0005, 100.0010], dtype=np.float64),
        "latitude": np.array([30.0, 30.0, 30.0], dtype=np.float64),
        "trace_distance_m": np.array([0.0, 55.6, 111.2], dtype=np.float32),
    }

    integrated = load_and_integrate_optional_sidecars(trace_metadata)

    assert set(integrated.keys()) == set(trace_metadata.keys())
    for key, values in trace_metadata.items():
        assert np.array_equal(integrated[key], values)


def test_load_and_integrate_optional_sidecars_parses_rtk_and_imu_files(tmp_path: Path):
    module = importlib.import_module("core.sidecar_integration")
    load_and_integrate_optional_sidecars = module.load_and_integrate_optional_sidecars

    trace_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "longitude": np.array([100.0, 100.0005, 100.0010], dtype=np.float64),
        "latitude": np.array([30.0, 30.0, 30.0], dtype=np.float64),
        "trace_distance_m": np.array([0.0, 55.6, 111.2], dtype=np.float32),
        "flight_height_m": np.array([5.0, 5.1, 5.2], dtype=np.float32),
    }
    trace_timestamps_s = np.array([10.0, 11.0, 12.0], dtype=np.float64)

    rtk_path = tmp_path / "rtk.csv"
    _write_csv(
        rtk_path,
        [
            {"gps_time": 12.0, "lon": 100.0010, "lat": 30.0, "elevation_m": 11.5, "height_m": 4.8},
            {"gps_time": 10.0, "lon": 100.0000, "lat": 30.0, "elevation_m": 11.0, "height_m": 5.0},
            {"gps_time": 11.0, "lon": 100.0005, "lat": 30.0, "elevation_m": 11.2, "height_m": 4.9},
        ],
    )
    imu_path = tmp_path / "imu.csv"
    _write_csv(
        imu_path,
        [
            {"timestamp": 12.0, "roll": 2.0, "pitch": 1.0, "yaw": 182.0},
            {"timestamp": 10.0, "roll": 0.0, "pitch": 0.0, "yaw": 180.0},
            {"timestamp": 11.0, "roll": 1.0, "pitch": 0.5, "yaw": 181.0},
        ],
    )

    integrated = load_and_integrate_optional_sidecars(
        trace_metadata,
        trace_timestamps_s=trace_timestamps_s,
        rtk_path=rtk_path,
        imu_path=imu_path,
    )

    assert np.array_equal(integrated["trace_timestamp_s"], trace_timestamps_s)
    assert integrated["local_x_m"].shape == (3,)
    assert integrated["local_y_m"].shape == (3,)
    assert set(integrated["alignment_status"].tolist()) == {"aligned"}
    assert np.allclose(integrated["roll_deg"], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(integrated["pitch_deg"], np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert np.allclose(integrated["yaw_deg"], np.array([180.0, 181.0, 182.0], dtype=np.float32))


def test_load_and_integrate_optional_sidecars_requires_timestamps_when_files_are_provided(tmp_path: Path):
    module = importlib.import_module("core.sidecar_integration")
    load_and_integrate_optional_sidecars = module.load_and_integrate_optional_sidecars

    trace_metadata = {
        "trace_index": np.array([0, 1], dtype=np.int32),
        "trace_distance_m": np.array([0.0, 1.0], dtype=np.float32),
    }
    rtk_path = tmp_path / "rtk.csv"
    _write_csv(rtk_path, [{"gps_time": 10.0, "lon": 100.0, "lat": 30.0}])

    with pytest.raises(ValueError, match="trace_timestamps_s"):
        load_and_integrate_optional_sidecars(trace_metadata, rtk_path=rtk_path)

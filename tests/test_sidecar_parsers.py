#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RED tests for normalized RTK/IMU sidecar parser boundaries."""

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


def test_parse_sidecar_csv_normalizes_rtk_fields_and_sorts_by_timestamp(tmp_path: Path):
    module = importlib.import_module("core.sidecar_parsers")
    parse_sidecar_csv = module.parse_sidecar_csv

    csv_path = tmp_path / "rtk.csv"
    _write_csv(
        csv_path,
        [
            {
                "gps_time": 12.0,
                "lon": 100.0010,
                "lat": 30.0000,
                "elevation_m": 11.5,
                "height_m": 4.8,
                "fix": 5,
                "satellites": 18,
                "hdop": 0.7,
            },
            {
                "gps_time": 10.0,
                "lon": 100.0000,
                "lat": 30.0000,
                "elevation_m": 11.0,
                "height_m": 5.0,
                "fix": 4,
                "satellites": 16,
                "hdop": 0.9,
            },
            {
                "gps_time": 11.0,
                "lon": 100.0005,
                "lat": 30.0000,
                "elevation_m": 11.2,
                "height_m": 4.9,
                "fix": 5,
                "satellites": 17,
                "hdop": 0.8,
            },
        ],
    )

    parsed = parse_sidecar_csv(csv_path, kind="rtk")

    assert parsed["source_kind"] == "rtk"
    assert np.array_equal(parsed["timestamp_s"], np.array([10.0, 11.0, 12.0], dtype=np.float64))
    assert np.allclose(parsed["longitude"], np.array([100.0, 100.0005, 100.0010]))
    assert np.allclose(parsed["latitude"], np.array([30.0, 30.0, 30.0]))
    assert np.allclose(parsed["ground_elevation_m"], np.array([11.0, 11.2, 11.5], dtype=np.float32))
    assert np.allclose(parsed["flight_height_m"], np.array([5.0, 4.9, 4.8], dtype=np.float32))
    assert np.array_equal(parsed["rtk_fix_type"], np.array([4, 5, 5], dtype=np.int32))
    assert np.array_equal(parsed["satellites"], np.array([16, 17, 18], dtype=np.int32))
    assert np.allclose(parsed["hdop"], np.array([0.9, 0.8, 0.7], dtype=np.float32))


def test_parse_sidecar_csv_normalizes_imu_fields_and_preserves_attitude_columns(tmp_path: Path):
    module = importlib.import_module("core.sidecar_parsers")
    parse_sidecar_csv = module.parse_sidecar_csv

    csv_path = tmp_path / "imu.csv"
    _write_csv(
        csv_path,
        [
            {"timestamp": 2.0, "roll": 1.0, "pitch": 0.1, "yaw": 180.0},
            {"timestamp": 1.0, "roll": 0.5, "pitch": 0.0, "yaw": 179.0},
            {"timestamp": 3.0, "roll": 1.5, "pitch": 0.2, "yaw": 181.0},
        ],
    )

    parsed = parse_sidecar_csv(csv_path, kind="imu")

    assert parsed["source_kind"] == "imu"
    assert np.array_equal(parsed["timestamp_s"], np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert np.allclose(parsed["roll_deg"], np.array([0.5, 1.0, 1.5], dtype=np.float32))
    assert np.allclose(parsed["pitch_deg"], np.array([0.0, 0.1, 0.2], dtype=np.float32))
    assert np.allclose(parsed["yaw_deg"], np.array([179.0, 180.0, 181.0], dtype=np.float32))


def test_parse_sidecar_csv_rejects_missing_timestamp_column(tmp_path: Path):
    module = importlib.import_module("core.sidecar_parsers")
    parse_sidecar_csv = module.parse_sidecar_csv

    csv_path = tmp_path / "bad_rtk.csv"
    _write_csv(
        csv_path,
        [
            {"lon": 100.0, "lat": 30.0, "height_m": 5.0},
        ],
    )

    with pytest.raises(ValueError, match="timestamp"):
        parse_sidecar_csv(csv_path, kind="rtk")


def test_parse_sidecar_csv_rejects_unsupported_kind(tmp_path: Path):
    module = importlib.import_module("core.sidecar_parsers")
    parse_sidecar_csv = module.parse_sidecar_csv

    csv_path = tmp_path / "unknown.csv"
    _write_csv(csv_path, [{"timestamp": 1.0}])

    with pytest.raises(ValueError, match="Unsupported sidecar kind"):
        parse_sidecar_csv(csv_path, kind="barometer")

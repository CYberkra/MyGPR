#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Airborne CSV ingest contract tests."""

from __future__ import annotations

import csv

import numpy as np

from core.gpr_io import (
    compute_trace_distance_m,
    extract_airborne_csv_payload,
    subset_trace_metadata,
)


def _write_csv(path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_extract_airborne_csv_payload_stacked_format_returns_trace_metadata():
    header_info = {
        "a_scan_length": 3,
        "num_traces": 2,
        "total_time_ns": 120.0,
        "trace_interval_m": 0.5,
    }
    raw = np.array(
        [
            [100.0, 30.0, 10.0, 1.0, 5.0],
            [100.0, 30.0, 10.0, 2.0, 5.0],
            [100.0, 30.0, 10.0, 3.0, 5.0],
            [100.001, 30.0, 11.0, 4.0, 6.0],
            [100.001, 30.0, 11.0, 5.0, 6.0],
            [100.001, 30.0, 11.0, 6.0, 6.0],
        ],
        dtype=np.float64,
    )

    data, metadata, updated_header = extract_airborne_csv_payload(raw, header_info)

    assert data.shape == (3, 2)
    assert data.dtype == np.float32
    assert np.array_equal(
        data,
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
    )
    assert metadata is not None
    assert np.array_equal(metadata["trace_index"], np.array([0, 1], dtype=np.int32))
    assert np.allclose(metadata["longitude"], np.array([100.0, 100.001]))
    assert np.allclose(metadata["latitude"], np.array([30.0, 30.0]))
    assert np.allclose(metadata["ground_elevation_m"], np.array([10.0, 11.0]))
    assert np.allclose(metadata["flight_height_m"], np.array([5.0, 6.0]))
    assert metadata["trace_distance_m"].shape == (2,)
    assert metadata["trace_distance_m"][0] == 0.0
    assert float(metadata["trace_distance_m"][1]) > 0.0
    assert metadata["time_window_ns"] == 120.0
    assert updated_header is not None
    assert updated_header["source"] == "airborne_csv"
    assert updated_header["has_airborne_metadata"] is True
    assert float(updated_header["trace_interval_m"]) > 0.0


def test_extract_airborne_csv_payload_reads_explicit_trace_timestamps():
    header_info = {
        "a_scan_length": 3,
        "num_traces": 2,
        "total_time_ns": 120.0,
        "trace_interval_m": 0.5,
    }
    raw = np.array(
        [
            [100.0, 30.0, 10.0, 1.0, 5.0, 10.0],
            [100.0, 30.0, 10.0, 2.0, 5.0, 10.0],
            [100.0, 30.0, 10.0, 3.0, 5.0, 10.0],
            [100.001, 30.0, 11.0, 4.0, 6.0, 11.0],
            [100.001, 30.0, 11.0, 5.0, 6.0, 11.0],
            [100.001, 30.0, 11.0, 6.0, 6.0, 11.0],
        ],
        dtype=np.float64,
    )

    data, metadata, updated_header = extract_airborne_csv_payload(raw, header_info)

    assert data.shape == (3, 2)
    assert metadata is not None
    assert np.array_equal(
        metadata["trace_timestamp_s"], np.array([10.0, 11.0], dtype=np.float64)
    )
    assert updated_header is not None
    assert updated_header["source"] == "airborne_csv"


def test_extract_airborne_csv_payload_plain_matrix_fallback_keeps_legacy_behavior():
    header_info = {
        "a_scan_length": 3,
        "num_traces": 2,
        "total_time_ns": 120.0,
        "trace_interval_m": 0.5,
    }
    raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    data, metadata, updated_header = extract_airborne_csv_payload(raw, header_info)

    assert np.array_equal(
        data,
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
    )
    assert metadata is None
    assert updated_header == header_info


def test_subset_trace_metadata_slices_all_keys_by_indices():
    metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "longitude": np.array([100.0, 101.0, 102.0], dtype=np.float64),
        "flight_height_m": np.array([5.0, 6.0, 7.0], dtype=np.float32),
    }

    subset = subset_trace_metadata(metadata, np.array([0, 2]))

    assert subset is not None
    assert np.array_equal(subset["trace_index"], np.array([0, 2], dtype=np.int32))
    assert np.array_equal(subset["longitude"], np.array([100.0, 102.0]))
    assert np.array_equal(subset["flight_height_m"], np.array([5.0, 7.0]))
    assert subset["trace_index"] is not metadata["trace_index"]


def test_compute_trace_distance_m_matches_known_haversine_scale():
    longitude = np.array([0.0, 0.001, 0.002], dtype=np.float64)
    latitude = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    distance_m = compute_trace_distance_m(longitude, latitude)

    assert distance_m.shape == (3,)
    assert distance_m[0] == 0.0
    assert np.all(np.diff(distance_m) > 100.0)
    assert abs(float(distance_m[1]) - 111.2) < 1.0
    assert abs(float(distance_m[2]) - 222.4) < 2.0


def test_extract_airborne_csv_payload_with_optional_sidecars_enriches_metadata(tmp_path):
    header_info = {
        "a_scan_length": 3,
        "num_traces": 2,
        "total_time_ns": 120.0,
        "trace_interval_m": 0.5,
    }
    raw = np.array(
        [
            [100.0, 30.0, 10.0, 1.0, 5.0],
            [100.0, 30.0, 10.0, 2.0, 5.0],
            [100.0, 30.0, 10.0, 3.0, 5.0],
            [100.001, 30.0, 11.0, 4.0, 6.0],
            [100.001, 30.0, 11.0, 5.0, 6.0],
            [100.001, 30.0, 11.0, 6.0, 6.0],
        ],
        dtype=np.float64,
    )
    trace_timestamps_s = np.array([10.0, 11.0], dtype=np.float64)

    rtk_path = tmp_path / "rtk.csv"
    _write_csv(
        rtk_path,
        [
            {"gps_time": 11.5, "lon": 100.0010, "lat": 30.0, "elevation_m": 11.0, "height_m": 5.8},
            {"gps_time": 9.5, "lon": 99.9995, "lat": 30.0, "elevation_m": 10.0, "height_m": 5.0},
        ],
    )

    imu_path = tmp_path / "imu.csv"
    _write_csv(
        imu_path,
        [
            {"timestamp": 11.5, "roll": 2.0, "pitch": 1.0, "yaw": 182.0},
            {"timestamp": 9.5, "roll": 0.0, "pitch": 0.0, "yaw": 180.0},
        ],
    )

    data, metadata, updated_header = extract_airborne_csv_payload(
        raw,
        header_info,
        trace_timestamps_s=trace_timestamps_s,
        rtk_path=rtk_path,
        imu_path=imu_path,
    )

    assert data.shape == (3, 2)
    assert updated_header is not None
    assert updated_header["source"] == "airborne_csv"
    assert metadata is not None
    assert np.array_equal(metadata["trace_timestamp_s"], trace_timestamps_s)
    assert metadata["local_x_m"].shape == (2,)
    assert metadata["local_y_m"].shape == (2,)
    assert set(metadata["alignment_status"].tolist()) == {"aligned"}
    assert np.allclose(metadata["roll_deg"], np.array([0.5, 1.5], dtype=np.float32))
    assert np.allclose(metadata["pitch_deg"], np.array([0.25, 0.75], dtype=np.float32))
    assert np.allclose(metadata["yaw_deg"], np.array([180.5, 181.5], dtype=np.float32))


def test_extract_airborne_csv_payload_requires_trace_timestamps_when_sidecars_are_provided(tmp_path):
    header_info = {
        "a_scan_length": 3,
        "num_traces": 2,
        "total_time_ns": 120.0,
        "trace_interval_m": 0.5,
    }
    raw = np.array(
        [
            [100.0, 30.0, 10.0, 1.0, 5.0],
            [100.0, 30.0, 10.0, 2.0, 5.0],
            [100.0, 30.0, 10.0, 3.0, 5.0],
            [100.001, 30.0, 11.0, 4.0, 6.0],
            [100.001, 30.0, 11.0, 5.0, 6.0],
            [100.001, 30.0, 11.0, 6.0, 6.0],
        ],
        dtype=np.float64,
    )

    rtk_path = tmp_path / "rtk.csv"
    _write_csv(
        rtk_path,
        [{"gps_time": 10.0, "lon": 100.0, "lat": 30.0, "elevation_m": 10.0, "height_m": 5.0}],
    )

    try:
        extract_airborne_csv_payload(raw, header_info, rtk_path=rtk_path)
        assert False, "expected sidecar integration to require trace_timestamps_s"
    except ValueError as exc:
        assert "trace_timestamps_s" in str(exc)

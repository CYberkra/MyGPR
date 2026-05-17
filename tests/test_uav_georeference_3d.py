#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for UAV-GPR 3D georeference preview/export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.uav_georeference_3d import (
    AIR_TWO_WAY_DEPTH_SCALE_M_PER_NS,
    build_airborne_georeference_3d_payload,
    export_airborne_georeference_3d_bundle,
)


def test_build_airborne_georeference_payload_with_limited_preview():
    data = np.arange(72, dtype=np.float32).reshape(12, 6)
    trace_metadata = {
        "trace_index": np.arange(6, dtype=np.int32),
        "longitude": np.linspace(102.0, 102.0005, 6),
        "latitude": np.linspace(30.0, 30.0004, 6),
        "ground_elevation_m": np.linspace(10.0, 10.5, 6),
        "flight_height_m": np.linspace(2.0, 2.5, 6),
        "trace_distance_m": np.linspace(0.0, 1.5, 6),
    }
    header_info = {"total_time_ns": 60.0}

    payload = build_airborne_georeference_3d_payload(
        data,
        header_info,
        trace_metadata,
    )

    assert payload is not None
    assert payload["trace_count"] == 6
    assert payload["sample_count"] == 12
    assert payload["depth_scale_m_per_ns"] == AIR_TWO_WAY_DEPTH_SCALE_M_PER_NS
    assert payload["has_longitude_latitude"] is True
    assert payload["preview"]["curtain_z_m"].shape == (12, 6)
    assert payload["preview"]["amplitude"].shape == (12, 6)
    assert payload["preview"]["trace_stride"] == 1
    assert payload["preview"]["sample_stride"] == 1
    assert payload["airborne_z_m"].shape == (6,)
    assert np.isclose(payload["airborne_z_m"][0], 12.0)


def test_build_airborne_georeference_payload_downsamples_without_losing_tail():
    data = np.arange(15 * 14, dtype=np.float32).reshape(15, 14)

    payload = build_airborne_georeference_3d_payload(
        data,
        {"total_time_ns": 70.0},
        {"trace_distance_m": np.linspace(0.0, 1.3, 14)},
        max_preview_traces=5,
        max_preview_samples=6,
    )

    assert payload is not None
    assert payload["preview"]["trace_indices"][-1] == 13
    assert payload["preview"]["sample_indices"][-1] == 14
    assert payload["preview"]["amplitude"][-1, -1] == data[-1, -1]
    assert "downsampled_preview" in payload["quality_flags"]


def test_export_airborne_georeference_bundle_writes_vtk_csv_json(tmp_path: Path):
    data = np.arange(200, dtype=np.float32).reshape(20, 10)
    trace_metadata = {
        "trace_distance_m": np.linspace(0.0, 4.5, 10),
        "flight_height_m": np.linspace(3.0, 4.0, 10),
    }
    payload = build_airborne_georeference_3d_payload(
        data,
        {"total_time_ns": 100.0},
        trace_metadata,
        selected_trace_index=3,
    )
    assert payload is not None

    result = export_airborne_georeference_3d_bundle(payload, tmp_path / "scene.vtk")

    assert Path(result["vtk_path"]).exists()
    assert Path(result["csv_path"]).exists()
    assert Path(result["json_path"]).exists()
    assert result["summary"]["trace_count"] == 10
    assert result["summary"]["preview"]["shape"] == [20, 10]
    vtk_lines = Path(result["vtk_path"]).read_text(encoding="utf-8").splitlines()
    point_header = vtk_lines.index("POINTS 200 float")
    assert vtk_lines[point_header + 1] == "0.000000 0.000000 3.000000"
    lookup_header = vtk_lines.index("LOOKUP_TABLE default")
    assert vtk_lines[lookup_header + 1] == "0.000000"


def test_build_airborne_georeference_payload_falls_back_to_trace_distance():
    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    trace_metadata = {
        "trace_distance_m": np.linspace(0.0, 0.9, 4),
    }

    payload = build_airborne_georeference_3d_payload(
        data,
        {},
        trace_metadata,
    )

    assert payload is not None
    assert "fallback_trace_distance_axis" in payload["quality_flags"]
    assert "missing_ground_elevation" in payload["quality_flags"]
    assert payload["has_longitude_latitude"] is False


def test_build_airborne_georeference_payload_handles_bad_numeric_metadata():
    data = np.arange(30, dtype=np.float32).reshape(6, 5)
    trace_metadata = {
        "trace_distance_m": np.array(["bad", "values"], dtype=object),
        "flight_height_m": np.array(["not-a-height"], dtype=object),
    }

    payload = build_airborne_georeference_3d_payload(
        data,
        {"total_time_ns": 60.0},
        trace_metadata,
    )

    assert payload is not None
    assert payload["trace_count"] == 5
    assert np.isfinite(payload["trace_distance_m"]).all()
    assert np.isfinite(payload["airborne_z_m"]).all()
    assert np.allclose(payload["trace_distance_m"], 0.0)

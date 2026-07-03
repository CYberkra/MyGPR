#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic UAV-GPR motion demo package tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.gpr_io import extract_airborne_csv_payload
from core.sidecar_parsers import parse_sidecar_csv
from core.uav_georeference_3d import build_airborne_georeference_3d_payload
from PythonModule.motion_compensation_v2 import method_motion_compensation_v2
from read_file_data import readcsv
from scripts.generate_uav_gpr_motion_demo_v1 import (
    SAMPLES,
    TRACES,
    generate_demo_package,
)


def _header() -> dict[str, float | int]:
    return {
        "a_scan_length": SAMPLES,
        "num_traces": TRACES,
        "total_time_ns": 95.0,
        "trace_interval_m": 0.34,
    }


def test_generate_uav_gpr_motion_demo_v1_files(tmp_path: Path):
    package_dir = tmp_path / "uav_gpr_motion_demo_v1"
    config_path = tmp_path / "uav_gpr_motion_demo_v1.json"

    result = generate_demo_package(package_dir, config_out=config_path)

    assert result.package_dir == package_dir.resolve()
    for name in [
        "main.csv",
        "trace_timestamps.csv",
        "rtk.csv",
        "imu.csv",
        "altimeter.csv",
        "manifest.json",
        "metadata.json",
        "README.md",
    ]:
        assert (package_dir / name).exists()

    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["recommended_workflow"] == "motion_compensation_v2"
    assert manifest["expected_target"]["type"] == "pipe_like_hyperbola"


def test_uav_gpr_motion_demo_v1_reads_and_runs_motion_v2(tmp_path: Path):
    package_dir = tmp_path / "uav_gpr_motion_demo_v1"
    generate_demo_package(package_dir, config_out=tmp_path / "demo.json")

    raw = readcsv(str(package_dir / "main.csv"))
    data, metadata, updated_header = extract_airborne_csv_payload(
        raw,
        _header(),
        rtk_path=package_dir / "rtk.csv",
        imu_path=package_dir / "imu.csv",
        altimeter_path=package_dir / "altimeter.csv",
    )

    assert data.shape == (SAMPLES, TRACES)
    assert metadata is not None
    assert updated_header is not None
    assert metadata["trace_timestamp_s"].shape == (TRACES,)
    assert metadata["local_x_m"].shape == (TRACES,)
    assert metadata["height_agl_m"].min() >= 0.07
    assert metadata["height_agl_m"].max() <= 0.17

    corrected, motion_meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        header_info=updated_header,
        max_shift_samples=8,
        max_shift_ns=12.0,
        max_amplitude_scale=1.8,
        resample_spacing_m=0.34,
    )

    assert corrected.ndim == 2
    assert corrected.shape[0] == SAMPLES
    assert motion_meta["height_correction_applied"] is True
    assert motion_meta["time_shift_correction_applied"] is True

    current_metadata = motion_meta.get("trace_metadata_out")
    if current_metadata is None:
        current_metadata = dict(metadata)
        current_metadata.update(motion_meta.get("trace_metadata_updates") or {})

    raw_payload = build_airborne_georeference_3d_payload(
        data,
        updated_header,
        metadata,
        max_preview_traces=240,
        max_preview_samples=160,
    )
    current_payload = build_airborne_georeference_3d_payload(
        corrected,
        updated_header,
        current_metadata,
        max_preview_traces=240,
        max_preview_samples=160,
    )

    assert raw_payload is not None
    assert current_payload is not None
    assert raw_payload["preview"]["amplitude"].shape[0] <= 160
    assert current_payload["preview"]["amplitude"].shape[1] <= 240


def test_uav_gpr_motion_demo_v1_sidecars_are_parser_compatible(tmp_path: Path):
    package_dir = tmp_path / "uav_gpr_motion_demo_v1"
    generate_demo_package(package_dir, config_out=tmp_path / "demo.json")

    rtk = parse_sidecar_csv(package_dir / "rtk.csv", kind="rtk")
    imu = parse_sidecar_csv(package_dir / "imu.csv", kind="imu")
    altimeter = parse_sidecar_csv(package_dir / "altimeter.csv", kind="altimeter")

    assert rtk["timestamp_s"].shape == (TRACES,)
    assert "local_x_m" in rtk
    assert imu["roll_deg"].shape == (TRACES,)
    assert altimeter["height_agl_m"].shape == (TRACES,)
    assert np.isfinite(altimeter["height_agl_m"]).all()

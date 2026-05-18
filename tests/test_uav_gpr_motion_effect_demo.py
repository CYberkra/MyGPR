#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visible UAV-GPR motion compensation effect demo tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.gpr_io import extract_airborne_csv_payload
from core.uav_georeference_3d import build_airborne_georeference_3d_payload
from read_file_data import readcsv
from scripts.generate_uav_gpr_motion_effect_demo import (
    SAMPLES,
    TRACES,
    generate_motion_effect_demo,
)


def _header() -> dict[str, float | int]:
    return {
        "a_scan_length": SAMPLES,
        "num_traces": TRACES,
        "total_time_ns": 120.0,
        "trace_interval_m": 0.42,
    }


def test_generate_motion_effect_demo_outputs_visible_artifacts(tmp_path: Path):
    output_dir = tmp_path / "motion_effect_demo"
    result = generate_motion_effect_demo(output_dir)

    for name in [
        "main.csv",
        "trace_timestamps.csv",
        "rtk.csv",
        "imu.csv",
        "altimeter.csv",
        "manifest.json",
        "metadata.json",
        "processing_summary.json",
        "raw_bscan.png",
        "final_motion_bscan.png",
        "bscan_motion_comparison.png",
        "raw_3d_preview.png",
        "final_3d_preview.png",
        "README.md",
    ]:
        assert (output_dir / name).exists()

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["pipeline"] == [
        "trajectory_smoothing",
        "motion_compensation_speed",
        "motion_compensation_attitude",
        "motion_compensation_height",
    ]
    assert summary["bscan_rms_delta"] > 0.02
    assert summary["after_speed_spacing_std_m"] < summary["before_spacing_std_m"] * 0.20
    assert summary["top_interface_std_after"] <= summary["top_interface_std_before"] + 2.0


def test_motion_effect_demo_sidecars_align_without_explicit_timestamps(tmp_path: Path):
    output_dir = tmp_path / "motion_effect_demo"
    generate_motion_effect_demo(output_dir)

    raw = readcsv(str(output_dir / "main.csv"))
    data, metadata, header = extract_airborne_csv_payload(
        raw,
        _header(),
        rtk_path=output_dir / "rtk.csv",
        imu_path=output_dir / "imu.csv",
        altimeter_path=output_dir / "altimeter.csv",
    )

    assert data.shape == (SAMPLES, TRACES)
    assert metadata is not None
    assert header is not None
    assert metadata["trace_timestamp_s"].shape == (TRACES,)
    assert metadata["local_x_m"].shape == (TRACES,)
    assert metadata["roll_deg"].shape == (TRACES,)
    assert metadata["height_agl_m"].shape == (TRACES,)
    assert np.isfinite(metadata["trace_timestamp_s"]).all()

    payload = build_airborne_georeference_3d_payload(
        data,
        header,
        metadata,
        max_preview_traces=260,
        max_preview_samples=180,
    )
    assert payload is not None
    assert payload["preview"]["amplitude"].shape[0] <= 180
    assert payload["preview"]["amplitude"].shape[1] <= 260

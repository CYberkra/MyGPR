#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gprMax-derived UAV motion validation package tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from core.gpr_io import extract_airborne_csv_payload
from read_file_data import readcsv
from scripts.gprmax_benchmark.generate_gprmax_motion_validation_package import (
    generate_gprmax_motion_validation_package,
)


def _write_gprmax_fixture(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    samples = 72
    traces = 18
    sample_axis = np.arange(samples, dtype=np.float64)[:, None]
    trace_axis = np.arange(traces, dtype=np.float64)[None, :]
    data = 0.08 * np.sin(2.0 * np.pi * sample_axis / 8.0)
    data = np.repeat(data, traces, axis=1)
    hyperbola = 22.0 + 0.18 * (trace_axis.reshape(-1) - 9.0) ** 2
    for idx, row in enumerate(hyperbola):
        rr = int(round(row))
        data[rr : rr + 3, idx] += np.array([0.35, 1.1, 0.35])
    data += 0.02 * np.random.default_rng(42).normal(size=data.shape)

    out_path = dataset_dir / "pipe_merged.out"
    with h5py.File(out_path, "w") as handle:
        rx_group = handle.create_group("rxs").create_group("rx1")
        rx_group.create_dataset("Ez", data=data.astype(np.float32))
        handle.attrs["Iterations"] = samples
        handle.attrs["dt"] = 1e-10

    (dataset_dir / "pipe.in").write_text(
        "\n".join(
            [
                "#title: motion_pipe",
                "#domain: 1.000 0.500 0.010",
                "#dx_dy_dz: 0.010 0.010 0.010",
                "#time_window: 7.200e-09",
                "#src_steps: 0.100 0.000 0.000",
                "#rx_steps: 0.100 0.000 0.000",
            ]
        ),
        encoding="utf-8",
    )
    (dataset_dir / "metadata.json").write_text(
        json.dumps({"scenario_id": "motion_pipe", "runs": traces}, ensure_ascii=False),
        encoding="utf-8",
    )
    (dataset_dir / "ground_truth.yaml").write_text(
        "\n".join(
            [
                "schema: gprmax_ground_truth_v1",
                "scenario_id: motion_pipe",
                "target:",
                "  type: pipe",
                "  material: pec",
                "  depth_m: 0.35",
                "  center_x_m: 0.9",
                "  center_y_m: 0.2",
                "  radius_m: 0.04",
                "target_roi:",
                "  sample_range: [20, 32]",
                "  trace_range: [6, 12]",
                "analysis_roi:",
                "  sample_range: [12, 42]",
                "  trace_range: [3, 15]",
                "background_roi:",
                "  sample_range: [4, 12]",
                "  trace_range: [0, 5]",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "gprmax_dataset_manifest_v1",
        "scenario_id": "motion_pipe",
        "primary_out_file": "pipe_merged.out",
        "metadata_file": "metadata.json",
        "ground_truth_file": "ground_truth.yaml",
    }
    manifest_path = dataset_dir / "pipe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest_path


@pytest.fixture(scope="module")
def default_motion_validation_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Any]:
    root = tmp_path_factory.mktemp("gprmax_motion_validation_default")
    manifest_path = _write_gprmax_fixture(root / "dataset")
    output_dir = root / "motion_validation"
    result = generate_gprmax_motion_validation_package(manifest_path, output_dir, seed=7)
    return output_dir, result


def test_generate_gprmax_motion_validation_package(default_motion_validation_package):
    output_dir, result = default_motion_validation_package

    for name in [
        "main.csv",
        "trace_timestamps.csv",
        "rtk.csv",
        "imu.csv",
        "altimeter.csv",
        "manifest.json",
        "metadata.json",
        "processing_summary.json",
        "motion_validation_report.md",
        "source_gprmax_bscan.png",
        "motion_injected_raw_bscan.png",
        "atomic_motion_final_bscan.png",
        "motion_v2_final_bscan.png",
        "bscan_motion_validation_comparison.png",
        "paper_motion_validation_comparison.png",
        "raw_3d_preview.png",
        "motion_v2_3d_preview.png",
        "README.md",
    ]:
        assert (output_dir / name).exists(), name

    assert result.raw_shape == (72, 18)
    assert result.atomic_shape[0] == 72
    assert result.v2_shape[0] == 72
    assert result.spacing_std_before_m > 0.0
    assert result.spacing_std_atomic_after_speed_m < result.spacing_std_before_m * 0.60
    assert result.spacing_std_v2_m < result.spacing_std_before_m * 0.20
    assert result.target_ratio_raw is not None
    assert result.target_ratio_atomic is not None
    assert result.target_ratio_v2 is not None

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["source"]["scenario_id"] == "motion_pipe"
    assert summary["pipeline"]["atomic"] == [
        "trajectory_smoothing",
        "motion_compensation_attitude",
        "motion_compensation_speed",
        "motion_compensation_height",
    ]
    assert "raw_vs_source_rms" in summary["metrics"]
    assert "ridge_rmse_samples_atomic" in summary["metrics"]
    assert "target_apex_error_samples_v2" in summary["metrics"]
    assert "target_roi_energy_preservation_atomic" in summary["metrics"]
    assert "trace_spacing_cv_before" in summary["metrics"]
    assert "max_gap_ratio_v2" in summary["metrics"]
    assert "target_traces" in summary["metrics"]
    assert summary["metrics"]["atomic_vs_source_rms"] < summary["metrics"]["raw_vs_source_rms"]
    assert summary["metrics"]["v2_vs_source_rms"] < summary["metrics"]["raw_vs_source_rms"]
    assert summary["metrics"]["trace_spacing_cv_atomic"] < summary["metrics"]["trace_spacing_cv_before"]
    assert summary["metrics"]["spacing_std_atomic_m"] <= summary["metrics"]["spacing_std_atomic_after_speed_m"] * 1.25
    assert "motion_v2_3d_preview.png" in summary["artifacts"]["images"]
    assert "paper_motion_validation_comparison.png" in summary["artifacts"]["images"]


def test_generate_gprmax_motion_validation_longline_report(tmp_path: Path):
    manifest_path = _write_gprmax_fixture(tmp_path / "dataset")
    output_dir = tmp_path / "motion_validation_longline"

    result = generate_gprmax_motion_validation_package(
        manifest_path,
        output_dir,
        seed=9,
        target_traces=120,
    )

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    report = result.report_md.read_text(encoding="utf-8")

    assert result.raw_shape == (72, 120)
    assert summary["source"]["original_gprmax_shape"] == [72, 18]
    assert summary["source"]["derived_longline"] is True
    assert summary["shapes"]["source"] == [72, 120]
    assert summary["metrics"]["raw_vs_source_rms"] > summary["metrics"]["atomic_vs_source_rms"]
    assert summary["metrics"]["raw_vs_source_rms"] > summary["metrics"]["v2_vs_source_rms"]
    for key in [
        "ridge_rmse_samples_raw",
        "ridge_rmse_samples_atomic",
        "ridge_rmse_samples_v2",
        "target_apex_error_samples_raw",
        "target_apex_error_samples_atomic",
        "target_apex_error_samples_v2",
        "target_roi_energy_preservation_raw",
        "target_roi_energy_preservation_atomic",
        "target_roi_energy_preservation_v2",
        "trace_spacing_cv_before",
        "trace_spacing_cv_v2",
        "max_gap_ratio_before",
        "max_gap_ratio_v2",
        "resample_spacing_m",
        "target_traces",
    ]:
        assert key in summary["metrics"], key
    assert summary["metrics"]["target_traces"] == summary["shapes"]["v2"][1]
    assert "V2 Resampling Explanation" in report
    assert "Validation Notes" in report
    assert "target_traces" in report
    assert "resampled to the processed trace axis" in report
    assert (output_dir / "paper_motion_validation_comparison.png").exists()
    assert (output_dir / "source_manifest.json").exists()
    assert (output_dir / "source_ground_truth.yaml").exists()
    assert (output_dir / "source_model_in.in").exists()


def test_generated_gprmax_motion_package_sidecars_align(default_motion_validation_package):
    output_dir, _result = default_motion_validation_package

    raw = readcsv(str(output_dir / "main.csv"))
    data, metadata, header = extract_airborne_csv_payload(
        raw,
        {
            "a_scan_length": 72,
            "num_traces": 18,
            "total_time_ns": 7.2,
            "trace_interval_m": 0.1,
        },
        rtk_path=output_dir / "rtk.csv",
        imu_path=output_dir / "imu.csv",
        altimeter_path=output_dir / "altimeter.csv",
    )

    assert data.shape == (72, 18)
    assert metadata is not None
    assert header is not None
    assert metadata["trace_timestamp_s"].shape == (18,)
    assert metadata["height_agl_m"].shape == (18,)
    assert metadata["roll_deg"].shape == (18,)
    assert metadata["local_x_m"].shape == (18,)
    assert np.isfinite(metadata["trace_distance_m"]).all()

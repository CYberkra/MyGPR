#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPRMAX benchmark package generation tests."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from core.auto_tune_comparison import run_auto_tune_comparison
from core.auto_tune_comparison_export import export_auto_tune_comparison_artifacts
from scripts.gprmax_benchmark.generate_cylinder_single_v1 import generate_package


def _write_gprmax_out(path: Path, data: np.ndarray, *, dt: float = 5e-11) -> None:
    with h5py.File(path, "w") as handle:
        rx_group = handle.create_group("rxs").create_group("rx1")
        rx_group.create_dataset("Ez", data=np.asarray(data, dtype=np.float32))
        handle.attrs["Iterations"] = int(np.asarray(data).shape[0])
        handle.attrs["dt"] = float(dt)
        handle.attrs["nx_ny_nz"] = [1, 1, 1]


def test_generate_cylinder_single_v1_writes_clean_benchmark_package(tmp_path: Path):
    result = generate_package(tmp_path / "cylinder_single_v1")
    package_dir = result.package_dir

    expected = {
        "scenario.json",
        "model.in",
        "ground_truth.json",
        "mygpr_bscan.csv",
        "preview.png",
        "README.md",
    }
    assert expected <= {path.name for path in package_dir.iterdir()}

    scenario = json.loads(result.scenario_path.read_text(encoding="utf-8"))
    assert scenario["schema"] == "mygpr_gprmax_scenario_v1"
    assert scenario["scenario_id"] == "cylinder_single_v1"
    assert scenario["simulation"]["trace_count"] > 20
    assert scenario["target"]["type"] == "metal_cylinder"

    ground_truth = json.loads(result.ground_truth_path.read_text(encoding="utf-8"))
    assert ground_truth["schema"] == "mygpr_gprmax_ground_truth_v1"
    assert ground_truth["scenario_id"] == "cylinder_single_v1"
    assert len(ground_truth["targets"]) == 1
    target = ground_truth["targets"][0]
    assert target["type"] == "hyperbola"
    assert target["must_preserve"] is True
    roi = target["roi"]
    assert 0 <= roi["time_start_idx"] < roi["time_end_idx"] <= scenario["simulation"]["sample_count"]
    assert 0 <= roi["dist_start_idx"] < roi["dist_end_idx"] <= scenario["simulation"]["trace_count"]

    bscan = np.loadtxt(result.bscan_csv_path, delimiter=",")
    assert bscan.shape == (
        scenario["simulation"]["sample_count"],
        scenario["simulation"]["trace_count"],
    )
    assert np.isfinite(bscan).all()
    assert result.preview_path.stat().st_size > 0
    assert "#title: MyGPR cylinder_single_v1" in result.model_in_path.read_text(
        encoding="utf-8"
    )


def test_convert_gprmax_out_to_mygpr_csv_preserves_numeric_trace_order(tmp_path: Path):
    raw_dir = tmp_path / "raw_out"
    raw_dir.mkdir()
    traces = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
    ]
    _write_gprmax_out(raw_dir / "cylinder_single_v110.out", traces[1])
    _write_gprmax_out(raw_dir / "cylinder_single_v12.out", traces[0])
    _write_gprmax_out(raw_dir / "cylinder_single_v130.out", traces[2])

    result = generate_package(
        tmp_path / "converted",
        raw_out_path=raw_dir / "cylinder_single_v12.out",
    )

    bscan = np.loadtxt(result.bscan_csv_path, delimiter=",")
    assert bscan.shape == (4, 3)
    assert np.array_equal(bscan, np.column_stack(traces))

    scenario = json.loads(result.scenario_path.read_text(encoding="utf-8"))
    assert scenario["source"]["kind"] == "gprmax_out"
    assert scenario["simulation"]["sample_count"] == 4
    assert scenario["simulation"]["trace_count"] == 3
    assert scenario["simulation"]["time_step_s"] == 5e-11


def test_gprmax_benchmark_can_export_auto_tune_comparison_evidence(tmp_path: Path):
    package = generate_package(tmp_path / "cylinder_single_v1")
    bscan = np.loadtxt(package.bscan_csv_path, delimiter=",").astype(np.float32)
    ground_truth = json.loads(package.ground_truth_path.read_text(encoding="utf-8"))
    roi = ground_truth["targets"][0]["roi"]

    comparison = run_auto_tune_comparison(
        bscan,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        roi_spec={
            "mode": "manual",
            "bounds": roi,
            "label": "cylinder target ROI",
        },
        search_mode="fast",
    )
    exported = export_auto_tune_comparison_artifacts(
        comparison,
        out_dir=tmp_path / "evidence",
        bundle_name="cylinder_single_v1",
        input_ref=str(package.bscan_csv_path),
        notes=["GPRMAX scenario: cylinder_single_v1"],
    )

    summary_path = Path(exported["artifacts"]["summary_json"])
    report_path = Path(exported["artifacts"]["report_md"])
    assert summary_path.exists()
    assert report_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_ref"] == str(package.bscan_csv_path)
    assert summary["roi_info"]["label"] == "cylinder target ROI"
    assert summary["display_spec"]["locked_scale"] is True

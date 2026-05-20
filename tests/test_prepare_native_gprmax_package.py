#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for native gprMax-to-CSV package preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.gprmax_benchmark.prepare_native_gprmax_package import h5py, prepare_native_package


def _write_model(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#title: test native",
                "#domain: 0.600 0.400 0.002",
                "#dx_dy_dz: 0.002 0.002 0.002",
                "#time_window: 1.2e-08",
                "#hertzian_dipole: z 0.080 0.300 0 my_ricker",
                "#rx: 0.120 0.300 0",
                "#src_steps: 0.004 0 0",
                "#rx_steps: 0.004 0 0",
            ]
        ),
        encoding="utf-8",
    )


def _write_out(path: Path) -> np.ndarray:
    if h5py is None:
        pytest.skip("h5py is unavailable")
    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    with h5py.File(path, "w") as handle:
        handle.attrs["Iterations"] = data.shape[0]
        handle.attrs["dt"] = 2e-10
        handle.attrs["nx_ny_nz"] = np.array([300, 200, 1])
        group = handle.create_group("rxs").create_group("rx1")
        group.create_dataset("Ez", data=data)
    return data


def _write_ground_truth(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "schema: gprmax_ground_truth_v1",
                "dataset_id: native_test",
                "model_file: native_test.in",
                "output_file: native_test_merged.out",
                "target_roi:",
                "  sample_range: [1, 3]",
                "  trace_range: [1, 2]",
                "background_roi:",
                "  sample_range: [4, 5]",
                "  trace_range: [0, 1]",
                "target:",
                "  type: pipe",
                "  material: pec",
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_native_package_converts_out_to_csv_and_audits(tmp_path: Path):
    model = tmp_path / "native_test.in"
    raw_out = tmp_path / "native_test_merged.out"
    truth = tmp_path / "ground_truth.yaml"
    output_dir = tmp_path / "package"
    data = _write_out(raw_out)
    _write_model(model)
    _write_ground_truth(truth)

    result = prepare_native_package(
        model_in=model,
        out_path=raw_out,
        output_dir=output_dir,
        receiver="rx1",
        component="Ez",
        scenario_id="native_test",
        ground_truth=truth,
    )

    assert result["status"] == "native_gprmax_converted"
    assert result["shape"] == [6, 4]
    assert Path(result["csv_path"]).exists()
    assert Path(result["preview_path"]).exists()
    assert np.loadtxt(result["csv_path"], delimiter=",").shape == data.shape

    scenario = json.loads((output_dir / "scenario.json").read_text(encoding="utf-8"))
    assert scenario["source"]["kind"] == "native_gprmax_converted"
    assert scenario["source"]["raw_out_hash"]
    assert scenario["conversion"]["hdf5_dataset"] == "/rxs/rx1/Ez"

    manifest = json.loads((output_dir / "native_gprmax_package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_file"] == "mygpr_bscan.csv"
    assert manifest["receiver"] == "rx1"
    assert manifest["component"] == "Ez"
    assert manifest["converted_ground_truth_file"] == "ground_truth.json"

    audit = json.loads((output_dir / "gprmax_package_audit.json").read_text(encoding="utf-8"))
    assert audit["source"]["kind"] == "native_gprmax_converted"
    assert audit["source"]["native_gprmax_verified"] is True
    assert audit["files"]["raw_out_exists"] is True
    assert audit["source"]["dt_source"] == "native gprMax .out"
    assert audit["ground_truth"]["roi_inside_bscan"] is True


def test_prepare_native_package_writes_pending_when_out_missing(tmp_path: Path):
    model = tmp_path / "native_test.in"
    _write_model(model)

    result = prepare_native_package(
        model_in=model,
        out_path=tmp_path / "missing.out",
        output_dir=tmp_path / "pending",
        receiver="rx1",
        component="Ez",
        scenario_id="pending_case",
    )

    assert result["status"] == "pending_native_out"
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "pending_native_out"
    assert "native .out missing" in manifest["missing_reason"]

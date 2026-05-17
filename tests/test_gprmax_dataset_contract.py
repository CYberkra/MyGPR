#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for external gprMax dataset contract loading."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from core.auto_tune_pipeline import run_auto_tune_pipeline, to_summary_dict
from core.gprmax_dataset_contract import (
    adapt_gprmax_ground_truth,
    load_gprmax_dataset_contract,
)
from core.gprmax_truth_metrics import compute_ground_truth_metrics


def _write_gprmax_out(path: Path, data: np.ndarray, *, dt: float = 1e-10) -> None:
    with h5py.File(path, "w") as handle:
        rx_group = handle.create_group("rxs").create_group("rx1")
        rx_group.create_dataset("Ez", data=np.asarray(data, dtype=np.float32))
        handle.attrs["Iterations"] = int(np.asarray(data).shape[0])
        handle.attrs["dt"] = float(dt)
        handle.attrs["nx_ny_nz"] = [1, 1, 1]


def _write_contract_files(tmp_path: Path) -> Path:
    data = np.zeros((48, 16), dtype=np.float32)
    data[12:19, 5:10] = 3.0
    data[25:32, 1:4] = 0.3
    out_path = tmp_path / "pipe_merged.out"
    _write_gprmax_out(out_path, data)
    (tmp_path / "pipe.in").write_text(
        "\n".join(
            [
                "#title: pipe",
                "#domain: 1.000 0.500 0.010",
                "#dx_dy_dz: 0.010 0.010 0.010",
                "#time_window: 8.000e-10",
                "#waveform: impulse 1 1.0 my_impulse",
                "#hertzian_dipole: z 0.100 0.200 0.100 my_impulse",
                "#rx: 0.100 0.300 0.100",
                "#src_steps: 0.050 0.000 0.000",
                "#rx_steps: 0.050 0.000 0.000",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"scenario_id": "pipe_demo", "runs": 16}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "ground_truth.yaml").write_text(
        "\n".join(
            [
                "schema: gprmax_ground_truth_v1",
                "scenario_id: pipe_demo",
                "target_id: pipe_01",
                "target_type: hyperbola",
                "target_roi:",
                "  sample_range: [12, 18]",
                "  trace_range: [5, 9]",
                "analysis_roi:",
                "  sample_range: [8, 24]",
                "  trace_range: [3, 12]",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "gprmax_dataset_manifest_v1",
        "scenario_id": "pipe_demo",
        "primary_out_file": "pipe_merged.out",
        "metadata_file": "metadata.json",
        "ground_truth_file": "ground_truth.yaml",
    }
    manifest_path = tmp_path / "pipe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def test_adapt_gprmax_ground_truth_converts_closed_roi_to_slice_roi():
    converted = adapt_gprmax_ground_truth(
        {
            "schema": "gprmax_ground_truth_v1",
            "scenario_id": "pipe_demo",
            "target_id": "pipe_01",
            "target_type": "hyperbola",
            "target_roi": {
                "sample_range": [12, 18],
                "trace_range": [5, 9],
            },
            "analysis_roi": {
                "sample_range": [8, 24],
                "trace_range": [3, 12],
            },
        },
        data_shape=(48, 16),
    )

    assert converted["schema"] == "mygpr_gprmax_ground_truth_v1"
    assert converted["scenario_id"] == "pipe_demo"
    assert converted["analysis_roi"] == {
        "time_start_idx": 8,
        "time_end_idx": 25,
        "dist_start_idx": 3,
        "dist_end_idx": 13,
    }
    assert converted["targets"][0]["target_id"] == "pipe_01"
    assert converted["targets"][0]["type"] == "hyperbola"
    assert converted["targets"][0]["roi"] == {
        "time_start_idx": 12,
        "time_end_idx": 19,
        "dist_start_idx": 5,
        "dist_end_idx": 10,
    }


def test_adapt_gprmax_ground_truth_converts_wavefield_rois():
    converted = adapt_gprmax_ground_truth(
        {
            "schema": "gprmax_ground_truth_v1",
            "scenario_id": "airborne_demo",
            "wavefield_rois": {
                "direct_air_wave": {
                    "sample_range": [2, 4],
                    "trace_range": [0, 99],
                    "time_ns": 0.7,
                },
                "air_ground_reflection": {
                    "roi": {
                        "sample_range": [10, 12],
                        "trace_range": [1, 6],
                    },
                    "time_ns": 3.2,
                },
                "subsurface_target": {
                    "sample_range": [20, 28],
                    "trace_range": [3, 5],
                },
            },
        },
        data_shape=(24, 8),
    )

    rois = converted["wavefield_rois"]
    assert rois["direct_air_wave"]["roi"] == {
        "time_start_idx": 2,
        "time_end_idx": 5,
        "dist_start_idx": 0,
        "dist_end_idx": 8,
    }
    assert rois["air_ground_reflection"]["time_ns"] == 3.2
    assert rois["air_ground_reflection"]["roi"]["time_end_idx"] == 13
    assert rois["subsurface_target"]["roi"] == {
        "time_start_idx": 20,
        "time_end_idx": 24,
        "dist_start_idx": 3,
        "dist_end_idx": 6,
    }


def test_load_gprmax_dataset_contract_reads_manifest_out_and_ground_truth(tmp_path: Path):
    manifest_path = _write_contract_files(tmp_path)

    package = load_gprmax_dataset_contract(manifest_path)

    assert package.scenario_id == "pipe_demo"
    assert package.data.shape == (48, 16)
    assert package.primary_out_file == tmp_path / "pipe_merged.out"
    assert package.metadata["runs"] == 16
    assert package.header_info["data_context"] == "gprmax_impulse"
    assert package.trace_metadata is not None
    assert np.allclose(
        package.trace_metadata["trace_distance_m"][:3],
        [0.0, 0.05, 0.1],
    )
    assert package.ground_truth["targets"][0]["roi"]["time_end_idx"] == 19
    assert package.header_info["ground_truth"] is package.ground_truth
    assert package.ground_truth["source_paths"]["manifest_file"] == str(manifest_path)
    assert package.ground_truth["source_paths"]["ground_truth_file"] == str(
        tmp_path / "ground_truth.yaml"
    )
    assert package.ground_truth_paths["ground_truth_file"] == tmp_path / "ground_truth.yaml"


def test_gprmax_contract_feeds_truth_metrics_and_auto_tune_pipeline(tmp_path: Path):
    manifest_path = _write_contract_files(tmp_path)
    package = load_gprmax_dataset_contract(manifest_path)

    metrics = compute_ground_truth_metrics(
        package.data,
        package.data,
        package.ground_truth,
    )

    assert metrics["truth_target_count"] == 1.0
    assert metrics["truth_target_energy_preservation"] > 0.95

    result = run_auto_tune_pipeline(
        package.data,
        header_info=package.header_info,
        trace_metadata=package.trace_metadata,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 3}},
        ground_truth=package.ground_truth,
        search_mode="fast",
    )
    summary = to_summary_dict(result)

    assert result.input_shape == package.data.shape
    assert summary["ground_truth_info"]["scenario_id"] == "pipe_demo"
    assert summary["ground_truth_info"]["target_count"] == 1
    assert "truth_score" in summary["manual"]["metrics"]
    assert "truth_score" in summary["automatic"]["metrics"]

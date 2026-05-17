#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax ground_truth.yaml adaptation into MyGPR truth metrics."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from core.auto_tune_pipeline import run_auto_tune_pipeline, to_summary_dict
from core.gpr_io import read_gprmax_out
from core.gprmax_ground_truth import (
    convert_gprmax_ground_truth_to_mygpr,
    load_gprmax_ground_truth,
    load_ground_truth_from_manifest,
)
from core.gprmax_truth_metrics import compute_ground_truth_metrics


def _write_gprmax_out(path: Path, data: np.ndarray, *, dt: float = 1e-10) -> None:
    with h5py.File(path, "w") as handle:
        rx_group = handle.create_group("rxs").create_group("rx1")
        rx_group.create_dataset("Ez", data=np.asarray(data, dtype=np.float32))
        handle.attrs["Iterations"] = int(np.asarray(data).shape[0])
        handle.attrs["dt"] = float(dt)
        handle.attrs["nx_ny_nz"] = [1, 1, 1]


def _sidecar_text(*, output_file: str = "pipe_merged.out") -> str:
    return "\n".join(
        [
            "schema: gprmax_ground_truth_v1",
            "dataset_id: pipe_dataset_001",
            f"output_file: {output_file}",
            "target_roi:",
            "  sample_range: [720, 860]",
            "  trace_range: [42, 48]",
            "type: pipe",
            "material: metal",
            "depth_m: 0.42",
            "center_x_m: 1.4",
            "center_y_m: 0.42",
            "radius_m: 0.05",
            "background_roi:",
            "  sample_range: [100, 180]",
            "  trace_range: [4, 18]",
            "metrics_contract:",
            "  target_energy_preservation_min: 0.70",
        ]
    )


def test_load_gprmax_ground_truth_reads_yaml(tmp_path: Path):
    sidecar_path = tmp_path / "ground_truth.yaml"
    sidecar_path.write_text(_sidecar_text(), encoding="utf-8")

    sidecar = load_gprmax_ground_truth(str(sidecar_path))

    assert sidecar["schema"] == "gprmax_ground_truth_v1"
    assert sidecar["dataset_id"] == "pipe_dataset_001"
    assert sidecar["target_roi"]["sample_range"] == [720, 860]


def test_convert_gprmax_ground_truth_uses_half_open_roi_and_preserves_raw_sidecar():
    sidecar = yaml.safe_load(_sidecar_text())

    converted = convert_gprmax_ground_truth_to_mygpr(
        sidecar,
        data_shape=(1000, 96),
    )

    assert converted["scenario_id"] == "pipe_dataset_001"
    assert converted["analysis_roi"] == {
        "time_start_idx": 720,
        "time_end_idx": 861,
        "dist_start_idx": 42,
        "dist_end_idx": 49,
    }
    target = converted["targets"][0]
    assert target["id"] == "target_0"
    assert target["type"] == "pipe"
    assert target["material"] == "metal"
    assert target["depth_m"] == 0.42
    assert target["center_x_m"] == 1.4
    assert target["center_y_m"] == 0.42
    assert target["radius_m"] == 0.05
    assert target["must_preserve"] is True
    assert target["roi"] == {
        "time_start_idx": 720,
        "time_end_idx": 861,
        "dist_start_idx": 42,
        "dist_end_idx": 49,
    }
    assert converted["background_rois"] == [
        {
            "time_start_idx": 100,
            "time_end_idx": 181,
            "dist_start_idx": 4,
            "dist_end_idx": 19,
        }
    ]
    assert converted["metrics_contract"]["target_energy_preservation_min"] == 0.70
    assert converted["raw_sidecar"]["target_roi"]["trace_range"] == [42, 48]


def test_convert_gprmax_ground_truth_preserves_nested_target_metadata():
    sidecar = {
        "schema": "gprmax_ground_truth_v1",
        "dataset_id": "nested_target_demo",
        "target_roi": {
            "sample_range": [720, 860],
            "trace_range": [42, 48],
        },
        "type": "legacy_top_level_type",
        "material": "legacy_top_level_material",
        "target": {
            "type": "pipe",
            "material": "metal",
            "depth_m": 0.42,
            "center_x_m": 1.4,
            "center_y_m": 0.42,
            "radius_m": 0.05,
        },
    }

    converted = convert_gprmax_ground_truth_to_mygpr(
        sidecar,
        data_shape=(1000, 96),
    )

    target = converted["targets"][0]
    assert target["type"] == "pipe"
    assert target["material"] == "metal"
    assert target["depth_m"] == 0.42
    assert target["center_x_m"] == 1.4
    assert target["center_y_m"] == 0.42
    assert target["radius_m"] == 0.05
    assert target["roi"] == {
        "time_start_idx": 720,
        "time_end_idx": 861,
        "dist_start_idx": 42,
        "dist_end_idx": 49,
    }


def test_convert_gprmax_ground_truth_preserves_nested_target_metadata_in_target_list():
    sidecar = {
        "schema": "gprmax_ground_truth_v1",
        "dataset_id": "target_list_demo",
        "targets": [
            {
                "id": "pipe_a",
                "target_roi": {
                    "sample_range": [120, 180],
                    "trace_range": [7, 11],
                },
                "type": "legacy_type",
                "material": "legacy_material",
                "target": {
                    "type": "pipe",
                    "material": "metal",
                    "depth_m": 0.35,
                    "center_x_m": 0.7,
                    "center_y_m": 0.35,
                    "radius_m": 0.04,
                },
            }
        ],
    }

    converted = convert_gprmax_ground_truth_to_mygpr(
        sidecar,
        data_shape=(300, 32),
    )

    target = converted["targets"][0]
    assert target["id"] == "pipe_a"
    assert target["type"] == "pipe"
    assert target["material"] == "metal"
    assert target["depth_m"] == 0.35
    assert target["center_x_m"] == 0.7
    assert target["center_y_m"] == 0.35
    assert target["radius_m"] == 0.04
    assert target["roi"] == {
        "time_start_idx": 120,
        "time_end_idx": 181,
        "dist_start_idx": 7,
        "dist_end_idx": 12,
    }


def test_load_ground_truth_from_manifest_reads_nested_path_and_warns_on_output_mismatch(
    tmp_path: Path,
):
    sidecar_path = tmp_path / "ground_truth.yaml"
    sidecar_path.write_text(
        _sidecar_text(output_file="old_name.out"),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "pipe_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "gprmax_dataset_manifest_v1",
                "paths_relative_to_output_dir": {
                    "primary_out_file": "pipe_merged.out",
                    "ground_truth_file": "ground_truth.yaml",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="output_file does not match"):
        converted = load_ground_truth_from_manifest(str(manifest_path))

    assert converted is not None
    assert converted["scenario_id"] == "pipe_dataset_001"
    assert converted["source_paths"]["ground_truth_file"] == str(sidecar_path.resolve())
    assert "conversion_warnings" in converted


def test_converted_ground_truth_feeds_metrics_and_auto_tune_pipeline():
    raw = np.zeros((1000, 96), dtype=np.float32)
    raw[720:861, 42:49] = 5.0
    raw[100:181, 4:19] = 1.5
    processed = raw.copy()
    processed[100:181, 4:19] *= 0.1
    ground_truth = convert_gprmax_ground_truth_to_mygpr(
        yaml.safe_load(_sidecar_text()),
        data_shape=raw.shape,
    )

    metrics = compute_ground_truth_metrics(raw, processed, ground_truth)
    result = run_auto_tune_pipeline(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 3}},
        ground_truth=ground_truth,
        search_mode="fast",
    )
    summary = to_summary_dict(result)

    assert metrics["truth_target_count"] == 1.0
    assert metrics["truth_background_energy_reduction"] > 0.8
    assert summary["ground_truth_info"]["scenario_id"] == "pipe_dataset_001"
    assert "truth_score" in summary["manual"]["metrics"]


def test_metrics_prefer_explicit_background_rois_over_analysis_minus_target():
    raw = np.zeros((40, 20), dtype=np.float32)
    raw[15:20, 8:12] = 10.0
    raw[5:10, 1:4] = 1.0
    raw[25:32, 14:18] = 100.0
    processed = raw.copy()
    processed[5:10, 1:4] *= 0.1
    ground_truth = {
        "scenario_id": "explicit_background_demo",
        "analysis_roi": {
            "time_start_idx": 0,
            "time_end_idx": 35,
            "dist_start_idx": 0,
            "dist_end_idx": 20,
        },
        "targets": [
            {
                "id": "target_0",
                "must_preserve": True,
                "roi": {
                    "time_start_idx": 15,
                    "time_end_idx": 20,
                    "dist_start_idx": 8,
                    "dist_end_idx": 12,
                },
            }
        ],
    }
    explicit_background = {
        **ground_truth,
        "background_rois": [
            {
                "time_start_idx": 5,
                "time_end_idx": 10,
                "dist_start_idx": 1,
                "dist_end_idx": 4,
            }
        ],
    }

    fallback_metrics = compute_ground_truth_metrics(raw, processed, ground_truth)
    explicit_metrics = compute_ground_truth_metrics(
        raw,
        processed,
        explicit_background,
    )

    assert explicit_metrics["truth_background_energy_reduction"] > 0.8
    assert (
        explicit_metrics["truth_background_energy_reduction"]
        > fallback_metrics["truth_background_energy_reduction"] + 0.5
    )


def test_read_gprmax_out_attaches_manifest_ground_truth(tmp_path: Path):
    data = np.zeros((32, 12), dtype=np.float32)
    data[10:15, 4:8] = 3.0
    _write_gprmax_out(tmp_path / "pipe_merged.out", data)
    (tmp_path / "ground_truth.yaml").write_text(
        "\n".join(
            [
                "schema: gprmax_ground_truth_v1",
                "dataset_id: pipe_manifest_demo",
                "output_file: pipe_merged.out",
                "target_roi:",
                "  sample_range: [10, 14]",
                "  trace_range: [4, 7]",
                "background_roi:",
                "  sample_range: [0, 5]",
                "  trace_range: [0, 3]",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "pipe_manifest.json").write_text(
        json.dumps(
            {
                "schema": "gprmax_dataset_manifest_v1",
                "primary_out_file": "pipe_merged.out",
                "ground_truth_file": "ground_truth.yaml",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = read_gprmax_out(str(tmp_path / "pipe_merged.out"))

    ground_truth = payload["header_info"]["ground_truth"]
    assert ground_truth["scenario_id"] == "pipe_manifest_demo"
    assert ground_truth["targets"][0]["roi"] == {
        "time_start_idx": 10,
        "time_end_idx": 15,
        "dist_start_idx": 4,
        "dist_end_idx": 8,
    }
    assert ground_truth["background_rois"][0]["time_end_idx"] == 6

    result = run_auto_tune_pipeline(
        payload["data"],
        header_info=payload["header_info"],
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 3}},
        search_mode="fast",
    )
    assert result.ground_truth_info["enabled"] is True
    assert result.ground_truth_info["scenario_id"] == "pipe_manifest_demo"

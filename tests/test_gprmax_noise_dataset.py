#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for deterministic noisy gprMax dataset generation."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from core.gprmax_dataset_contract import load_gprmax_dataset_contract
from scripts.gprmax_benchmark.add_noise_to_gprmax_dataset import create_noisy_dataset


def _write_gprmax_out(path: Path, data: np.ndarray, *, dt: float = 1e-10) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["Title"] = "noise fixture"
        handle.attrs["Iterations"] = data.shape[0]
        handle.attrs["dt"] = dt
        handle.attrs["nrx"] = 1
        handle.attrs["nsrc"] = 1
        handle.attrs["rxsteps"] = [1, 0, 0]
        handle.attrs["srcsteps"] = [1, 0, 0]
        rx = handle.create_group("rxs").create_group("rx1")
        rx.create_dataset("Ez", data=data.astype(np.float32))


def _write_fixture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = np.linspace(-1.0, 1.0, 80, dtype=np.float32).reshape(10, 8)
    _write_gprmax_out(tmp_path / "pipe_merged.out", data)
    (tmp_path / "pipe.in").write_text("#title: pipe\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"scenario_id": "pipe_fixture"}, indent=2),
        encoding="utf-8",
    )
    (tmp_path / "ground_truth.yaml").write_text(
        "\n".join(
            [
                "schema: gprmax_ground_truth_v1",
                "dataset_id: pipe_fixture",
                "model_file: pipe.in",
                "output_file: pipe_merged.out",
                "target_roi:",
                "  sample_range: [2, 4]",
                "  trace_range: [2, 4]",
                "background_roi:",
                "  sample_range: [6, 8]",
                "  trace_range: [0, 2]",
                "target:",
                "  type: pipe",
                "  material: pec",
                "  depth_m: 0.3",
                "  center_x_m: 0.5",
                "  center_y_m: 0.2",
                "  radius_m: 0.03",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "gprmax_dataset_manifest_v1",
        "scenario_id": "pipe_fixture",
        "primary_out_file": "pipe_merged.out",
        "metadata_file": "metadata.json",
        "ground_truth_file": "ground_truth.yaml",
    }
    manifest_path = tmp_path / "pipe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def test_create_noisy_dataset_preserves_contract_and_updates_sidecars(tmp_path: Path):
    manifest_path = _write_fixture(tmp_path / "source")
    output_root = tmp_path / "out"

    summary = create_noisy_dataset(
        manifest_path,
        output_root,
        output_name="pipe_low_snr_v1",
        target_snr_db=6.0,
        seed=123,
    )

    dataset_dir = output_root / "pipe_low_snr_v1"
    assert Path(summary["primary_out_file"]).exists()
    assert Path(summary["manifest_file"]).exists()
    assert Path(summary["ground_truth_file"]).exists()
    assert summary["noise"]["target_snr_db"] == 6.0
    assert abs(summary["noise"]["actual_snr_db"] - 6.0) < 0.5

    package = load_gprmax_dataset_contract(dataset_dir / "pipe_low_snr_v1_manifest.json")
    assert package.scenario_id == "pipe_low_snr_v1"
    assert package.ground_truth["scenario_id"] == "pipe_low_snr_v1"
    assert package.ground_truth_raw["output_file"] == "pipe_low_snr_v1_merged.out"
    assert package.metadata["noise_augmentation"]["type"] == "additive_gaussian"

    clean = load_gprmax_dataset_contract(manifest_path).data
    assert package.data.shape == clean.shape
    assert not np.allclose(package.data, clean)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end smoke for gprMax dataset contract and AutoTune Evidence export."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np

from core.auto_tune_comparison import run_auto_tune_comparison, to_summary_dict
from core.auto_tune_comparison_export import export_auto_tune_comparison_artifacts
from core.gprmax_dataset_contract import load_gprmax_dataset_contract


ROOT = Path(__file__).resolve().parents[1]


def _write_gprmax_out(path: Path, data: np.ndarray, *, dt: float = 1e-10) -> None:
    with h5py.File(path, "w") as handle:
        rx_group = handle.create_group("rxs").create_group("rx1")
        rx_group.create_dataset("Ez", data=np.asarray(data, dtype=np.float32))
        handle.attrs["Iterations"] = int(np.asarray(data).shape[0])
        handle.attrs["dt"] = float(dt)
        handle.attrs["nx_ny_nz"] = [1, 1, 1]


def _build_pipe_demo_data(samples: int = 48, traces: int = 16) -> np.ndarray:
    rng = np.random.default_rng(170)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(-1.0, 1.0, traces, dtype=np.float64)[None, :]
    data = 0.20 * np.sin(2.0 * np.pi * 7.0 * t)
    data = np.repeat(data, traces, axis=1)
    data += 0.08 * np.sin(2.0 * np.pi * 2.0 * x)
    data += 0.025 * rng.normal(size=(samples, traces))
    center = traces // 2
    for trace_idx in range(traces):
        lateral = (trace_idx - center) / max(center, 1)
        row = int(round(12 + 6.0 * lateral * lateral))
        data[row : row + 3, trace_idx] += np.array([0.35, 1.2, 0.35])
    data[6:11, 1:4] += 0.18
    return data.astype(np.float32)


def _write_dataset(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = _build_pipe_demo_data()
    _write_gprmax_out(tmp_path / "pipe_merged.out", data)
    (tmp_path / "pipe.in").write_text(
        "\n".join(
            [
                "#title: pipe_demo",
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
                "background_roi:",
                "  sample_range: [2, 8]",
                "  trace_range: [0, 4]",
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


def test_gprmax_autotune_evidence_smoke_exports_truth_bundle(tmp_path: Path):
    manifest_path = _write_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "evidence"

    package = load_gprmax_dataset_contract(manifest_path)

    assert package.data.shape == (48, 16)
    assert package.header_info["ground_truth"]["scenario_id"] == "pipe_demo"
    assert package.header_info["ground_truth"]["targets"][0]["roi"] == {
        "time_start_idx": 12,
        "time_end_idx": 19,
        "dist_start_idx": 5,
        "dist_end_idx": 10,
    }

    comparison = run_auto_tune_comparison(
        package.data,
        header_info=package.header_info,
        trace_metadata=package.trace_metadata,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 3}},
        roi_spec={
            "mode": "manual",
            "bounds": package.ground_truth["analysis_roi"],
            "label": "ground_truth:pipe_demo",
        },
        search_mode="fast",
    )
    summary = to_summary_dict(comparison)
    for key in (
        "truth_score",
        "truth_target_energy_preservation",
        "truth_target_saliency_gain",
        "truth_background_energy_reduction",
        "truth_false_positive_ratio",
    ):
        assert key in summary["manual"]["metrics"]
        assert key in summary["automatic"]["metrics"]

    bundle = export_auto_tune_comparison_artifacts(
        comparison,
        out_dir=output_dir,
        bundle_name="pipe_demo_smoke",
        input_ref=str(package.primary_out_file),
    )
    artifacts = {key: Path(value) for key, value in bundle["artifacts"].items()}
    for key in (
        "evidence_manifest_json",
        "summary_json",
        "truth_metrics_json",
        "workflow_params_json",
        "trial_table_csv",
        "trial_table_json",
        "manual_png",
        "auto_png",
        "side_by_side_png",
        "report_md",
        "evidence_zip",
    ):
        assert artifacts[key].exists(), key
        assert artifacts[key].stat().st_size > 0, key

    report = artifacts["report_md"].read_text(encoding="utf-8")
    assert "pipe_demo" in report
    assert "truth_score" in report
    assert "truth_target_energy_preservation" in report

    with artifacts["trial_table_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["branch"] == "automatic" for row in rows)

    with zipfile.ZipFile(artifacts["evidence_zip"]) as zf:
        names = set(zf.namelist())
    assert {
        "evidence_manifest.json",
        "comparison_summary.json",
        "truth_metrics.json",
        "workflow_params.json",
        "trial_table.csv",
        "trial_table.json",
        "comparison_report.md",
    } <= names


def test_gprmax_autotune_evidence_smoke_cli(tmp_path: Path):
    manifest_path = _write_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "cli_evidence"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "gprmax_benchmark" / "run_autotune_evidence_smoke.py"),
            "--dataset",
            str(manifest_path),
            "--output",
            str(output_dir),
            "--bundle-name",
            "pipe_demo_cli_smoke",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "truth_score manual:" in completed.stdout
    assert "truth_score AutoTune:" in completed.stdout
    assert "comparison_score manual:" in completed.stdout
    assert "comparison_score AutoTune:" in completed.stdout
    assert (output_dir / "pipe_demo_cli_smoke" / "comparison_report.md").exists()
    assert (output_dir / "pipe_demo_cli_smoke" / "evidence_bundle.zip").exists()

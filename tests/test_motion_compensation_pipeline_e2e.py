#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end evidence export coverage for the motion compensation V1 benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cli_batch
from core.benchmark_registry import generate_benchmark_sample
from core.evidence_export import export_motion_compensation_benchmark
from core.quality_metrics import compute_motion_quality_metrics


def _assert_motion_improvement(summary: dict) -> None:
    _, context = generate_benchmark_sample("motion_compensation_v1", seed=42)
    metric_config = context["expected_metrics"]["metric_config"]
    final_metrics = compute_motion_quality_metrics(
        summary["final_data"],
        summary["corrected_trace_metadata"],
        context["ground_truth_trace_metadata"],
        ground_truth_data=context["ground_truth_data"],
        ridge_row_range=tuple(metric_config["ridge_row_range"]),
        target_row_range=tuple(metric_config["target_row_range"]),
        banding_trace_band=tuple(metric_config["banding_trace_band"]),
        banding_row_range=tuple(metric_config["banding_row_range"]),
    )

    assert final_metrics["raw_ridge_rmse_samples"] < context["expected_metrics"]["raw"]["raw_ridge_rmse_samples"]
    assert final_metrics["trace_spacing_std_m"] < context["expected_metrics"]["raw"]["trace_spacing_std_m"]
    assert final_metrics["path_rmse_m"] < context["expected_metrics"]["raw"]["path_rmse_m"]
    assert final_metrics["footprint_rmse_m"] < context["expected_metrics"]["raw"]["footprint_rmse_m"]
    assert final_metrics["periodic_banding_ratio"] < context["expected_metrics"]["raw"]["periodic_banding_ratio"]
    assert final_metrics["target_preservation_ratio"] >= context["expected_metrics"]["raw"]["target_preservation_ratio"]


def test_export_motion_compensation_benchmark_writes_required_artifacts(tmp_path: Path):
    summary = export_motion_compensation_benchmark(tmp_path)

    expected_files = [
        tmp_path / "before.png",
        tmp_path / "after.png",
        tmp_path / "difference.png",
        tmp_path / "motion_metrics.json",
        tmp_path / "corrected_trace_metadata.csv",
        tmp_path / "motion_compensation_v1-summary.json",
    ]
    for path in expected_files:
        assert path.exists(), f"missing artifact: {path}"

    with (tmp_path / "motion_metrics.json").open("r", encoding="utf-8") as handle:
        metrics_payload = json.load(handle)
    assert all(metrics_payload["objective_checks"].values())

    for step in summary["steps"]:
        assert "trace_metadata" not in step["params"]
        assert "header_info" not in step["params"]
        assert "runtime_context" in step
        assert step["runtime_context"]["trace_metadata"]["trace_count"] > 0

    with (tmp_path / "corrected_trace_metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows
    assert "trace_distance_m" in rows[0]
    assert "local_y_m" in rows[0]
    assert len(rows) > 2

    _assert_motion_improvement(summary)


def test_cli_run_job_exports_motion_benchmark_evidence(tmp_path: Path):
    cfg = cli_batch.load_config(str(Path(cli_batch.BASE_DIR) / "config" / "motion_compensation_v1_benchmark.json"))

    result = cli_batch.run_job(
        cfg["jobs"][0],
        repo_root=str(Path(cli_batch.BASE_DIR)),
        output_dir=str(tmp_path),
    )

    assert result["status"] == "ok"
    assert result["benchmark_sample"] == "motion_compensation_v1"
    assert all(result["objective_checks"].values())
    assert (tmp_path / "before.png").exists()
    assert (tmp_path / "after.png").exists()
    assert (tmp_path / "difference.png").exists()
    assert (tmp_path / "motion_metrics.json").exists()
    assert (tmp_path / "corrected_trace_metadata.csv").exists()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-BG-002 diagnostic-only background suppression harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from core.autotune_background_suppression import (
    CandidateSpec,
    load_csv_2d,
    run_background_suppression_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "autotune_background_suppression_diagnostic.py"


def _build_arrays() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    clutter = np.sin(x)[:, None] * np.ones((64, 21))
    target = np.zeros((64, 21), dtype=np.float64)
    target[20:30, 8:13] = 0.5
    raw = clutter + target
    target_response = target.copy()
    return raw, target_response


def test_global_mean_candidate_works(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_mean_global"
    result = run_background_suppression_diagnostic(
        raw=raw,
        target_response=target,
        output_dir=out,
        artifact_id="a",
        scene_id="s",
        candidate_specs=[
            CandidateSpec(
                method="mean_background_subtraction",
                parameter_set={"mode": "global_mean", "axis": "trace"},
                candidate_group="mean",
            )
        ],
    )
    assert result["status"] == "success"
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    assert rows[0]["method"] == "mean_background_subtraction"
    assert rows[0]["mae"] is not None


def test_moving_window_mean_candidate_works(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_mean_window"
    run_background_suppression_diagnostic(
        raw=raw,
        target_response=target,
        output_dir=out,
        candidate_specs=[
            CandidateSpec(
                method="mean_background_subtraction",
                parameter_set={"mode": "moving_window_mean", "window_size": 9, "axis": "trace"},
                candidate_group="mean",
            )
        ],
    )
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    assert rows[0]["parameter_set"]["window_size"] == 9


def test_global_median_candidate_works(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_median_global"
    run_background_suppression_diagnostic(
        raw=raw,
        target_response=target,
        output_dir=out,
        candidate_specs=[
            CandidateSpec(
                method="median_background_subtraction",
                parameter_set={"mode": "global_median", "axis": "trace"},
                candidate_group="median",
            )
        ],
    )
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    assert rows[0]["method"] == "median_background_subtraction"


def test_moving_window_median_candidate_works(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_median_window"
    run_background_suppression_diagnostic(
        raw=raw,
        target_response=target,
        output_dir=out,
        candidate_specs=[
            CandidateSpec(
                method="median_background_subtraction",
                parameter_set={"mode": "moving_window_median", "window_size": 15, "axis": "trace"},
                candidate_group="median",
            )
        ],
    )
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    assert rows[0]["parameter_set"]["window_size"] == 15


def test_svd_remove_rank_candidate_works(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_svd"
    run_background_suppression_diagnostic(
        raw=raw,
        target_response=target,
        output_dir=out,
        candidate_specs=[
            CandidateSpec(
                method="svd_background_suppression",
                parameter_set={"remove_rank": 2},
                candidate_group="svd",
            )
        ],
    )
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    assert rows[0]["method"] == "svd_background_suppression"


def test_trial_table_fields_exist(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_fields"
    run_background_suppression_diagnostic(raw=raw, target_response=target, output_dir=out)
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    assert rows
    for key in (
        "trial_id",
        "artifact_id",
        "scene_id",
        "method",
        "parameter_set",
        "candidate_group",
        "mae",
        "mse",
        "rmse",
        "psnr",
        "warnings",
        "recommendation_label",
    ):
        assert key in rows[0]


def test_shape_mismatch_rejected():
    raw = np.ones((8, 4), dtype=np.float64)
    target = np.ones((9, 4), dtype=np.float64)
    try:
        run_background_suppression_diagnostic(
            raw=raw,
            target_response=target,
            output_dir=Path.cwd() / "output" / "tmp_should_not_exist",
        )
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for shape mismatch")


def test_nan_inf_warning(tmp_path):
    raw, target = _build_arrays()
    raw[0, 0] = np.nan
    out = tmp_path / "out_nan"
    run_background_suppression_diagnostic(
        raw=raw,
        target_response=target,
        output_dir=out,
        candidate_specs=[
            CandidateSpec(
                method="mean_background_subtraction",
                parameter_set={"mode": "global_mean", "axis": "trace"},
                candidate_group="mean",
            )
        ],
    )
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    codes = {item["code"] for item in rows[0]["warnings"]}
    assert "processed_bscan_nan_or_inf" in codes


def test_roi_invalid_warning(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_roi_bad"
    run_background_suppression_diagnostic(
        raw=raw,
        target_response=target,
        output_dir=out,
        roi={"sample_range": [5, 2], "trace_range": [0, 10]},
        candidate_specs=[
            CandidateSpec(
                method="mean_background_subtraction",
                parameter_set={"mode": "global_mean", "axis": "trace"},
                candidate_group="mean",
            )
        ],
    )
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    codes = {item["code"] for item in rows[0]["warnings"]}
    assert "roi_invalid" in codes


def test_recommendation_label_generated(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_label"
    run_background_suppression_diagnostic(raw=raw, target_response=target, output_dir=out)
    rows = json.loads((out / "trial_table.json").read_text(encoding="utf-8"))
    labels = {row["recommendation_label"] for row in rows}
    assert any(label in labels for label in {"recommended", "acceptable_alternative", "manual_review_recommended"})


def test_selected_parameters_generated(tmp_path):
    raw, target = _build_arrays()
    out = tmp_path / "out_selected"
    run_background_suppression_diagnostic(raw=raw, target_response=target, output_dir=out)
    selected_path = out / "selected_parameters.json"
    assert selected_path.exists()
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    assert "selected" in payload


def test_csv_loading_preserves_2d_shape(tmp_path):
    csv_path = tmp_path / "single_col.csv"
    np.savetxt(csv_path, np.array([[1.0], [2.0], [3.0]]), delimiter=",", fmt="%.10g")
    arr = load_csv_2d(csv_path)
    assert arr.shape == (3, 1)


def test_cli_runs_without_gprmax_or_evidence_repo(tmp_path):
    raw, target = _build_arrays()
    raw_path = tmp_path / "raw.csv"
    target_path = tmp_path / "target.csv"
    out_dir = tmp_path / "cli_out"
    np.savetxt(raw_path, raw, delimiter=",")
    np.savetxt(target_path, target, delimiter=",")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--raw",
            str(raw_path),
            "--target-response",
            str(target_path),
            "--output-dir",
            str(out_dir),
            "--artifact-id",
            "artifact_test",
            "--scene-id",
            "scene_test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "success"
    assert (out_dir / "trial_table.json").exists()


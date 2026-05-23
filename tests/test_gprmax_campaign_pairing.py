#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for paired output validation and target_response generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from core.gprmax_campaign.pairing import (
    PairedOutputSpec,
    discover_converted_pair_paths,
    generate_target_response,
    validate_paired_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "gprmax_campaign" / "pairing"
RUNNER = ROOT / "scripts" / "gprmax_campaign_runner.py"


def test_valid_csv_pair_generates_target_response(tmp_path):
    spec = PairedOutputSpec(
        campaign_id="GX-003",
        scene_id="scene_a",
        raw_output_path=FIXTURE_DIR / "raw_valid.csv",
        background_output_path=FIXTURE_DIR / "background_valid.csv",
        output_dir=tmp_path / "pair_ok",
        source_format="csv",
    )
    result = generate_target_response(spec)
    assert result.status == "success"
    assert result.target_response_npy_path and result.target_response_npy_path.exists()
    assert result.target_response_csv_path and result.target_response_csv_path.exists()
    assert result.validation_summary_path.exists()
    assert result.metrics_path and result.metrics_path.exists()
    raw = np.genfromtxt(FIXTURE_DIR / "raw_valid.csv", delimiter=",")
    bg = np.genfromtxt(FIXTURE_DIR / "background_valid.csv", delimiter=",")
    generated = np.load(result.target_response_npy_path)
    np.testing.assert_allclose(generated, raw - bg)
    summary = json.loads(result.validation_summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ready"
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["raw_shape"] == [3, 3]
    assert "target_response_energy" in metrics
    assert "raw_background_mae" in metrics
    assert "raw_background_mse" in metrics
    assert "raw_background_rmse" in metrics
    assert "raw_background_psnr" in metrics
    assert "target_to_raw_energy_ratio" in metrics
    assert "sparsity_or_concentration_proxy" in metrics


def test_shape_mismatch_invalid(tmp_path):
    spec = PairedOutputSpec(
        campaign_id="GX-003",
        scene_id="scene_mismatch",
        raw_output_path=FIXTURE_DIR / "raw_shape_mismatch.csv",
        background_output_path=FIXTURE_DIR / "background_shape_mismatch.csv",
        output_dir=tmp_path / "pair_bad_shape",
        source_format="csv",
    )
    result = generate_target_response(spec)
    assert result.status == "invalid"
    issue = next(item for item in result.issues if item["code"] == "shape_mismatch")
    assert "raw=" in issue["message"]
    assert "background=" in issue["message"]
    assert "raw_shape_mismatch.csv" in issue["message"]
    assert "background_shape_mismatch.csv" in issue["message"]


def test_missing_file_invalid(tmp_path):
    spec = PairedOutputSpec(
        campaign_id="GX-003",
        scene_id="scene_missing",
        raw_output_path=FIXTURE_DIR / "raw_valid.csv",
        background_output_path=FIXTURE_DIR / "missing.csv",
        output_dir=tmp_path / "pair_missing",
        source_format="csv",
    )
    validation, _, _ = validate_paired_outputs(spec)
    assert validation.status == "invalid"
    assert any(item["code"] == "background_missing" for item in validation.issues)


def test_nan_invalid(tmp_path):
    spec = PairedOutputSpec(
        campaign_id="GX-003",
        scene_id="scene_nan",
        raw_output_path=FIXTURE_DIR / "raw_nan.csv",
        background_output_path=FIXTURE_DIR / "background_valid.csv",
        output_dir=tmp_path / "pair_nan",
        source_format="csv",
    )
    validation, _, _ = validate_paired_outputs(spec)
    assert validation.status == "invalid"
    assert any(item["code"] == "raw_nan_or_inf" for item in validation.issues)


def test_npy_pair_works(tmp_path):
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    bg = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    raw_path = tmp_path / "raw.npy"
    bg_path = tmp_path / "bg.npy"
    np.save(raw_path, raw)
    np.save(bg_path, bg)
    spec = PairedOutputSpec(
        campaign_id="GX-003",
        scene_id="scene_npy",
        raw_output_path=raw_path,
        background_output_path=bg_path,
        output_dir=tmp_path / "pair_npy",
        source_format="npy",
    )
    result = generate_target_response(spec)
    assert result.status == "success"
    arr = np.load(result.target_response_npy_path)
    np.testing.assert_allclose(arr, raw - bg)


def test_single_column_csv_pair_preserves_2d_shape(tmp_path):
    raw = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    bg = np.array([[0.25], [0.5], [0.75]], dtype=np.float64)
    raw_path = tmp_path / "raw_single_column.csv"
    bg_path = tmp_path / "background_single_column.csv"
    np.savetxt(raw_path, raw, delimiter=",", fmt="%.10g")
    np.savetxt(bg_path, bg, delimiter=",", fmt="%.10g")
    spec = PairedOutputSpec(
        campaign_id="GX-003",
        scene_id="scene_single_column_csv",
        raw_output_path=raw_path,
        background_output_path=bg_path,
        output_dir=tmp_path / "pair_single_column_csv",
        source_format="csv",
    )
    result = generate_target_response(spec)
    assert result.status == "success"
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["target_response_shape"] == [3, 1]
    generated = np.load(result.target_response_npy_path)
    np.testing.assert_allclose(generated, raw - bg)


def test_pairing_with_roi_json_path_adds_roi_metric(tmp_path):
    raw = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    bg = np.zeros((2, 2), dtype=np.float64)
    raw_path = tmp_path / "raw.csv"
    bg_path = tmp_path / "bg.csv"
    roi_path = tmp_path / "roi.json"
    np.savetxt(raw_path, raw, delimiter=",", fmt="%.10g")
    np.savetxt(bg_path, bg, delimiter=",", fmt="%.10g")
    roi_path.write_text(
        json.dumps({"sample_range": [0, 1], "trace_range": [0, 2]}),
        encoding="utf-8",
    )

    spec = PairedOutputSpec(
        campaign_id="GX-003",
        scene_id="scene_roi",
        raw_output_path=raw_path,
        background_output_path=bg_path,
        output_dir=tmp_path / "pair_roi",
        target_roi=str(roi_path),
        source_format="csv",
    )
    result = generate_target_response(spec)
    assert result.status == "success"
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["roi_energy_ratio"] == 0.5


def test_cli_pair_outputs_valid(tmp_path):
    json_path = tmp_path / "pair_valid.json"
    out_dir = tmp_path / "pair_valid_out"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--pair-outputs",
        "--campaign-id",
        "GX-003",
        "--scene-id",
        "scene_cli_ok",
        "--raw-output",
        str(FIXTURE_DIR / "raw_valid.csv"),
        "--background-output",
        str(FIXTURE_DIR / "background_valid.csv"),
        "--output-dir",
        str(out_dir),
        "--source-format",
        "csv",
        "--json",
        str(json_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert "target_response_npy_path" in payload


def test_cli_pair_outputs_shape_mismatch_nonzero(tmp_path):
    out_dir = tmp_path / "pair_bad_cli"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--pair-outputs",
        "--campaign-id",
        "GX-003",
        "--scene-id",
        "scene_cli_bad",
        "--raw-output",
        str(FIXTURE_DIR / "raw_shape_mismatch.csv"),
        "--background-output",
        str(FIXTURE_DIR / "background_shape_mismatch.csv"),
        "--output-dir",
        str(out_dir),
        "--source-format",
        "csv",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "shape_mismatch" in proc.stdout


def test_cli_pair_outputs_json_parseable_invalid(tmp_path):
    json_path = tmp_path / "pair_invalid.json"
    out_dir = tmp_path / "pair_invalid_out"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--pair-outputs",
        "--campaign-id",
        "GX-003",
        "--scene-id",
        "scene_cli_invalid_json",
        "--raw-output",
        str(FIXTURE_DIR / "raw_nan.csv"),
        "--background-output",
        str(FIXTURE_DIR / "background_valid.csv"),
        "--output-dir",
        str(out_dir),
        "--source-format",
        "csv",
        "--json",
        str(json_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "invalid"
    assert any(issue["code"] == "raw_nan_or_inf" for issue in payload["issues"])


def test_discover_converted_pair_paths_prefers_npy(tmp_path):
    scene_root = tmp_path / "scene"
    raw_dir = scene_root / "raw_with_target" / "converted"
    bg_dir = scene_root / "background_only" / "converted"
    raw_dir.mkdir(parents=True)
    bg_dir.mkdir(parents=True)
    np.save(raw_dir / "raw_bscan.npy", np.ones((2, 2), dtype=np.float64))
    np.save(bg_dir / "background_bscan.npy", np.zeros((2, 2), dtype=np.float64))

    raw, bg = discover_converted_pair_paths(scene_root, prefer_format="npy")
    assert raw.name == "raw_bscan.npy"
    assert bg.name == "background_bscan.npy"


def test_discover_converted_pair_paths_windows_style_string(tmp_path):
    scene_root = tmp_path / "scene_win"
    raw_dir = scene_root / "raw_with_target" / "converted"
    bg_dir = scene_root / "background_only" / "converted"
    raw_dir.mkdir(parents=True)
    bg_dir.mkdir(parents=True)
    np.savetxt(raw_dir / "raw_bscan.csv", np.ones((3, 1), dtype=np.float64), delimiter=",")
    np.savetxt(bg_dir / "background_bscan.csv", np.zeros((3, 1), dtype=np.float64), delimiter=",")

    win_like = str(scene_root).replace("/", "\\")
    raw, bg = discover_converted_pair_paths(win_like, prefer_format="csv")
    assert raw.name == "raw_bscan.csv"
    assert bg.name == "background_bscan.csv"

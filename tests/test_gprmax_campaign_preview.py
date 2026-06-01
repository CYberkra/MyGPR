#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for paired preview PNG generation and lightweight report stubs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from core.gprmax_campaign.preview import generate_pair_preview_report


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "gprmax_campaign" / "pairing"
RUNNER = ROOT / "scripts" / "gprmax_campaign_runner.py"
PAIR_CONVERTED = ROOT / "scripts" / "gprmax_campaign_pair_converted.py"


def test_preview_generation_valid_csv_pair(tmp_path):
    out_dir = tmp_path / "preview_valid"
    result = generate_pair_preview_report(
        campaign_id="GX-004",
        scene_id="scene_preview_ok",
        raw_output_path=FIXTURE_DIR / "raw_valid.csv",
        background_output_path=FIXTURE_DIR / "background_valid.csv",
        output_dir=out_dir,
        source_format="csv",
    )
    assert result.status == "success"
    assert result.raw_preview_path and result.raw_preview_path.exists()
    assert result.background_preview_path and result.background_preview_path.exists()
    assert result.target_response_preview_path and result.target_response_preview_path.exists()
    assert result.paired_preview_panel_path and result.paired_preview_panel_path.exists()
    assert result.report_md_path and result.report_md_path.exists()
    assert result.summary_json_path and result.summary_json_path.exists()
    assert result.paired_preview_panel_path.stat().st_size > 0
    report_text = result.report_md_path.read_text(encoding="utf-8")
    assert "campaign_id" in report_text
    assert "scene_id" in report_text
    assert "Claim Boundary" in report_text


def test_constant_arrays_do_not_crash_preview(tmp_path):
    out_dir = tmp_path / "preview_constant"
    result = generate_pair_preview_report(
        campaign_id="GX-004",
        scene_id="scene_constant",
        raw_output_path=FIXTURE_DIR / "raw_constant.csv",
        background_output_path=FIXTURE_DIR / "background_constant.csv",
        output_dir=out_dir,
        source_format="csv",
    )
    assert result.status == "success"
    assert result.paired_preview_panel_path and result.paired_preview_panel_path.exists()


def test_preview_accepts_existing_npy_target_with_csv_sources(tmp_path):
    out_dir = tmp_path / "preview_mixed_target"
    target_response = tmp_path / "target_response.npy"

    raw = np.genfromtxt(FIXTURE_DIR / "raw_valid.csv", delimiter=",")
    bg = np.genfromtxt(FIXTURE_DIR / "background_valid.csv", delimiter=",")
    np.save(target_response, raw - bg)
    result = generate_pair_preview_report(
        campaign_id="GX-004",
        scene_id="scene_mixed_target",
        raw_output_path=FIXTURE_DIR / "raw_valid.csv",
        background_output_path=FIXTURE_DIR / "background_valid.csv",
        target_response_path=target_response,
        output_dir=out_dir,
        source_format="csv",
    )
    assert result.status == "success"
    assert result.target_response_preview_path and result.target_response_preview_path.exists()


def test_preview_existing_target_shape_mismatch_reports_paths(tmp_path):
    out_dir = tmp_path / "preview_bad_target_shape"
    target_response = tmp_path / "target_response.npy"
    np.save(target_response, np.zeros((1, 2), dtype=np.float64))

    result = generate_pair_preview_report(
        campaign_id="GX-004",
        scene_id="scene_bad_target_shape",
        raw_output_path=FIXTURE_DIR / "raw_valid.csv",
        background_output_path=FIXTURE_DIR / "background_valid.csv",
        target_response_path=target_response,
        output_dir=out_dir,
        source_format="csv",
    )

    assert result.status == "invalid"
    issue = result.issues[0]
    assert issue["code"] == "target_response_shape_mismatch"
    assert "target_response.npy" in issue["message"]
    assert "raw_valid.csv" in issue["message"]


def test_preview_existing_target_unsupported_suffix_writes_json_summary(tmp_path):
    out_dir = tmp_path / "preview_bad_target_suffix"
    target_response = tmp_path / "target_response.txt"
    target_response.write_text("1,2\n3,4\n", encoding="utf-8")

    result = generate_pair_preview_report(
        campaign_id="GX-004",
        scene_id="scene_bad_target_suffix",
        raw_output_path=FIXTURE_DIR / "raw_valid.csv",
        background_output_path=FIXTURE_DIR / "background_valid.csv",
        target_response_path=target_response,
        output_dir=out_dir,
        source_format="csv",
    )

    assert result.status == "invalid"
    assert result.summary_json_path and result.summary_json_path.exists()
    payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert payload["issues"][0]["code"] == "target_response_load_failed"


def test_cli_preview_pair_valid(tmp_path):
    out_dir = tmp_path / "cli_preview_ok"
    json_path = tmp_path / "cli_preview_ok.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--preview-pair",
        "--campaign-id",
        "GX-004",
        "--scene-id",
        "scene_cli_preview_ok",
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
    assert payload["paired_preview_panel_path"] is not None


def test_cli_preview_pair_shape_mismatch_nonzero(tmp_path):
    out_dir = tmp_path / "cli_preview_bad"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--preview-pair",
        "--campaign-id",
        "GX-004",
        "--scene-id",
        "scene_cli_preview_bad",
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


def test_cli_preview_pair_json_parseable_invalid(tmp_path):
    out_dir = tmp_path / "cli_preview_invalid_json"
    json_path = tmp_path / "cli_preview_invalid.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--preview-pair",
        "--campaign-id",
        "GX-004",
        "--scene-id",
        "scene_cli_preview_invalid",
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


def test_pair_converted_script_gx008_like_scene_root(tmp_path):
    scene_root = tmp_path / "fixture_pair_scene"
    raw_dir = scene_root / "raw_with_target" / "converted"
    bg_dir = scene_root / "background_only" / "converted"
    raw_dir.mkdir(parents=True)
    bg_dir.mkdir(parents=True)
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    bg = np.array([[0.5, 1.0], [1.0, 1.5]], dtype=np.float64)
    np.save(raw_dir / "raw_bscan.npy", raw)
    np.save(bg_dir / "background_bscan.npy", bg)
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(
        json.dumps({"sample_range": [0, 2], "trace_range": [0, 2]}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "paired_outputs"
    json_path = tmp_path / "pair_preview_summary.json"
    cmd = [
        sys.executable,
        str(PAIR_CONVERTED),
        "--scene-root",
        str(scene_root),
        "--output-dir",
        str(out_dir),
        "--campaign-id",
        "GX-008",
        "--scene-id",
        "fixture_pair_scene",
        "--roi-json",
        str(roi_path),
        "--json",
        str(json_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["pair_result"]["status"] == "success"
    assert payload["preview_result"]["status"] == "success"
    assert "pair_metrics_keys" in payload
    assert "roi_energy_ratio" in payload["pair_metrics_keys"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for paired preview PNG generation and lightweight report stubs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from core.gprmax_campaign.preview import generate_pair_preview_report


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "gprmax_campaign" / "pairing"
RUNNER = ROOT / "scripts" / "gprmax_campaign_runner.py"


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

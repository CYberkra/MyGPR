#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI tests for gprMax campaign dry-run runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "gprmax_campaign"
RUNNER = ROOT / "scripts" / "gprmax_campaign_runner.py"


def test_cli_dry_run_prints_summary():
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(FIXTURE_DIR / "campaign_valid.yaml"),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "campaign_id: GX-RUN-001_valid" in proc.stdout
    assert "campaign_status: ready" in proc.stdout
    assert "scene_valid_01: ready" in proc.stdout


def test_cli_json_writes_parseable_report(tmp_path):
    report_path = tmp_path / "campaign_report.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(FIXTURE_DIR / "campaign_valid.yaml"),
        "--dry-run",
        "--json",
        str(report_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == "GX-RUN-001_valid"
    assert payload["status"] == "ready"
    assert payload["ready_count"] == 1

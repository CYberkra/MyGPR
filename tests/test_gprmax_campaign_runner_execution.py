#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax single-task execution wrapper and CLI execution mode."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from core.gprmax_campaign.runner import run_gprmax_task
from core.gprmax_campaign.schema import GprMaxTaskSpec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "gprmax_campaign"
RUNNER = ROOT / "scripts" / "gprmax_campaign_runner.py"
MISSING_EXPECTED = FIXTURE_DIR / "campaign_missing_expected_outputs.yaml"


def test_runner_success_writes_logs_and_manifest(tmp_path):
    task = GprMaxTaskSpec(
        campaign_id="C1",
        scene_id="S1",
        variant="raw_with_target",
        model_path=FIXTURE_DIR / "fake_gprmax_success.py",
        output_dir=tmp_path / "run_success",
        gprmax_executable=sys.executable,
    )
    result = run_gprmax_task(task)
    assert result.status == "success"
    assert result.return_code == 0
    assert result.stdout_path.exists()
    assert result.stderr_path.exists()
    assert result.manifest_path.exists()
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert "manifest_path" in payload
    assert str(payload["manifest_path"]).endswith("run_manifest.json")
    assert payload["manifest_path"] == result.to_dict()["manifest_path"]


def test_runner_failure_return_code_recorded(tmp_path):
    task = GprMaxTaskSpec(
        campaign_id="C1",
        scene_id="S1",
        variant="raw_with_target",
        model_path=FIXTURE_DIR / "fake_gprmax_fail.py",
        output_dir=tmp_path / "run_fail",
        gprmax_executable=sys.executable,
    )
    result = run_gprmax_task(task)
    assert result.status == "failed"
    assert result.return_code == 7
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["return_code"] == 7


def test_runner_timeout_recorded(tmp_path):
    task = GprMaxTaskSpec(
        campaign_id="C1",
        scene_id="S1",
        variant="raw_with_target",
        model_path=FIXTURE_DIR / "fake_gprmax_sleep.py",
        output_dir=tmp_path / "run_timeout",
        gprmax_executable=sys.executable,
        timeout_seconds=0.5,
    )
    result = run_gprmax_task(task)
    assert result.status == "timeout"
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "timeout"
    assert payload["timeout_seconds"] == 0.5


def test_runner_cancelled_recorded(tmp_path):
    cancel_event = threading.Event()
    task = GprMaxTaskSpec(
        campaign_id="C1",
        scene_id="S1",
        variant="raw_with_target",
        model_path=FIXTURE_DIR / "fake_gprmax_sleep.py",
        output_dir=tmp_path / "run_cancel",
        gprmax_executable=sys.executable,
    )

    def _cancel_soon():
        time.sleep(0.35)
        cancel_event.set()

    t = threading.Thread(target=_cancel_soon, daemon=True)
    t.start()
    result = run_gprmax_task(task, cancel_event=cancel_event)
    t.join(timeout=0.2)
    assert result.status == "cancelled"
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"


def test_cli_refuses_invalid_scene_for_execution():
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(MISSING_EXPECTED),
        "--run-scene",
        "scene_missing_expected_outputs",
        "--variant",
        "raw_with_target",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "execution refused" in proc.stdout.lower()
    assert "expected_outputs_missing" in proc.stdout


def test_cli_run_valid_scene_with_fake_executable(tmp_path):
    campaign_path = tmp_path / "campaign_exec.yaml"
    output_root = tmp_path / "campaign_output"
    campaign_path.write_text(
        "\n".join(
            [
                "campaign_id: GX-RUN-002_cli_exec",
                f"output_root: {output_root.as_posix()}",
                f"gprmax_executable: {sys.executable}",
                "scenes:",
                "  - scene_id: scene_valid_01",
                f"    raw_model: {str((FIXTURE_DIR / 'fake_gprmax_success.py').resolve()).replace('\\', '/')}",
                f"    background_model: {str((FIXTURE_DIR / 'fake_gprmax_fail.py').resolve()).replace('\\', '/')}",
                f"    materials: {str((FIXTURE_DIR / 'models' / 'materials.txt').resolve()).replace('\\', '/')}",
                f"    target_roi: {str((FIXTURE_DIR / 'annotations' / 'target_roi.json').resolve()).replace('\\', '/')}",
                "    expected_outputs:",
                "      - raw_with_target",
                "      - background_only",
                "      - target_response",
                "    tags: [exec]",
            ]
        ),
        encoding="utf-8",
    )
    result_json = tmp_path / "result.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(campaign_path),
        "--run-scene",
        "scene_valid_01",
        "--variant",
        "raw_with_target",
        "--json",
        str(result_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["status"] == "success"


def test_cli_num_runs_forwards_gprmax_n_argument(tmp_path):
    campaign_path = tmp_path / "campaign_exec.yaml"
    output_root = tmp_path / "campaign_output"
    campaign_path.write_text(
        "\n".join(
            [
                "campaign_id: GX-RUN-002_cli_num_runs",
                f"output_root: {output_root.as_posix()}",
                f"gprmax_executable: {sys.executable}",
                "scenes:",
                "  - scene_id: scene_valid_01",
                f"    raw_model: {str((FIXTURE_DIR / 'fake_gprmax_success.py').resolve()).replace('\\', '/')}",
                f"    background_model: {str((FIXTURE_DIR / 'fake_gprmax_fail.py').resolve()).replace('\\', '/')}",
                f"    materials: {str((FIXTURE_DIR / 'models' / 'materials.txt').resolve()).replace('\\', '/')}",
                f"    target_roi: {str((FIXTURE_DIR / 'annotations' / 'target_roi.json').resolve()).replace('\\', '/')}",
                "    expected_outputs:",
                "      - raw_with_target",
                "      - background_only",
                "      - target_response",
            ]
        ),
        encoding="utf-8",
    )
    result_json = tmp_path / "result.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(campaign_path),
        "--run-scene",
        "scene_valid_01",
        "--variant",
        "raw_with_target",
        "--num-runs",
        "3",
        "--json",
        str(result_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["command"][-2:] == ["-n", "3"]


def test_cli_rejects_non_positive_num_runs(tmp_path):
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(MISSING_EXPECTED),
        "--run-scene",
        "scene_missing_expected_outputs",
        "--variant",
        "raw_with_target",
        "--num-runs",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "positive integer" in proc.stderr

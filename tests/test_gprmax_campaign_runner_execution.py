#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax single-task execution wrapper and CLI execution mode."""

from __future__ import annotations

import json
import os
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
    assert payload["gprmax_command_mode"] == "path_executable"
    assert "manifest_path" in payload
    assert str(payload["manifest_path"]).endswith("run_manifest.json")
    assert payload["manifest_path"] == result.to_dict()["manifest_path"]
    assert payload["gpu_requested"] is False
    assert payload["gpu_flag_emitted"] is False
    assert payload["gpu_device_ids"] == []


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
    assert payload["startup_error"] is False


def test_runner_supports_executable_with_inline_args(tmp_path):
    task = GprMaxTaskSpec(
        campaign_id="C1",
        scene_id="S1",
        variant="raw_with_target",
        model_path=FIXTURE_DIR / "fake_gprmax_success.py",
        output_dir=tmp_path / "run_exec_inline_args",
        gprmax_executable=f"{sys.executable} -u",
    )
    result = run_gprmax_task(task)
    assert result.status == "success"
    assert result.return_code == 0
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["command"][0].endswith("python.exe")
    assert payload["command"][1] == "-u"
    assert payload["gprmax_command_mode"] == "path_executable"


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
    assert payload["requested_num_runs"] == 3
    assert payload["gpu_requested"] is False


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


def test_cli_gpu_flag_passthrough_without_device(tmp_path):
    campaign_path = tmp_path / "campaign_gpu.yaml"
    output_root = tmp_path / "campaign_output"
    campaign_path.write_text(
        "\n".join(
            [
                "campaign_id: GX-RUN-GPU-001_cli_gpu",
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
    result_json = tmp_path / "result_gpu.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(campaign_path),
        "--run-scene",
        "scene_valid_01",
        "--variant",
        "raw_with_target",
        "--gpu",
        "--json",
        str(result_json),
    ]
    env = dict(os.environ)
    env["MYGPR_GPU_CHECK_NVCC"] = "0"
    env["MYGPR_GPU_CHECK_PYCUDA"] = "0"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["command"][-1] == "-gpu"
    assert payload["gpu_requested"] is True
    assert payload["gpu_flag_emitted"] is True
    assert payload["gpu_device_ids"] == []
    assert payload["nvcc_available"] is False
    assert payload["pycuda_available"] is False
    assert "gpu_warning" in payload and payload["gpu_warning"]


def test_cli_gpu_single_device_passthrough(tmp_path):
    campaign_path = tmp_path / "campaign_gpu_single.yaml"
    output_root = tmp_path / "campaign_output"
    campaign_path.write_text(
        "\n".join(
            [
                "campaign_id: GX-RUN-GPU-001_cli_gpu_single",
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
    result_json = tmp_path / "result_gpu_single.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(campaign_path),
        "--run-scene",
        "scene_valid_01",
        "--variant",
        "raw_with_target",
        "--gpu-device",
        "0",
        "--json",
        str(result_json),
    ]
    env = dict(os.environ)
    env["MYGPR_GPU_CHECK_NVCC"] = "1"
    env["MYGPR_GPU_CHECK_PYCUDA"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["command"][-2:] == ["-gpu", "0"]
    assert payload["gpu_device_ids"] == [0]
    assert payload["nvcc_available"] is True
    assert payload["pycuda_available"] is True
    assert payload["gpu_warning"] is None


def test_cli_gpu_multi_devices_passthrough(tmp_path):
    campaign_path = tmp_path / "campaign_gpu_multi.yaml"
    output_root = tmp_path / "campaign_output"
    campaign_path.write_text(
        "\n".join(
            [
                "campaign_id: GX-RUN-GPU-001_cli_gpu_multi",
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
    result_json = tmp_path / "result_gpu_multi.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(campaign_path),
        "--run-scene",
        "scene_valid_01",
        "--variant",
        "raw_with_target",
        "--gpu-devices",
        "0",
        "1",
        "2",
        "3",
        "--json",
        str(result_json),
    ]
    env = dict(os.environ)
    env["MYGPR_GPU_CHECK_NVCC"] = "1"
    env["MYGPR_GPU_CHECK_PYCUDA"] = "0"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["command"][-5:] == ["-gpu", "0", "1", "2", "3"]
    assert payload["gpu_device_ids"] == [0, 1, 2, 3]
    assert payload["gpu_warning"] and "pycuda" in payload["gpu_warning"]


def test_cli_num_runs_and_gpu_passthrough_combined(tmp_path):
    campaign_path = tmp_path / "campaign_gpu_num_runs.yaml"
    output_root = tmp_path / "campaign_output"
    campaign_path.write_text(
        "\n".join(
            [
                "campaign_id: GX-RUN-GPU-001_cli_gpu_combo",
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
    result_json = tmp_path / "result_gpu_combo.json"
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
        "60",
        "--gpu-device",
        "0",
        "--json",
        str(result_json),
    ]
    env = dict(os.environ)
    env["MYGPR_GPU_CHECK_NVCC"] = "1"
    env["MYGPR_GPU_CHECK_PYCUDA"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert "-n" in payload["command"]
    assert "60" in payload["command"]
    assert payload["command"][-2:] == ["-gpu", "0"]
    assert payload["requested_num_runs"] == 60


def test_cli_gprmax_python_builds_python_module_command(tmp_path):
    campaign_path = tmp_path / "campaign_gpu_py.yaml"
    output_root = tmp_path / "campaign_output"
    campaign_path.write_text(
        "\n".join(
            [
                "campaign_id: GX-RUN-GPRMAX-PYTHON-001_cli_py",
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
    result_json = tmp_path / "result_gpu_py.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(campaign_path),
        "--run-scene",
        "scene_valid_01",
        "--variant",
        "raw_with_target",
        "--gprmax-python",
        str(sys.executable),
        "--num-runs",
        "21",
        "--gpu-device",
        "0",
        "--json",
        str(result_json),
    ]
    env = dict(os.environ)
    env["MYGPR_GPU_CHECK_NVCC"] = "1"
    env["MYGPR_GPU_CHECK_PYCUDA"] = "0"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 1
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["gprmax_command_mode"] == "python_module"
    assert payload["gprmax_python"] == str(Path(sys.executable).resolve())
    assert payload["command"][:3] == [str(Path(sys.executable).resolve()), "-m", "gprMax"]
    assert payload["command"][-4:] == ["-n", "21", "-gpu", "0"]
    assert payload["gpu_warning"] and "external gprMax runtime python" in payload["gpu_warning"]


def test_cli_gprmax_python_missing_path_fails_fast(tmp_path):
    campaign_path = tmp_path / "campaign_gpu_py_missing.yaml"
    output_root = tmp_path / "campaign_output"
    campaign_path.write_text(
        "\n".join(
            [
                "campaign_id: GX-RUN-GPRMAX-PYTHON-001_cli_py_missing",
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
    missing_py = tmp_path / "not_found_python.exe"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(campaign_path),
        "--run-scene",
        "scene_valid_01",
        "--variant",
        "raw_with_target",
        "--gprmax-python",
        str(missing_py),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "gprMax runtime python not found" in proc.stdout


def test_cli_rejects_gpu_device_and_gpu_devices_together(tmp_path):
    cmd = [
        sys.executable,
        str(RUNNER),
        "--campaign",
        str(MISSING_EXPECTED),
        "--run-scene",
        "scene_missing_expected_outputs",
        "--variant",
        "raw_with_target",
        "--gpu-device",
        "0",
        "--gpu-devices",
        "1",
        "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "mutually exclusive" in proc.stdout

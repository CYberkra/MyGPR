#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run manifest writer for gprMax campaign task execution."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from core.gprmax_campaign.schema import GprMaxRunResult, GprMaxTaskSpec


def build_run_manifest_payload(
    task: GprMaxTaskSpec,
    result: GprMaxRunResult,
) -> dict[str, Any]:
    """Build a compact JSON payload for run_manifest.json."""
    gpu_device_ids = [int(i) for i in (task.gpu_device_ids or [])]
    command = list(result.command)
    gpu_flag_emitted = "-gpu" in command
    return {
        "schema_version": "gprmax_campaign_run_manifest_v1",
        "campaign_id": result.campaign_id,
        "scene_id": result.scene_id,
        "variant": result.variant,
        "model_path": str(result.model_path),
        "output_dir": str(result.output_dir),
        "command": command,
        "gprmax_python": str(task.gprmax_python) if task.gprmax_python else None,
        "gprmax_command_mode": result.command_mode,
        "status": result.status,
        "run_status": result.status,
        "return_code": result.return_code,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "runtime_seconds": result.runtime_seconds,
        "stdout_path": str(result.stdout_path),
        "stderr_path": str(result.stderr_path),
        "manifest_path": str(result.manifest_path),
        "timeout_seconds": task.timeout_seconds,
        "requested_num_runs": task.requested_num_runs,
        "gpu_requested": bool(task.gpu_requested),
        "gpu_flag_emitted": gpu_flag_emitted,
        "gpu_device_ids": gpu_device_ids,
        "nvcc_available": task.nvcc_available,
        "pycuda_available": task.pycuda_available,
        "gpu_warning": task.gpu_warning,
        "startup_error": result.startup_error,
        "error_message": result.error_message,
        "host": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
        },
    }


def write_run_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write run manifest JSON and return its resolved path."""
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path

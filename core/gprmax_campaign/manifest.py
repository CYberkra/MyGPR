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
    return {
        "schema_version": "gprmax_campaign_run_manifest_v1",
        "campaign_id": result.campaign_id,
        "scene_id": result.scene_id,
        "variant": result.variant,
        "model_path": str(result.model_path),
        "output_dir": str(result.output_dir),
        "command": list(result.command),
        "status": result.status,
        "return_code": result.return_code,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "runtime_seconds": result.runtime_seconds,
        "stdout_path": str(result.stdout_path),
        "stderr_path": str(result.stderr_path),
        "manifest_path": str(result.manifest_path),
        "timeout_seconds": task.timeout_seconds,
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

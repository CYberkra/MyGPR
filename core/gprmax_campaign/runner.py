#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-task local gprMax execution wrapper (backend-only)."""

from __future__ import annotations

import subprocess
import time
import shlex
from datetime import datetime, timezone
from pathlib import Path

from core.gprmax_campaign.manifest import (
    build_run_manifest_payload,
    write_run_manifest,
)
from core.gprmax_campaign.schema import GprMaxRunResult, GprMaxTaskSpec


def run_gprmax_task(task: GprMaxTaskSpec, cancel_event=None) -> GprMaxRunResult:
    """Run one local gprMax task and persist run manifest."""
    output_dir = Path(task.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(task.model_path).expanduser().resolve()
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    manifest_path = output_dir / "run_manifest.json"

    command = _build_command(task, model_path)
    start_ts = time.perf_counter()
    started_at = _iso_now()
    status = "failed"
    return_code = None
    error_message = None
    process = None
    grace_seconds = 1.5

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=str(output_dir),
                text=True,
            )
            deadline = (
                time.perf_counter() + float(task.timeout_seconds)
                if task.timeout_seconds is not None
                else None
            )
            while True:
                polled = process.poll()
                if polled is not None:
                    return_code = int(polled)
                    status = "success" if return_code == 0 else "failed"
                    break
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    _terminate_process(process, grace_seconds)
                    return_code = process.poll()
                    status = "cancelled"
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    _terminate_process(process, grace_seconds)
                    return_code = process.poll()
                    status = "timeout"
                    break
                time.sleep(0.1)
        except FileNotFoundError as exc:
            status = "failed"
            error_message = str(exc)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            status = "failed"
            error_message = str(exc)
            if process is not None and process.poll() is None:
                _terminate_process(process, grace_seconds)
                return_code = process.poll()

    ended_at = _iso_now()
    runtime_seconds = max(0.0, time.perf_counter() - start_ts)
    result = GprMaxRunResult(
        campaign_id=task.campaign_id,
        scene_id=task.scene_id,
        variant=task.variant,
        model_path=model_path,
        output_dir=output_dir,
        command=command,
        status=status,
        return_code=return_code,
        started_at=started_at,
        ended_at=ended_at,
        runtime_seconds=runtime_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        manifest_path=manifest_path,
        error_message=error_message,
    )
    payload = build_run_manifest_payload(task, result)
    write_run_manifest(manifest_path, payload)
    return result


def _terminate_process(process: subprocess.Popen, grace_seconds: float) -> None:
    try:
        process.terminate()
        process.wait(timeout=grace_seconds)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=grace_seconds)
        except Exception:
            return


def _build_command(task: GprMaxTaskSpec, model_path: Path) -> list[str]:
    raw_exec = str(task.gprmax_executable).strip()
    if not raw_exec:
        return [str(model_path), *list(task.extra_args or [])]
    try:
        exec_tokens = shlex.split(raw_exec, posix=False)
    except ValueError:
        exec_tokens = [raw_exec]
    if not exec_tokens:
        exec_tokens = [raw_exec]
    return [*exec_tokens, str(model_path), *list(task.extra_args or [])]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

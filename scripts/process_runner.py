#!/usr/bin/env python3
"""Cross-platform logged subprocess execution with process-tree timeouts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LoggedProcessResult:
    returncode: int
    duration_s: float
    timed_out: bool
    output_tail: str
    log_path: str


def safe_log_name(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return compact[:120] or "stage"


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_logged_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: int,
    log_path: Path,
    tail_chars: int = 6000,
    heartbeat_label: str | None = None,
    heartbeat_interval_s: float = 5.0,
) -> LoggedProcessResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
        deadline = started + float(timeout)
        next_heartbeat = started + max(1.0, float(heartbeat_interval_s))
        while True:
            polled = process.poll()
            if polled is not None:
                returncode = int(polled)
                break
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                _kill_process_tree(process)
                returncode = 124
                handle.write(f"\nTIMEOUT after {timeout}s\n")
                handle.flush()
                break
            if heartbeat_label and now >= next_heartbeat:
                elapsed = int(now - started)
                print(f"[runner] {heartbeat_label}: still running ({elapsed}s)", flush=True)
                next_heartbeat = now + max(1.0, float(heartbeat_interval_s))
            time.sleep(min(0.25, max(0.01, deadline - now)))
    try:
        output = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = ""
    return LoggedProcessResult(
        returncode=int(returncode),
        duration_s=round(time.monotonic() - started, 3),
        timed_out=timed_out,
        output_tail=output[-tail_chars:],
        log_path=str(log_path),
    )

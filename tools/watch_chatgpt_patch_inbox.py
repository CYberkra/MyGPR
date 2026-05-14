#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch .chatgpt_inbox and apply incoming patch files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INBOX = REPO_ROOT / ".chatgpt_inbox"
DEFAULT_LOG = REPO_ROOT / "output" / "chatgpt_patch_watcher.log"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(message: str, log_path: Path) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _wait_until_stable(path: Path, interval: float) -> bool:
    try:
        first = path.stat()
    except FileNotFoundError:
        return False
    time.sleep(max(0.2, interval))
    try:
        second = path.stat()
    except FileNotFoundError:
        return False
    return first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns


def _next_patch(inbox: Path) -> Path | None:
    patches = sorted(
        path for path in inbox.glob("*.patch") if path.is_file() and not path.name.startswith(".")
    )
    return patches[0] if patches else None


def _process_patch(patch: Path, args: argparse.Namespace, log_path: Path) -> bool:
    processing_dir = args.inbox / "_processing"
    done_dir = args.inbox / "_done"
    failed_dir = args.inbox / "_failed"
    for directory in [processing_dir, done_dir, failed_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    working_patch = processing_dir / f"{_timestamp()}_{patch.name}"
    shutil.move(str(patch), str(working_patch))
    _log(f"[patch] processing {working_patch}", log_path)

    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "apply_chatgpt_patch.py"),
        "--branch",
        args.branch,
        "--patch-file",
        str(working_patch),
    ]
    if args.mygpr_smoke:
        command.append("--mygpr-smoke")
    if args.allow_missing_base:
        command.append("--allow-missing-base")
    command.append("--push" if args.push else "--no-push")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        for line in result.stdout.splitlines():
            _log("  " + line, log_path)

    if result.returncode == 0:
        target = done_dir / working_patch.name
        shutil.move(str(working_patch), str(target))
        _log(f"[patch] done {target}", log_path)
        return True

    target = failed_dir / working_patch.name
    shutil.move(str(working_patch), str(target))
    _log(f"[patch] failed {target}", log_path)
    return False


def watch(args: argparse.Namespace) -> int:
    args.inbox.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    _log(
        f"[watch] inbox={args.inbox} branch={args.branch} push={args.push} smoke={args.mygpr_smoke}",
        args.log_path,
    )
    while True:
        patch = _next_patch(args.inbox)
        if patch is None:
            if args.once:
                _log("[watch] no patch found; exiting because --once was set", args.log_path)
                return 0
            time.sleep(args.poll_interval)
            continue

        if not _wait_until_stable(patch, args.poll_interval):
            continue

        ok = _process_patch(patch, args, args.log_path)
        if args.once:
            return 0 if ok else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--mygpr-smoke", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--allow-missing-base",
        action="store_true",
        help="Allow legacy patches without a Base-Commit header.",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    args.inbox = args.inbox.expanduser().resolve()
    args.log_path = args.log_path.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        return watch(_parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - watcher should log concise failures.
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create verified Git checkpoints for MyGPR development."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


class GitCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be created safely."""


def _flatten_file_args(items: list[list[str]] | None) -> list[str]:
    files: list[str] = []
    for group in items or []:
        files.extend(str(item).strip() for item in group if str(item).strip())
    return files


def _run_command(command: str, *, repo_root: Path) -> None:
    print(f"[verify] {command}")
    result = subprocess.run(command, cwd=repo_root, shell=True, check=False)
    if result.returncode != 0:
        raise GitCheckpointError(
            f"verification command failed with exit code {result.returncode}: {command}"
        )


def _safe_print(text: str, *, file=None) -> None:
    """Print subprocess output without failing on a narrow Windows console encoding."""
    if not text:
        return
    target = file or sys.stdout
    payload = text.rstrip()
    try:
        print(payload, file=target)
    except UnicodeEncodeError:
        encoding = getattr(target, "encoding", None) or "utf-8"
        safe_payload = payload.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
        print(safe_payload, file=target)


def _run_git(
    args: Sequence[str],
    *,
    repo_root: Path,
    capture: bool = True,
) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise GitCheckpointError(
            f"git {' '.join(args)} failed with exit code {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    return (result.stdout or "").strip() if capture else ""


def _print_status(repo_root: Path) -> None:
    branch = _run_git(["branch", "--show-current"], repo_root=repo_root)
    status = _run_git(["status", "--short"], repo_root=repo_root)
    print(f"[git] branch: {branch or '(detached)'}")
    print("[git] status:")
    print(status or "  clean")


def _ensure_no_preexisting_staged_changes(repo_root: Path) -> None:
    staged = _run_git(["diff", "--cached", "--name-only"], repo_root=repo_root)
    if staged.strip():
        raise GitCheckpointError(
            "pre-existing staged changes detected; unstage or commit them before using "
            "git_checkpoint.py:\n" + staged
        )


def _slugify_tag_topic(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", str(text).strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "checkpoint"


def _tag_exists(tag_name: str, *, repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode == 0


def _unique_tag_name(topic: str, *, repo_root: Path) -> str:
    base = f"v{datetime.now().strftime('%Y-%m-%d')}-{_slugify_tag_topic(topic)}"
    if not _tag_exists(base, repo_root=repo_root):
        return base
    suffix = 2
    while True:
        candidate = f"{base}-{suffix}"
        if not _tag_exists(candidate, repo_root=repo_root):
            return candidate
        suffix += 1


def _run_archive_checkpoint(
    *,
    repo_root: Path,
    summary: str,
    topic: str,
) -> str:
    script = repo_root / "scripts" / "archive_checkpoint.py"
    print(f"[archive] {script}")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary",
            summary,
            "--topic",
            topic,
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.stdout:
        _safe_print(result.stdout)
    if result.stderr:
        _safe_print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise GitCheckpointError(
            f"archive checkpoint failed with exit code {result.returncode}"
        )
    for line in (result.stdout or "").splitlines():
        if line.startswith("Archived note:"):
            return line.split(":", 1)[1].strip()
    return ""


def _run_meeting_progress_note(
    *,
    repo_root: Path,
    summary: str,
    progress: Sequence[str],
    results: Sequence[str],
    risks: Sequence[str],
    next_steps: Sequence[str],
    commit: str,
    tag: str,
    archive_note: str,
) -> None:
    script = repo_root / "scripts" / "meeting_progress_note.py"
    command = [
        sys.executable,
        str(script),
        "--summary",
        summary,
        "--commit",
        commit,
    ]
    if tag:
        command.extend(["--tag", tag])
    if archive_note:
        command.extend(["--archive-note", archive_note])
    for item in progress:
        command.extend(["--progress", item])
    for item in results:
        command.extend(["--result", item])
    for item in risks:
        command.extend(["--risk", item])
    for item in next_steps:
        command.extend(["--next-step", item])

    print(f"[meeting] {script}")
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.stdout:
        _safe_print(result.stdout)
    if result.stderr:
        _safe_print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise GitCheckpointError(
            f"meeting progress note failed with exit code {result.returncode}"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a verified Git checkpoint from explicit pathspecs."
    )
    parser.add_argument("--summary", required=True, help="Commit summary/message")
    parser.add_argument(
        "--files",
        action="append",
        nargs="+",
        required=True,
        help="Files or directories to stage. Can be repeated.",
    )
    parser.add_argument(
        "--verify",
        action="append",
        default=[],
        help="Verification command to run before staging. Can be repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "important"],
        default="normal",
        help="normal commits only; important also tags and archives.",
    )
    parser.add_argument(
        "--topic",
        default="",
        help="Important checkpoint topic for tag and Obsidian archive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without running verification, staging, or committing.",
    )
    parser.add_argument(
        "--meeting-progress",
        action="append",
        default=[],
        help="Important-mode meeting note progress bullet. Can be repeated.",
    )
    parser.add_argument(
        "--meeting-result",
        action="append",
        default=[],
        help="Important-mode meeting note displayable-result bullet. Can be repeated.",
    )
    parser.add_argument(
        "--meeting-risk",
        action="append",
        default=[],
        help="Important-mode meeting note risk bullet. Can be repeated.",
    )
    parser.add_argument(
        "--meeting-next",
        action="append",
        default=[],
        help="Important-mode meeting note next-step bullet. Can be repeated.",
    )
    return parser.parse_args(argv)


def create_checkpoint(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT) -> int:
    repo_root = Path(repo_root)
    files = _flatten_file_args(args.files)
    summary = str(args.summary).strip()
    topic = str(args.topic or summary).strip()
    if not summary:
        raise GitCheckpointError("--summary must not be empty")
    if not files:
        raise GitCheckpointError("--files must include at least one pathspec")

    _print_status(repo_root)

    if args.dry_run:
        print("[dry-run] verification commands:")
        for command in args.verify:
            print(f"  {command}")
        print("[dry-run] would run: git diff --check")
        print("[dry-run] would run: git add -- " + " ".join(files))
        print(f"[dry-run] would commit: {summary}")
        if args.mode == "important":
            tag = _unique_tag_name(topic, repo_root=repo_root)
            print(f"[dry-run] would tag: {tag}")
            print("[dry-run] would archive to Obsidian")
            print("[dry-run] would append meeting progress note")
        return 0

    _ensure_no_preexisting_staged_changes(repo_root)
    for command in args.verify:
        _run_command(command, repo_root=repo_root)

    print("[git] diff --check")
    _run_git(["diff", "--check"], repo_root=repo_root, capture=False)

    print("[git] add explicit pathspecs")
    _run_git(["add", "--", *files], repo_root=repo_root, capture=False)

    staged_files = _run_git(["diff", "--cached", "--name-only"], repo_root=repo_root)
    if not staged_files.strip():
        print("[git] no staged diff after git add; checkpoint skipped")
        return 0

    print("[git] staged files:")
    print(staged_files)
    _run_git(["commit", "-m", summary], repo_root=repo_root, capture=False)
    print(f"[git] committed: {summary}")

    if args.mode == "important":
        tag = _unique_tag_name(topic, repo_root=repo_root)
        _run_git(["tag", tag], repo_root=repo_root, capture=False)
        print(f"[git] tagged: {tag}")
        archive_note = _run_archive_checkpoint(
            repo_root=repo_root, summary=summary, topic=topic
        )
        commit = _run_git(["rev-parse", "--short", "HEAD"], repo_root=repo_root)
        _run_meeting_progress_note(
            repo_root=repo_root,
            summary=summary,
            progress=list(args.meeting_progress or []),
            results=list(args.meeting_result or []),
            risks=list(args.meeting_risk or []),
            next_steps=list(args.meeting_next or []),
            commit=commit,
            tag=tag,
            archive_note=archive_note,
        )

    return 0


def main(argv: Sequence[str] | None = None, *, repo_root: Path = REPO_ROOT) -> int:
    args = _parse_args(argv)
    try:
        return create_checkpoint(args, repo_root=repo_root)
    except GitCheckpointError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

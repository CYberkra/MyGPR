#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply a ChatGPT-generated patch with MyGPR smoke verification."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"[run] {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed with exit {result.returncode}: {' '.join(command)}")
    return result


def _git_output(args: list[str]) -> str:
    result = _run(["git", *args], check=True)
    return result.stdout or ""


def _ensure_clean() -> None:
    status = _git_output(["status", "--porcelain"])
    if status.strip():
        raise RuntimeError("working tree is not clean; refusing to apply patch\n" + status)


def _changed_files() -> list[str]:
    output = _git_output(["diff", "--name-only"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def _apply_patch_file(patch_file: Path) -> None:
    """Apply patch, falling back to git's 3-way merge for shifted context."""
    check_result = _run(["git", "apply", "--check", str(patch_file)], check=False)
    if check_result.returncode == 0:
        _run(["git", "apply", str(patch_file)])
        return

    print("[apply] direct git apply failed; trying git apply --3way", flush=True)
    three_way_result = _run(["git", "apply", "--3way", str(patch_file)], check=False)
    if three_way_result.returncode != 0:
        raise RuntimeError(
            "patch could not be applied directly or with --3way; "
            "it is likely based on stale file content"
        )

    unresolved = _git_output(["diff", "--name-only", "--diff-filter=U"])
    if unresolved.strip():
        raise RuntimeError("patch produced unresolved conflicts\n" + unresolved)


def _summary_from_patch(patch_file: Path) -> str:
    stem = patch_file.stem
    stem = re.sub(r"^\d+[_-]*", "", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return f"apply: {stem}" if stem else f"apply: {patch_file.name}"


def _verify_mygpr_smoke(changed: list[str]) -> None:
    py_files = [path for path in changed if path.endswith(".py") and Path(REPO_ROOT, path).exists()]
    if py_files:
        _run([sys.executable, "-m", "py_compile", *py_files])

    workflow_touched = any(
        path.startswith("ui/workflow")
        or path.startswith("ui\\workflow")
        or path in {"ui/gui_workflow_page.py", "ui\\gui_workflow_page.py", "app_qt.py"}
        for path in changed
    )
    if workflow_touched:
        _run(
            [
                "pytest",
                "tests\\test_workflow_page_editor.py",
                "tests\\test_workflow_ui_contracts.py",
                "-q",
            ]
        )

    _run([sys.executable, "scripts\\preflight_check.py"])


def apply_patch(args: argparse.Namespace) -> int:
    patch_file = Path(args.patch_file).expanduser().resolve()
    if not patch_file.exists():
        raise FileNotFoundError(patch_file)

    _ensure_clean()
    _run(["git", "checkout", args.branch])
    _run(["git", "pull", "--ff-only"])
    _ensure_clean()

    _apply_patch_file(patch_file)

    changed = _changed_files()
    if not changed:
        print("[apply] patch produced no changes; nothing to commit")
        return 0

    if args.mygpr_smoke:
        _verify_mygpr_smoke(changed)

    _run(["git", "diff", "--check"])
    _run(["git", "add", "--", *changed])
    _run(["git", "commit", "-m", args.summary or _summary_from_patch(patch_file)])

    if args.push:
        _run(["git", "push", "origin", args.branch])

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--patch-file", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--mygpr-smoke", action="store_true")
    push_group = parser.add_mutually_exclusive_group()
    push_group.add_argument("--push", action="store_true")
    push_group.add_argument("--no-push", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.no_push:
        args.push = False
    try:
        return apply_patch(args)
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure.
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

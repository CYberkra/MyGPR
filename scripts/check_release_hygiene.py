#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that a MyGPR source tree is safe to package for beta release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

FORBIDDEN_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_TOP_LEVEL = {"runtime_projects", "logs", "build", "dist"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".rej", ".orig"}
FORBIDDEN_FILE_NAMES = {"mygpr_handoff.zip"}

# These directories are legitimate local developer state and must not make a
# source checkout fail hygiene checks.  They are pruned before traversal so a
# project-local virtual environment does not generate thousands of false hits.
# Cache dirs (__pycache__/.mypy_cache/...) remain forbidden in a *release*
# package but are produced by any CI/dev run, so traversal skips them.
EXCLUDED_DIR_NAMES = {".venv", "venv", "env", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def scan(root: Path) -> dict:
    root = root.resolve()
    findings: list[str] = []

    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)

        # Do not descend into legitimate local environments or VCS metadata.
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIR_NAMES]

        for dirname in list(dirnames):
            path = current_path / dirname
            rel_parts = path.relative_to(root).parts
            rel = path.relative_to(root).as_posix()
            if dirname in FORBIDDEN_DIR_NAMES or (rel_parts and rel_parts[0] in FORBIDDEN_TOP_LEVEL):
                findings.append(rel)
                dirnames.remove(dirname)

        for filename in filenames:
            path = current_path / filename
            rel_path = path.relative_to(root)
            rel = rel_path.as_posix()
            if rel_path.parts and rel_path.parts[0] in FORBIDDEN_TOP_LEVEL:
                findings.append(rel)
                continue
            if filename in FORBIDDEN_FILE_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
                findings.append(rel)

    findings.sort()
    return {
        "schema": "mygpr.release_hygiene.v1",
        "project_root": str(root),
        "ok": not findings,
        "finding_count": len(findings),
        "findings": findings[:500],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check MyGPR release tree hygiene.")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    payload = scan(project_root())
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("MyGPR release hygiene:", "OK" if payload["ok"] else "FAILED")
        print("Findings:", payload["finding_count"])
        for item in payload["findings"][:50]:
            print("  -", item)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

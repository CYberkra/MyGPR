#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that a MyGPR source tree is safe to package for beta release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

FORBIDDEN_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_TOP_LEVEL = {"runtime_projects", "logs", "build", "dist"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def scan(root: Path) -> dict:
    findings: list[str] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if any(part in FORBIDDEN_DIR_NAMES for part in path.parts):
            findings.append(rel)
            continue
        if path.parts and path.relative_to(root).parts[0] in FORBIDDEN_TOP_LEVEL:
            findings.append(rel)
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(rel)
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

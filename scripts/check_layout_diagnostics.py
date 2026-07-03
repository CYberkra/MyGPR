#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check MyGPR layout_diagnostics.json against release layout rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.field_panels.layout_diagnostics_rules import check_layout_diagnostics_payload  # noqa: E402


def _resolve_input(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "layout_diagnostics.json"
    return candidate


def check_layout_diagnostics(path: str | Path) -> dict:
    source = _resolve_input(path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return check_layout_diagnostics_payload(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check MyGPR layout diagnostics pass/fail rules.")
    parser.add_argument("path", help="Path to layout_diagnostics.json or a capture directory containing it.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args(argv)

    try:
        report = check_layout_diagnostics(args.path)
    except Exception as exc:
        print(f"layout_check_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report.get("ok"):
        print(f"layout_check_ok: pages={report.get('pages_checked', 0)} issues=0")
    else:
        print(f"layout_check_failed: issues={report.get('issue_count', 0)}", file=sys.stderr)
        for issue in report.get("issues", [])[:20]:
            print(
                f"- [{issue.get('severity')}] {issue.get('page')}::{issue.get('widget')} "
                f"{issue.get('rule')} | {issue.get('observed')} | {issue.get('message')}",
                file=sys.stderr,
            )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

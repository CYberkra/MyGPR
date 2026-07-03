#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check MyGPR visual comfort diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.field_panels.visual_comfort_rules import check_visual_comfort_payload


def _resolve_payload_path(path: Path) -> Path:
    if path.is_dir():
        return path / "layout_diagnostics.json"
    return path


def check_visual_comfort(path: str | Path) -> dict:
    target = _resolve_payload_path(Path(path))
    payload = json.loads(target.read_text(encoding="utf-8"))
    return check_visual_comfort_payload(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check MyGPR visual comfort diagnostics.")
    parser.add_argument("path", help="layout_diagnostics.json or a capture directory containing it")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args(argv)
    report = check_visual_comfort(args.path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report.get("ok"):
        print(f"visual_check_ok: pages={report.get('pages_checked', 0)} issues=0")
    else:
        print(f"visual_check_failed: issues={report.get('issue_count', 0)}", file=sys.stderr)
        for issue in report.get("issues", [])[:20]:
            print(
                f"- [{issue.get('severity')}] {issue.get('page')} {issue.get('rule')} {issue.get('widget')}: "
                f"{issue.get('observed')} ({issue.get('message')})",
                file=sys.stderr,
            )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

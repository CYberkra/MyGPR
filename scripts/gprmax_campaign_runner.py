#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-only gprMax campaign loader and dry-run validator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gprmax_campaign import load_campaign_yaml, validate_campaign


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gprMax campaign backend runner (GX-RUN-001 dry-run only).",
    )
    parser.add_argument(
        "--campaign",
        required=True,
        help="Path to campaign YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate campaign only; do not execute gprMax.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional path to write dry-run JSON summary.",
    )
    return parser.parse_args(argv)


def _print_summary(result) -> None:
    print(f"campaign_id: {result.campaign_id}")
    print(f"campaign_status: {result.status}")
    print(f"total_scenes: {result.total_scenes}")
    print(f"ready_count: {result.ready_count}")
    print(f"warning_count: {result.warning_count}")
    print(f"invalid_count: {result.invalid_count}")
    if result.issues:
        print("campaign_issues:")
        for issue in result.issues:
            print(f"  - [{issue.level}] {issue.code}: {issue.message}")
    print("scene_status:")
    for scene in result.scenes:
        print(f"  - {scene.scene_id}: {scene.status}")
        for issue in scene.issues:
            print(f"      [{issue.level}] {issue.code}: {issue.message}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.dry_run:
        print(
            "GX-RUN-001 supports dry-run validation only. "
            "Execution mode is not implemented yet. Use --dry-run."
        )
        return 2

    try:
        campaign = load_campaign_yaml(args.campaign)
        result = validate_campaign(campaign)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    _print_summary(result)
    if args.json_path:
        output_path = Path(args.json_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"json_report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

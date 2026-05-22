#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-only gprMax campaign runner (dry-run + single task execution)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gprmax_campaign import (
    GprMaxTaskSpec,
    load_campaign_yaml,
    run_gprmax_task,
    validate_campaign,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gprMax campaign backend runner (GX-RUN-001/002).",
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
        "--run-scene",
        help="Run one scene_id after validation.",
    )
    parser.add_argument(
        "--variant",
        choices=["raw_with_target", "background_only"],
        help="Variant to run for --run-scene.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Optional timeout seconds for process execution.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra CLI argument forwarded to gprMax executable (repeatable).",
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
    try:
        campaign = load_campaign_yaml(args.campaign)
        result = validate_campaign(campaign)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.dry_run:
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

    if args.run_scene:
        if args.variant is None:
            print("ERROR: --variant is required when --run-scene is used.")
            return 2
        if result.status == "invalid":
            print("ERROR: Campaign validation is invalid; execution refused.")
            _print_summary(result)
            return 2
        scene = _find_scene_by_id(campaign.scenes, args.run_scene)
        if scene is None:
            print(f"ERROR: scene_id not found: {args.run_scene}")
            return 2
        scene_validation = _find_scene_validation(result, scene.scene_id)
        if scene_validation is None or scene_validation.status == "invalid":
            print(f"ERROR: Scene '{scene.scene_id}' is invalid; execution refused.")
            if scene_validation is not None:
                for issue in scene_validation.issues:
                    print(f"  [{issue.level}] {issue.code}: {issue.message}")
            return 2

        model_path = (
            scene.raw_model
            if args.variant == "raw_with_target"
            else scene.background_model
        )
        output_dir = campaign.output_root / scene.scene_id / args.variant
        task = GprMaxTaskSpec(
            campaign_id=campaign.campaign_id,
            scene_id=scene.scene_id,
            variant=args.variant,
            model_path=model_path,
            output_dir=output_dir,
            gprmax_executable=campaign.gprmax_executable,
            timeout_seconds=args.timeout_seconds,
            extra_args=list(args.extra_arg or []),
        )
        run_result = run_gprmax_task(task)
        print(f"campaign_id: {run_result.campaign_id}")
        print(f"scene_id: {run_result.scene_id}")
        print(f"variant: {run_result.variant}")
        print(f"status: {run_result.status}")
        print(f"return_code: {run_result.return_code}")
        print(f"runtime_seconds: {run_result.runtime_seconds:.3f}")
        print(f"stdout_path: {run_result.stdout_path}")
        print(f"stderr_path: {run_result.stderr_path}")
        print(f"manifest_path: {run_result.manifest_path}")
        if run_result.error_message:
            print(f"error_message: {run_result.error_message}")
        if args.json_path:
            output_path = Path(args.json_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(run_result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"json_report: {output_path}")
        return 0 if run_result.status == "success" else 1

    print("ERROR: specify either --dry-run or --run-scene.")
    return 2


def _find_scene_by_id(scenes, scene_id: str):
    wanted = str(scene_id).strip()
    for scene in scenes:
        if scene.scene_id == wanted:
            return scene
    return None


def _find_scene_validation(result, scene_id: str):
    for scene in result.scenes:
        if scene.scene_id == scene_id:
            return scene
    return None


if __name__ == "__main__":
    raise SystemExit(main())

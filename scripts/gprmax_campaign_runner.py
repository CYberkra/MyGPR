#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-only gprMax campaign runner (dry-run/scene-run/pair-outputs)."""

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
    PairedOutputSpec,
    generate_pair_preview_report,
    generate_target_response,
    load_campaign_yaml,
    run_gprmax_task,
    validate_campaign,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gprMax campaign backend runner (GX-RUN-001/002/003/004).",
    )
    parser.add_argument("--campaign", help="Path to campaign YAML file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate campaign only; do not execute gprMax.",
    )
    parser.add_argument("--run-scene", help="Run one scene_id after validation.")
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
        "--pair-outputs",
        action="store_true",
        help="Validate raw/background outputs and generate target_response.",
    )
    parser.add_argument(
        "--preview-pair",
        action="store_true",
        help="Generate preview PNGs and lightweight paired report stubs.",
    )
    parser.add_argument("--campaign-id", help="Campaign id for --pair-outputs mode.")
    parser.add_argument("--scene-id", help="Scene id for --pair-outputs mode.")
    parser.add_argument("--raw-output", help="Raw output path for --pair-outputs mode.")
    parser.add_argument(
        "--background-output",
        help="Background output path for --pair-outputs mode.",
    )
    parser.add_argument("--output-dir", help="Output directory for generated artifacts.")
    parser.add_argument(
        "--source-format",
        choices=["auto", "csv", "npy"],
        default="auto",
        help="Input source format for --pair-outputs.",
    )
    parser.add_argument(
        "--target-roi",
        help="Optional target ROI path/label for --pair-outputs metadata.",
    )
    parser.add_argument(
        "--target-response",
        help="Optional existing target_response input path for --preview-pair mode.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional path to write mode result JSON summary.",
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
    if args.preview_pair:
        return _run_preview_pair_mode(args)
    if args.pair_outputs:
        return _run_pair_outputs_mode(args)
    if args.dry_run or args.run_scene:
        return _run_campaign_mode(args)
    print("ERROR: specify one mode: --dry-run, --run-scene, or --pair-outputs.")
    return 2


def _run_campaign_mode(args: argparse.Namespace) -> int:
    if not args.campaign:
        print("ERROR: --campaign is required for --dry-run and --run-scene mode.")
        return 2
    try:
        campaign = load_campaign_yaml(args.campaign)
        result = validate_campaign(campaign)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.dry_run:
        _print_summary(result)
        if args.json_path:
            _write_json(args.json_path, result.to_dict())
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
            _write_json(args.json_path, run_result.to_dict())
        return 0 if run_result.status == "success" else 1

    print("ERROR: unsupported campaign mode combination.")
    return 2


def _run_pair_outputs_mode(args: argparse.Namespace) -> int:
    missing = []
    for key in ["campaign_id", "scene_id", "raw_output", "background_output", "output_dir"]:
        if not getattr(args, key):
            missing.append(f"--{key.replace('_', '-')}")
    if missing:
        print("ERROR: --pair-outputs mode missing required args: " + ", ".join(missing))
        return 2

    spec = PairedOutputSpec(
        campaign_id=str(args.campaign_id),
        scene_id=str(args.scene_id),
        raw_output_path=Path(args.raw_output),
        background_output_path=Path(args.background_output),
        output_dir=Path(args.output_dir),
        target_roi=args.target_roi,
        source_format=args.source_format,
    )
    result = generate_target_response(spec)
    print(f"campaign_id: {result.campaign_id}")
    print(f"scene_id: {result.scene_id}")
    print(f"status: {result.status}")
    print(f"validation_summary_path: {result.validation_summary_path}")
    if result.status == "success":
        print(f"target_response_npy_path: {result.target_response_npy_path}")
        print(f"target_response_csv_path: {result.target_response_csv_path}")
        print(f"metrics_path: {result.metrics_path}")
        if result.metrics:
            print(
                "target_response_energy: "
                f"{result.metrics.get('target_response_energy')}"
            )
    else:
        for issue in result.issues:
            print(f"[{issue.get('level')}] {issue.get('code')}: {issue.get('message')}")

    if args.json_path:
        _write_json(args.json_path, result.to_dict())
    return 0 if result.status == "success" else 1


def _run_preview_pair_mode(args: argparse.Namespace) -> int:
    missing = []
    for key in ["campaign_id", "scene_id", "raw_output", "background_output", "output_dir"]:
        if not getattr(args, key):
            missing.append(f"--{key.replace('_', '-')}")
    if missing:
        print("ERROR: --preview-pair mode missing required args: " + ", ".join(missing))
        return 2

    result = generate_pair_preview_report(
        campaign_id=str(args.campaign_id),
        scene_id=str(args.scene_id),
        raw_output_path=Path(args.raw_output),
        background_output_path=Path(args.background_output),
        target_response_path=Path(args.target_response) if args.target_response else None,
        output_dir=Path(args.output_dir),
        source_format=args.source_format,
        target_roi=args.target_roi,
    )
    print(f"campaign_id: {result.campaign_id}")
    print(f"scene_id: {result.scene_id}")
    print(f"status: {result.status}")
    if result.raw_preview_path:
        print(f"raw_preview_path: {result.raw_preview_path}")
    if result.background_preview_path:
        print(f"background_preview_path: {result.background_preview_path}")
    if result.target_response_preview_path:
        print(f"target_response_preview_path: {result.target_response_preview_path}")
    if result.paired_preview_panel_path:
        print(f"paired_preview_panel_path: {result.paired_preview_panel_path}")
    if result.report_md_path:
        print(f"report_md_path: {result.report_md_path}")
    if result.summary_json_path:
        print(f"summary_json_path: {result.summary_json_path}")
    if result.status != "success":
        for issue in result.issues:
            print(f"[{issue.get('level')}] {issue.get('code')}: {issue.get('message')}")
    if args.json_path:
        _write_json(args.json_path, result.to_dict())
    return 0 if result.status == "success" else 1


def _write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json_report: {output_path}")


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

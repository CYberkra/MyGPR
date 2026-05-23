#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run diagnostic-only background suppression AutoTune trials on synthetic paired data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.autotune_background_suppression import (
    load_csv_2d,
    load_roi_json,
    parse_candidate_config,
    run_background_suppression_diagnostic,
)


def _parse_bool(value: str) -> bool:
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only background suppression AutoTune harness (AT-BG-002 draft)."
    )
    parser.add_argument("--raw", required=True, help="Path to raw_bscan.csv")
    parser.add_argument(
        "--target-response",
        required=True,
        help="Path to target_response.csv",
    )
    parser.add_argument("--background", default="", help="Optional background_bscan.csv path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--artifact-id", default="unknown_artifact")
    parser.add_argument("--scene-id", default="unknown_scene")
    parser.add_argument("--roi-json", default="", help="Optional ROI JSON path")
    parser.add_argument(
        "--candidate-config",
        default="",
        help="Optional candidate config JSON path (default uses AT-BG-001 v1 grid).",
    )
    parser.add_argument(
        "--write-arrays",
        type=_parse_bool,
        default=False,
        help="Whether to write processed candidate arrays (default: false).",
    )
    parser.add_argument(
        "--max-preview-candidates",
        type=int,
        default=None,
        help="Reserved optional integer for future preview limit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    raw = load_csv_2d(args.raw)
    target = load_csv_2d(args.target_response)
    roi = load_roi_json(args.roi_json) if args.roi_json else None
    candidates = parse_candidate_config(args.candidate_config or None)

    summary = run_background_suppression_diagnostic(
        raw=raw,
        target_response=target,
        output_dir=args.output_dir,
        artifact_id=args.artifact_id,
        scene_id=args.scene_id,
        roi=roi,
        candidate_specs=candidates,
        write_arrays=bool(args.write_arrays),
        max_preview_candidates=args.max_preview_candidates,
        input_paths={
            "raw": str(Path(args.raw).expanduser().resolve()),
            "target_response": str(Path(args.target_response).expanduser().resolve()),
            "background": str(Path(args.background).expanduser().resolve())
            if args.background
            else "",
            "roi_json": str(Path(args.roi_json).expanduser().resolve())
            if args.roi_json
            else "",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pair and preview converted gprMax arrays from a scene root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gprmax_campaign import (
    PairedOutputSpec,
    discover_converted_pair_paths,
    generate_pair_preview_report,
    generate_target_response,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair and preview converted scene outputs.")
    parser.add_argument("--scene-root", required=True, help="Scene root containing raw_with_target/background_only converted dirs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for paired artifacts.")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--raw", default="", help="Optional explicit raw converted array path.")
    parser.add_argument("--background", default="", help="Optional explicit background converted array path.")
    parser.add_argument("--prefer-format", choices=["npy", "csv"], default="npy")
    parser.add_argument(
        "--roi-json",
        default="",
        help="Optional ROI json file path with trace_range/sample_range.",
    )
    parser.add_argument("--json", default="", help="Optional summary json output path.")
    return parser.parse_args(argv)


def _load_roi_payload(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    roi_path = Path(path).expanduser().resolve()
    payload = json.loads(roi_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ROI payload must be a JSON object: {roi_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    roi_payload = _load_roi_payload(args.roi_json)
    raw, background = discover_converted_pair_paths(
        args.scene_root,
        prefer_format=args.prefer_format,
        raw_path=args.raw or None,
        background_path=args.background or None,
    )

    spec = PairedOutputSpec(
        campaign_id=args.campaign_id,
        scene_id=args.scene_id,
        raw_output_path=raw,
        background_output_path=background,
        output_dir=Path(args.output_dir),
        target_roi=roi_payload,
        source_format="auto",
    )
    pair_result = generate_target_response(spec)
    preview_result = generate_pair_preview_report(
        campaign_id=args.campaign_id,
        scene_id=args.scene_id,
        raw_output_path=raw,
        background_output_path=background,
        target_response_path=pair_result.target_response_npy_path,
        output_dir=Path(args.output_dir),
        source_format="auto",
    )
    summary = {
        "raw_path": str(raw),
        "background_path": str(background),
        "roi_json": args.roi_json or None,
        "pair_result": pair_result.to_dict(),
        "preview_result": preview_result.to_dict(),
        "pair_metrics_keys": (
            sorted(pair_result.metrics.keys()) if pair_result.metrics else []
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        out = Path(args.json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if pair_result.status != "success" or preview_result.status != "success":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

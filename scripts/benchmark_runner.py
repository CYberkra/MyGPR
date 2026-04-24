#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for deterministic benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.benchmark_registry import list_benchmark_sample_ids
from core.benchmark_runner import run_benchmark_sample


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic GPR benchmark samples."
    )
    parser.add_argument("--sample", required=True, choices=list_benchmark_sample_ids())
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "output" / "benchmarks"),
        help="Directory for benchmark PNG and JSON artifacts.",
    )
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.sample
    summary = run_benchmark_sample(
        sample_id=args.sample,
        method_keys=args.methods,
        out_dir=out_dir,
        seed=args.seed,
        save_images=not args.no_images,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

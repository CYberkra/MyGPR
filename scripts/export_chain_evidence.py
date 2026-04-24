#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry for reproducible standard-chain evidence export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark_registry import list_benchmark_sample_ids
from core.evidence_export import STANDARD_CHAIN_SPECS, export_standard_chain_for_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, choices=list_benchmark_sample_ids())
    parser.add_argument(
        "--chain", required=True, choices=sorted(STANDARD_CHAIN_SPECS.keys())
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="output/chain_evidence")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_dir) / args.sample / args.chain
    summary = export_standard_chain_for_sample(
        sample_id=args.sample,
        chain_key=args.chain,
        out_dir=out_root,
        seed=args.seed,
        save_images=not args.no_images,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

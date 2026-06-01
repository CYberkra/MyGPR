#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-AT-SCORE-001 paired AutoTune smoke runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.app_paths import expand_path_template
from core.autotune_paired_scoring_smoke import (
    build_inventory,
    ensure_scene_arrays,
    score_scene_candidates,
    select_stable_pairs,
    write_inventory_outputs,
    write_scoring_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GX-AT-SCORE-001 paired autotune smoke.")
    parser.add_argument(
        "--root-single",
        default=os.environ.get("MYGPR_GPRMAX_SINGLE_ROOT", "${MYGPR_GPR_RESULT_RUNS}/01_single"),
        help="Output V5 single-run root.",
    )
    parser.add_argument(
        "--root-batch",
        default=os.environ.get("MYGPR_GPRMAX_BATCH_ROOT", "${MYGPR_GPR_RESULT_RUNS}/02_batch"),
        help="Output V5 batch root.",
    )
    parser.add_argument(
        "--docs-dir",
        default="docs/autotune",
        help="Repo docs output directory.",
    )
    parser.add_argument(
        "--component",
        default="Ey",
        help="Preferred component for scoring.",
    )
    return parser.parse_args()


def _load_roi_from_task(task_dir: Path) -> dict[str, Any] | None:
    model_dir = task_dir / "1_模型输入"
    for name in ("roi.json", "roi_draft.json", "target_roi.json", "roi_manifest.json"):
        path = model_dir / name
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    return None


def main() -> int:
    args = _parse_args()
    docs_dir = Path(args.docs_dir).resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(expand_path_template(args.root_single)), Path(expand_path_template(args.root_batch))]
    inventories = build_inventory(roots, component_preference=args.component)
    inventory_md = docs_dir / "gprmax_paired_inventory.md"
    inventory_json = docs_dir / "gprmax_paired_inventory.json"
    write_inventory_outputs(inventories, output_md=inventory_md, output_json=inventory_json)

    stable_pairs = select_stable_pairs(inventories)
    all_rows = []
    scene_summaries = []
    for item in stable_pairs:
        try:
            raw, _bg, target, conversion_summary = ensure_scene_arrays(item, component=args.component)
        except Exception as exc:
            scene_summaries.append(
                {
                    "scene_id": item.scene_id_guess,
                    "task_dir": str(item.task_dir),
                    "status": "array_prepare_failed",
                    "error": str(exc),
                }
            )
            continue

        roi = _load_roi_from_task(item.task_dir)
        rows = score_scene_candidates(
            scene_id=item.scene_id_guess,
            task_dir=item.task_dir,
            raw=raw,
            target_response=target,
            roi=roi,
            component=args.component,
        )
        all_rows.extend(rows)
        scene_summaries.append(
            {
                "scene_id": item.scene_id_guess,
                "task_dir": str(item.task_dir),
                "status": "scored",
                "trial_count": len(rows),
                "raw_shape": list(raw.shape),
                "target_shape": list(target.shape),
                "conversion_summary": conversion_summary,
            }
        )

    write_scoring_outputs(
        all_rows,
        report_md=docs_dir / "gx_at_score_001_report.md",
        trial_csv=docs_dir / "gx_at_score_001_trial_table.csv",
        metrics_json=docs_dir / "gx_at_score_001_metrics_summary.json",
        selected_json=docs_dir / "gx_at_score_001_selected_parameters.json",
        claim_md=docs_dir / "gx_at_score_001_claim_boundary.md",
    )

    run_summary = {
        "roots": [str(p) for p in roots],
        "inventory_count": len(inventories),
        "stable_pair_count": len(stable_pairs),
        "scored_scene_count": len([s for s in scene_summaries if s.get("status") == "scored"]),
        "scene_summaries": scene_summaries,
    }
    (docs_dir / "gx_at_score_001_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"inventory_count={len(inventories)} stable_pairs={len(stable_pairs)}")
    print(f"inventory_md={inventory_md}")
    print(f"inventory_json={inventory_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-AT-SCORE-002 scoring hardening runner."""

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
    CLAIM_BOUNDARY_LINES,
    build_inventory,
    ensure_scene_arrays,
    score_scene_candidates,
    select_scoreable_pairs,
    summarize_candidate_aggregates,
    summarize_warning_counts,
    top_k_candidates_for_scene,
    write_inventory_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GX-AT-SCORE-002 scoring hardening.")
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
    parser.add_argument("--component", default="Ey", help="Preferred scoring component.")
    return parser.parse_args()


def _load_roi_from_task(task_dir: Path) -> dict[str, Any] | None:
    model_dir = task_dir / "1_模型输入"
    for name in ("roi.json", "roi_draft.json", "target_roi.json", "roi_manifest.json"):
        path = model_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _write_trial_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    fieldnames = [
        "scene_id",
        "task_dir",
        "component",
        "candidate",
        "parameters",
        "roi_mode",
        "roi_sample_range",
        "roi_trace_range",
        "mae",
        "mse",
        "rmse",
        "psnr",
        "ssim",
        "roi_energy_retention",
        "outside_roi_residual_energy",
        "cnr_proxy",
        "cnr_gain_vs_baseline",
        "selected",
        "status",
        "warnings",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = row.copy()
            flat["parameters"] = json.dumps(flat["parameters"], ensure_ascii=False)
            flat["roi_sample_range"] = json.dumps(flat["roi_sample_range"], ensure_ascii=False)
            flat["roi_trace_range"] = json.dumps(flat["roi_trace_range"], ensure_ascii=False)
            flat["warnings"] = json.dumps(flat["warnings"], ensure_ascii=False)
            flat.pop("claim_boundary", None)
            writer.writerow(flat)


def main() -> int:
    args = _parse_args()
    docs_dir = Path(args.docs_dir).resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(expand_path_template(args.root_single)), Path(expand_path_template(args.root_batch))]
    inventories = build_inventory(roots, component_preference=args.component)
    write_inventory_outputs(
        inventories,
        output_md=docs_dir / "gprmax_paired_inventory.md",
        output_json=docs_dir / "gprmax_paired_inventory.json",
    )

    stable_count = sum(1 for i in inventories if i.status == "stable_completed")
    convertible_count = sum(1 for i in inventories if i.status == "convertible_pair")
    scoreable = select_scoreable_pairs(inventories)

    all_rows = []
    scene_errors: list[dict[str, Any]] = []
    for item in scoreable:
        try:
            raw, _bg, target, _summary = ensure_scene_arrays(item, component=args.component)
        except Exception as exc:
            scene_errors.append(
                {
                    "scene_id": item.scene_id_guess,
                    "task_dir": str(item.task_dir),
                    "status": "array_prepare_failed",
                    "error": str(exc),
                }
            )
            continue
        roi = _load_roi_from_task(item.task_dir)
        scene_rows = score_scene_candidates(
            scene_id=item.scene_id_guess,
            task_dir=item.task_dir,
            raw=raw,
            target_response=target,
            roi=roi,
            component=args.component,
        )
        all_rows.extend(scene_rows)

    rows_dict = [r.to_dict() for r in all_rows]
    trial_csv = docs_dir / "gx_at_score_002_trial_table.csv"
    _write_trial_csv(rows_dict, trial_csv)

    per_scene_selected = {row["scene_id"]: row for row in rows_dict if row.get("selected")}
    candidate_aggregate = summarize_candidate_aggregates(all_rows)
    warning_counts = summarize_warning_counts(all_rows)
    top3 = top_k_candidates_for_scene(all_rows, k=3)
    top3_dict = {scene_id: [item.to_dict() for item in entries] for scene_id, entries in top3.items()}

    metrics_summary = {
        "task_id": "GX-AT-SCORE-002-SCORING-HARDENING",
        "total_inventory_count": len(inventories),
        "stable_completed_count": stable_count,
        "convertible_pair_count": convertible_count,
        "scored_scene_count": len({row["scene_id"] for row in rows_dict}),
        "trial_count": len(rows_dict),
        "per_candidate_aggregate": candidate_aggregate,
        "warnings_summary": warning_counts,
        "scene_errors": scene_errors,
        "claim_boundary": CLAIM_BOUNDARY_LINES,
    }
    metrics_path = docs_dir / "gx_at_score_002_metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_payload = {
        "task_id": "GX-AT-SCORE-002-SCORING-HARDENING",
        "selection_rule": "min MAE, then min RMSE, then max PSNR",
        "selected_per_scene": per_scene_selected,
        "top3_per_scene": top3_dict,
        "claim_boundary": CLAIM_BOUNDARY_LINES,
    }
    selected_path = docs_dir / "gx_at_score_002_selected_parameters.json"
    selected_path.write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# GX-AT-SCORE-002 Scoring Hardening Report",
        "",
        f"- total_inventory_count: `{len(inventories)}`",
        f"- stable_completed_count: `{stable_count}`",
        f"- convertible_pair_count: `{convertible_count}`",
        f"- scored_scene_count: `{metrics_summary['scored_scene_count']}`",
        f"- trial_count: `{len(rows_dict)}`",
        "",
        "## Per-scene Selection + Top-3",
        "",
    ]
    for scene_id in sorted(top3_dict):
        top_entries = top3_dict[scene_id]
        selected = per_scene_selected.get(scene_id)
        report_lines.append(f"### {scene_id}")
        if selected:
            report_lines.append(f"- selected_candidate: `{selected['candidate']}`")
        report_lines.append("- top_3_candidates:")
        for idx, row in enumerate(top_entries, start=1):
            report_lines.append(
                f"  - {idx}. `{row['candidate']}` | MAE={row['mae']:.10g} RMSE={row['rmse']:.10g} "
                f"PSNR={row['psnr']} SSIM={row['ssim']} roi_mode={row['roi_mode']} warnings={row['warnings']}"
            )
        report_lines.append(f"- claim_boundary: `{'; '.join(CLAIM_BOUNDARY_LINES)}`")
        report_lines.append("")
    report_path = docs_dir / "gx_at_score_002_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    claim_lines = ["# GX-AT-SCORE-002 Claim Boundary", ""]
    claim_lines.extend([f"- {line}" for line in CLAIM_BOUNDARY_LINES])
    claim_lines.append("- ROI energy/CNR are diagnostic proxies only, not detection accuracy evidence.")
    claim_path = docs_dir / "gx_at_score_002_claim_boundary.md"
    claim_path.write_text("\n".join(claim_lines), encoding="utf-8")

    print(f"inventory={len(inventories)} stable={stable_count} convertible={convertible_count}")
    print(f"scored_scene_count={metrics_summary['scored_scene_count']} trial_count={len(rows_dict)}")
    print(f"trial_table={trial_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

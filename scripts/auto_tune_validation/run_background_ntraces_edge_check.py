#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-010 background ntraces edge-check evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auto_tune_validation import run_no_zerotime_gain_validation as at005a
from scripts.auto_tune_validation.run_post_zero_time_policy_rerun import DEFAULT_GX003_DATASET
from scripts.auto_tune_validation.run_stepwise_validation import _git_rev_parse, _json_safe, _load_dataset, _write_json

DEFAULT_AT009_SUMMARY = (
    ROOT.parent
    / "MyGPR-Evidence"
    / "autotune"
    / "AT-009_background_gain_policy_refinement"
    / "manifests"
    / "background_gain_policy_summary.json"
)

BASE_CANDIDATES = [57, 65, 73, 81, 89, 97, 105, 113, 121]
OPTIONAL_CANDIDATES = [129, 145]
GAIN_STEP = ("energy_decay_gain", {"strength": 0.8, "smoothing_samples": 31, "max_gain": 6.0})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--dataset", default=str(DEFAULT_GX003_DATASET))
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--historical-at009", default=str(DEFAULT_AT009_SUMMARY))
    parser.add_argument("--include-optional-candidates", action="store_true")
    args = parser.parse_args(argv)

    result = run_validation(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
        historical_at009=Path(args.historical_at009),
        include_optional_candidates=args.include_optional_candidates,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_validation(
    *,
    evidence_root: Path,
    dataset: str,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
    historical_at009: Path = DEFAULT_AT009_SUMMARY,
    include_optional_candidates: bool = False,
) -> dict[str, Any]:
    source_commit = source_commit or _git_rev_parse(ROOT)
    package = _load_dataset(dataset)
    evidence_root.mkdir(parents=True, exist_ok=True)
    figures_dir = evidence_root / "figures"
    tables_dir = evidence_root / "tables"
    reports_dir = evidence_root / "reports"
    manifests_dir = evidence_root / "manifests"
    for d in (figures_dir, tables_dir, reports_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    candidates = list(BASE_CANDIDATES)
    if include_optional_candidates:
        candidates.extend(OPTIONAL_CANDIDATES)

    raw = np.asarray(package["data"], dtype=np.float64)
    header_info = dict(package["header_info"])
    trace_metadata = dict(package["trace_metadata"])
    roi = at005a._target_roi(package.get("ground_truth") or {}, raw.shape)
    color_limit = at005a._global_color_limit(raw)
    at005a._save_overlay(
        raw,
        figures_dir / "input_bscan_roi_overlay.png",
        roi=roi,
        color_limit=color_limit,
        title="GX-003 input with ROI",
    )

    rows: list[dict[str, Any]] = []
    lane_rows: list[dict[str, Any]] = []
    for ntraces in candidates:
        lane = at005a._run_lane(
            raw=raw,
            header_info=header_info,
            trace_metadata=trace_metadata,
            roi=roi,
            figures_dir=figures_dir,
            color_limit=color_limit,
            dataset_kind="gx003_ground_truth",
            lane_id=f"edge_bg{ntraces}_energy_decay_gain",
            branch="manual",
            description=f"edge check ntraces={ntraces}",
            pre_gain_steps=[("subtracting_average_2D", {"ntraces": int(ntraces)})],
            gain_step=GAIN_STEP,
            auto_tune=False,
        )
        row = lane["row"]
        row["ntraces"] = int(ntraces)
        row["zero_time_policy"] = "excluded"
        row["dewow_policy"] = "excluded_primary"
        row["candidate_score"] = _candidate_score(row)
        lane_rows.append(row)
        rows.append(
            {
                "ntraces": int(ntraces),
                "candidate_score": row["candidate_score"],
                "pipeline": "background_suppression->energy_decay_gain",
                "gain_method": row["gain_method"],
                "gain_params": row["gain_params"],
                "zero_time_policy": "excluded",
                "dewow_policy": "excluded_primary",
                "branch_validity": row["branch_validity"],
                "roi_energy": row["after_gain_metrics"].get("roi_energy"),
                "roi_to_local_background_contrast": row["after_gain_metrics"].get("roi_to_local_background_contrast"),
                "target_preservation": row["after_gain_metrics"].get("roi_energy_ratio_to_input"),
                "background_energy_reduction": row["before_gain_metrics"].get("background_energy_reduction"),
                "false_positive_proxy": row["after_gain_metrics"].get("false_positive_proxy"),
                "global_energy_ratio": row["after_gain_metrics"].get("global_energy_ratio_to_input"),
                "clipping_ratio": row["after_gain_metrics"].get("clipping_ratio"),
                "deep_zone_visibility_proxy": row["after_gain_metrics"].get("deep_zone_visibility_proxy"),
                "local_contrast": row["after_gain_metrics"].get("roi_to_local_background_contrast"),
                "sanity_warnings": row["sanity_warnings"],
                "preview_image_path": row["figure"],
                "roi_crop_path": row["roi_crop"],
            }
        )

    rows_sorted = sorted(rows, key=lambda r: float(r["candidate_score"]), reverse=True)
    best = rows_sorted[0]
    best_n = int(best["ntraces"])
    top = float(best["candidate_score"])
    plateau = [int(r["ntraces"]) for r in rows if float(r["candidate_score"]) >= top * 0.98]
    rec_range = f"{min(plateau)}-{max(plateau)}" if plateau else str(best_n)
    edge_flags = _edge_risk_flags(rows, candidates)
    preset_status = _preset_status(edge_flags, plateau)
    n73_supported = any(int(r["ntraces"]) == 73 and float(r["candidate_score"]) >= top * 0.95 for r in rows)

    decision = {
        "preset_candidacy_classification": preset_status,
        "best_ntraces": best_n,
        "recommended_ntraces_range": rec_range,
        "ntraces_73_supported": bool(n73_supported),
        "edge_risk_flags": edge_flags,
        "notes": _decision_note(preset_status),
    }

    _save_sweep_plot(rows, figures_dir / "extended_ntraces_sweep_summary.png")
    _save_risk_plot(edge_flags, figures_dir / "edge_risk_summary.png")
    _copy_recommended(lane_rows, best_n, figures_dir)
    _write_csv(tables_dir / "extended_ntraces_sweep_metrics.csv", rows)
    _write_csv(tables_dir / "preset_candidate_decision.csv", [decision])

    historical = _safe_read_json(historical_at009)
    summary = {
        "artifact_id": "AT-010",
        "task_id": "AT-010_background_ntraces_edge_check",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": package["dataset_name"],
            "shape": package["dataset_shape"],
            "hash": package["dataset_hash"],
            "source_evidence": "gprmax/GX-003_audited_native_gprmax_benchmark/",
        },
        "ground_truth_available": True,
        "gx003_roi": roi.as_dict() if roi else None,
        "zero_time_policy": "excluded",
        "dewow_policy": "excluded_primary",
        "extended_candidate_domain": candidates,
        "historical_at009_recommended_ntraces_range": historical.get("recommended_ntraces_range"),
        "best_ntraces": best_n,
        "recommended_ntraces_range": rec_range,
        "preset_candidate_decision": decision,
        "ntraces_73_supported": bool(n73_supported),
        "rows": rows,
        "claim_boundary": {
            "ground_truth": "GX-003 metrics are ground-truth-backed.",
            "heuristic_qc": "Display-oriented metrics are diagnostic only.",
            "autotune": "No overall AutoTune superiority claim in this artifact.",
        },
        "known_risks": [
            "Single native gprMax scene cannot establish broad preset generalization.",
            "Edge-check may still be domain-limited without wider scenario coverage.",
            "Dewow remains excluded only for this primary lane policy.",
        ],
    }
    manifest = {
        "artifact_id": "AT-010",
        "task_id": "AT-010_background_ntraces_edge_check",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending_self_reference",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_background_ntraces_edge_check.py",
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": True,
        "metric_type": "ground_truth_ntraces_edge_check",
        "zero_time_policy": "excluded",
        "dewow_policy": "excluded_primary",
        "artifacts": {
            "markdown_report": "reports/background_ntraces_edge_check_report.md",
            "html_report": "reports/background_ntraces_edge_check_report.html",
            "summary": "manifests/background_ntraces_edge_check_summary.json",
            "sweep_metrics": "tables/extended_ntraces_sweep_metrics.csv",
            "preset_decision": "tables/preset_candidate_decision.csv",
        },
        "limitations": summary["known_risks"],
    }
    _write_json(manifests_dir / "background_ntraces_edge_check_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)

    md = _render_md(summary)
    html_body = _render_html(summary)
    (reports_dir / "background_ntraces_edge_check_report.md").write_text(md, encoding="utf-8")
    (reports_dir / "background_ntraces_edge_check_report.html").write_text(html_body, encoding="utf-8")

    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "extended_candidate_domain": candidates,
        "best_ntraces": best_n,
        "recommended_ntraces_range": rec_range,
        "preset_candidacy_classification": preset_status,
        "edge_risk_flags": edge_flags,
        "ntraces_73_supported": bool(n73_supported),
        "html_report": str((reports_dir / "background_ntraces_edge_check_report.html").resolve()),
    }


def _candidate_score(row: dict[str, Any]) -> float:
    m = row.get("after_gain_metrics") or {}
    contrast = float(m.get("roi_to_local_background_contrast") or 0.0)
    preserve = float(m.get("roi_energy_ratio_to_input") or 0.0)
    clip = float(m.get("clipping_ratio") or 0.0)
    return contrast * (1.0 + min(preserve, 1.0)) - (3.0 * clip)


def _edge_risk_flags(rows: list[dict[str, Any]], candidates: list[int]) -> list[str]:
    flags: list[str] = []
    sorted_rows = sorted(rows, key=lambda r: int(r["ntraces"]))
    scores = [float(r["candidate_score"]) for r in sorted_rows]
    best = max(rows, key=lambda r: float(r["candidate_score"]))
    if int(best["ntraces"]) == max(candidates):
        flags.append("best_params_at_edge")
    if all(b >= a for a, b in zip(scores[:-1], scores[1:])):
        flags.append("monotonic_rightward_trend")
    top = float(best["candidate_score"])
    plateau = [r for r in rows if float(r["candidate_score"]) >= top * 0.98]
    if len(plateau) <= 1:
        flags.append("no_stable_plateau")
    baseline_fp = float(next(r for r in sorted_rows if int(r["ntraces"]) == min(candidates))["false_positive_proxy"] or 0.0)
    best_fp = float(best.get("false_positive_proxy") or 0.0)
    if best_fp > baseline_fp * 1.2:
        flags.append("false_positive_increase")
    best_bg = float(best.get("background_energy_reduction") or 0.0)
    if best_bg < 0.02:
        flags.append("weak_background_suppression")
    ranked = sorted(rows, key=lambda r: float(r["candidate_score"]), reverse=True)
    if len(ranked) > 1:
        margin = float(ranked[0]["candidate_score"]) - float(ranked[1]["candidate_score"])
        conf = float(ranked[0]["candidate_score"]) / (abs(float(ranked[0]["candidate_score"])) + abs(float(ranked[1]["candidate_score"])) + 1e-9)
        if margin < 0.10:
            flags.append("multiple_near_optima")
        if conf < 0.2:
            flags.append("low_selection_confidence")
    return sorted(set(flags))


def _preset_status(flags: list[str], plateau: list[int]) -> str:
    if any(f in flags for f in ("best_params_at_edge", "monotonic_rightward_trend", "no_stable_plateau")):
        return "not_ready_edge_limited"
    if any(f in flags for f in ("false_positive_increase", "weak_background_suppression")):
        return "not_ready_metric_conflict"
    if len(plateau) >= 2:
        return "provisional_single_scene_preset"
    return "preset_candidate_ready"


def _decision_note(status: str) -> str:
    mapping = {
        "preset_candidate_ready": "Stable local plateau observed; still needs multi-scene confirmation.",
        "provisional_single_scene_preset": "Plausible GX-003 range found, but more native scenes are required.",
        "not_ready_edge_limited": "Best candidate remains boundary-limited; extend domain or add scenes.",
        "not_ready_metric_conflict": "Metric conflict blocks preset finalization.",
    }
    return mapping.get(status, "inconclusive")


def _save_sweep_plot(rows: list[dict[str, Any]], path: Path) -> None:
    xs = [int(r["ntraces"]) for r in sorted(rows, key=lambda x: int(x["ntraces"]))]
    ys = [float(r["candidate_score"]) for r in sorted(rows, key=lambda x: int(x["ntraces"]))]
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=150)
    try:
        ax.plot(xs, ys, "o-", color="#0f766e")
        ax.set_xlabel("ntraces")
        ax.set_ylabel("candidate_score")
        ax.set_title("AT-010 Extended ntraces Edge Sweep")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_risk_plot(flags: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.8), dpi=150)
    try:
        ax.axis("off")
        text = "Edge-risk flags:\n- " + ("\n- ".join(flags) if flags else "none")
        ax.text(0.02, 0.95, text, ha="left", va="top", fontsize=11, family="monospace")
        ax.set_title("AT-010 Edge-risk Summary")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _copy_recommended(lane_rows: list[dict[str, Any]], best_n: int, figures_dir: Path) -> None:
    target = next((r for r in lane_rows if int(r.get("ntraces", -1)) == int(best_n)), None)
    if not target:
        return
    src = figures_dir / Path(target["figure"]).name
    crop = figures_dir / Path(target["roi_crop"]).name if target.get("roi_crop") else None
    if src.exists():
        shutil.copy2(src, figures_dir / "recommended_candidate_bscan.png")
    if crop and crop.exists():
        shutil.copy2(crop, figures_dir / "recommended_candidate_roi_crop.png")


def _render_md(summary: dict[str, Any]) -> str:
    d = summary["preset_candidate_decision"]
    lines = [
        "# AT-010 Background ntraces Edge Check and Preset Candidate",
        "",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Dataset: `{summary['dataset']['name']}` shape `{summary['dataset']['shape']}`",
        "- Ground truth: `available`",
        f"- Zero-time policy: `{summary['zero_time_policy']}`",
        f"- Dewow policy: `{summary['dewow_policy']}`",
        "- GX-003 ROI is used as-is.",
        "",
        "## Decision",
        f"- Preset candidacy classification: `{d['preset_candidacy_classification']}`",
        f"- Best ntraces: `{d['best_ntraces']}`",
        f"- Recommended ntraces range: `{d['recommended_ntraces_range']}`",
        f"- ntraces=73 remains supported: `{d['ntraces_73_supported']}`",
        f"- Edge-risk flags: `{','.join(d['edge_risk_flags']) if d['edge_risk_flags'] else 'none'}`",
        "",
        "## Claim boundary",
        "- This is an edge-check following AT-009; no overall AutoTune superiority claim is made.",
        "- AGC is not used for physical correctness claims here.",
        "- Dewow remains optional/diagnostic and excluded in this primary lane.",
        "",
        "## Next recommended task",
        "- more native gprMax model generation + constrained preset cross-scene validation.",
        "",
        "## Known risks",
    ]
    lines.extend(f"- {k}" for k in summary["known_risks"])
    lines.append("")
    return "\n".join(lines)


def _render_html(summary: dict[str, Any]) -> str:
    d = summary["preset_candidate_decision"]
    risks = "\n".join(f"<li>{html.escape(k)}</li>" for k in summary["known_risks"])
    flags = ", ".join(d["edge_risk_flags"]) if d["edge_risk_flags"] else "none"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT-010 ntraces Edge Check</title>
  <style>
    body {{ margin: 0; background: #f5f7fb; color: #162033; font-family: "Segoe UI", "PingFang SC", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    .card {{ background: #fff; border: 1px solid #d9e2ee; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }}
    .warning {{ border-color: #f59e0b; background: #fff9ea; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #d9e2ee; border-radius: 8px; padding: 10px; }}
    figure img {{ width: 100%; display: block; }}
    figcaption {{ margin-top: 6px; font-size: 13px; color: #495a75; }}
    code {{ background: #edf2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>AT-010 Background ntraces Edge Check</h1>
  <div class="card">
    <div><b>Source commit:</b> <code>{summary["source_commit"]}</code></div>
    <div><b>Dataset:</b> {summary["dataset"]["name"]} {summary["dataset"]["shape"]}</div>
    <div><b>Zero-time policy:</b> <code>{summary["zero_time_policy"]}</code></div>
    <div><b>Dewow policy:</b> <code>{summary["dewow_policy"]}</code></div>
    <div><b>Extended domain:</b> <code>{summary["extended_candidate_domain"]}</code></div>
  </div>
  <div class="card warning">
    <b>Decision:</b> <code>{d['preset_candidacy_classification']}</code> |
    best ntraces=<code>{d['best_ntraces']}</code> |
    recommended range=<code>{d['recommended_ntraces_range']}</code> |
    73 supported=<code>{d['ntraces_73_supported']}</code> |
    flags=<code>{flags}</code>
  </div>
  <div class="grid">
    <figure><img src="../figures/input_bscan_roi_overlay.png"><figcaption>Input with ROI</figcaption></figure>
    <figure><img src="../figures/extended_ntraces_sweep_summary.png"><figcaption>Extended ntraces sweep</figcaption></figure>
    <figure><img src="../figures/edge_risk_summary.png"><figcaption>Edge-risk summary</figcaption></figure>
    <figure><img src="../figures/recommended_candidate_bscan.png"><figcaption>Recommended candidate B-scan</figcaption></figure>
    <figure><img src="../figures/recommended_candidate_roi_crop.png"><figcaption>Recommended candidate ROI crop</figcaption></figure>
  </div>
  <h2>Known risks</h2><ul>{risks}</ul>
</main>
</body>
</html>
"""


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_json_safe(row.get(k)), ensure_ascii=False) for k in fields})


if __name__ == "__main__":
    raise SystemExit(main())


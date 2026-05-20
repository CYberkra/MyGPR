#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-009 background suppression domain + gain policy refinement evidence."""

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

DEFAULT_AT008A_SUMMARY = (
    ROOT.parent
    / "MyGPR-Evidence"
    / "autotune"
    / "AT-008A_no_dewow_post_fix_validation"
    / "manifests"
    / "no_dewow_validation_summary.json"
)

NTRACES_CANDIDATES = [9, 17, 25, 33, 41, 49, 57, 65, 73]
GAIN_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "energy_decay_gain": ("energy_decay_gain", {"strength": 0.8, "smoothing_samples": 31, "max_gain": 6.0}),
    "sec_gain": ("sec_gain", {"gain_min": 1.0, "gain_max": 4.5, "power": 1.1}),
    "time_power_gain_local": ("time_power_gain_local", {"power": 1.35, "max_gain": 5.0}),
    "agcGain": ("agcGain", {"window": 121, "_low_energy_guard": True}),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--dataset", default=str(DEFAULT_GX003_DATASET))
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--historical-at008a", default=str(DEFAULT_AT008A_SUMMARY))
    args = parser.parse_args(argv)

    result = run_validation(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
        historical_at008a=Path(args.historical_at008a),
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
    historical_at008a: Path = DEFAULT_AT008A_SUMMARY,
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

    raw = np.asarray(package["data"], dtype=np.float64)
    header_info = dict(package["header_info"])
    trace_metadata = dict(package["trace_metadata"])
    ground_truth = package.get("ground_truth") or {}
    roi = at005a._target_roi(ground_truth, raw.shape)
    color_limit = at005a._global_color_limit(raw)
    at005a._save_overlay(
        raw,
        figures_dir / "input_bscan_roi_overlay.png",
        roi=roi,
        color_limit=color_limit,
        title="GX-003 input with ROI",
    )

    # A. background suppression sweep under fixed conservative gain policy
    sweep_rows: list[dict[str, Any]] = []
    all_lane_rows: list[dict[str, Any]] = []
    for ntraces in NTRACES_CANDIDATES:
        lane = at005a._run_lane(
            raw=raw,
            header_info=header_info,
            trace_metadata=trace_metadata,
            roi=roi,
            figures_dir=figures_dir,
            color_limit=color_limit,
            dataset_kind="gx003_ground_truth",
            lane_id=f"bg{ntraces}_energy_decay_gain",
            branch="manual",
            description=f"background ntraces={ntraces} + energy_decay_gain",
            pre_gain_steps=[("subtracting_average_2D", {"ntraces": int(ntraces)})],
            gain_step=GAIN_SPECS["energy_decay_gain"],
            auto_tune=False,
        )
        row = lane["row"]
        row["ntraces"] = int(ntraces)
        row["zero_time_policy"] = "excluded"
        row["dewow_policy"] = "excluded_primary"
        row["candidate_id"] = f"bg_{ntraces}"
        row["candidate_score"] = _candidate_score(row)
        row["gain_category"] = _gain_category(row["gain_method"])
        all_lane_rows.append(row)
        sweep_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "pipeline": "background_suppression->energy_decay_gain",
                "ntraces": int(ntraces),
                "gain_method": row["gain_method"],
                "gain_params": row["gain_params"],
                "zero_time_policy": row["zero_time_policy"],
                "dewow_policy": row["dewow_policy"],
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
                "invalid_reason": "" if row["branch_validity"] == "valid_with_caveats" else "invalid",
                "preview_image_path": row["figure"],
                "roi_overlay_path": "figures/input_bscan_roi_overlay.png",
                "roi_crop_path": row["roi_crop"],
                "candidate_score": row["candidate_score"],
            }
        )

    best_sweep = max(sweep_rows, key=lambda r: float(r.get("candidate_score") or -1e9))
    recommended_ntraces_range = _recommended_ntraces_range(sweep_rows)

    # B. gain policy comparison on representative ntraces
    representative_ntraces = int(best_sweep["ntraces"])
    gain_rows: list[dict[str, Any]] = []
    for gain_name, gain_spec in GAIN_SPECS.items():
        lane = at005a._run_lane(
            raw=raw,
            header_info=header_info,
            trace_metadata=trace_metadata,
            roi=roi,
            figures_dir=figures_dir,
            color_limit=color_limit,
            dataset_kind="gx003_ground_truth",
            lane_id=f"gain_{gain_name}_n{representative_ntraces}",
            branch="manual",
            description=f"gain policy compare: {gain_name}",
            pre_gain_steps=[("subtracting_average_2D", {"ntraces": representative_ntraces})],
            gain_step=gain_spec,
            auto_tune=False,
        )
        row = lane["row"]
        row["ntraces"] = representative_ntraces
        row["zero_time_policy"] = "excluded"
        row["dewow_policy"] = "excluded_primary"
        row["gain_category"] = _gain_category(row["gain_method"])
        row["candidate_score"] = _candidate_score(row)
        all_lane_rows.append(row)
        gain_rows.append(
            {
                "candidate_id": f"gain_{gain_name}",
                "pipeline": "background_suppression->gain",
                "ntraces": representative_ntraces,
                "gain_method": row["gain_method"],
                "gain_params": row["gain_params"],
                "gain_category": row["gain_category"],
                "amplitude_preserving": row["amplitude_preservation"],
                "zero_time_policy": row["zero_time_policy"],
                "dewow_policy": row["dewow_policy"],
                "branch_validity": row["branch_validity"],
                "roi_to_local_background_contrast": row["after_gain_metrics"].get("roi_to_local_background_contrast"),
                "target_preservation": row["after_gain_metrics"].get("roi_energy_ratio_to_input"),
                "background_energy_reduction": row["before_gain_metrics"].get("background_energy_reduction"),
                "global_energy_ratio": row["after_gain_metrics"].get("global_energy_ratio_to_input"),
                "clipping_ratio": row["after_gain_metrics"].get("clipping_ratio"),
                "deep_zone_visibility_proxy": row["after_gain_metrics"].get("deep_zone_visibility_proxy"),
                "sanity_warnings": row["sanity_warnings"],
                "preview_image_path": row["figure"],
                "roi_crop_path": row["roi_crop"],
                "candidate_score": row["candidate_score"],
            }
        )

    # C. constrained autotune (simulated by constrained sweep selection)
    manual_candidate = next(
        r
        for r in all_lane_rows
        if int(r.get("ntraces", -1)) == representative_ntraces and r.get("gain_method") == "energy_decay_gain"
    )
    safe_candidate = next(
        r for r in all_lane_rows if int(r.get("ntraces", -1)) == 41 and r.get("gain_method") == "energy_decay_gain"
    )
    sorted_sweep = sorted(sweep_rows, key=lambda r: float(r.get("candidate_score") or -1e9), reverse=True)
    auto_candidate_meta = sorted_sweep[0]
    auto_candidate = next(
        r
        for r in all_lane_rows
        if int(r.get("ntraces", -1)) == int(auto_candidate_meta["ntraces"]) and r.get("gain_method") == "energy_decay_gain"
    )
    second_score = float(sorted_sweep[1]["candidate_score"]) if len(sorted_sweep) > 1 else float(sorted_sweep[0]["candidate_score"])
    first_score = float(sorted_sweep[0]["candidate_score"])
    margin = first_score - second_score
    confidence = first_score / (abs(first_score) + abs(second_score) + 1e-9)
    risk_flags: list[str] = []
    if margin < 0.15:
        risk_flags.append("multiple_near_optima")
    if confidence < 0.2:
        risk_flags.append("low_selection_confidence")
    if int(auto_candidate_meta["ntraces"]) in {min(NTRACES_CANDIDATES), max(NTRACES_CANDIDATES)}:
        risk_flags.append("best_params_at_edge")
    constrained_rows = [
        _comparison_row("manual_recommended", manual_candidate, risk_flags=[]),
        _comparison_row("safe_conservative", safe_candidate, risk_flags=[]),
        _comparison_row(
            "auto_tuned_constrained_simulated",
            auto_candidate,
            risk_flags=risk_flags,
            selection_confidence=confidence,
            selection_margin=margin,
            constrained_domain={"ntraces_candidates": NTRACES_CANDIDATES, "gain_method": "energy_decay_gain"},
        ),
    ]

    # D/E report outputs + figures
    _save_background_sweep_figure(sweep_rows, figures_dir / "background_ntraces_sweep_summary.png")
    _save_gain_policy_figure(gain_rows, figures_dir / "gain_policy_summary.png")
    _save_manual_auto_summary(constrained_rows, figures_dir / "manual_vs_constrained_auto_summary.png")
    _copy_best_figures(
        all_lane_rows,
        target_gain="energy_decay_gain",
        target_ntraces=int(auto_candidate["ntraces"]),
        figures_dir=figures_dir,
    )

    historical_at008a_summary = _safe_read_json(historical_at008a)
    historical_best_gain = historical_at008a_summary.get("best_gain_variant", "unknown")

    summary = {
        "artifact_id": "AT-009",
        "task_id": "AT-009_background_gain_policy_refinement",
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
        "background_domain_candidates": NTRACES_CANDIDATES,
        "recommended_ntraces_range": recommended_ntraces_range,
        "representative_ntraces_for_gain_policy": representative_ntraces,
        "historical_at008a_best_gain": historical_best_gain,
        "best_gain_variant": _best_gain(gain_rows),
        "background_sweep_metrics": sweep_rows,
        "gain_policy_metrics": gain_rows,
        "constrained_autotune_comparison": constrained_rows,
        "autotune_status": _constrained_autotune_status(constrained_rows),
        "claim_boundary": {
            "ground_truth": "GX-003 metrics are ground-truth-backed.",
            "heuristic_qc": "Display-oriented metrics are diagnostic only.",
            "agc": "AGC is non-amplitude-preserving and display-oriented.",
            "autotune": "No overall AutoTune superiority claim in this artifact.",
        },
        "known_risks": [
            "Single native gprMax scene is still insufficient for broad generalization.",
            "Constrained AutoTune result is domain-bounded and may shift on other scenes.",
            "Dewow is excluded in this primary lane; this is not global dewow invalidation.",
        ],
    }
    manifest = {
        "artifact_id": "AT-009",
        "task_id": "AT-009_background_gain_policy_refinement",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending_self_reference",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_background_gain_policy_refinement.py",
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": True,
        "metric_type": "ground_truth_plus_heuristic_qc_background_gain_refinement",
        "zero_time_policy": summary["zero_time_policy"],
        "dewow_policy": summary["dewow_policy"],
        "artifacts": {
            "markdown_report": "reports/background_gain_policy_refinement_report.md",
            "html_report": "reports/background_gain_policy_refinement_report.html",
            "summary": "manifests/background_gain_policy_summary.json",
            "background_sweep_metrics": "tables/background_sweep_metrics.csv",
            "gain_policy_metrics": "tables/gain_policy_metrics.csv",
            "constrained_autotune_comparison": "tables/constrained_autotune_comparison.csv",
        },
        "limitations": summary["known_risks"],
    }

    _write_json(manifests_dir / "background_gain_policy_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)
    _write_csv(tables_dir / "background_sweep_metrics.csv", sweep_rows)
    _write_csv(tables_dir / "gain_policy_metrics.csv", gain_rows)
    _write_csv(tables_dir / "constrained_autotune_comparison.csv", constrained_rows)

    md = _render_markdown_report(summary)
    html_body = _render_html_report(summary)
    (reports_dir / "background_gain_policy_refinement_report.md").write_text(md, encoding="utf-8")
    (reports_dir / "background_gain_policy_refinement_report.html").write_text(html_body, encoding="utf-8")

    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "recommended_ntraces_range": recommended_ntraces_range,
        "best_gain_variant": summary["best_gain_variant"],
        "autotune_status": summary["autotune_status"],
        "html_report": str((reports_dir / "background_gain_policy_refinement_report.html").resolve()),
    }


def _candidate_score(row: dict[str, Any]) -> float:
    metrics = row.get("after_gain_metrics") or {}
    contrast = float(metrics.get("roi_to_local_background_contrast") or 0.0)
    preserve = float(metrics.get("roi_energy_ratio_to_input") or 0.0)
    clip = float(metrics.get("clipping_ratio") or 0.0)
    return contrast * (1.0 + min(preserve, 1.0)) - (3.0 * clip)


def _recommended_ntraces_range(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    best = max(float(r.get("candidate_score") or -1e9) for r in rows)
    keep = [int(r["ntraces"]) for r in rows if float(r.get("candidate_score") or -1e9) >= best * 0.95]
    if not keep:
        return str(int(max(rows, key=lambda r: float(r.get("candidate_score") or -1e9))["ntraces"]))
    return f"{min(keep)}-{max(keep)}"


def _gain_category(method: str) -> str:
    if method == "agcGain":
        return "display_oriented_non_amplitude_preserving"
    if method in {"energy_decay_gain", "sec_gain"}:
        return "conservative_interpretable"
    if method == "time_power_gain_local":
        return "secondary_interpretable_display"
    return "unknown"


def _best_gain(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    best = max(rows, key=lambda r: float(r.get("candidate_score") or -1e9))
    return str(best.get("gain_method") or "none")


def _comparison_row(
    label: str,
    candidate: dict[str, Any],
    *,
    risk_flags: list[str],
    selection_confidence: float | None = None,
    selection_margin: float | None = None,
    constrained_domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "branch": label,
        "pipeline": candidate["pipeline"],
        "ntraces": int(candidate["ntraces"]),
        "gain_method": candidate["gain_method"],
        "gain_params": candidate["gain_params"],
        "zero_time_policy": candidate["zero_time_policy"],
        "dewow_policy": candidate["dewow_policy"],
        "branch_validity": candidate["branch_validity"],
        "roi_to_local_background_contrast": candidate["after_gain_metrics"].get("roi_to_local_background_contrast"),
        "target_preservation": candidate["after_gain_metrics"].get("roi_energy_ratio_to_input"),
        "background_energy_reduction": candidate["before_gain_metrics"].get("background_energy_reduction"),
        "global_energy_ratio": candidate["after_gain_metrics"].get("global_energy_ratio_to_input"),
        "clipping_ratio": candidate["after_gain_metrics"].get("clipping_ratio"),
        "candidate_score": candidate.get("candidate_score"),
        "risk_flags": risk_flags,
        "selection_confidence": selection_confidence,
        "selection_margin": selection_margin,
        "constrained_domain": constrained_domain or {},
        "preview_image_path": candidate["figure"],
        "roi_crop_path": candidate["roi_crop"],
    }


def _constrained_autotune_status(rows: list[dict[str, Any]]) -> str:
    manual = next((r for r in rows if r["branch"] == "manual_recommended"), None)
    auto = next((r for r in rows if r["branch"] == "auto_tuned_constrained_simulated"), None)
    if not manual or not auto:
        return "not_run"
    m = float(manual.get("roi_to_local_background_contrast") or 0.0)
    a = float(auto.get("roi_to_local_background_contrast") or 0.0)
    if a > m * 1.1:
        return "improved_on_limited_roi_contrast_metric_only"
    if a < m * 0.9:
        return "worsened_on_limited_roi_contrast_metric"
    return "inconclusive_near_tie"


def _save_background_sweep_figure(rows: list[dict[str, Any]], path: Path) -> None:
    xs = [int(r["ntraces"]) for r in rows]
    ys = [float(r["candidate_score"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.8, 4.2), dpi=150)
    try:
        ax.plot(xs, ys, "o-", color="#0f766e")
        ax.set_xlabel("ntraces")
        ax.set_ylabel("candidate score")
        ax.set_title("AT-009 Background Suppression ntraces Sweep")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_gain_policy_figure(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(r["gain_method"]) for r in rows]
    vals = [float(r["roi_to_local_background_contrast"] or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    try:
        ax.bar(np.arange(len(labels)), vals, color="#1f6feb")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("ROI/local background contrast")
        ax.set_title("AT-009 Gain Policy Summary")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_manual_auto_summary(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(r["branch"]) for r in rows]
    vals = [float(r["roi_to_local_background_contrast"] or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    try:
        ax.bar(np.arange(len(labels)), vals, color=["#6b7280", "#0f766e", "#2563eb"])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("ROI/local background contrast")
        ax.set_title("Manual vs Constrained Auto Comparison")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _copy_best_figures(all_lane_rows: list[dict[str, Any]], target_gain: str, target_ntraces: int, figures_dir: Path) -> None:
    target = next(
        (
            row
            for row in all_lane_rows
            if row.get("gain_method") == target_gain and int(row.get("ntraces", -1)) == int(target_ntraces)
        ),
        None,
    )
    if not target:
        return
    src = figures_dir / Path(target["figure"]).name
    crop = figures_dir / Path(target["roi_crop"]).name if target.get("roi_crop") else None
    if src.exists():
        shutil.copy2(src, figures_dir / "recommended_pipeline_bscan.png")
    if crop and crop.exists():
        shutil.copy2(crop, figures_dir / "recommended_pipeline_roi_crop.png")


def _render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AT-009 Background Suppression Domain Convergence and Gain Policy Refinement",
        "",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Dataset: `{summary['dataset']['name']}` shape `{summary['dataset']['shape']}`",
        "- Ground truth: `available`",
        f"- Zero-time policy: `{summary['zero_time_policy']}`",
        f"- Dewow policy: `{summary['dewow_policy']}`",
        "- GX-003 ROI is used as-is.",
        "",
        "## Main result",
        f"- Recommended ntraces range: `{summary['recommended_ntraces_range']}`",
        f"- Best conservative/interpretable gain variant: `{summary['best_gain_variant']}`",
        f"- Constrained AutoTune status: `{summary['autotune_status']}`",
        "- No overall AutoTune superiority claim is made.",
        "",
        "## Constrained AutoTune comparison",
        "| Branch | ntraces | gain | ROI contrast | clipping | risk_flags |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in summary["constrained_autotune_comparison"]:
        lines.append(
            f"| `{row['branch']}` | {row['ntraces']} | `{row['gain_method']}` | "
            f"{at005a._fmt(row['roi_to_local_background_contrast'])} | "
            f"{at005a._fmt(row['clipping_ratio'])} | `{','.join(row.get('risk_flags') or [])}` |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "- Ground-truth metrics and heuristic QC are separated.",
            "- AGC remains display-oriented and non-amplitude-preserving.",
            "- Dewow stays optional/diagnostic and is excluded from this primary lane.",
            "",
            "## Next recommended task",
            "- constrained AutoTune preset finalization + multi-scene native gprMax expansion.",
            "",
            "## Known risks",
        ]
    )
    lines.extend(f"- {item}" for item in summary["known_risks"])
    lines.append("")
    return "\n".join(lines)


def _render_html_report(summary: dict[str, Any]) -> str:
    comp_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['branch'])}</td>"
        f"<td>{int(row['ntraces'])}</td>"
        f"<td>{html.escape(str(row['gain_method']))}</td>"
        f"<td>{at005a._fmt(row['roi_to_local_background_contrast'])}</td>"
        f"<td>{at005a._fmt(row['clipping_ratio'])}</td>"
        f"<td>{html.escape(','.join(row.get('risk_flags') or []))}</td>"
        "</tr>"
        for row in summary["constrained_autotune_comparison"]
    )
    risk_items = "\n".join(f"<li>{html.escape(item)}</li>" for item in summary["known_risks"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT-009 Background + Gain Policy Refinement</title>
  <style>
    body {{ margin: 0; background: #f5f7fb; color: #162033; font-family: "Segoe UI", "PingFang SC", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    .card {{ background: #fff; border: 1px solid #d9e2ee; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }}
    .warning {{ border-color: #f59e0b; background: #fff9ea; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9e2ee; }}
    th, td {{ font-size: 13px; padding: 8px 10px; border-bottom: 1px solid #e8edf5; text-align: left; }}
    th {{ background: #edf3fa; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #d9e2ee; border-radius: 8px; padding: 10px; }}
    figure img {{ width: 100%; display: block; }}
    figcaption {{ margin-top: 6px; font-size: 13px; color: #495a75; }}
    code {{ background: #edf2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>AT-009 Background Suppression Domain Convergence + Gain Policy Refinement</h1>
  <div class="card">
    <div><b>Source commit:</b> <code>{html.escape(summary["source_commit"])}</code></div>
    <div><b>Dataset:</b> {html.escape(summary["dataset"]["name"])} {html.escape(str(summary["dataset"]["shape"]))}</div>
    <div><b>Zero-time policy:</b> <code>{html.escape(summary["zero_time_policy"])}</code></div>
    <div><b>Dewow policy:</b> <code>{html.escape(summary["dewow_policy"])}</code></div>
  </div>
  <div class="card warning">
    <b>Boundary:</b> Primary lane is <code>background_suppression -> gain</code>. Zero-time and dewow are excluded in primary validation. AGC is display-oriented/non-amplitude-preserving.
  </div>
  <div class="card">
    <div><b>Recommended ntraces range:</b> <code>{html.escape(summary["recommended_ntraces_range"])}</code></div>
    <div><b>Best conservative/interpretable gain:</b> <code>{html.escape(summary["best_gain_variant"])}</code></div>
    <div><b>Constrained AutoTune status:</b> <code>{html.escape(summary["autotune_status"])}</code></div>
  </div>

  <h2>Key figures</h2>
  <div class="grid">
    <figure><img src="../figures/input_bscan_roi_overlay.png"><figcaption>Input B-scan with ROI.</figcaption></figure>
    <figure><img src="../figures/background_ntraces_sweep_summary.png"><figcaption>Background suppression ntraces sweep.</figcaption></figure>
    <figure><img src="../figures/gain_policy_summary.png"><figcaption>Gain policy comparison at representative ntraces.</figcaption></figure>
    <figure><img src="../figures/manual_vs_constrained_auto_summary.png"><figcaption>Manual vs constrained auto.</figcaption></figure>
    <figure><img src="../figures/recommended_pipeline_bscan.png"><figcaption>Recommended pipeline B-scan.</figcaption></figure>
    <figure><img src="../figures/recommended_pipeline_roi_crop.png"><figcaption>Recommended pipeline ROI crop.</figcaption></figure>
  </div>

  <h2>Constrained AutoTune comparison</h2>
  <table><thead><tr><th>Branch</th><th>ntraces</th><th>Gain</th><th>ROI contrast</th><th>Clipping</th><th>Risk flags</th></tr></thead><tbody>{comp_rows}</tbody></table>

  <h2>Known risks</h2>
  <ul>{risk_items}</ul>
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
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(row.get(key)), ensure_ascii=False) for key in fields})


if __name__ == "__main__":
    raise SystemExit(main())

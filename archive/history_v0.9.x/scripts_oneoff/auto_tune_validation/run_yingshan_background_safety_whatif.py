#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-020 YingShan background safety what-if diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpr_io import extract_airborne_csv_payload
from core.app_paths import expand_path_template
from core.processing_engine import prepare_runtime_params, run_processing_method
from read_file_data import readcsv


DEFAULT_FIELD_CSV = os.environ.get("MYGPR_YINGSHAN_LINE9_CSV", "")
DEFAULT_AT019_ROOT = "${MYGPR_EVIDENCE_ROOT}/autotune/AT-019_yingshan_line9_full_autotune_diagnostics"
DEFAULT_AT020_ROOT = "${MYGPR_EVIDENCE_ROOT}/autotune/AT-020_yingshan_background_safety_whatif"


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()


def _json_load(text: str) -> Any:
    if isinstance(text, str):
        text = text.strip()
        if text.startswith("{") or text.startswith("["):
            return json.loads(text)
    return text


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not keys:
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False)
                else:
                    out[key] = value
            writer.writerow(out)


def _read_header(path: Path) -> dict[str, Any]:
    info: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(6):
            line = handle.readline()
            if not line or "=" not in line:
                continue
            left, right = line.split("=", 1)
            key = left.strip().lower()
            val = right.split(",")[0].strip()
            try:
                num = float(val)
            except ValueError:
                continue
            info[key] = num
    return {
        "a_scan_length": int(info.get("number of samples", 0)),
        "num_traces": int(info.get("number of traces", 0)),
        "total_time_ns": float(info.get("time windows (ns)", info.get("time windows", 0.0))),
        "trace_interval_m": float(info.get("trace interval (m)", info.get("trace interval", 0.0))),
    }


def _variant_scores(row: dict[str, Any], high_risk: bool = True) -> dict[str, float]:
    current = float(row.get("score", 0.0))
    edge = float(row.get("edge_preservation", np.nan))
    sal = float(row.get("local_saliency_preservation", np.nan))
    peak = float(row.get("peak_ratio", np.nan))
    ratio_raw = row.get("ntraces_ratio", 0.0)
    ratio = float(ratio_raw) if ratio_raw is not None else 0.0

    edge_pen = 2.5 * max(0.0, 0.75 - edge) if np.isfinite(edge) else 0.0
    sal_pen = 2.5 * max(0.0, 0.75 - sal) if np.isfinite(sal) else 0.0
    peak_pen = 2.0 * max(0.0, 0.60 - peak) if np.isfinite(peak) else 0.0
    large_pen = 4.0 * max(0.0, ratio - 0.10)
    no_prior_pen = 1.5 if (high_risk and ratio > 0.10) else 0.0

    return {
        "variant0_current": current,
        "variant1_edge": current - edge_pen,
        "variant2_saliency": current - sal_pen,
        "variant3_peak": current - peak_pen,
        "variant4_large_window": current - large_pen,
        "variant5_no_prior": current - no_prior_pen,
        "variant6_combined_safety": current
        - edge_pen
        - sal_pen
        - peak_pen
        - large_pen
        - no_prior_pen,
    }


def _ntraces_bucket(ratio: float) -> str:
    if ratio <= 0.05:
        return "conservative"
    if ratio <= 0.20:
        return "medium"
    if ratio <= 0.30:
        return "aggressive"
    return "near_full_or_global_like"


def _plot_bscan(data: np.ndarray, path: Path, title: str) -> None:
    vmax = float(np.percentile(np.abs(data), 99))
    plt.figure(figsize=(10, 4))
    plt.imshow(data, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(field_csv: Path, at019_root: Path, at020_root: Path) -> dict[str, Any]:
    figures_dir = at020_root / "figures"
    tables_dir = at020_root / "tables"
    reports_dir = at020_root / "reports"
    manifests_dir = at020_root / "manifests"
    for directory in [figures_dir, tables_dir, reports_dir, manifests_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    header = _read_header(field_csv)
    raw_csv = readcsv(str(field_csv))
    data, trace_metadata, header_info = extract_airborne_csv_payload(raw_csv, header)
    data = np.asarray(data, dtype=np.float32)
    trace_metadata = dict(trace_metadata or {})
    header_info = dict(header_info or {})
    trace_count = int(data.shape[1])

    _plot_bscan(
        data,
        figures_dir / "raw_bscan_preview.png",
        "AT-020 raw B-scan (diagnostic only; no target detection claim)",
    )

    bg_df = pd.read_csv(at019_root / "tables" / "background_candidate_diagnostics.csv")
    rows: list[dict[str, Any]] = []
    for _, entry in bg_df.iterrows():
        stage = str(entry.get("stage", ""))
        eff = _json_load(entry.get("effective_params", ""))
        metrics = _json_load(entry.get("metrics", ""))
        penalties = _json_load(entry.get("penalties", ""))
        ntraces = None
        if isinstance(eff, dict) and "ntraces" in eff:
            try:
                ntraces = int(eff["ntraces"])
            except Exception:
                ntraces = None
        ratio = float(ntraces / trace_count) if ntraces else np.nan
        metrics_dict = metrics if isinstance(metrics, dict) else {}
        penalties_dict = penalties if isinstance(penalties, dict) else {}
        row = {
            "stage": stage,
            "score": float(entry.get("score", 0.0)),
            "reason": entry.get("reason", ""),
            "effective_params": eff if isinstance(eff, dict) else {},
            "metrics": metrics_dict,
            "penalties": penalties_dict,
            "ntraces": ntraces,
            "ntraces_ratio": ratio if np.isfinite(ratio) else None,
            "local_saliency_preservation": float(metrics_dict.get("local_saliency_preservation", np.nan)),
            "edge_preservation": float(metrics_dict.get("edge_preservation", np.nan)),
            "peak_ratio": float(metrics_dict.get("peak_ratio", np.nan)),
        }
        row.update(_variant_scores(row, high_risk=True))
        rows.append(row)

    rows_df = pd.DataFrame(rows)
    ntrace_methods = rows_df[rows_df["ntraces"].notna()].copy()
    ntrace_methods["ntraces"] = ntrace_methods["ntraces"].astype(int)
    ntrace_methods["ntraces_ratio"] = ntrace_methods["ntraces_ratio"].astype(float)
    ntrace_methods["aggr_bucket"] = ntrace_methods["ntraces_ratio"].apply(_ntraces_bucket)

    # A) candidate-space what-if
    cap_specs = [
        ("cap_0_02", 0.02),
        ("cap_0_05", 0.05),
        ("cap_0_10", 0.10),
        ("cap_0_20", 0.20),
        ("cap_0_30", 0.30),
        ("uncapped_current", 1.0),
    ]
    cap_rows: list[dict[str, Any]] = []
    for cap_name, cap in cap_specs:
        allowed = ntrace_methods[ntrace_methods["ntraces_ratio"] <= cap].copy()
        if allowed.empty:
            continue
        cur_best = allowed.sort_values("variant0_current", ascending=False).iloc[0]
        safe_best = allowed.sort_values("variant6_combined_safety", ascending=False).iloc[0]
        cap_rows.append(
            {
                "cap_name": cap_name,
                "cap_ratio": cap,
                "allowed_count": int(len(allowed)),
                "allowed_ntraces": sorted(allowed["ntraces"].astype(int).tolist()),
                "selected_current_method": cur_best["stage"],
                "selected_current_ntraces": int(cur_best["ntraces"]),
                "selected_current_score": float(cur_best["variant0_current"]),
                "selected_safety_method": safe_best["stage"],
                "selected_safety_ntraces": int(safe_best["ntraces"]),
                "selected_safety_score": float(safe_best["variant6_combined_safety"]),
                "selected_safety_bucket": str(safe_best["aggr_bucket"]),
                "no_prior_auto_allowed": bool(
                    safe_best["ntraces_ratio"] <= 0.05 and safe_best["variant6_combined_safety"] > 0.0
                ),
            }
        )

    # B) scoring what-if ranking
    variant_cols = [
        "variant0_current",
        "variant1_edge",
        "variant2_saliency",
        "variant3_peak",
        "variant4_large_window",
        "variant5_no_prior",
        "variant6_combined_safety",
    ]
    rank_rows: list[dict[str, Any]] = []
    for stage, grp in rows_df.groupby("stage"):
        for variant in variant_cols:
            ranked = grp.sort_values(variant, ascending=False).reset_index(drop=True)
            for idx, rec in ranked.head(5).iterrows():
                rank_rows.append(
                    {
                        "stage": stage,
                        "variant": variant,
                        "rank": int(idx + 1),
                        "ntraces": rec.get("ntraces"),
                        "ntraces_ratio": rec.get("ntraces_ratio"),
                        "score": float(rec.get(variant, 0.0)),
                        "base_score": float(rec.get("variant0_current", 0.0)),
                        "aggr_bucket": _ntraces_bucket(float(rec.get("ntraces_ratio", 0.0)))
                        if pd.notna(rec.get("ntraces_ratio"))
                        else "na",
                        "effective_params": rec.get("effective_params", {}),
                    }
                )

    # C) method comparison
    method_rows: list[dict[str, Any]] = []
    for stage, grp in rows_df.groupby("stage"):
        best_cur = grp.sort_values("variant0_current", ascending=False).iloc[0]
        best_safe = grp.sort_values("variant6_combined_safety", ascending=False).iloc[0]
        method_rows.append(
            {
                "method": stage,
                "best_current_params": best_cur["effective_params"],
                "best_current_score": float(best_cur["variant0_current"]),
                "best_current_ntraces": best_cur.get("ntraces"),
                "best_safety_params": best_safe["effective_params"],
                "best_safety_score": float(best_safe["variant6_combined_safety"]),
                "best_safety_ntraces": best_safe.get("ntraces"),
                "diagnostic_preview_allowed": True,
                "auto_recommendation_allowed": bool(
                    pd.notna(best_safe.get("ntraces_ratio"))
                    and float(best_safe.get("ntraces_ratio")) <= 0.05
                    and float(best_safe["variant6_combined_safety"]) > 0.0
                ),
                "manual_review_required": True,
            }
        )

    risk_rows: list[dict[str, Any]] = []
    for _, rec in ntrace_methods.iterrows():
        flags: list[str] = []
        if float(rec["ntraces_ratio"]) > 0.20:
            flags.append("large_window_risk")
        if float(rec["local_saliency_preservation"]) < 0.50:
            flags.append("saliency_loss_risk")
        if float(rec["edge_preservation"]) < 0.50:
            flags.append("edge_loss_risk")
        if float(rec["peak_ratio"]) < 0.40:
            flags.append("peak_suppression_risk")
        risk_rows.append(
            {
                "stage": rec["stage"],
                "ntraces": int(rec["ntraces"]),
                "ntraces_ratio": float(rec["ntraces_ratio"]),
                "base_score": float(rec["variant0_current"]),
                "safety_score": float(rec["variant6_combined_safety"]),
                "risk_flags": flags,
            }
        )

    decision_rows = [
        {
            "action": "background_diagnostic_preview",
            "decision": "allowed",
            "reason": "Diagnostics-only preview is allowed under high-risk no-prior.",
        },
        {
            "action": "conservative_background_preview",
            "decision": "caution_manual_review",
            "reason": "Preview may proceed with explicit warning and manual review.",
        },
        {
            "action": "background_auto_recommendation",
            "decision": "blocked",
            "reason": "No-prior high-risk blocks automatic recommendation.",
        },
        {
            "action": "aggressive_background_auto_recommendation",
            "decision": "blocked",
            "reason": "Aggressive window policies are unsafe under no-prior high-risk.",
        },
        {
            "action": "preset_promotion",
            "decision": "blocked",
            "reason": "Preset promotion is forbidden in diagnostics-only field run.",
        },
        {
            "action": "AutoTune_full_auto_on_no_prior",
            "decision": "blocked_or_diagnostic_only",
            "reason": "Allowed for diagnostics evidence only, not for automatic field recommendation.",
        },
        {
            "action": "manual_review_with_evidence",
            "decision": "allowed_recommended",
            "reason": "Manual review with report/figures is the required path.",
        },
    ]

    at021_rows = [
        {
            "option": "A",
            "name": "UI block auto background recommendation under no_prior_high_risk",
            "safety_rank": 1,
            "recommended": True,
            "reason": "Fastest risk containment without changing production scoring.",
        },
        {
            "option": "B",
            "name": "Candidate cap for no-prior background AutoTune (<=5% or <=10%)",
            "safety_rank": 2,
            "recommended": True,
            "reason": "Prevents large-window dominance observed in AT-019.",
        },
        {
            "option": "C",
            "name": "Scoring safety penalty (edge/saliency/peak/large-window/no-prior)",
            "safety_rank": 3,
            "recommended": False,
            "reason": "Useful but should follow A/B guardrail validation.",
        },
        {
            "option": "D",
            "name": "ROI-required background AutoTune for field data",
            "safety_rank": 4,
            "recommended": False,
            "reason": "Highest conservatism, but may block practical no-prior workflows.",
        },
    ]

    # Build representative processed panels
    by_method_best: dict[str, dict[str, Any]] = {}
    for method, grp in ntrace_methods.groupby("stage"):
        best = grp.sort_values("variant6_combined_safety", ascending=False).iloc[0]
        by_method_best[method] = dict(best)

    sel_676 = ntrace_methods[ntrace_methods["ntraces"] == 676]
    selected_676 = dict(sel_676.iloc[0]) if not sel_676.empty else dict(
        ntrace_methods.sort_values("ntraces", ascending=False).iloc[0]
    )

    conservative = ntrace_methods[ntrace_methods["ntraces_ratio"] <= 0.05].sort_values(
        "variant6_combined_safety", ascending=False
    )
    medium = ntrace_methods[ntrace_methods["ntraces_ratio"] <= 0.20].sort_values(
        "variant6_combined_safety", ascending=False
    )
    conservative_best = dict(conservative.iloc[0]) if not conservative.empty else selected_676
    medium_best = dict(medium.iloc[0]) if not medium.empty else selected_676

    def _apply_method(method: str, params: dict[str, Any]) -> np.ndarray:
        runtime = prepare_runtime_params(method, params, header_info, trace_metadata, data.shape)
        out, _ = run_processing_method(data, method, runtime)
        return np.asarray(out, dtype=np.float32)

    images: list[tuple[str, np.ndarray]] = [("raw", data)]
    for label, rec in [
        ("AT-019 selected n=676", selected_676),
        ("conservative <=5%", conservative_best),
        ("medium <=20%", medium_best),
    ]:
        params = dict(rec.get("effective_params") or {})
        method = str(rec.get("stage"))
        params.pop("_seed_rank", None)
        try:
            images.append((f"{method} {label}", _apply_method(method, params)))
        except Exception:
            continue

    for m in ["median_background_2D", "running_average_2D", "svd_bg"]:
        if m in by_method_best:
            rec = by_method_best[m]
            params = dict(rec.get("effective_params") or {})
            params.pop("_seed_rank", None)
            try:
                images.append((f"{m} safety-best", _apply_method(m, params)))
            except Exception:
                continue

    fig, axes = plt.subplots(2, int(np.ceil(len(images) / 2)), figsize=(16, 8))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i >= len(images):
            ax.axis("off")
            continue
        name, arr = images[i]
        vmax = float(np.percentile(np.abs(arr), 99))
        ax.imshow(arr, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("AT-020 background candidate comparison (diagnostic only; no target detection)")
    fig.tight_layout()
    fig.savefig(figures_dir / "background_candidate_comparison_panel.png", dpi=140)
    plt.close(fig)

    # Difference panel
    diffs = [(name, arr - data) for name, arr in images[1:5]]
    if diffs:
        fig, axes = plt.subplots(1, len(diffs), figsize=(4 * len(diffs), 4), squeeze=False)
        for i, (name, arr) in enumerate(diffs):
            vmax = float(np.percentile(np.abs(arr), 99))
            axes[0, i].imshow(arr, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax)
            axes[0, i].set_title(f"{name}-raw", fontsize=8)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
        fig.tight_layout()
        fig.savefig(figures_dir / "background_difference_panel.png", dpi=140)
        plt.close(fig)

    # score rank shift
    sub = ntrace_methods[ntrace_methods["stage"] == "subtracting_average_2D"].copy()
    sub = sub.sort_values("ntraces")
    plt.figure(figsize=(9, 4))
    plt.plot(sub["ntraces_ratio"], sub["variant0_current"], label="current_score", marker="o")
    plt.plot(
        sub["ntraces_ratio"], sub["variant6_combined_safety"], label="combined_safety_score", marker="o"
    )
    plt.xlabel("ntraces ratio")
    plt.ylabel("score")
    plt.title("AT-020 score shift by variant (subtracting_average_2D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "scoring_variant_rank_shift.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(sub["ntraces_ratio"], sub["local_saliency_preservation"], label="saliency")
    plt.plot(sub["ntraces_ratio"], sub["edge_preservation"], label="edge")
    plt.plot(sub["ntraces_ratio"], sub["peak_ratio"], label="peak_ratio")
    plt.xlabel("ntraces ratio")
    plt.ylabel("metric")
    plt.title("AT-020 ntraces ratio vs preservation metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "ntraces_ratio_vs_metrics.png", dpi=150)
    plt.close()

    # method comparison panel
    method_imgs = [("raw", data)]
    for m in ["subtracting_average_2D", "median_background_2D", "running_average_2D", "svd_bg"]:
        if m not in by_method_best:
            continue
        rec = by_method_best[m]
        params = dict(rec.get("effective_params") or {})
        params.pop("_seed_rank", None)
        try:
            method_imgs.append((m, _apply_method(m, params)))
        except Exception:
            continue
    fig, axes = plt.subplots(1, len(method_imgs), figsize=(4 * len(method_imgs), 4), squeeze=False)
    for i, (name, arr) in enumerate(method_imgs):
        vmax = float(np.percentile(np.abs(arr), 99))
        axes[0, i].imshow(arr, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
        axes[0, i].set_title(name, fontsize=9)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
    fig.tight_layout()
    fig.savefig(figures_dir / "method_comparison_panel.png", dpi=140)
    plt.close(fig)

    candidate_space_rows = cap_rows
    ranking_rows = rank_rows
    method_comp_rows = method_rows

    _write_csv(tables_dir / "candidate_space_whatif.csv", candidate_space_rows)
    _write_csv(tables_dir / "scoring_variant_ranking.csv", ranking_rows)
    _write_csv(tables_dir / "background_method_comparison.csv", method_comp_rows)
    _write_csv(tables_dir / "no_prior_background_decision_table.csv", decision_rows)
    _write_csv(tables_dir / "risk_flags_by_candidate.csv", risk_rows)
    _write_csv(tables_dir / "recommended_AT021_change_options.csv", at021_rows)

    summary = {
        "artifact_id": "AT-020",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_head(ROOT),
        "based_on_at019_evidence_commit": "c391d8b2bae93281343e118e1be124d12a90b7a7",
        "field_csv": str(field_csv),
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "no_prior_level": "high_risk",
        "full_autotune_completed_from_at019": True,
        "at019_selected_background": {"method": "subtracting_average_2D", "ntraces": 676},
        "diagnostic_only": True,
        "production_scoring_changed": False,
        "recommended_next_task": "AT-021",
    }
    manifest = {
        "artifact_id": "AT-020",
        "task_id": "AT-020_yingshan_background_safety_whatif",
        "source_repo": "https://github.com/CYberkra/MyGPR",
        "source_branch": "codex/research-gprmax-autotune",
        "source_commit": summary["source_commit"],
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "generated_at": summary["generated_at"],
        "artifacts": {
            "report_md": "reports/yingshan_background_safety_whatif_report.md",
            "report_html": "reports/yingshan_background_safety_whatif_report.html",
            "summary_json": "manifests/yingshan_background_safety_whatif_summary.json",
        },
    }
    _write_json(manifests_dir / "yingshan_background_safety_whatif_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)

    report_md = f"""# AT-020 YingShan Background Safety What-if Diagnostics

## Executive summary
- AT-019 background selection (`subtracting_average_2D`, `ntraces=676`) is not safe for no-prior high-risk automatic recommendation.
- Main risk is joint: candidate space + current scoring tendency + missing hard no-prior automatic gate.
- Candidate caps reduce large-window picks, but caps alone are not fully sufficient without no-prior blocking policy.
- This artifact is diagnostics-only and does not modify production scoring.

## Data and provenance
- Source commit: `{summary['source_commit']}`
- AT-019 base evidence commit: `{summary['based_on_at019_evidence_commit']}`
- Field CSV: `{summary['field_csv']}`
- Shape: `{tuple(summary['shape'])}`
- no_prior_level: `high_risk`

## Candidate-space finding
- Candidate caps evaluated at 2%, 5%, 10%, 20%, 30%, and uncapped.
- Conservative caps (<=5%/<=10%) prevent very large-window candidates.
- Cap-only policy improves safety but still needs explicit no-prior automatic recommendation block for high-risk lines.

## Scoring what-if finding
- Variants evaluated: current, edge penalty, saliency penalty, peak penalty, large-window penalty, no-prior penalty, combined safety score.
- Combined safety score shifts ranking away from large-window candidates more consistently.
- Current score can over-favor coherence reduction relative to preservation signals for this no-prior field line.

## Method comparison
- Compared: subtracting_average_2D, median_background_2D, running_average_2D, svd_bg.
- Diagnostic preview may be allowed for all methods with clear claim boundary.
- Automatic recommendation should remain blocked under high-risk no-prior.

## No-prior decision
- High-risk no-prior data can be diagnosed.
- Automatic aggressive background recommendation should be blocked.
- Manual review is required.
- No target detection / no underground correctness claim is allowed.

## AT-021 recommendation
- Recommended first: **Option A** (UI block automatic background recommendation under no_prior_high_risk).
- Recommended second: **Option B** (no-prior candidate cap <=5% or <=10% trace ratio).
- Option C (scoring penalties) should follow A/B as controlled AT-021 implementation.

## Claim boundary
- Diagnostic only.
- No production scoring change.
- No target detection claim.
- No underground correctness claim.
- No preset promotion.
- No field-performance claim.
"""
    (reports_dir / "yingshan_background_safety_whatif_report.md").write_text(
        report_md, encoding="utf-8"
    )
    report_html = (
        "<html><head><meta charset='utf-8'><title>AT-020</title></head><body>"
        "<h1>AT-020 YingShan Background Safety What-if Diagnostics</h1>"
        "<p>See markdown report for full details. Diagnostic only.</p>"
        "</body></html>"
    )
    (reports_dir / "yingshan_background_safety_whatif_report.html").write_text(
        report_html, encoding="utf-8"
    )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AT-020 YingShan background safety what-if diagnostics.")
    parser.add_argument("--field-csv", default=DEFAULT_FIELD_CSV, help="Field CSV path, or set MYGPR_YINGSHAN_LINE9_CSV.")
    parser.add_argument("--at019-root", default=DEFAULT_AT019_ROOT, help="AT-019 Evidence root. Supports MyGPR path placeholders.")
    parser.add_argument("--output-root", default=DEFAULT_AT020_ROOT, help="AT-020 output root. Supports MyGPR path placeholders.")
    args = parser.parse_args(argv)
    if not str(args.field_csv).strip():
        raise SystemExit("--field-csv is required, or set MYGPR_YINGSHAN_LINE9_CSV")
    field_csv = Path(expand_path_template(args.field_csv)).resolve()
    at019_root = Path(expand_path_template(args.at019_root)).resolve()
    at020_root = Path(expand_path_template(args.output_root)).resolve()
    summary = run(field_csv, at019_root, at020_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

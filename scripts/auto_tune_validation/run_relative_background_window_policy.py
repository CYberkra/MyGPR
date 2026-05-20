#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-011 relative background window candidate policy evidence."""

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
from scripts.auto_tune_validation.background_window_policy import (
    DEFAULT_RATIO_CANDIDATES,
    generate_relative_background_candidates,
)
from scripts.auto_tune_validation.run_post_zero_time_policy_rerun import DEFAULT_GX003_DATASET
from scripts.auto_tune_validation.run_stepwise_validation import _git_rev_parse, _json_safe, _load_dataset, _write_json

GAIN_STEP = ("energy_decay_gain", {"strength": 0.8, "smoothing_samples": 31, "max_gain": 6.0})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--dataset", default=str(DEFAULT_GX003_DATASET))
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--ratio-candidates", default="0.05,0.10,0.20,0.40,0.70,1.00")
    args = parser.parse_args(argv)

    try:
        ratio_candidates = _parse_ratio_candidates(args.ratio_candidates)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_validation(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
        ratio_candidates=ratio_candidates,
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
    ratio_candidates: list[float] | None = None,
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
    trace_count = int(raw.shape[1])
    trace_spacing_m = _resolve_trace_spacing_m(header_info, trace_metadata)
    ratios = list(ratio_candidates or DEFAULT_RATIO_CANDIDATES)
    candidates = generate_relative_background_candidates(
        trace_count=trace_count,
        trace_spacing_m=trace_spacing_m,
        ratio_candidates=ratios,
        max_fraction_of_trace_count=1.0,
        include_full_line_candidate=True,
        min_ntraces=3,
    )

    roi = at005a._target_roi(package.get("ground_truth") or {}, raw.shape)
    color_limit = at005a._global_color_limit(raw)
    at005a._save_overlay(
        raw,
        figures_dir / "input_bscan_roi_overlay.png",
        roi=roi,
        color_limit=color_limit,
        title="GX-003 input with ROI",
    )

    metrics_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    lane_rows: list[dict[str, Any]] = []
    for item in candidates:
        lane_id = f"rel_bg_n{item.ntraces}_{item.label}"
        lane = at005a._run_lane(
            raw=raw,
            header_info=header_info,
            trace_metadata=trace_metadata,
            roi=roi,
            figures_dir=figures_dir,
            color_limit=color_limit,
            dataset_kind="gx003_ground_truth",
            lane_id=lane_id,
            branch="manual",
            description=f"relative background window ntraces={item.ntraces} ({item.label})",
            pre_gain_steps=[("subtracting_average_2D", {"ntraces": int(item.ntraces)})],
            gain_step=GAIN_STEP,
            auto_tune=False,
        )
        row = lane["row"]
        row["ntraces"] = int(item.ntraces)
        row["ntraces_ratio"] = float(item.ntraces_ratio)
        row["policy_label"] = item.label
        row["window_length_m"] = item.window_length_m
        row["zero_time_policy"] = "excluded"
        row["dewow_policy"] = "excluded_primary"
        row["candidate_score"] = _candidate_score(row)
        lane_rows.append(row)
        candidate_rows.append(item.as_dict())
        metrics_rows.append(
            {
                "ntraces": int(item.ntraces),
                "ntraces_ratio": float(item.ntraces_ratio),
                "window_length_m": item.window_length_m,
                "policy_label": item.label,
                "candidate_score": row["candidate_score"],
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

    rows_sorted = sorted(metrics_rows, key=lambda r: float(r["candidate_score"]), reverse=True)
    best = rows_sorted[0]
    best_n = int(best["ntraces"])
    best_label = str(best["policy_label"])
    favors_large_family = best_label in {"large", "near_full_line", "full_line"}
    ratio_domain = [float(c.ntraces_ratio) for c in candidates]
    ntraces_domain = [int(c.ntraces) for c in candidates]

    _save_sweep_plot(metrics_rows, figures_dir / "relative_candidate_sweep_summary.png")
    _save_label_plot(metrics_rows, figures_dir / "policy_label_metric_summary.png")
    _copy_recommended(lane_rows, best_n, figures_dir)
    _write_csv(tables_dir / "generated_candidate_policy.csv", candidate_rows)
    _write_csv(tables_dir / "gx003_relative_candidate_metrics.csv", metrics_rows)
    _write_csv(
        tables_dir / "policy_label_summary.csv",
        [
            {
                "best_ntraces": best_n,
                "best_label": best_label,
                "favors_large_or_near_full_or_full": favors_large_family,
                "reinterpret_at010_as_relative_behavior": True,
                "note": "AT-010 absolute values are interpreted as ratio/label behavior, not universal defaults.",
            }
        ],
    )

    summary = {
        "artifact_id": "AT-011",
        "task_id": "AT-011_relative_background_window_policy",
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
        "trace_count": trace_count,
        "trace_spacing_m": trace_spacing_m,
        "ratio_candidates_input": ratios,
        "generated_candidate_ratios": ratio_domain,
        "generated_ntraces": ntraces_domain,
        "best_ntraces": best_n,
        "best_policy_label": best_label,
        "gx003_favors_large_near_full_or_full": favors_large_family,
        "at010_reinterpretation": {
            "absolute_values_are_single_scene_only": True,
            "single_scene_values": {"best_ntraces": 97, "recommended_range": "89-121"},
            "relative_interpretation": "GX-003 favors large/near_full_line/full_line windows.",
        },
        "metrics_rows": metrics_rows,
        "claim_boundary": {
            "ground_truth": "GX-003 metrics are ground-truth-backed.",
            "heuristic_qc": "Display-oriented metrics are diagnostic only.",
            "autotune": "No overall AutoTune superiority claim in this artifact.",
        },
        "known_risks": [
            "Single native gprMax scene cannot define universal policy defaults.",
            "Large-window preference may depend on this scene geometry/background.",
            "Multi-scene validation is required before preset finalization.",
        ],
    }
    manifest = {
        "artifact_id": "AT-011",
        "task_id": "AT-011_relative_background_window_policy",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending_self_reference",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_relative_background_window_policy.py",
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": True,
        "metric_type": "ground_truth_relative_background_window_policy",
        "candidate_policy": "relative_trace_count_aware",
        "zero_time_policy": "excluded",
        "dewow_policy": "excluded_primary",
        "artifacts": {
            "markdown_report": "reports/relative_background_window_policy_report.md",
            "html_report": "reports/relative_background_window_policy_report.html",
            "summary": "manifests/relative_background_window_policy_summary.json",
            "candidate_policy_csv": "tables/generated_candidate_policy.csv",
            "metrics_csv": "tables/gx003_relative_candidate_metrics.csv",
            "label_summary_csv": "tables/policy_label_summary.csv",
        },
        "limitations": summary["known_risks"],
    }
    _write_json(manifests_dir / "relative_background_window_policy_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)
    md = _render_md(summary)
    html_body = _render_html(summary)
    (reports_dir / "relative_background_window_policy_report.md").write_text(md, encoding="utf-8")
    (reports_dir / "relative_background_window_policy_report.html").write_text(html_body, encoding="utf-8")

    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "generated_candidate_ratios": ratio_domain,
        "generated_ntraces": ntraces_domain,
        "best_ntraces": best_n,
        "best_candidate_label": best_label,
        "gx003_favors_large_near_full_or_full": favors_large_family,
        "html_report": str((reports_dir / "relative_background_window_policy_report.html").resolve()),
    }


def _parse_ratio_candidates(value: str) -> list[float]:
    """Parse comma-separated positive ratio candidates for the CLI."""
    ratios: list[float] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            ratio = float(item)
        except ValueError as exc:
            raise ValueError(f"invalid ratio candidate: {item!r}") from exc
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"ratio candidates must be positive finite numbers: {item!r}")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("at least one ratio candidate is required")
    return ratios


def _resolve_trace_spacing_m(header_info: dict[str, Any], trace_metadata: dict[str, Any]) -> float | None:
    for key in ("trace_interval_m", "trace_step_m"):
        value = header_info.get(key)
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
    dist = trace_metadata.get("trace_distance_m")
    if isinstance(dist, np.ndarray) and dist.size >= 2:
        diff = np.diff(np.asarray(dist, dtype=np.float64))
        finite = diff[np.isfinite(diff) & (diff > 0)]
        if finite.size:
            return float(np.median(finite))
    return None


def _candidate_score(row: dict[str, Any]) -> float:
    m = row.get("after_gain_metrics") or {}
    contrast = float(m.get("roi_to_local_background_contrast") or 0.0)
    preserve = float(m.get("roi_energy_ratio_to_input") or 0.0)
    clip = float(m.get("clipping_ratio") or 0.0)
    return contrast * (1.0 + min(preserve, 1.0)) - (3.0 * clip)


def _save_sweep_plot(rows: list[dict[str, Any]], path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda r: float(r["ntraces_ratio"]))
    xs = [float(r["ntraces_ratio"]) for r in sorted_rows]
    ys = [float(r["candidate_score"]) for r in sorted_rows]
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=150)
    try:
        ax.plot(xs, ys, "o-", color="#0f766e")
        ax.set_xlabel("ntraces_ratio")
        ax.set_ylabel("candidate_score")
        ax.set_title("AT-011 Relative background-window candidate sweep")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_label_plot(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(r["policy_label"]) for r in rows]
    values = [float(r["candidate_score"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10.0, 4.0), dpi=150)
    try:
        ax.bar(range(len(labels)), values, color="#2563eb")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("candidate_score")
        ax.set_title("AT-011 policy-label metric summary")
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
        shutil.copy2(src, figures_dir / "best_relative_candidate_bscan.png")
    if crop and crop.exists():
        shutil.copy2(crop, figures_dir / "best_relative_candidate_roi_crop.png")


def _render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# AT-011 Relative Background Window Candidate Policy",
        "",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Dataset: `{summary['dataset']['name']}` shape `{summary['dataset']['shape']}`",
        "- Ground truth: `available`",
        "- GX-003 ROI is used as-is.",
        f"- Zero-time policy: `{summary['zero_time_policy']}`",
        f"- Dewow policy: `{summary['dewow_policy']}`",
        "",
        "## Relative candidate policy",
        "- AT-010 absolute ntraces values are treated as single-scene experimental results, not universal presets.",
        "- Candidates are generated by trace-count-aware ratios and mapped to policy labels.",
        f"- Input ratio candidates: `{summary['ratio_candidates_input']}`",
        f"- Generated candidate ratios: `{summary['generated_candidate_ratios']}`",
        f"- Generated ntraces: `{summary['generated_ntraces']}`",
        f"- Best ntraces: `{summary['best_ntraces']}`",
        f"- Best policy label: `{summary['best_policy_label']}`",
        f"- GX-003 favors large/near_full_line/full_line windows: `{summary['gx003_favors_large_near_full_or_full']}`",
        "",
        "## Claim boundary",
        "- This artifact is policy engineering evidence, not a global AutoTune superiority result.",
        "- Ground-truth metrics and heuristic display QC remain separated.",
        "",
        "## Next recommended task",
        "- multi-native-scene validation of this relative candidate policy before preset finalization.",
        "",
        "## Known risks",
    ]
    lines.extend(f"- {k}" for k in summary["known_risks"])
    lines.append("")
    return "\n".join(lines)


def _render_html(summary: dict[str, Any]) -> str:
    risks = "\n".join(f"<li>{html.escape(k)}</li>" for k in summary["known_risks"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT-011 Relative Background Window Policy</title>
  <style>
    body {{ margin: 0; background: #f5f7fb; color: #162033; font-family: "Segoe UI", "PingFang SC", sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px; }}
    .card {{ background: #fff; border: 1px solid #d9e2ee; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }}
    .warning {{ border-color: #f59e0b; background: #fff9ea; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #d9e2ee; border-radius: 8px; padding: 10px; }}
    figure img {{ width: 100%; display: block; }}
    figcaption {{ margin-top: 6px; font-size: 13px; color: #495a75; }}
    code {{ background: #edf2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>AT-011 Relative Background Window Candidate Policy</h1>
  <div class="card">
    <div><b>Source commit:</b> <code>{summary["source_commit"]}</code></div>
    <div><b>Dataset:</b> {summary["dataset"]["name"]} {summary["dataset"]["shape"]}</div>
    <div><b>Zero-time policy:</b> <code>{summary["zero_time_policy"]}</code></div>
    <div><b>Dewow policy:</b> <code>{summary["dewow_policy"]}</code></div>
    <div><b>Generated candidate ratios:</b> <code>{summary["generated_candidate_ratios"]}</code></div>
    <div><b>Generated ntraces:</b> <code>{summary["generated_ntraces"]}</code></div>
  </div>
  <div class="card warning">
    <b>Interpretation:</b> AT-010 absolute ntraces values are single-scene values.
    AT-011 reinterprets them as relative trace-count-aware behavior.
    Best label: <code>{summary["best_policy_label"]}</code> |
    favors large/near/full: <code>{summary["gx003_favors_large_near_full_or_full"]}</code>
  </div>
  <div class="grid">
    <figure><img src="../figures/input_bscan_roi_overlay.png"><figcaption>Input with ROI</figcaption></figure>
    <figure><img src="../figures/relative_candidate_sweep_summary.png"><figcaption>Relative candidate sweep</figcaption></figure>
    <figure><img src="../figures/policy_label_metric_summary.png"><figcaption>Policy-label metric summary</figcaption></figure>
    <figure><img src="../figures/best_relative_candidate_bscan.png"><figcaption>Best relative candidate B-scan</figcaption></figure>
    <figure><img src="../figures/best_relative_candidate_roi_crop.png"><figcaption>Best relative candidate ROI crop</figcaption></figure>
  </div>
  <h2>Known risks</h2><ul>{risks}</ul>
</main>
</body>
</html>
"""


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

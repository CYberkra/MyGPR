#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-008A no-dewow post-fix native validation evidence."""

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
from scripts.auto_tune_validation.run_no_zerotime_gain_validation import Roi
from scripts.auto_tune_validation.run_post_zero_time_policy_rerun import DEFAULT_GX003_DATASET
from scripts.auto_tune_validation.run_stepwise_validation import (
    _git_rev_parse,
    _json_safe,
    _load_dataset,
    _write_json,
)

DEFAULT_AT007_SUMMARY = (
    ROOT.parent
    / "MyGPR-Evidence"
    / "autotune"
    / "AT-007_post_zero_time_policy_rerun"
    / "manifests"
    / "post_fix_signal_loss_summary.json"
)

DEFAULT_AT005A_SUMMARY = (
    ROOT.parent
    / "MyGPR-Evidence"
    / "autotune"
    / "AT-005A_no_zerotime_gain_validation"
    / "manifests"
    / "validation_summary.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="AT-008A evidence output directory")
    parser.add_argument("--dataset", default=str(DEFAULT_GX003_DATASET), help="GX-003 evidence dir or dataset dir")
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--historical-at007", default=str(DEFAULT_AT007_SUMMARY))
    parser.add_argument("--historical-at005a", default=str(DEFAULT_AT005A_SUMMARY))
    parser.add_argument("--include-dewow-side-lanes", action="store_true")
    args = parser.parse_args(argv)

    result = run_validation(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
        historical_at007=Path(args.historical_at007),
        historical_at005a=Path(args.historical_at005a),
        include_dewow_side_lanes=args.include_dewow_side_lanes,
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
    historical_at007: Path = DEFAULT_AT007_SUMMARY,
    historical_at005a: Path = DEFAULT_AT005A_SUMMARY,
    include_dewow_side_lanes: bool = True,
) -> dict[str, Any]:
    """Run AT-008A no-dewow post-fix validation and write evidence artifacts."""
    source_commit = source_commit or _git_rev_parse(ROOT)
    package = _load_dataset(dataset)
    evidence_root.mkdir(parents=True, exist_ok=True)
    figures_dir = evidence_root / "figures"
    tables_dir = evidence_root / "tables"
    reports_dir = evidence_root / "reports"
    manifests_dir = evidence_root / "manifests"
    for directory in (figures_dir, tables_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(package["data"], dtype=np.float64)
    header_info = dict(package["header_info"])
    trace_metadata = dict(package["trace_metadata"])
    ground_truth = package.get("ground_truth") or {}
    gx003_roi = at005a._target_roi(ground_truth, raw.shape)
    color_limit = at005a._global_color_limit(raw)
    at005a._save_overlay(
        raw,
        figures_dir / "input_bscan_roi_overlay.png",
        roi=gx003_roi,
        color_limit=color_limit,
        title="GX-003 input with ground-truth ROI",
    )

    lane_specs = _lane_specs(include_dewow_side_lanes=include_dewow_side_lanes)
    lane_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for spec in lane_specs:
        lane = _run_lane(
            raw=raw,
            header_info=header_info,
            trace_metadata=trace_metadata,
            roi=gx003_roi,
            figures_dir=figures_dir,
            color_limit=color_limit,
            **spec,
        )
        lane_rows.append(lane["row"])
        trial_rows.extend(lane["trials"])

    gain_table = _build_gain_variant_table(lane_rows)
    before_after_rows = _build_before_after_rows(
        lane_rows=lane_rows,
        at007_summary_path=historical_at007,
        at005a_summary_path=historical_at005a,
    )

    _save_gain_variant_summary(lane_rows, figures_dir / "gain_variant_summary.png")
    _save_lane_summary_plot(lane_rows, figures_dir / "no_dewow_lane_summary.png")
    _save_aliases(figures_dir, lane_rows)
    _save_before_after_plot(before_after_rows, figures_dir / "before_after_dewow_policy.png")

    summary = {
        "artifact_id": "AT-008A",
        "task_id": "AT-008A_no_dewow_post_fix_validation",
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
        "gx003_roi": gx003_roi.as_dict() if gx003_roi else None,
        "zero_time_policy": "excluded",
        "dewow_policy": "excluded_primary_optional_diagnostic_side_lanes",
        "lane_rows": lane_rows,
        "gain_variant_table": gain_table,
        "before_after_dewow_comparison": before_after_rows,
        "best_gain_variant": _best_gain_variant(lane_rows),
        "autotune_status": _autotune_status(lane_rows),
        "primary_branch_validity": _primary_validity(lane_rows),
        "historical_references": {
            "AT-007": str(historical_at007),
            "AT-005A": str(historical_at005a),
        },
        "claim_boundary": {
            "ground_truth": "GX-003 metrics are ground-truth-backed.",
            "heuristic_qc": "Display-oriented metrics are diagnostic only.",
            "agc": "AGC is non-amplitude-preserving and display-oriented.",
            "autotune": "No overall AutoTune superiority claim in this artifact.",
        },
        "known_risks": [
            "Single native gprMax scene remains insufficient for broad generalization.",
            "Dewow is excluded in the primary lane for this artifact; this is not a global dewow deprecation.",
            "AGC visual clarity cannot be treated as physical amplitude correctness.",
        ],
    }

    manifest = {
        "artifact_id": "AT-008A",
        "task_id": "AT-008A_no_dewow_post_fix_validation",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending_self_reference",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_no_dewow_post_fix_validation.py",
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": True,
        "metric_type": "ground_truth_plus_heuristic_qc_with_no_dewow_primary_lane",
        "zero_time_policy": summary["zero_time_policy"],
        "dewow_policy": summary["dewow_policy"],
        "artifacts": {
            "markdown_report": "reports/no_dewow_post_fix_validation_report.md",
            "html_report": "reports/no_dewow_post_fix_validation_report.html",
            "validation_summary": "manifests/no_dewow_validation_summary.json",
            "lane_metrics": "tables/lane_metrics.csv",
            "gain_variant_table": "tables/gain_variant_table.csv",
            "before_after": "tables/before_after_dewow_comparison.csv",
        },
        "limitations": summary["known_risks"],
    }

    _write_json(manifests_dir / "no_dewow_validation_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)
    _write_csv(tables_dir / "lane_metrics.csv", lane_rows)
    _write_csv(tables_dir / "gain_variant_table.csv", gain_table)
    _write_csv(tables_dir / "before_after_dewow_comparison.csv", before_after_rows)
    _write_csv(tables_dir / "trial_table.csv", trial_rows)

    md = _render_markdown_report(summary)
    html_body = _render_html_report(summary)
    (reports_dir / "no_dewow_post_fix_validation_report.md").write_text(md, encoding="utf-8")
    (reports_dir / "no_dewow_post_fix_validation_report.html").write_text(html_body, encoding="utf-8")
    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "zero_time_policy": summary["zero_time_policy"],
        "dewow_policy": summary["dewow_policy"],
        "best_gain_variant": summary["best_gain_variant"],
        "autotune_status": summary["autotune_status"],
        "primary_branch_validity": summary["primary_branch_validity"],
        "html_report": str((reports_dir / "no_dewow_post_fix_validation_report.html").resolve()),
    }


def _lane_specs(*, include_dewow_side_lanes: bool) -> list[dict[str, Any]]:
    bg = [("subtracting_average_2D", {"ntraces": 41})]
    specs = [
        {
            "lane_id": "lane_0_raw_input",
            "branch": "raw",
            "description": "Raw input only.",
            "pre_gain_steps": [],
            "gain_step": None,
            "auto_tune": False,
            "dewow_policy": "excluded_primary",
        },
        {
            "lane_id": "lane_1_background_only",
            "branch": "manual",
            "description": "Primary lane: background suppression only.",
            "pre_gain_steps": bg,
            "gain_step": None,
            "auto_tune": False,
            "dewow_policy": "excluded_primary",
        },
        {
            "lane_id": "lane_2_background_energy_decay_gain",
            "branch": "manual",
            "description": "Primary lane: background suppression then energy-decay gain.",
            "pre_gain_steps": bg,
            "gain_step": ("energy_decay_gain", {"strength": 0.8, "smoothing_samples": 31, "max_gain": 6.0}),
            "auto_tune": False,
            "dewow_policy": "excluded_primary",
        },
        {
            "lane_id": "lane_3_background_sec_gain",
            "branch": "manual",
            "description": "Primary lane: background suppression then SEC gain.",
            "pre_gain_steps": bg,
            "gain_step": ("sec_gain", {"gain_min": 1.0, "gain_max": 4.5, "power": 1.1}),
            "auto_tune": False,
            "dewow_policy": "excluded_primary",
        },
        {
            "lane_id": "lane_4_background_time_power_gain",
            "branch": "manual",
            "description": "Primary lane: background suppression then validation-local time-power gain.",
            "pre_gain_steps": bg,
            "gain_step": ("time_power_gain_local", {"power": 1.35, "max_gain": 5.0}),
            "auto_tune": False,
            "dewow_policy": "excluded_primary",
        },
        {
            "lane_id": "lane_5_background_agc_gain",
            "branch": "manual",
            "description": "Primary lane: background suppression then AGC display gain.",
            "pre_gain_steps": bg,
            "gain_step": ("agcGain", {"window": 121, "_low_energy_guard": True}),
            "auto_tune": False,
            "dewow_policy": "excluded_primary",
        },
        {
            "lane_id": "lane_auto_background_energy_decay",
            "branch": "auto_tuned",
            "description": "Primary lane AutoTune: tune background suppression and energy-decay gain.",
            "pre_gain_steps": bg,
            "gain_step": ("energy_decay_gain", {"strength": 0.8, "smoothing_samples": 31, "max_gain": 6.0}),
            "auto_tune": True,
            "dewow_policy": "excluded_primary",
        },
    ]
    if include_dewow_side_lanes:
        specs.extend(
            [
                {
                    "lane_id": "lane_6_dewow256_background_energy_decay",
                    "branch": "diagnostic",
                    "description": "Diagnostic side lane: dewow 256 -> background -> energy-decay gain.",
                    "pre_gain_steps": [("dewow", {"window": 256}), *bg],
                    "gain_step": ("energy_decay_gain", {"strength": 0.8, "smoothing_samples": 31, "max_gain": 6.0}),
                    "auto_tune": False,
                    "dewow_policy": "fixed_diagnostic_window_256",
                },
                {
                    "lane_id": "lane_7_dewow512_background_energy_decay",
                    "branch": "diagnostic",
                    "description": "Diagnostic side lane: dewow 512 -> background -> energy-decay gain.",
                    "pre_gain_steps": [("dewow", {"window": 512}), *bg],
                    "gain_step": ("energy_decay_gain", {"strength": 0.8, "smoothing_samples": 31, "max_gain": 6.0}),
                    "auto_tune": False,
                    "dewow_policy": "fixed_diagnostic_window_512",
                },
            ]
        )
    return specs


def _run_lane(
    *,
    raw: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, Any],
    roi: Roi | None,
    figures_dir: Path,
    color_limit: float,
    lane_id: str,
    branch: str,
    description: str,
    pre_gain_steps: list[tuple[str, dict[str, Any]]],
    gain_step: tuple[str, dict[str, Any]] | None,
    auto_tune: bool,
    dewow_policy: str,
) -> dict[str, Any]:
    lane = at005a._run_lane(
        raw=raw,
        header_info=header_info,
        trace_metadata=trace_metadata,
        roi=roi,
        figures_dir=figures_dir,
        color_limit=color_limit,
        dataset_kind="gx003_ground_truth",
        lane_id=lane_id,
        branch=branch,
        description=description,
        pre_gain_steps=pre_gain_steps,
        gain_step=gain_step,
        auto_tune=auto_tune,
    )
    row = lane["row"]
    row["zero_time_policy"] = "excluded"
    row["dewow_policy"] = dewow_policy
    row["primary_lane"] = branch in {"raw", "manual", "auto_tuned"} and "dewow" not in row["pipeline"]
    row["claim_boundary"] = "ground_truth_metrics_plus_heuristic_qc"
    if row["gain_method"] == "agcGain":
        row["amplitude_preserving"] = False
    return lane


def _build_gain_variant_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        if row.get("gain_method") == "none":
            continue
        table.append(
            {
                "lane_id": row["lane_id"],
                "primary_lane": bool(row.get("primary_lane")),
                "dewow_policy": row.get("dewow_policy"),
                "gain_method": row["gain_method"],
                "gain_semantics": row["gain_semantics"],
                "roi_contrast": row["after_gain_metrics"].get("roi_to_local_background_contrast"),
                "global_energy_ratio": row["after_gain_metrics"].get("global_energy_ratio_to_input"),
                "clipping_ratio": row["after_gain_metrics"].get("clipping_ratio"),
                "deep_zone_visibility_proxy": row["after_gain_metrics"].get("deep_zone_visibility_proxy"),
                "amplitude_preservation": row["amplitude_preservation"],
            }
        )
    return table


def _build_before_after_rows(
    *,
    lane_rows: list[dict[str, Any]],
    at007_summary_path: Path,
    at005a_summary_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    at007 = _safe_read_json(at007_summary_path)
    at005a = _safe_read_json(at005a_summary_path)

    at007_first = ((at007.get("post_fix") or {}).get("first_failing_step") or {})
    rows.append(
        {
            "comparison": "AT-007_first_failing_step",
            "before": at007_first.get("method_key", "unknown"),
            "after": "no_dewow_primary_lane",
            "interpretation": "AT-007 first failing step was dewow; AT-008A primary lane excludes dewow.",
        }
    )

    at005a_best = at005a.get("best_visual_gain_variant", "unknown")
    now_best = _best_gain_variant(lane_rows)
    rows.append(
        {
            "comparison": "Best visual gain variant",
            "before": at005a_best,
            "after": now_best,
            "interpretation": "Cross-check against AT-005A no-zero-time benchmark.",
        }
    )

    primary_valid = _primary_validity(lane_rows)
    rows.append(
        {
            "comparison": "Primary lane validity (AT-008A)",
            "before": "AT-007 validity remained 0->0",
            "after": primary_valid,
            "interpretation": "Checks whether no-dewow primary lanes avoid AT-007 failure mode.",
        }
    )
    return rows


def _best_gain_variant(rows: list[dict[str, Any]]) -> str:
    candidates = [r for r in rows if r.get("primary_lane") and r.get("gain_method") not in {"none", None}]
    if not candidates:
        return "none"
    best = max(
        candidates,
        key=lambda row: float(row["after_gain_metrics"].get("roi_to_local_background_contrast") or 0.0)
        * (1.0 - min(float(row["after_gain_metrics"].get("clipping_ratio") or 0.0), 0.5)),
    )
    return str(best["gain_method"])


def _autotune_status(rows: list[dict[str, Any]]) -> str:
    manual = next((r for r in rows if r["lane_id"] == "lane_2_background_energy_decay_gain"), None)
    auto = next((r for r in rows if r["lane_id"] == "lane_auto_background_energy_decay"), None)
    if not manual or not auto:
        return "not_run"
    m = float(manual["after_gain_metrics"].get("roi_to_local_background_contrast") or 0.0)
    a = float(auto["after_gain_metrics"].get("roi_to_local_background_contrast") or 0.0)
    if a > m * 1.1:
        return "improved_on_limited_contrast_metric_only"
    if a < m * 0.9:
        return "worsened_on_limited_contrast_metric"
    return "inconclusive_near_tie"


def _primary_validity(rows: list[dict[str, Any]]) -> str:
    primaries = [r for r in rows if r.get("primary_lane")]
    if not primaries:
        return "not_available"
    invalid = [r for r in primaries if r.get("branch_validity") != "valid_with_caveats"]
    if not invalid:
        return "all_valid_with_caveats"
    return f"{len(primaries) - len(invalid)}/{len(primaries)}_valid_with_caveats"


def _save_gain_variant_summary(rows: list[dict[str, Any]], path: Path) -> None:
    gain_rows = [r for r in rows if r.get("primary_lane") and r.get("gain_method") not in {"none", None}]
    labels = [r["gain_method"] for r in gain_rows]
    contrasts = [float(r["after_gain_metrics"].get("roi_to_local_background_contrast") or 0.0) for r in gain_rows]
    clipping = [float(r["after_gain_metrics"].get("clipping_ratio") or 0.0) for r in gain_rows]
    fig, ax1 = plt.subplots(figsize=(8.8, 4.2), dpi=150)
    try:
        x = np.arange(len(labels))
        ax1.bar(x - 0.16, contrasts, width=0.32, color="#1f6feb", label="ROI/local contrast")
        ax1.set_ylabel("ROI contrast")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(x + 0.16, clipping, "o-", color="#d04e4e", label="clipping ratio")
        ax2.set_ylabel("Clipping")
        ax1.set_title("AT-008A Primary Gain Variant Summary")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_lane_summary_plot(rows: list[dict[str, Any]], path: Path) -> None:
    primaries = [r for r in rows if r.get("primary_lane")]
    labels = [r["lane_id"] for r in primaries]
    values = [float(r["after_gain_metrics"].get("roi_to_local_background_contrast") or 0.0) for r in primaries]
    fig, ax = plt.subplots(figsize=(10.0, 4.0), dpi=150)
    try:
        x = np.arange(len(labels))
        ax.bar(x, values, color="#0f766e")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("ROI/local background contrast")
        ax.set_title("AT-008A No-Dewow Primary Lane Summary")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_before_after_plot(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=150)
    try:
        ax.axis("off")
        text = "\n".join(
            f"{idx + 1}. {row['comparison']}: {row['before']} -> {row['after']}" for idx, row in enumerate(rows)
        )
        ax.text(0.01, 0.96, text, ha="left", va="top", fontsize=10, family="monospace")
        ax.set_title("AT-008A vs historical AT-007/AT-005A")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_aliases(figures_dir: Path, lane_rows: list[dict[str, Any]]) -> None:
    alias_map = {
        "lane_1_background_only": "background_suppression_only.png",
        "lane_2_background_energy_decay_gain": "energy_decay_gain_comparison.png",
        "lane_3_background_sec_gain": "sec_gain_comparison.png",
        "lane_4_background_time_power_gain": "time_power_gain_comparison.png",
        "lane_5_background_agc_gain": "agc_gain_comparison.png",
    }
    lane_to_file = {row["lane_id"]: Path(row["figure"]).name for row in lane_rows}
    for lane_id, alias in alias_map.items():
        file_name = lane_to_file.get(lane_id)
        if not file_name:
            continue
        src = figures_dir / file_name
        if src.exists():
            shutil.copy2(src, figures_dir / alias)
    if "lane_6_dewow256_background_energy_decay" in lane_to_file or "lane_7_dewow512_background_energy_decay" in lane_to_file:
        for lane_id in ("lane_6_dewow256_background_energy_decay", "lane_7_dewow512_background_energy_decay"):
            name = lane_to_file.get(lane_id)
            if name and (figures_dir / name).exists():
                shutil.copy2(figures_dir / name, figures_dir / "dewow_side_lane_comparison.png")
                break


def _render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AT-008A No-Dewow Post-Fix Native Validation",
        "",
        "## Metadata",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Dataset: `{summary['dataset']['name']}` shape `{summary['dataset']['shape']}`",
        "- Ground truth: `available`",
        f"- Zero-time policy: `{summary['zero_time_policy']}`",
        f"- Dewow policy: `{summary['dewow_policy']}`",
        "- GX-003 ROI is used as-is.",
        "",
        "## Core conclusion",
        "- Primary lane excludes both `set_zero_time` and `dewow`.",
        "- Dewow remains optional diagnostic side-lane; it is not deleted or globally invalidated.",
        f"- Primary branch validity: `{summary['primary_branch_validity']}`",
        f"- Best gain variant (primary lanes): `{summary['best_gain_variant']}`",
        f"- AutoTune status: `{summary['autotune_status']}`",
        "",
        "## Lane summary",
        "| Lane | Branch | Dewow policy | Gain | ROI contrast | Clipping | Validity | Figure |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in summary["lane_rows"]:
        lines.append(
            f"| `{row['lane_id']}` | `{row['branch']}` | `{row['dewow_policy']}` | "
            f"`{row['gain_method']}` | {at005a._fmt(row['after_gain_metrics'].get('roi_to_local_background_contrast'))} | "
            f"{at005a._fmt(row['after_gain_metrics'].get('clipping_ratio'))} | "
            f"`{row['branch_validity']}` | `{row['figure']}` |"
        )

    lines.extend(
        [
            "",
            "## Before / after reference",
            "| Comparison | Before | After | Interpretation |",
            "|---|---|---|---|",
        ]
    )
    for row in summary["before_after_dewow_comparison"]:
        lines.append(
            f"| {row['comparison']} | {row['before']} | {row['after']} | {row['interpretation']} |"
        )
    lines.extend(["", "## Known risks"])
    lines.extend(f"- {item}" for item in summary["known_risks"])
    lines.append("")
    return "\n".join(lines)


def _render_html_report(summary: dict[str, Any]) -> str:
    lane_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['lane_id'])}</td>"
        f"<td>{html.escape(row['branch'])}</td>"
        f"<td>{html.escape(row['dewow_policy'])}</td>"
        f"<td>{html.escape(row['gain_method'])}</td>"
        f"<td>{at005a._fmt(row['after_gain_metrics'].get('roi_to_local_background_contrast'))}</td>"
        f"<td>{at005a._fmt(row['after_gain_metrics'].get('clipping_ratio'))}</td>"
        f"<td>{html.escape(row['branch_validity'])}</td>"
        f"<td><a href='../{html.escape(row['figure'])}'>figure</a></td>"
        "</tr>"
        for row in summary["lane_rows"]
    )
    before_after_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['comparison']))}</td>"
        f"<td>{html.escape(str(row['before']))}</td>"
        f"<td>{html.escape(str(row['after']))}</td>"
        f"<td>{html.escape(str(row['interpretation']))}</td>"
        "</tr>"
        for row in summary["before_after_dewow_comparison"]
    )
    risk_items = "\n".join(f"<li>{html.escape(item)}</li>" for item in summary["known_risks"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT-008A No-Dewow Post-Fix Native Validation</title>
  <style>
    body {{ margin: 0; background: #f5f7fb; color: #162033; font-family: "Segoe UI", "PingFang SC", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    .card {{ background: #fff; border: 1px solid #d9e2ee; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }}
    .warning {{ border-color: #f59e0b; background: #fff9ea; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9e2ee; }}
    th, td {{ font-size: 13px; padding: 8px 10px; border-bottom: 1px solid #e8edf5; text-align: left; }}
    th {{ background: #edf3fa; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #d9e2ee; border-radius: 8px; padding: 10px; }}
    figure img {{ width: 100%; display: block; }}
    figcaption {{ margin-top: 6px; font-size: 13px; color: #495a75; }}
    code {{ background: #edf2f7; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>AT-008A No-Dewow Post-Fix Native Validation</h1>
  <div class="card">
    <div><b>Source commit:</b> <code>{html.escape(summary["source_commit"])}</code></div>
    <div><b>Dataset:</b> {html.escape(summary["dataset"]["name"])} {html.escape(str(summary["dataset"]["shape"]))}</div>
    <div><b>Zero-time policy:</b> <code>{html.escape(summary["zero_time_policy"])}</code></div>
    <div><b>Dewow policy:</b> <code>{html.escape(summary["dewow_policy"])}</code></div>
    <div><b>Ground truth:</b> GX-003 ROI used as-is.</div>
  </div>
  <div class="card warning">
    <b>Boundary:</b> Primary AT-008A lane excludes both zero-time and dewow. Dewow side lanes are diagnostic only. AGC is display-oriented and non-amplitude-preserving.
  </div>
  <div class="card">
    <div><b>Primary validity:</b> <code>{html.escape(summary["primary_branch_validity"])}</code></div>
    <div><b>Best gain variant:</b> <code>{html.escape(summary["best_gain_variant"])}</code></div>
    <div><b>AutoTune status:</b> <code>{html.escape(summary["autotune_status"])}</code></div>
  </div>

  <h2>Key figures</h2>
  <div class="grid">
    <figure><img src="../figures/input_bscan_roi_overlay.png"><figcaption>Input with ROI overlay.</figcaption></figure>
    <figure><img src="../figures/no_dewow_lane_summary.png"><figcaption>No-dewow primary lane summary.</figcaption></figure>
    <figure><img src="../figures/energy_decay_gain_comparison.png"><figcaption>Energy-decay gain lane.</figcaption></figure>
    <figure><img src="../figures/agc_gain_comparison.png"><figcaption>AGC lane (display-oriented).</figcaption></figure>
    <figure><img src="../figures/gain_variant_summary.png"><figcaption>Gain variant metrics summary.</figcaption></figure>
    <figure><img src="../figures/before_after_dewow_policy.png"><figcaption>AT-008A vs AT-007/AT-005A summary.</figcaption></figure>
  </div>

  <h2>Lane summary table</h2>
  <table><thead><tr><th>Lane</th><th>Branch</th><th>Dewow policy</th><th>Gain</th><th>ROI contrast</th><th>Clipping</th><th>Validity</th><th>Figure</th></tr></thead><tbody>{lane_rows}</tbody></table>

  <h2>Before / after comparison</h2>
  <table><thead><tr><th>Comparison</th><th>Before</th><th>After</th><th>Interpretation</th></tr></thead><tbody>{before_after_rows}</tbody></table>

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

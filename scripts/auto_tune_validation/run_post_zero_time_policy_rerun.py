#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run AT-007 post zero-time policy rerun (ablation + diagnosis + before/after)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auto_tune_validation.run_native_ablation import run_ablation
from scripts.auto_tune_validation.run_signal_loss_diagnosis import run_diagnosis
from scripts.auto_tune_validation.run_stepwise_validation import _git_rev_parse


DEFAULT_GX003_DATASET = (
    ROOT.parent / "MyGPR-Evidence" / "gprmax" / "GX-003_audited_native_gprmax_benchmark"
)
DEFAULT_AT002_SUMMARY = (
    ROOT.parent
    / "MyGPR-Evidence"
    / "autotune"
    / "AT-002_native_ablation"
    / "manifests"
    / "ablation_summary.json"
)
DEFAULT_AT003_SUMMARY = (
    ROOT.parent
    / "MyGPR-Evidence"
    / "autotune"
    / "AT-003_signal_loss_diagnosis"
    / "manifests"
    / "step_diagnostics.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="AT-007 evidence output directory")
    parser.add_argument("--dataset", default=str(DEFAULT_GX003_DATASET), help="GX-003 evidence dir or dataset dir")
    parser.add_argument("--mode", choices=["smoke", "normal"], default="normal")
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--historical-at002-summary", default=str(DEFAULT_AT002_SUMMARY))
    parser.add_argument("--historical-at003-summary", default=str(DEFAULT_AT003_SUMMARY))
    args = parser.parse_args(argv)

    result = run_post_rerun(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        mode=args.mode,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
        historical_at002_summary=Path(args.historical_at002_summary),
        historical_at003_summary=Path(args.historical_at003_summary),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_post_rerun(
    *,
    evidence_root: Path,
    dataset: str,
    mode: str,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
    historical_at002_summary: Path,
    historical_at003_summary: Path,
) -> dict[str, Any]:
    source_commit = source_commit or _git_rev_parse(ROOT)
    evidence_root.mkdir(parents=True, exist_ok=True)
    figures_dir = evidence_root / "figures"
    reports_dir = evidence_root / "reports"
    manifests_dir = evidence_root / "manifests"
    tables_dir = evidence_root / "tables"
    post_fix_ablation_dir = evidence_root / "post_fix_ablation"
    post_fix_diagnosis_dir = evidence_root / "post_fix_signal_loss"
    for directory in (
        figures_dir,
        reports_dir,
        manifests_dir,
        tables_dir,
        post_fix_ablation_dir,
        post_fix_diagnosis_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    ablation = run_ablation(
        evidence_root=post_fix_ablation_dir,
        dataset=dataset,
        mode=mode,
        source_repo=source_repo,
        source_branch=source_branch,
        source_commit=source_commit,
    )
    diagnosis = run_diagnosis(
        evidence_root=post_fix_diagnosis_dir,
        dataset=dataset,
        mode=mode,
        source_repo=source_repo,
        source_branch=source_branch,
        source_commit=source_commit,
    )

    post_ablation_summary = _read_json(post_fix_ablation_dir / "manifests" / "ablation_summary.json")
    post_diagnosis_summary = _read_json(post_fix_diagnosis_dir / "manifests" / "step_diagnostics.json")
    historical_ablation_summary = _read_json(historical_at002_summary)
    historical_diagnosis_summary = _read_json(historical_at003_summary)

    post_step_rows = list(post_diagnosis_summary.get("diagnostics") or [])
    _write_csv(tables_dir / "post_fix_step_diagnostics.csv", post_step_rows)
    _write_csv(tables_dir / "post_fix_ablation_table.csv", list(post_ablation_summary.get("stage_ablation_table") or []))

    validity_rows = _build_before_after_validity_rows(
        historical_ablation_summary=historical_ablation_summary,
        post_ablation_summary=post_ablation_summary,
    )
    _write_csv(tables_dir / "before_after_validity_comparison.csv", validity_rows)

    metric_rows = _build_before_after_metric_rows(
        historical_diagnosis_summary=historical_diagnosis_summary,
        post_diagnosis_summary=post_diagnosis_summary,
    )
    _write_csv(tables_dir / "before_after_metric_comparison.csv", metric_rows)

    input_overlay = post_fix_diagnosis_dir / "figures" / "input_bscan_roi_overlay.png"
    if input_overlay.exists():
        shutil.copy2(input_overlay, figures_dir / "input_bscan_roi_overlay.png")
    _copy_if_exists(
        post_fix_ablation_dir / "figures" / "manual_vs_auto_side_by_side.png",
        figures_dir / "post_fix_manual_vs_auto_summary.png",
    )
    _copy_if_exists(
        post_fix_diagnosis_dir / "figures" / "auto_tuned_stepwise_energy_curve.png",
        figures_dir / "post_fix_stepwise_energy_curve.png",
    )
    _copy_step_images(post_fix_diagnosis_dir / "figures", figures_dir / "steps")

    _save_zero_time_policy_comparison_figure(
        historical_diagnosis_summary=historical_diagnosis_summary,
        post_diagnosis_summary=post_diagnosis_summary,
        output_path=figures_dir / "before_after_zero_time_policy.png",
    )
    _save_branch_validity_figure(
        validity_rows=validity_rows,
        output_path=figures_dir / "before_after_branch_validity.png",
    )

    post_fix_signal_loss_summary = {
        "artifact_id": "AT-007",
        "task_id": "AT-007_post_zero_time_policy_rerun",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "mode": mode,
        "zero_time_policy": post_diagnosis_summary.get("zero_time_policy"),
        "dataset": post_diagnosis_summary.get("dataset"),
        "historical_pre_fix": {
            "at002_summary": str(historical_at002_summary),
            "at003_summary": str(historical_at003_summary),
            "first_failing_step": historical_diagnosis_summary.get("first_failing_step"),
        },
        "post_fix": {
            "first_failing_step": post_diagnosis_summary.get("first_failing_step"),
            "branch_first_failures": post_diagnosis_summary.get("branch_first_failures"),
            "likely_root_cause": post_diagnosis_summary.get("likely_root_cause"),
        },
        "zero_time_shift_eliminated": _is_zero_time_shift_eliminated(post_step_rows),
    }
    _write_json(manifests_dir / "post_fix_signal_loss_summary.json", post_fix_signal_loss_summary)
    _write_json(manifests_dir / "post_fix_ablation_summary.json", post_ablation_summary)

    conclusion = _build_conclusion(
        historical_ablation_summary=historical_ablation_summary,
        post_ablation_summary=post_ablation_summary,
        historical_diagnosis_summary=historical_diagnosis_summary,
        post_diagnosis_summary=post_diagnosis_summary,
    )
    summary = {
        "artifact_id": "AT-007",
        "task_id": "AT-007_post_zero_time_policy_rerun",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": post_ablation_summary.get("dataset"),
        "ground_truth_available": bool(post_ablation_summary.get("ground_truth_available")),
        "roi_policy": "GX-003 ROI used as-is",
        "zero_time_policy": post_ablation_summary.get("zero_time_policy"),
        "historical_baseline": {
            "at002_source_commit": historical_ablation_summary.get("source_commit"),
            "at003_source_commit": historical_diagnosis_summary.get("source_commit"),
            "at002_conclusion": "inconclusive/all-invalid historical pre-fix",
        },
        "post_fix_ablation": {
            "source_commit": post_ablation_summary.get("source_commit"),
            "stage_winners": post_ablation_summary.get("stage_winners"),
        },
        "post_fix_signal_loss": {
            "source_commit": post_diagnosis_summary.get("source_commit"),
            "first_failing_step": post_diagnosis_summary.get("first_failing_step"),
            "likely_root_cause": post_diagnosis_summary.get("likely_root_cause"),
        },
        "before_after": conclusion,
        "claim_boundary": {
            "ground_truth_claims": "GX-003 metrics are ground-truth-based.",
            "heuristic_claims": "Heuristic QC metrics are diagnostic only and not physical truth proof.",
            "not_allowed": "No overall AutoTune superiority claim unless broad multi-scene evidence supports it.",
        },
        "next_recommended_task": conclusion.get("next_recommended_task"),
        "known_risks": conclusion.get("known_risks"),
    }
    _write_json(manifests_dir / "evidence_manifest.json", _build_evidence_manifest(summary))

    md = _render_markdown_report(summary)
    html = _render_html_report(summary)
    (reports_dir / "post_zero_time_policy_rerun_report.md").write_text(md, encoding="utf-8")
    (reports_dir / "post_zero_time_policy_rerun_report.html").write_text(html, encoding="utf-8")

    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "zero_time_policy": summary["zero_time_policy"],
        "zero_time_shift_eliminated": summary["before_after"]["implicit_423_shift_eliminated"],
        "first_failing_step_after_fix": summary["before_after"]["first_failing_step_after_fix"],
        "autotune_status": summary["before_after"]["autotune_status"],
        "report_markdown": str((reports_dir / "post_zero_time_policy_rerun_report.md").resolve()),
        "report_html": str((reports_dir / "post_zero_time_policy_rerun_report.html").resolve()),
        "ablation_report": ablation.get("report"),
        "diagnosis_report": diagnosis.get("report"),
    }


def _build_before_after_validity_rows(
    *,
    historical_ablation_summary: dict[str, Any],
    post_ablation_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    hist_rows = {str(row.get("branch")): row for row in historical_ablation_summary.get("stage_ablation_table", [])}
    post_rows = {str(row.get("branch")): row for row in post_ablation_summary.get("stage_ablation_table", [])}
    branches = sorted(set(hist_rows) | set(post_rows))
    rows: list[dict[str, Any]] = []
    for branch in branches:
        hist = hist_rows.get(branch, {})
        post = post_rows.get(branch, {})
        rows.append(
            {
                "branch": branch,
                "valid_before": bool(hist.get("valid")),
                "valid_after": bool(post.get("valid")),
                "branch_invalid_reason_before": hist.get("branch_invalid_reason", ""),
                "branch_invalid_reason_after": post.get("branch_invalid_reason", ""),
                "truth_score_before": hist.get("truth_score"),
                "truth_score_after": post.get("truth_score"),
                "delta_truth_score_after_minus_before": _delta(post.get("truth_score"), hist.get("truth_score")),
            }
        )
    return rows


def _build_before_after_metric_rows(
    *,
    historical_diagnosis_summary: dict[str, Any],
    post_diagnosis_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    hist_first = historical_diagnosis_summary.get("first_failing_step") or {}
    post_first = post_diagnosis_summary.get("first_failing_step") or {}
    rows = [
        {
            "metric": "first_failing_step_method",
            "before": hist_first.get("method_key"),
            "after": post_first.get("method_key"),
        },
        {
            "metric": "first_failing_step_branch",
            "before": hist_first.get("branch"),
            "after": post_first.get("branch"),
        },
        {
            "metric": "first_failing_step_global_energy_ratio",
            "before": hist_first.get("global_energy_ratio"),
            "after": post_first.get("global_energy_ratio"),
            "delta_after_minus_before": _delta(post_first.get("global_energy_ratio"), hist_first.get("global_energy_ratio")),
        },
    ]
    return rows


def _is_zero_time_shift_eliminated(rows: list[dict[str, Any]]) -> bool:
    zero_rows = [row for row in rows if str(row.get("method_key")) == "set_zero_time"]
    if not zero_rows:
        return True
    for row in zero_rows:
        if int(row.get("zero_time_shift_samples") or 0) >= 423:
            return False
    return True


def _build_conclusion(
    *,
    historical_ablation_summary: dict[str, Any],
    post_ablation_summary: dict[str, Any],
    historical_diagnosis_summary: dict[str, Any],
    post_diagnosis_summary: dict[str, Any],
) -> dict[str, Any]:
    hist_first = historical_diagnosis_summary.get("first_failing_step") or {}
    post_first = post_diagnosis_summary.get("first_failing_step") or {}
    hist_valid = sum(1 for row in historical_ablation_summary.get("stage_ablation_table", []) if row.get("valid"))
    post_valid = sum(1 for row in post_ablation_summary.get("stage_ablation_table", []) if row.get("valid"))
    improved = post_valid > hist_valid
    zero_eliminated = not (
        str(hist_first.get("method_key")) == "set_zero_time"
        and int(hist_first.get("global_energy_ratio") or 0.0) < 0.01
        and str(post_first.get("method_key")) == "set_zero_time"
    )
    if post_first.get("method_key") == "dewow":
        bottleneck = "dewow parameter-domain refinement"
        next_task = "dewow_parameter_domain_refinement"
    elif post_first.get("method_key") == "subtracting_average_2D":
        bottleneck = "background suppression tuning"
        next_task = "background_suppression_tuning"
    else:
        bottleneck = "candidate-space / scoring refinement"
        next_task = "autotune_candidate_space_or_scoring_refinement"

    if improved:
        autotune_status = "improved_but_not_overall_superiority"
    else:
        autotune_status = "remains_inconclusive"
    return {
        "implicit_423_shift_eliminated": bool(zero_eliminated),
        "branch_validity_before_count": int(hist_valid),
        "branch_validity_after_count": int(post_valid),
        "branch_validity_improved": bool(improved),
        "first_failing_step_before_fix": hist_first,
        "first_failing_step_after_fix": post_first,
        "next_bottleneck": bottleneck,
        "autotune_status": autotune_status,
        "next_recommended_task": next_task,
        "known_risks": [
            "Single native gprMax scene is insufficient for broad claim changes.",
            "Ground-truth metrics and heuristic QC can disagree under aggressive denoising.",
            "Historical AT-002/AT-003 remain valid pre-fix records and must not be reinterpreted as post-fix.",
        ],
    }


def _save_zero_time_policy_comparison_figure(
    *,
    historical_diagnosis_summary: dict[str, Any],
    post_diagnosis_summary: dict[str, Any],
    output_path: Path,
) -> None:
    hist_first = historical_diagnosis_summary.get("first_failing_step") or {}
    post_first = post_diagnosis_summary.get("first_failing_step") or {}
    hist_val = float(hist_first.get("global_energy_ratio") or 0.0)
    post_val = float(post_first.get("global_energy_ratio") or 0.0)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    try:
        labels = ["AT-003 pre-fix", "AT-007 post-fix"]
        vals = [hist_val, post_val]
        ax.bar(labels, vals, color=["#b45309", "#0f766e"])
        ax.set_title("First-failing-step global energy ratio")
        ax.set_ylabel("global_energy_ratio")
        for idx, val in enumerate(vals):
            ax.text(idx, val, f"{val:.6f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path)
    finally:
        plt.close(fig)


def _save_branch_validity_figure(*, validity_rows: list[dict[str, Any]], output_path: Path) -> None:
    branches = [str(row.get("branch")) for row in validity_rows]
    before = [1.0 if row.get("valid_before") else 0.0 for row in validity_rows]
    after = [1.0 if row.get("valid_after") else 0.0 for row in validity_rows]
    x = np.arange(len(branches), dtype=float)
    fig, ax = plt.subplots(figsize=(max(8.0, len(branches) * 1.1), 4.2), dpi=150)
    try:
        ax.bar(x - 0.18, before, width=0.36, label="before", color="#9ca3af")
        ax.bar(x + 0.18, after, width=0.36, label="after", color="#0ea5e9")
        ax.set_ylim(0.0, 1.15)
        ax.set_yticks([0.0, 1.0], labels=["invalid", "valid"])
        ax.set_xticks(x, labels=branches, rotation=25, ha="right")
        ax.set_title("Branch validity before vs after AT-006 policy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path)
    finally:
        plt.close(fig)


def _copy_step_images(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.glob("*_roi_overlay.png"):
        shutil.copy2(path, dst_dir / path.name)
    for path in src_dir.glob("*_roi_crop.png"):
        shutil.copy2(path, dst_dir / path.name)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _build_evidence_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    dataset = summary.get("dataset") or {}
    return {
        "artifact_id": "AT-007",
        "task_id": "AT-007_post_zero_time_policy_rerun",
        "source_repo": summary.get("source_repo"),
        "source_branch": summary.get("source_branch"),
        "source_commit": summary.get("source_commit"),
        "evidence_repo": summary.get("evidence_repo"),
        "evidence_commit": "pending_self_reference",
        "generated_at": summary.get("generated_at"),
        "dataset_name": dataset.get("name"),
        "dataset_shape": dataset.get("shape"),
        "dataset_hash": dataset.get("hash"),
        "ground_truth_available": bool(summary.get("ground_truth_available")),
        "metric_type": "ground_truth_with_heuristic_diagnostics",
        "zero_time_policy": summary.get("zero_time_policy"),
        "artifacts": {
            "markdown_report": "reports/post_zero_time_policy_rerun_report.md",
            "html_report": "reports/post_zero_time_policy_rerun_report.html",
            "post_fix_ablation_summary": "manifests/post_fix_ablation_summary.json",
            "post_fix_signal_loss_summary": "manifests/post_fix_signal_loss_summary.json",
            "post_fix_ablation_table": "tables/post_fix_ablation_table.csv",
            "post_fix_step_diagnostics": "tables/post_fix_step_diagnostics.csv",
            "before_after_validity": "tables/before_after_validity_comparison.csv",
            "before_after_metric": "tables/before_after_metric_comparison.csv",
        },
        "limitations": [
            "Post-fix rerun evaluates policy impact; it does not redesign AutoTune scoring.",
            "Historical AT-002/AT-003 are preserved as pre-fix references.",
            "No field-data ground-truth claim is made.",
        ],
        "known_risks": summary.get("before_after", {}).get("known_risks", []),
    }


def _render_markdown_report(summary: dict[str, Any]) -> str:
    before_after = summary.get("before_after", {})
    lines = [
        "# AT-007 Post Zero-Time Policy Rerun",
        "",
        f"- Source commit: `{summary.get('source_commit')}`",
        f"- Dataset: `{(summary.get('dataset') or {}).get('name')}`",
        f"- Ground truth available: `{summary.get('ground_truth_available')}`",
        f"- Zero-time policy: `{summary.get('zero_time_policy')}`",
        "- GX-003 ROI is used as-is.",
        "- Historical AT-002/AT-003 are preserved as pre-fix baselines.",
        "",
        "## Key Findings",
        f"- Implicit 423-sample shift eliminated: `{before_after.get('implicit_423_shift_eliminated')}`",
        f"- Branch validity before -> after: `{before_after.get('branch_validity_before_count')}` -> `{before_after.get('branch_validity_after_count')}`",
        f"- First failing step before: `{(before_after.get('first_failing_step_before_fix') or {}).get('method_key')}`",
        f"- First failing step after: `{(before_after.get('first_failing_step_after_fix') or {}).get('method_key')}`",
        f"- AutoTune status: `{before_after.get('autotune_status')}`",
        f"- Next bottleneck: `{before_after.get('next_bottleneck')}`",
        "",
        "## Claim Boundary",
        "- This rerun does not claim overall AutoTune superiority.",
        "- Ground-truth metrics are used on GX-003; heuristic metrics remain diagnostic.",
        "",
        "## Next Recommended Task",
        f"- `{summary.get('next_recommended_task')}`",
        "",
        "## Evidence",
        "- `tables/before_after_validity_comparison.csv`",
        "- `tables/before_after_metric_comparison.csv`",
        "- `figures/before_after_zero_time_policy.png`",
        "- `figures/before_after_branch_validity.png`",
    ]
    return "\n".join(lines) + "\n"


def _render_html_report(summary: dict[str, Any]) -> str:
    before_after = summary.get("before_after", {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AT-007 Post Zero-Time Policy Rerun</title>
  <style>
    body {{ font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-top: 1.2em; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px 14px; background: #f9fafb; margin-top: 8px; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
    ul {{ line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>AT-007 Post Zero-Time Policy Rerun</h1>
  <div class="card">
    <div><b>Source commit:</b> <code>{summary.get("source_commit")}</code></div>
    <div><b>Dataset:</b> <code>{(summary.get("dataset") or {}).get("name")}</code></div>
    <div><b>Ground truth:</b> <code>{summary.get("ground_truth_available")}</code></div>
    <div><b>Zero-time policy:</b> <code>{summary.get("zero_time_policy")}</code></div>
  </div>
  <h2>Key Findings</h2>
  <ul>
    <li>Implicit 423-sample shift eliminated: <code>{before_after.get("implicit_423_shift_eliminated")}</code></li>
    <li>Branch validity before - after: <code>{before_after.get("branch_validity_before_count")}</code> -> <code>{before_after.get("branch_validity_after_count")}</code></li>
    <li>First failing step before: <code>{(before_after.get("first_failing_step_before_fix") or {}).get("method_key")}</code></li>
    <li>First failing step after: <code>{(before_after.get("first_failing_step_after_fix") or {}).get("method_key")}</code></li>
    <li>AutoTune status: <code>{before_after.get("autotune_status")}</code></li>
    <li>Next bottleneck: <code>{before_after.get("next_bottleneck")}</code></li>
  </ul>
  <h2>Claim Boundary</h2>
  <ul>
    <li>No overall AutoTune superiority claim is made.</li>
    <li>GX-003 ground-truth metrics and heuristic diagnostics are separated.</li>
    <li>Historical AT-002/AT-003 remain pre-fix evidence.</li>
  </ul>
  <h2>Next Task</h2>
  <div class="card"><code>{summary.get("next_recommended_task")}</code></div>
</body>
</html>
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _delta(after: Any, before: Any) -> float | None:
    try:
        if after is None or before is None:
            return None
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

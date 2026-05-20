#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run AT-002 native gprMax AutoTune per-stage ablation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auto_tune_validation.run_stepwise_validation import (
    AUTO_TUNE_SEARCH_MODE,
    MANUAL_EXPERT_PARAMS,
    _build_trial_table,
    _common_heuristic_metrics,
    _git_rev_parse,
    _json_safe,
    _load_dataset,
    _metric_delta,
    _read_json_optional,
    _rel,
    _save_bscan_png,
    _save_side_by_side,
    _write_json,
    _write_trial_csv,
    _run_branch,
)

DEFAULT_GX003_DATASET = (
    ROOT.parent / "MyGPR-Evidence" / "gprmax" / "GX-003_audited_native_gprmax_benchmark"
)
DEFAULT_PIPELINE = [
    "set_zero_time",
    "dewow",
    "frequency_filter_1d",
    "subtracting_average_2D",
    "energy_decay_gain",
]
ABLATION_STAGES = {
    "dewow": "dewow",
    "frequency_filter_1d": "frequency_filter_1d",
    "background_suppression": "subtracting_average_2D",
    "gain": "energy_decay_gain",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="AT-002 evidence output directory")
    parser.add_argument("--dataset", default=str(DEFAULT_GX003_DATASET), help="GX-003 evidence dir or dataset dir")
    parser.add_argument("--mode", choices=["smoke", "normal"], default="normal")
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)

    result = run_ablation(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        mode=args.mode,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_ablation(
    *,
    evidence_root: Path,
    dataset: str,
    mode: str,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Run AT-002 per-stage ablation on a native gprMax-converted benchmark."""
    source_commit = source_commit or _git_rev_parse(ROOT)
    package = _load_dataset(dataset)
    evidence_root.mkdir(parents=True, exist_ok=True)
    figures_dir = evidence_root / "figures"
    tables_dir = evidence_root / "tables"
    reports_dir = evidence_root / "reports"
    manifests_dir = evidence_root / "manifests"
    for directory in (figures_dir, tables_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = package["data"]
    header_info = package["header_info"]
    trace_metadata = package["trace_metadata"]
    ground_truth = package.get("ground_truth")
    metric_type = "ground_truth" if ground_truth else "heuristic_qc"
    search_mode = AUTO_TUNE_SEARCH_MODE[mode]

    input_png = figures_dir / "input_bscan.png"
    _save_bscan_png(raw, input_png, title="AT-002 input B-scan")

    branches: dict[str, dict[str, Any]] = {}
    branch_specs = {
        "expert_manual": {"auto_tune": False, "manual_params": MANUAL_EXPERT_PARAMS, "tune_methods": None},
        "safe_default": {"auto_tune": False, "manual_params": {}, "tune_methods": None},
        "auto_tuned": {"auto_tune": True, "manual_params": MANUAL_EXPERT_PARAMS, "tune_methods": None},
    }
    for branch, spec in branch_specs.items():
        branches[branch] = _run_branch(
            branch=branch,
            raw=raw,
            header_info=header_info,
            trace_metadata=trace_metadata,
            ground_truth=ground_truth,
            figures_dir=figures_dir,
            auto_tune=bool(spec["auto_tune"]),
            search_mode=search_mode,
            pipeline=DEFAULT_PIPELINE,
            manual_params=dict(spec["manual_params"]),
            tune_methods=spec["tune_methods"],
        )

    ablations: dict[str, dict[str, Any]] = {}
    for stage_name, method_key in ABLATION_STAGES.items():
        branch_name = f"ablation_{stage_name}"
        ablations[stage_name] = _run_branch(
            branch=branch_name,
            raw=raw,
            header_info=header_info,
            trace_metadata=trace_metadata,
            ground_truth=ground_truth,
            figures_dir=figures_dir,
            auto_tune=True,
            search_mode=search_mode,
            pipeline=DEFAULT_PIPELINE,
            manual_params=MANUAL_EXPERT_PARAMS,
            tune_methods={method_key},
        )

    for branch, result in {**branches, **{f"ablation_{k}": v for k, v in ablations.items()}}.items():
        _save_bscan_png(result["result"], figures_dir / f"{branch}_bscan.png", title=branch)

    _save_side_by_side(
        raw,
        branches["expert_manual"]["result"],
        branches["auto_tuned"]["result"],
        figures_dir / "manual_vs_auto_side_by_side.png",
    )
    _save_side_by_side(
        branches["expert_manual"]["result"],
        branches["safe_default"]["result"],
        branches["auto_tuned"]["result"],
        figures_dir / "manual_safe_auto_summary.png",
    )

    table = _build_ablation_table(branches, ablations)
    summary = _build_summary(
        package=package,
        source_repo=source_repo,
        source_branch=source_branch,
        source_commit=source_commit,
        mode=mode,
        metric_type=metric_type,
        ground_truth=ground_truth,
        branches=branches,
        ablations=ablations,
        table=table,
    )
    trial_rows: list[dict[str, Any]] = []
    for result in [branches["auto_tuned"], *ablations.values()]:
        trial_rows.extend(_build_trial_table(result))

    evidence_manifest = {
        "artifact_id": "AT-002",
        "task_id": "AT-002_native_benchmark_autotune_ablation",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/auto_tune_validation/run_native_ablation.py",
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": bool(ground_truth),
        "metric_type": metric_type,
        "artifacts": {
            "ablation_report": "reports/ablation_report.md",
            "ablation_summary": "manifests/ablation_summary.json",
            "stage_ablation_table": "tables/stage_ablation_table.csv",
            "trial_table": "tables/trial_table.csv",
            "input_bscan": "figures/input_bscan.png",
            "manual_vs_auto": "figures/manual_vs_auto_side_by_side.png",
        },
        "limitations": summary["limitations"],
        "known_risks": summary["known_risks"],
    }

    _write_json(manifests_dir / "ablation_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", evidence_manifest)
    _write_csv(tables_dir / "stage_ablation_table.csv", table)
    _write_json(tables_dir / "stage_ablation_table.json", table)
    _write_json(tables_dir / "trial_table.json", trial_rows)
    _write_trial_csv(tables_dir / "trial_table.csv", trial_rows)
    (reports_dir / "ablation_report.md").write_text(_render_report(summary), encoding="utf-8")

    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "dataset_name": package["dataset_name"],
        "ground_truth_available": bool(ground_truth),
        "metric_type": metric_type,
        "stage_winners": summary["stage_winners"],
        "report": str((reports_dir / "ablation_report.md").resolve()),
    }


def _build_ablation_table(
    branches: dict[str, dict[str, Any]],
    ablations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = branches["expert_manual"]
    candidates = {
        "expert_manual": baseline,
        "safe_default": branches["safe_default"],
        "auto_tuned": branches["auto_tuned"],
        **{f"only_{stage}_auto_tuned": result for stage, result in ablations.items()},
    }
    for label, result in candidates.items():
        truth = result.get("ground_truth_metrics", {})
        heuristic = result.get("heuristic_metrics", {})
        rows.append(
            {
                "branch": label,
                "valid": not bool(result.get("branch_invalid_reason")),
                "branch_invalid_reason": result.get("branch_invalid_reason", ""),
                "truth_score": truth.get("truth_score"),
                "truth_target_energy_preservation": truth.get("truth_target_energy_preservation"),
                "truth_target_saliency_gain": truth.get("truth_target_saliency_gain"),
                "truth_background_energy_reduction": truth.get("truth_background_energy_reduction"),
                "truth_false_positive_ratio": truth.get("truth_false_positive_ratio"),
                "heuristic_target_band_energy_ratio": heuristic.get("target_band_energy_ratio"),
                "heuristic_edge_preservation": heuristic.get("edge_preservation"),
                "heuristic_clipping_ratio_after": heuristic.get("clipping_ratio_after"),
                "delta_truth_score_vs_manual": _delta(truth.get("truth_score"), baseline.get("ground_truth_metrics", {}).get("truth_score")),
                "selected_auto_params": json.dumps(_selected_params_summary(result), ensure_ascii=False, sort_keys=True),
                "sanity_warnings": "; ".join(result.get("sanity_warnings", [])),
            }
        )
    return rows


def _build_summary(
    *,
    package: dict[str, Any],
    source_repo: str,
    source_branch: str,
    source_commit: str,
    mode: str,
    metric_type: str,
    ground_truth: dict[str, Any] | None,
    branches: dict[str, dict[str, Any]],
    ablations: dict[str, dict[str, Any]],
    table: list[dict[str, Any]],
) -> dict[str, Any]:
    winners = _rank_stage_winners(table, metric_type)
    return {
        "artifact_id": "AT-002",
        "task_id": "AT-002_native_benchmark_autotune_ablation",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "mode": mode,
        "dataset": {
            "name": package["dataset_name"],
            "path": package["dataset_path"],
            "shape": package["dataset_shape"],
            "hash": package["dataset_hash"],
            "source_evidence": "gprmax/GX-003_audited_native_gprmax_benchmark/",
        },
        "pipeline": DEFAULT_PIPELINE,
        "ablation_stages": ABLATION_STAGES,
        "metric_type": metric_type,
        "ground_truth_available": bool(ground_truth),
        "heuristic_qc_only": not bool(ground_truth),
        "branches": {key: _branch_summary(value) for key, value in branches.items()},
        "ablations": {key: _branch_summary(value) for key, value in ablations.items()},
        "stage_ablation_table": table,
        "stage_winners": winners,
        "metric_delta_auto_minus_manual": {
            "heuristic": _metric_delta(branches["expert_manual"]["heuristic_metrics"], branches["auto_tuned"]["heuristic_metrics"]),
            "ground_truth": _metric_delta(branches["expert_manual"]["ground_truth_metrics"], branches["auto_tuned"]["ground_truth_metrics"]),
        },
        "limitations": [
            "AT-002 diagnoses stage-level behavior on one native gprMax benchmark; it is not a global AutoTune redesign.",
            "AutoTune scoring is unchanged; ground truth is used only after selection for validation and evidence.",
            "pipe_demo_longline_v1 is paper-usable for limited native gprMax claims, not field-data generalization.",
        ],
        "known_risks": [
            "A stage can improve heuristic QC while degrading ground-truth target preservation.",
            "Manual-vs-auto conclusions are invalid if branch sanity warnings indicate early signal loss.",
            "One 90-trace pipe scenario is insufficient to redesign scoring without more scenes.",
        ],
    }


def _branch_summary(branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "params_by_method": _json_safe(branch["params_by_method"]),
        "heuristic_metrics": _json_safe(branch["heuristic_metrics"]),
        "ground_truth_metrics": _json_safe(branch["ground_truth_metrics"]),
        "sanity_warnings": branch["sanity_warnings"],
        "branch_invalid_reason": branch["branch_invalid_reason"],
        "auto_tune_results": _json_safe(branch.get("auto_tune_results", {})),
    }


def _rank_stage_winners(table: list[dict[str, Any]], metric_type: str) -> dict[str, Any]:
    score_key = "truth_score" if metric_type == "ground_truth" else "heuristic_target_band_energy_ratio"
    all_ranked = [row for row in table if _is_number(row.get(score_key))]
    all_ranked.sort(key=lambda row: float(row[score_key]), reverse=True)
    valid = [row for row in all_ranked if row.get("valid")]
    ranked = [row for row in valid if _is_number(row.get(score_key))]
    return {
        "metric_used": score_key,
        "winner": ranked[0]["branch"] if ranked else "inconclusive",
        "loser": ranked[-1]["branch"] if ranked else "inconclusive",
        "diagnostic_winner_including_invalid": all_ranked[0]["branch"] if all_ranked else "inconclusive",
        "diagnostic_loser_including_invalid": all_ranked[-1]["branch"] if all_ranked else "inconclusive",
        "ranking": [
            {"branch": row["branch"], "score": row.get(score_key), "valid": row.get("valid")}
            for row in ranked
        ],
        "ranking_including_invalid": [
            {"branch": row["branch"], "score": row.get(score_key), "valid": row.get("valid")}
            for row in all_ranked
        ],
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AT-002 Native Benchmark AutoTune Ablation",
        "",
        "## Dataset And Scope",
        f"- Scenario: `{summary['dataset']['name']}`",
        f"- Shape: `{summary['dataset']['shape']}`",
        f"- Source evidence: `{summary['dataset']['source_evidence']}`",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Ground truth available: `{summary['ground_truth_available']}`",
        f"- Metric type: `{summary['metric_type']}`",
        "",
        "AT-002 uses GX-003 native gprMax-to-CSV provenance to diagnose which processing stages help or hurt before any AutoTune scoring redesign.",
        "Ground truth metrics and heuristic QC are stored separately. Ground truth is not used as AutoTune search input.",
        "",
        "## Claim Boundary",
        "- This is a per-stage ablation diagnostic, not a claim that AutoTune globally beats manual processing.",
        "- Motion compensation, processing_engine behavior, and AutoTune scoring are frozen for this task.",
        "- If a branch has sanity warnings or an invalid reason, manual-vs-auto claims must be treated as inconclusive.",
        "",
        "## Stage-Level Ranking",
        f"- Metric used: `{summary['stage_winners']['metric_used']}`",
        f"- Winner: `{summary['stage_winners']['winner']}`",
        f"- Loser: `{summary['stage_winners']['loser']}`",
        f"- Diagnostic winner including invalid branches: `{summary['stage_winners']['diagnostic_winner_including_invalid']}`",
        f"- Diagnostic loser including invalid branches: `{summary['stage_winners']['diagnostic_loser_including_invalid']}`",
        "",
        "| Branch | Valid | truth_score | target preservation | background reduction | false positive | heuristic target ratio | Invalid reason |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["stage_ablation_table"]:
        lines.append(
            "| {branch} | {valid} | {truth} | {preserve} | {bg} | {fp} | {heur} | {reason} |".format(
                branch=row["branch"],
                valid=row["valid"],
                truth=_fmt(row.get("truth_score")),
                preserve=_fmt(row.get("truth_target_energy_preservation")),
                bg=_fmt(row.get("truth_background_energy_reduction")),
                fp=_fmt(row.get("truth_false_positive_ratio")),
                heur=_fmt(row.get("heuristic_target_band_energy_ratio")),
                reason=row.get("branch_invalid_reason") or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Selected Auto Parameters",
        ]
    )
    for branch_name in ["auto_tuned", *[f"only_{stage}_auto_tuned" for stage in ABLATION_STAGES]]:
        row = next((item for item in summary["stage_ablation_table"] if item["branch"] == branch_name), None)
        if row:
            lines.extend([f"### {branch_name}", "```json", row["selected_auto_params"], "```", ""])
    lines.extend(
        [
            "## Sanity Warnings",
        ]
    )
    for row in summary["stage_ablation_table"]:
        lines.append(f"- `{row['branch']}`: {row.get('sanity_warnings') or 'none'}")
    lines.extend(["", "## Interpretation"])
    winner = summary["stage_winners"]["winner"]
    if winner == "inconclusive":
        lines.append("- Result is inconclusive because no valid scored branch was available.")
        lines.append(
            "- The diagnostic ranking still records metric highs/lows, but those cannot be used as stage winners until sanity failures are resolved."
        )
    elif winner == "expert_manual":
        lines.append("- Expert manual remains strongest under the selected metric; AutoTune should not be claimed as superior on this run.")
    elif winner == "auto_tuned":
        lines.append("- All-stage AutoTune ranks highest under the selected metric; inspect previews and sanity warnings before claiming improvement.")
    else:
        lines.append(f"- `{winner}` ranks highest, suggesting this stage deserves focused scoring/parameter-domain review.")
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend(["", "## Known Risks"])
    lines.extend(f"- {item}" for item in summary["known_risks"])
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def _selected_params_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method_key, tune_result in (result.get("auto_tune_results") or {}).items():
        summary[method_key] = {
            "recommended_params": tune_result.get("recommended_params") or tune_result.get("best_params") or {},
            "best_score": tune_result.get("best_score"),
            "risk_flags": tune_result.get("risk_flags", []),
            "trial_count": len(tune_result.get("all_trials") or []),
        }
    return _json_safe(summary)


def _delta(value: Any, baseline: Any) -> float | None:
    if not (_is_number(value) and _is_number(baseline)):
        return None
    return float(value) - float(baseline)


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}" if _is_number(value) else "-"


def _is_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())

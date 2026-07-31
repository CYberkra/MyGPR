#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Human-readable Markdown report projection for comparison evidence."""
from __future__ import annotations

from typing import Any

from core.auto_tune_comparison_export_common import _artifact_name, _csv_value
from core.auto_tune_comparison_export_tables import _build_autotune_v1_evidence_context

def _build_report_markdown(summary: dict[str, Any]) -> str:
    manual = summary.get("manual") or {}
    automatic = summary.get("automatic") or {}
    delta = summary.get("metric_delta") or {}
    roi = summary.get("roi_info") or {}
    artifacts = summary.get("artifacts") or {}
    ground_truth = summary.get("ground_truth_info") or {}
    v1_context = _build_autotune_v1_evidence_context(summary)
    notes = [str(item) for item in (summary.get("notes") or [])]
    lines = [
        "# AutoTune gprMax Evidence Report",
        "",
        "## 1. Experiment Summary",
        "",
        f"- Input: {summary.get('input_ref') or '--'}",
        f"- Verdict: {summary.get('verdict') or '--'}",
        f"- ROI: {roi.get('label') or roi.get('source') or '--'}",
        f"- Baseline profile: {summary.get('baseline_profile_key') or '--'}",
        f"- Manual score: {_csv_value((manual.get('metrics') or {}).get('comparison_score'))}",
        f"- Auto score: {_csv_value((automatic.get('metrics') or {}).get('comparison_score'))}",
        f"- Delta score: {_csv_value(delta.get('comparison_score'))}",
        "",
        "## 2. Ground Truth",
        "",
        f"- Enabled: {bool(ground_truth.get('enabled'))}",
        f"- Scenario ID: {ground_truth.get('scenario_id') or '--'}",
        f"- Target count: {ground_truth.get('target_count', 0)}",
        f"- Target: {_format_ground_truth_target(ground_truth)}",
        f"- Background ROI: {_format_background_rois(ground_truth)}",
        f"- Warnings: {_format_list(ground_truth.get('conversion_warnings') or [])}",
        "",
        "## 3. Metrics Table",
        "",
        _markdown_metrics_table(manual.get("metrics") or {}, automatic.get("metrics") or {}, delta),
        "",
        "## 4. Parameter Table",
        "",
        _markdown_params_table(manual, automatic),
        "",
        "## 5. Trial Summary",
        "",
        _markdown_trial_summary(summary),
        "",
        "## 5.1 AutoTune V1 Evidence Boundary",
        "",
        _markdown_autotune_v1_boundary(v1_context),
        "",
        "## 6. Figures",
        "",
        f"![Manual B-scan]({_artifact_name(artifacts.get('manual_png'))})",
        f"![AutoTune B-scan]({_artifact_name(artifacts.get('auto_png'))})",
        f"![Side-by-side]({_artifact_name(artifacts.get('side_by_side_png'))})",
        "",
        "## 7. Reproducibility",
        "",
        f"- Pipeline: {' -> '.join(str(item) for item in (manual.get('pipeline') or [])) or '--'}",
        f"- Summary JSON: `{_artifact_name(artifacts.get('summary_json'))}`",
        f"- Evidence manifest: `evidence_manifest.json`",
        f"- Converted ground truth: `converted_ground_truth.json`",
        f"- Trial table: `trial_table.csv` / `trial_table.json`",
        f"- Evidence bundle: `evidence_bundle.zip`",
        "",
        "### Artifacts",
        "",
    ]
    for key, path in artifacts.items():
        lines.append(f"- {key}: `{_artifact_name(path)}`")
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    evidence_warnings = list(summary.get("evidence_warnings") or [])
    if evidence_warnings:
        lines.extend(["", "### Export Warnings", ""])
        for warning in evidence_warnings:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## 8. Research Boundary",
            "",
            "This result is validated under controlled gprMax / selected ROI conditions. It does not automatically prove performance on all field data.",
            "",
            "AGC, background suppression, migration, and gain can alter amplitude or geometry. Truth metrics should be interpreted together, not as a single brightness score.",
            "",
        ]
    )
    return "\n".join(lines)

def _markdown_autotune_v1_boundary(context: dict[str, Any]) -> str:
    if not context.get("enabled"):
        return "- AutoTune V1 candidate-space metadata: not present in this export."
    lines = [
        f"- Candidate-space hash: `{_format_list(context.get('candidate_space_hashes') or [])}`",
        f"- Profile: `{_format_list(context.get('profile_ids') or [])}`",
        f"- Config version: `{_format_list(context.get('config_versions') or [])}`",
        f"- Recipe ids: `{_format_list(context.get('recipe_ids') or [])}`",
        f"- Scoring boundary: `{context.get('scoring_boundary')}`",
        f"- Target response available: {bool(context.get('target_response_available'))}",
        f"- Manual review required: {bool(context.get('manual_review_required'))}",
        f"- Claim boundary: {context.get('claim_boundary')}",
    ]
    forbidden = context.get("forbidden_metrics") or []
    if forbidden:
        lines.append(f"- Forbidden full-reference metrics under this boundary: `{_format_list(forbidden)}`")
    return "\n".join(lines)

def _format_ground_truth_target(ground_truth: dict[str, Any]) -> str:
    targets = ground_truth.get("targets") or []
    if not targets:
        return "--"
    target = targets[0] if isinstance(targets[0], dict) else {}
    roi = target.get("roi") if isinstance(target.get("roi"), dict) else {}
    return (
        f"type={target.get('type', '--')}, material={target.get('material', '--')}, "
        f"depth_m={target.get('depth_m', '--')}, roi={_format_roi_inline(roi)}"
    )

def _format_background_rois(ground_truth: dict[str, Any]) -> str:
    rois = [
        _format_roi_inline(roi)
        for roi in ground_truth.get("background_rois", []) or []
        if isinstance(roi, dict)
    ]
    return "; ".join(rois) if rois else "--"

def _format_roi_inline(roi: dict[str, Any]) -> str:
    if not roi:
        return "--"
    return (
        f"time=[{roi.get('time_start_idx')},{roi.get('time_end_idx')}), "
        f"trace=[{roi.get('dist_start_idx')},{roi.get('dist_end_idx')})"
    )

def _format_list(values: list[Any]) -> str:
    return "; ".join(str(item) for item in values) if values else "--"

def _markdown_metrics_table(
    manual_metrics: dict[str, Any],
    auto_metrics: dict[str, Any],
    delta: dict[str, Any],
) -> str:
    keys = sorted(set(manual_metrics) | set(auto_metrics) | set(delta))
    lines = ["| Metric | Manual | AutoTune | Delta |", "|---|---:|---:|---:|"]
    for key in keys:
        lines.append(
            f"| {key} | {_csv_value(manual_metrics.get(key)) or '--'} | "
            f"{_csv_value(auto_metrics.get(key)) or '--'} | "
            f"{_csv_value(delta.get(key)) or '--'} |"
        )
    return "\n".join(lines)

def _markdown_params_table(manual: dict[str, Any], automatic: dict[str, Any]) -> str:
    manual_params = manual.get("params_by_method") or {}
    auto_params = automatic.get("params_by_method") or {}
    pipeline = list(manual.get("pipeline") or automatic.get("pipeline") or [])
    if not pipeline:
        pipeline = sorted(set(manual_params) | set(auto_params))
    lines = ["| Method | Manual params | AutoTune params |", "|---|---|---|"]
    for method_key in pipeline:
        lines.append(
            f"| {method_key} | `{_csv_value(manual_params.get(method_key, {}))}` | "
            f"`{_csv_value(auto_params.get(method_key, {}))}` |"
        )
    return "\n".join(lines)

def _markdown_trial_summary(summary: dict[str, Any]) -> str:
    automatic = summary.get("automatic") or {}
    auto_results = automatic.get("auto_tune_results") or {}
    lines: list[str] = []
    if not auto_results:
        return "- total trials: 0\n- warnings: AutoTune trial data unavailable."
    for method_key, result in auto_results.items():
        if not isinstance(result, dict):
            continue
        stats = result.get("execution_stats") or {}
        trials = result.get("all_trials") or []
        total = stats.get("total_trial_count", len(trials))
        lines.append(f"### {method_key}")
        lines.append(f"- total trials: {total}")
        lines.append(f"- selected params: `{_csv_value(result.get('recommended_params') or result.get('best_params') or {})}`")
        lines.append(f"- best score: {_csv_value(result.get('best_score')) or '--'}")
        warnings_json = _csv_value(result.get("constraint_warnings") or [])
        lines.append(f"- warnings: {warnings_json or '--'}")
        lines.append("")
    return "\n".join(lines).strip()

__all__ = ['_build_report_markdown', '_markdown_autotune_v1_boundary', '_format_ground_truth_target', '_format_background_rois', '_format_roi_inline', '_format_list', '_markdown_metrics_table', '_markdown_params_table', '_markdown_trial_summary']

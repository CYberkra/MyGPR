#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose AT-002 ROI, zero-time, and pipeline signal-loss failure modes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.auto_tune import AutoTuneError, auto_tune_method
from core.processing_engine import (
    clone_header_info,
    clone_trace_metadata,
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from scripts.auto_tune_validation.run_native_ablation import (
    ABLATION_STAGES,
    DEFAULT_GX003_DATASET,
    DEFAULT_PIPELINE,
)
from scripts.auto_tune_validation.run_stepwise_validation import (
    AUTO_TUNE_SEARCH_MODE,
    MANUAL_EXPERT_PARAMS,
    _branch_invalid_reason,
    _common_heuristic_metrics,
    _git_rev_parse,
    _json_safe,
    _load_dataset,
    _runtime_warning_codes,
    _sanity_warnings,
    _write_json,
)


@dataclass(frozen=True)
class Roi:
    """Python half-open ROI bounds in sample/trace coordinates."""

    time_start_idx: int
    time_end_idx: int
    dist_start_idx: int
    dist_end_idx: int

    def as_dict(self) -> dict[str, int]:
        return {
            "time_start_idx": int(self.time_start_idx),
            "time_end_idx": int(self.time_end_idx),
            "dist_start_idx": int(self.dist_start_idx),
            "dist_end_idx": int(self.dist_end_idx),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="AT-003 evidence output directory")
    parser.add_argument("--dataset", default=str(DEFAULT_GX003_DATASET), help="GX-003 evidence dir or dataset dir")
    parser.add_argument("--mode", choices=["smoke", "normal"], default="normal")
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)

    result = run_diagnosis(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        mode=args.mode,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_diagnosis(
    *,
    evidence_root: Path,
    dataset: str,
    mode: str,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Replay AT-002 branches and export ROI/signal-loss diagnostics."""
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
    ground_truth = package.get("ground_truth") or {}
    target_roi = _target_roi(ground_truth, raw.shape)
    color_limit = _global_color_limit(raw)
    _save_overlay(
        raw,
        figures_dir / "input_bscan_roi_overlay.png",
        roi=target_roi,
        color_limit=color_limit,
        title="Input B-scan with ground-truth ROI",
    )

    branch_specs = _branch_specs()
    rows: list[dict[str, Any]] = []
    branch_summaries: dict[str, Any] = {}
    for branch, spec in branch_specs.items():
        branch_rows, branch_summary = _run_branch_diagnostics(
            branch=branch,
            raw=raw,
            header_info=package["header_info"],
            trace_metadata=package["trace_metadata"],
            target_roi=target_roi,
            figures_dir=figures_dir,
            color_limit=color_limit,
            mode=mode,
            auto_tune=bool(spec["auto_tune"]),
            manual_params=dict(spec["manual_params"]),
            tune_methods=spec["tune_methods"],
        )
        rows.extend(branch_rows)
        branch_summaries[branch] = branch_summary

    first_failure = _first_failure(rows)
    branch_failure_map = _branch_first_failures(rows)
    root_cause = _infer_root_cause(rows, first_failure)
    roi_alignment = _assess_roi_alignment(raw, target_roi)
    summary = {
        "artifact_id": "AT-003",
        "task_id": "AT-003_signal_loss_diagnosis",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": package["dataset_name"],
            "path": package["dataset_path"],
            "shape": package["dataset_shape"],
            "hash": package["dataset_hash"],
            "source_evidence": "gprmax/GX-003_audited_native_gprmax_benchmark/",
        },
        "pipeline": DEFAULT_PIPELINE,
        "target_roi_initial": target_roi.as_dict(),
        "roi_alignment_assessment": roi_alignment,
        "first_failing_step": first_failure,
        "branch_first_failures": branch_failure_map,
        "likely_root_cause": root_cause,
        "branches": branch_summaries,
        "diagnostics": rows,
        "conclusion": "AT-002 remains inconclusive; this diagnosis identifies where signal-loss checks first fail without changing scoring.",
    }
    manifest = {
        "artifact_id": "AT-003",
        "task_id": "AT-003_signal_loss_diagnosis",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_signal_loss_diagnosis.py",
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": bool(package.get("ground_truth")),
        "metric_type": "ground_truth_and_heuristic_diagnostics",
        "artifacts": {
            "report": "reports/signal_loss_diagnosis_report.md",
            "step_diagnostics": "manifests/step_diagnostics.json",
            "step_diagnostics_csv": "tables/step_diagnostics.csv",
            "input_overlay": "figures/input_bscan_roi_overlay.png",
        },
        "known_risks": [
            "ROI-following diagnostics infer zero-time shifts from method metadata; they do not alter processing outputs.",
            "Energy-ratio thresholds diagnose AT-002 invalidation and are not a replacement for visual review.",
        ],
    }
    _write_json(manifests_dir / "step_diagnostics.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)
    _write_csv(tables_dir / "step_diagnostics.csv", rows)
    (reports_dir / "signal_loss_diagnosis_report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "dataset_name": package["dataset_name"],
        "first_failing_step": first_failure,
        "likely_root_cause": root_cause,
        "report": str((reports_dir / "signal_loss_diagnosis_report.md").resolve()),
    }


def _branch_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "expert_manual": {"auto_tune": False, "manual_params": MANUAL_EXPERT_PARAMS, "tune_methods": None},
        "safe_default": {"auto_tune": False, "manual_params": {}, "tune_methods": None},
        "auto_tuned": {"auto_tune": True, "manual_params": MANUAL_EXPERT_PARAMS, "tune_methods": None},
    }
    for stage_name, method_key in ABLATION_STAGES.items():
        specs[f"only_{stage_name}_auto_tuned"] = {
            "auto_tune": True,
            "manual_params": MANUAL_EXPERT_PARAMS,
            "tune_methods": {method_key},
        }
    return specs


def _run_branch_diagnostics(
    *,
    branch: str,
    raw: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    target_roi: Roi,
    figures_dir: Path,
    color_limit: float,
    mode: str,
    auto_tune: bool,
    manual_params: dict[str, dict[str, Any]],
    tune_methods: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    search_mode = AUTO_TUNE_SEARCH_MODE[mode]
    current = np.array(raw, copy=True)
    current_header = clone_header_info(header_info)
    current_trace_metadata = clone_trace_metadata(trace_metadata)
    current_roi = target_roi
    rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    branch_invalid_reason = ""
    energy_curve: list[dict[str, Any]] = [
        {"step_index": 0, "method_key": "input", "roi_energy": _roi_energy(current, current_roi)}
    ]
    for step_index, method_key in enumerate(DEFAULT_PIPELINE, start=1):
        before = np.array(current, copy=True)
        before_roi = current_roi
        params = dict(manual_params.get(method_key, {}))
        should_tune = auto_tune and (tune_methods is None or method_key in tune_methods)
        if should_tune:
            try:
                tune_result = auto_tune_method(
                    current,
                    method_key,
                    header_info=current_header,
                    trace_metadata=current_trace_metadata,
                    base_params=params,
                    search_mode=search_mode,
                )
                params.update(dict(tune_result.get("recommended_params") or tune_result.get("best_params") or {}))
            except AutoTuneError as exc:
                all_warnings.append(f"{method_key}: auto-tune failed and manual params were used: {exc}")
        runtime_params = prepare_runtime_params(
            method_key,
            params,
            current_header,
            current_trace_metadata,
            current.shape,
        )
        current, meta = run_processing_method(current, method_key, runtime_params)
        current_header = merge_result_header_info(current_header, meta, current.shape)
        current_trace_metadata = merge_result_trace_metadata(current_trace_metadata, meta)
        shift_samples = int(meta.get("shift_samples") or 0) if method_key == "set_zero_time" else 0
        step_s = float(meta.get("time_step_s") or current_header.get("time_step_s") or 0.0)
        after_roi = _shift_roi_up(before_roi, shift_samples, current.shape) if shift_samples else before_roi

        heuristic = _common_heuristic_metrics(before, current)
        sanity = _sanity_warnings(
            before=before,
            after=current,
            heuristic_metrics=heuristic,
            ground_truth_metrics={},
        )
        all_warnings.extend(f"{method_key}: {item}" for item in sanity)
        if not branch_invalid_reason:
            branch_invalid_reason = _branch_invalid_reason(sanity)
        overlay_path = figures_dir / f"{branch}_step_{step_index:02d}_{method_key}_roi_overlay.png"
        crop_path = figures_dir / f"{branch}_step_{step_index:02d}_{method_key}_roi_crop.png"
        curve_path = figures_dir / f"{branch}_stepwise_energy_curve.png"
        _save_overlay(
            current,
            overlay_path,
            roi=after_roi,
            color_limit=color_limit,
            title=f"{branch} step {step_index}: {method_key}",
        )
        _save_crop(
            current,
            crop_path,
            roi=after_roi,
            color_limit=color_limit,
            title=f"{branch} step {step_index}: {method_key} ROI crop",
        )
        before_energy = _roi_energy(before, before_roi)
        after_energy = _roi_energy(current, after_roi)
        global_before = _global_energy(before)
        global_after = _global_energy(current)
        row = {
            "branch": branch,
            "step_index": step_index,
            "method_key": method_key,
            "input_shape": [int(before.shape[0]), int(before.shape[1])],
            "output_shape": [int(current.shape[0]), int(current.shape[1])],
            "params": _json_safe(params),
            "zero_time_shift_samples": int(shift_samples),
            "zero_time_shift_ns": float(shift_samples * step_s * 1e9) if step_s > 0 else 0.0,
            "roi_before": before_roi.as_dict(),
            "roi_after": after_roi.as_dict(),
            "roi_before_energy": before_energy,
            "roi_after_energy": after_energy,
            "roi_energy_ratio": _ratio(after_energy, before_energy),
            "global_energy_before": global_before,
            "global_energy_after": global_after,
            "global_energy_ratio": _ratio(global_after, global_before),
            "target_band_fidelity": heuristic.get("target_band_energy_ratio"),
            "edge_preservation": heuristic.get("edge_preservation"),
            "local_saliency_preservation": heuristic.get("local_saliency_preservation"),
            "sanity_warnings": sanity,
            "runtime_warnings": _runtime_warning_codes(meta),
            "branch_invalid_reason": branch_invalid_reason,
            "preview_png": overlay_path.name,
            "roi_overlay_png": overlay_path.name,
            "roi_crop_png": crop_path.name,
        }
        rows.append(row)
        current_roi = after_roi
        energy_curve.append({"step_index": step_index, "method_key": method_key, "roi_energy": after_energy})
        _save_energy_curve(energy_curve, curve_path, title=f"{branch} ROI energy by step")
    return rows, {
        "branch_invalid_reason": branch_invalid_reason,
        "sanity_warnings": sorted(set(all_warnings)),
        "final_roi": current_roi.as_dict(),
        "energy_curve_png": f"{branch}_stepwise_energy_curve.png",
    }


def _target_roi(ground_truth: dict[str, Any], shape: tuple[int, int]) -> Roi:
    targets = ground_truth.get("targets") or []
    roi = targets[0].get("roi") if targets and isinstance(targets[0], dict) else None
    if not isinstance(roi, dict):
        roi = ground_truth.get("analysis_roi") or {}
    return _clamp_roi(
        Roi(
            int(roi.get("time_start_idx", 0)),
            int(roi.get("time_end_idx", shape[0])),
            int(roi.get("dist_start_idx", 0)),
            int(roi.get("dist_end_idx", shape[1])),
        ),
        shape,
    )


def _clamp_roi(roi: Roi, shape: tuple[int, int]) -> Roi:
    samples, traces = int(shape[0]), int(shape[1])
    t0 = max(0, min(samples, int(roi.time_start_idx)))
    t1 = max(t0 + 1, min(samples, int(roi.time_end_idx)))
    x0 = max(0, min(traces, int(roi.dist_start_idx)))
    x1 = max(x0 + 1, min(traces, int(roi.dist_end_idx)))
    return Roi(t0, t1, x0, x1)


def _shift_roi_up(roi: Roi, shift_samples: int, shape: tuple[int, int]) -> Roi:
    return _clamp_roi(
        Roi(
            roi.time_start_idx - int(shift_samples),
            roi.time_end_idx - int(shift_samples),
            roi.dist_start_idx,
            roi.dist_end_idx,
        ),
        shape,
    )


def _roi_energy(data: np.ndarray, roi: Roi) -> float:
    crop = np.asarray(data[roi.time_start_idx : roi.time_end_idx, roi.dist_start_idx : roi.dist_end_idx], dtype=np.float64)
    if crop.size == 0:
        return 0.0
    return float(np.mean(np.nan_to_num(crop) ** 2))


def _global_energy(data: np.ndarray) -> float:
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.nan_to_num(arr) ** 2))


def _ratio(after: float, before: float) -> float:
    return float(after / max(before, 1.0e-30))


def _global_color_limit(data: np.ndarray) -> float:
    finite = np.asarray(data, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    vmax = float(np.percentile(np.abs(finite), 98.0)) if finite.size else 1.0
    return max(vmax, 1.0e-12)


def _save_overlay(data: np.ndarray, path: Path, *, roi: Roi, color_limit: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=150)
    try:
        ax.imshow(data, cmap="gray", aspect="auto", vmin=-color_limit, vmax=color_limit)
        ax.add_patch(
            Rectangle(
                (roi.dist_start_idx - 0.5, roi.time_start_idx - 0.5),
                roi.dist_end_idx - roi.dist_start_idx,
                roi.time_end_idx - roi.time_start_idx,
                fill=False,
                edgecolor="#ff3b30",
                linewidth=1.6,
            )
        )
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_crop(data: np.ndarray, path: Path, *, roi: Roi, color_limit: float, title: str) -> None:
    pad_t = max(6, (roi.time_end_idx - roi.time_start_idx) // 2)
    pad_x = max(4, (roi.dist_end_idx - roi.dist_start_idx) // 2)
    crop_roi = _clamp_roi(
        Roi(
            roi.time_start_idx - pad_t,
            roi.time_end_idx + pad_t,
            roi.dist_start_idx - pad_x,
            roi.dist_end_idx + pad_x,
        ),
        data.shape,
    )
    crop = data[crop_roi.time_start_idx : crop_roi.time_end_idx, crop_roi.dist_start_idx : crop_roi.dist_end_idx]
    shifted_roi = Roi(
        roi.time_start_idx - crop_roi.time_start_idx,
        roi.time_end_idx - crop_roi.time_start_idx,
        roi.dist_start_idx - crop_roi.dist_start_idx,
        roi.dist_end_idx - crop_roi.dist_start_idx,
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=150)
    try:
        ax.imshow(crop, cmap="gray", aspect="auto", vmin=-color_limit, vmax=color_limit)
        ax.add_patch(
            Rectangle(
                (shifted_roi.dist_start_idx - 0.5, shifted_roi.time_start_idx - 0.5),
                shifted_roi.dist_end_idx - shifted_roi.dist_start_idx,
                shifted_roi.time_end_idx - shifted_roi.time_start_idx,
                fill=False,
                edgecolor="#ff3b30",
                linewidth=1.5,
            )
        )
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_energy_curve(points: list[dict[str, Any]], path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=150)
    try:
        xs = [int(item["step_index"]) for item in points]
        ys = [float(item["roi_energy"]) for item in points]
        labels = [str(item["method_key"]) for item in points]
        ax.plot(xs, ys, marker="o", color="#0f766e")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("ROI mean squared amplitude")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _assess_roi_alignment(data: np.ndarray, roi: Roi) -> dict[str, Any]:
    roi_energy = _roi_energy(data, roi)
    global_energy = _global_energy(data)
    contrast = _ratio(roi_energy, global_energy)
    return {
        "roi_energy": roi_energy,
        "global_energy": global_energy,
        "roi_to_global_energy_ratio": contrast,
        "appears_energy_aligned": bool(contrast > 1.0),
        "note": "Energy ratio > 1 suggests ROI intersects stronger-than-average signal; visual overlay remains required.",
    }


def _first_failure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    severe = []
    for row in rows:
        if row.get("branch_invalid_reason") or float(row.get("roi_energy_ratio") or 1.0) < 0.25:
            severe.append(row)
    if not severe:
        return {"branch": "", "step_index": None, "method_key": "", "reason": "none"}
    first = sorted(severe, key=lambda item: (int(item["step_index"]), str(item["branch"])))[0]
    return {
        "branch": first["branch"],
        "step_index": first["step_index"],
        "method_key": first["method_key"],
        "reason": first.get("branch_invalid_reason") or "roi_energy_ratio_below_0.25",
        "roi_energy_ratio": first.get("roi_energy_ratio"),
        "global_energy_ratio": first.get("global_energy_ratio"),
    }


def _branch_first_failures(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the first severe diagnostic event in each branch."""
    failures: dict[str, dict[str, Any]] = {}
    for branch in sorted({str(row["branch"]) for row in rows}):
        branch_rows = sorted(
            (row for row in rows if row["branch"] == branch),
            key=lambda item: int(item["step_index"]),
        )
        for row in branch_rows:
            if row.get("branch_invalid_reason") or float(row.get("roi_energy_ratio") or 1.0) < 0.25:
                failures[branch] = {
                    "step_index": row["step_index"],
                    "method_key": row["method_key"],
                    "reason": row.get("branch_invalid_reason") or "roi_energy_ratio_below_0.25",
                    "zero_time_shift_samples": row.get("zero_time_shift_samples"),
                    "roi_energy_ratio": row.get("roi_energy_ratio"),
                    "global_energy_ratio": row.get("global_energy_ratio"),
                }
                break
        if branch not in failures:
            failures[branch] = {"step_index": None, "method_key": "", "reason": "none"}
    return failures


def _infer_root_cause(rows: list[dict[str, Any]], first_failure: dict[str, Any]) -> dict[str, Any]:
    step = first_failure.get("method_key")
    zero_rows = [row for row in rows if row["method_key"] == "set_zero_time"]
    dewow_rows = [row for row in rows if row["method_key"] == "dewow"]
    background_rows = [row for row in rows if row["method_key"] == "subtracting_average_2D"]
    shifted_zero_rows = [row for row in zero_rows if int(row.get("zero_time_shift_samples") or 0) > 0]
    dewow_collapse_rows = [row for row in dewow_rows if float(row.get("roi_energy_ratio") or 1.0) < 0.25]
    background_collapse_rows = [row for row in background_rows if float(row.get("roi_energy_ratio") or 1.0) < 0.25]
    if step == "set_zero_time" and shifted_zero_rows and dewow_collapse_rows:
        cause = "mixed_zero_time_default_and_dewow_signal_loss"
        note = (
            "A default zero-time shift causes the earliest global-energy collapse in at least one branch, "
            "while dewow is the first ROI-energy collapse in most fixed-parameter branches."
        )
    elif step == "set_zero_time" and shifted_zero_rows:
        cause = "zero_time_default_or_roi_shift"
        note = "Zero-time changed the sample axis before later steps; verify ROI alignment and default zero-time parameters."
    elif step == "dewow" or any(float(row.get("roi_energy_ratio") or 1.0) < 0.25 for row in dewow_rows):
        cause = "dewow_signal_loss_or_sanity_threshold"
        note = "Dewow is the first step where target ROI energy collapses under the current sanity thresholds."
    elif step == "subtracting_average_2D" or any(float(row.get("roi_energy_ratio") or 1.0) < 0.25 for row in background_rows):
        cause = "background_suppression_target_damage"
        note = "Background suppression is the first step that damages the tracked ROI under this diagnostic."
    else:
        cause = "threshold_policy_or_roi_definition"
        note = "No single processing step dominates; inspect ROI definition and sanity thresholds."
    return {
        "likely_cause": cause,
        "evidence": first_failure,
        "note": note,
        "zero_time_shifted_branches": [row["branch"] for row in shifted_zero_rows],
        "dewow_roi_collapse_branches": [row["branch"] for row in dewow_collapse_rows],
        "background_roi_collapse_branches": [row["branch"] for row in background_collapse_rows],
        "at002_conclusion": "inconclusive",
    }


def _render_report(summary: dict[str, Any]) -> str:
    first = summary["first_failing_step"]
    root = summary["likely_root_cause"]
    lines = [
        "# AT-003 Signal Loss Diagnosis",
        "",
        "## Dataset",
        f"- Scenario: `{summary['dataset']['name']}`",
        f"- Shape: `{summary['dataset']['shape']}`",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Initial target ROI: `{summary['target_roi_initial']}`",
        f"- ROI energy alignment ratio: `{summary['roi_alignment_assessment']['roi_to_global_energy_ratio']:.4g}`",
        f"- ROI appears energy aligned before processing: `{summary['roi_alignment_assessment']['appears_energy_aligned']}`",
        "",
        "## First Failing Step",
        f"- Branch: `{first.get('branch')}`",
        f"- Step: `{first.get('step_index')}`",
        f"- Method: `{first.get('method_key')}`",
        f"- Reason: `{first.get('reason')}`",
        f"- ROI energy ratio: `{first.get('roi_energy_ratio')}`",
        f"- Global energy ratio: `{first.get('global_energy_ratio')}`",
        "",
        "## Likely Root Cause",
        f"- `{root['likely_cause']}`",
        f"- Note: {root.get('note', '')}",
        f"- Zero-time shifted branches: `{root.get('zero_time_shifted_branches', [])}`",
        f"- Dewow ROI-collapse branches: `{root.get('dewow_roi_collapse_branches', [])}`",
        f"- Background ROI-collapse branches: `{root.get('background_roi_collapse_branches', [])}`",
        "",
        "AT-002 remains inconclusive. This report diagnoses where the failure starts; it does not change AutoTune scoring or hide invalid results.",
        "",
        "## Per-Branch First Failures",
        "| Branch | Step | Method | Reason | Shift samples | ROI energy ratio | Global energy ratio |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for branch, failure in summary["branch_first_failures"].items():
        lines.append(
            "| {branch} | {step} | `{method}` | {reason} | {shift} | {roi} | {global_ratio} |".format(
                branch=branch,
                step=failure.get("step_index"),
                method=failure.get("method_key"),
                reason=failure.get("reason"),
                shift=failure.get("zero_time_shift_samples"),
                roi=failure.get("roi_energy_ratio"),
                global_ratio=failure.get("global_energy_ratio"),
            )
        )
    lines.extend(
        [
        "",
        "## Branch Summary",
        "| Branch | Invalid reason | Final ROI | Energy curve |",
        "|---|---|---|---|",
        ]
    )
    for branch, info in summary["branches"].items():
        lines.append(
            f"| `{branch}` | {info.get('branch_invalid_reason') or '-'} | `{info.get('final_roi')}` | `figures/{info.get('energy_curve_png')}` |"
        )
    lines.extend(
        [
            "",
            "## Step Diagnostics",
            "| Branch | Step | Method | Shift samples | ROI before | ROI after | ROI energy ratio | Global energy ratio | Sanity warnings | Overlay | Crop |",
            "|---|---:|---|---:|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in summary["diagnostics"]:
        lines.append(
            "| {branch} | {step} | `{method}` | {shift} | `{before}` | `{after}` | {roi_ratio:.4g} | {global_ratio:.4g} | {warnings} | `{overlay}` | `{crop}` |".format(
                branch=row["branch"],
                step=row["step_index"],
                method=row["method_key"],
                shift=row["zero_time_shift_samples"],
                before=row["roi_before"],
                after=row["roi_after"],
                roi_ratio=float(row["roi_energy_ratio"]),
                global_ratio=float(row["global_energy_ratio"]),
                warnings="<br>".join(row["sanity_warnings"]) or "-",
                overlay=f"figures/{row['roi_overlay_png']}",
                crop=f"figures/{row['roi_crop_png']}",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "- Consistent color scale is used for stepwise B-scan and crop images.",
            "- Red rectangles show the ROI used for that step's after-state diagnosis.",
            "- ROI-following after zero-time is diagnostic only; it does not alter processing results.",
            "- Ground truth metrics and heuristic sanity checks remain separate.",
            "- AT-002 conclusion remains `inconclusive`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "branch",
        "step_index",
        "method_key",
        "input_shape",
        "output_shape",
        "params",
        "zero_time_shift_samples",
        "zero_time_shift_ns",
        "roi_before",
        "roi_after",
        "roi_before_energy",
        "roi_after_energy",
        "roi_energy_ratio",
        "global_energy_before",
        "global_energy_after",
        "global_energy_ratio",
        "target_band_fidelity",
        "edge_preservation",
        "local_saliency_preservation",
        "sanity_warnings",
        "branch_invalid_reason",
        "preview_png",
        "roi_overlay_png",
        "roi_crop_png",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(row.get(key)), ensure_ascii=False) for key in fields})


if __name__ == "__main__":
    raise SystemExit(main())

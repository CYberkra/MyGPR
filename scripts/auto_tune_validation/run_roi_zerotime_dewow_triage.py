#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run AT-004 ROI, zero-time, and dewow root-cause triage diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
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

from core.processing_engine import prepare_runtime_params, run_processing_method
from scripts.auto_tune_validation.run_native_ablation import DEFAULT_GX003_DATASET
from scripts.auto_tune_validation.run_stepwise_validation import (
    _git_rev_parse,
    _json_safe,
    _load_dataset,
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

    @property
    def height(self) -> int:
        return int(self.time_end_idx - self.time_start_idx)

    @property
    def width(self) -> int:
        return int(self.dist_end_idx - self.dist_start_idx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="AT-004 evidence output directory")
    parser.add_argument("--dataset", default=str(DEFAULT_GX003_DATASET), help="GX-003 evidence dir or dataset dir")
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)

    result = run_triage(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_triage(
    *,
    evidence_root: Path,
    dataset: str,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Run root-cause triage without changing AutoTune scoring or processing algorithms."""
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
    target_roi = _target_roi(ground_truth, raw.shape)
    candidate_roi = _find_candidate_roi(raw, target_roi)
    color_limit = _global_color_limit(raw)

    _save_overlay(
        raw,
        figures_dir / "input_bscan_roi_overlay.png",
        rois=[("GX-003 ROI", target_roi, "red"), ("candidate ROI", candidate_roi, "lime")],
        color_limit=color_limit,
        title="Input B-scan ROI review",
    )
    _save_crop(raw, figures_dir / "input_gx003_roi_crop.png", target_roi, color_limit, "GX-003 ROI crop")
    _save_crop(raw, figures_dir / "input_candidate_roi_crop.png", candidate_roi, color_limit, "Candidate ROI crop")

    experiments = _experiment_specs()
    rows: list[dict[str, Any]] = []
    for spec in experiments:
        result = _run_experiment(
            raw=raw,
            header_info=header_info,
            trace_metadata=trace_metadata,
            target_roi=target_roi,
            candidate_roi=candidate_roi,
            figures_dir=figures_dir,
            color_limit=color_limit,
            **spec,
        )
        rows.append(result)

    root_cause = _classify_root_cause(rows, target_roi, candidate_roi)
    summary = {
        "artifact_id": "AT-004",
        "task_id": "AT-004_roi_zerotime_dewow_triage",
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
        "gx003_roi": target_roi.as_dict(),
        "candidate_roi": candidate_roi.as_dict(),
        "roi_review": {
            "gx003": _roi_metrics(raw, target_roi),
            "candidate": _roi_metrics(raw, candidate_roi),
            "candidate_is_recommended": bool(root_cause["corrected_roi_recommended"]),
        },
        "experiments": rows,
        "root_cause_classification": root_cause,
        "conclusion": "AT-002 remains inconclusive; AT-004 diagnoses candidate next fixes only.",
    }
    manifest = {
        "artifact_id": "AT-004",
        "task_id": "AT-004_roi_zerotime_dewow_triage",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_roi_zerotime_dewow_triage.py",
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": bool(ground_truth),
        "metric_type": "ground_truth_roi_and_heuristic_qc_triage",
        "artifacts": {
            "report": "reports/roi_zerotime_dewow_triage_report.md",
            "summary": "manifests/triage_summary.json",
            "csv": "tables/triage_results.csv",
            "input_overlay": "figures/input_bscan_roi_overlay.png",
        },
        "limitations": [
            "Candidate ROI is diagnostic only and does not replace GX-003 ground truth.",
            "No AutoTune scoring, motion compensation, or processing_engine behavior is changed.",
        ],
    }
    _write_json(manifests_dir / "triage_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)
    _write_csv(tables_dir / "triage_results.csv", rows)
    (reports_dir / "roi_zerotime_dewow_triage_report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "dataset_name": package["dataset_name"],
        "root_cause_classification": root_cause,
        "report": str((reports_dir / "roi_zerotime_dewow_triage_report.md").resolve()),
    }


def _experiment_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "experiment_id": "no_zero_time_no_dewow",
            "description": "Raw data only; isolates GX-003 ROI definition before processing.",
            "steps": [],
        },
        {
            "experiment_id": "zero_time_fixed_0_dewow_off",
            "description": "Zero-time correction fixed at 0 ns, with dewow disabled.",
            "steps": [("set_zero_time", {"new_zero_time": 0.0})],
        },
        {
            "experiment_id": "safe_default_zero_time_only",
            "description": "Registry default zero-time behavior isolated from dewow.",
            "steps": [("set_zero_time", {})],
        },
    ]
    for window in [5, 11, 23, 64, 128, 256, 512]:
        specs.append(
            {
                "experiment_id": f"zero0_dewow_window_{window}",
                "description": f"Zero-time fixed at 0 ns, then dewow window={window}.",
                "steps": [("set_zero_time", {"new_zero_time": 0.0}), ("dewow", {"window": window})],
            }
        )
    return specs


def _run_experiment(
    *,
    raw: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, Any],
    target_roi: Roi,
    candidate_roi: Roi,
    figures_dir: Path,
    color_limit: float,
    experiment_id: str,
    description: str,
    steps: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    current = np.array(raw, copy=True)
    current_roi = target_roi
    zero_time_shift_samples = 0
    zero_time_shift_ns = 0.0
    applied_steps: list[dict[str, Any]] = []
    for method_key, params in steps:
        runtime_params = prepare_runtime_params(method_key, dict(params), header_info, trace_metadata, current.shape)
        current, meta = run_processing_method(current, method_key, runtime_params)
        shift_samples = int(meta.get("shift_samples") or 0) if method_key == "set_zero_time" else 0
        if shift_samples:
            zero_time_shift_samples += shift_samples
            step_s = float(meta.get("time_step_s") or header_info.get("time_step_s") or 0.0)
            zero_time_shift_ns += float(shift_samples * step_s * 1e9) if step_s > 0 else 0.0
            current_roi = _shift_roi_up(current_roi, shift_samples, current.shape)
        applied_steps.append({"method_key": method_key, "params": _json_safe(runtime_params), "metadata": _json_safe(meta)})

    overlay_name = f"{experiment_id}_roi_overlay.png"
    crop_name = f"{experiment_id}_gx003_roi_crop.png"
    candidate_crop_name = f"{experiment_id}_candidate_roi_crop.png"
    _save_overlay(
        current,
        figures_dir / overlay_name,
        rois=[("tracked GX-003 ROI", current_roi, "red"), ("candidate ROI", candidate_roi, "lime")],
        color_limit=color_limit,
        title=experiment_id,
    )
    _save_crop(current, figures_dir / crop_name, current_roi, color_limit, f"{experiment_id} tracked GX-003 ROI")
    _save_crop(current, figures_dir / candidate_crop_name, candidate_roi, color_limit, f"{experiment_id} candidate ROI")

    input_roi = _roi_metrics(raw, target_roi)
    output_roi = _roi_metrics(current, current_roi)
    output_candidate = _roi_metrics(current, candidate_roi)
    return {
        "experiment_id": experiment_id,
        "description": description,
        "steps": applied_steps,
        "input_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "output_shape": [int(current.shape[0]), int(current.shape[1])],
        "zero_time_shift_samples": int(zero_time_shift_samples),
        "zero_time_shift_ns": float(zero_time_shift_ns),
        "tracked_roi_after": current_roi.as_dict(),
        "gx003_roi_input": input_roi,
        "gx003_roi_output": output_roi,
        "candidate_roi_output": output_candidate,
        "roi_energy_ratio": _ratio(output_roi["roi_energy"], input_roi["roi_energy"]),
        "global_energy_ratio": _ratio(_global_energy(current), _global_energy(raw)),
        "roi_local_background_contrast": output_roi["roi_to_local_background_contrast"],
        "candidate_roi_local_background_contrast": output_candidate["roi_to_local_background_contrast"],
        "overlay_png": f"figures/{overlay_name}",
        "roi_crop_png": f"figures/{crop_name}",
        "candidate_roi_crop_png": f"figures/{candidate_crop_name}",
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


def _find_candidate_roi(data: np.ndarray, reference: Roi) -> Roi:
    """Find a diagnostic high-energy ROI near the reference trace band."""
    samples, traces = data.shape
    height = max(8, min(reference.height, samples))
    width = max(3, min(reference.width, traces))
    x_center = (reference.dist_start_idx + reference.dist_end_idx) // 2
    x0 = max(0, min(traces - width, x_center - width // 2))
    x1 = x0 + width
    search_margin = max(80, height)
    t0 = max(0, reference.time_start_idx - search_margin)
    t1 = min(samples - height, reference.time_end_idx + search_margin)
    if t1 <= t0:
        t0, t1 = 0, max(0, samples - height)

    arr = np.nan_to_num(np.asarray(data[:, x0:x1], dtype=np.float64)) ** 2
    best_t = int(reference.time_start_idx)
    best_energy = -1.0
    for start in range(t0, t1 + 1):
        energy = float(np.mean(arr[start : start + height, :]))
        if energy > best_energy:
            best_energy = energy
            best_t = start
    return _clamp_roi(Roi(best_t, best_t + height, x0, x1), data.shape)


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


def _roi_metrics(data: np.ndarray, roi: Roi) -> dict[str, Any]:
    roi_energy = _roi_energy(data, roi)
    global_energy = _global_energy(data)
    background = _local_background_roi(data.shape, roi)
    background_energy = _roi_energy(data, background)
    return {
        "roi": roi.as_dict(),
        "roi_energy": roi_energy,
        "global_energy": global_energy,
        "roi_to_global_energy_ratio": _ratio(roi_energy, global_energy),
        "local_background_roi": background.as_dict(),
        "local_background_energy": background_energy,
        "roi_to_local_background_contrast": _ratio(roi_energy, background_energy),
    }


def _local_background_roi(shape: tuple[int, int], roi: Roi) -> Roi:
    samples, traces = shape
    height = roi.height
    gap = max(4, height // 4)
    if roi.time_end_idx + gap + height <= samples:
        return Roi(roi.time_end_idx + gap, roi.time_end_idx + gap + height, roi.dist_start_idx, roi.dist_end_idx)
    if roi.time_start_idx - gap - height >= 0:
        return Roi(roi.time_start_idx - gap - height, roi.time_start_idx - gap, roi.dist_start_idx, roi.dist_end_idx)
    x0 = min(traces - roi.width, max(0, roi.dist_end_idx + 2))
    return _clamp_roi(Roi(roi.time_start_idx, roi.time_end_idx, x0, x0 + roi.width), shape)


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


def _save_overlay(
    data: np.ndarray,
    path: Path,
    *,
    rois: list[tuple[str, Roi, str]],
    color_limit: float,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=150)
    try:
        ax.imshow(data, cmap="gray", aspect="auto", vmin=-color_limit, vmax=color_limit)
        for label, roi, color in rois:
            ax.add_patch(
                Rectangle(
                    (roi.dist_start_idx - 0.5, roi.time_start_idx - 0.5),
                    roi.width,
                    roi.height,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5,
                    label=label,
                )
            )
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_crop(data: np.ndarray, path: Path, roi: Roi, color_limit: float, title: str) -> None:
    pad_t = max(4, roi.height // 5)
    pad_x = max(2, roi.width // 2)
    padded = _clamp_roi(
        Roi(
            roi.time_start_idx - pad_t,
            roi.time_end_idx + pad_t,
            roi.dist_start_idx - pad_x,
            roi.dist_end_idx + pad_x,
        ),
        data.shape,
    )
    crop = data[padded.time_start_idx : padded.time_end_idx, padded.dist_start_idx : padded.dist_end_idx]
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=150)
    try:
        ax.imshow(crop, cmap="gray", aspect="auto", vmin=-color_limit, vmax=color_limit)
        ax.add_patch(
            Rectangle(
                (roi.dist_start_idx - padded.dist_start_idx - 0.5, roi.time_start_idx - padded.time_start_idx - 0.5),
                roi.width,
                roi.height,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )
        )
        ax.set_title(title)
        ax.set_xlabel("Trace crop")
        ax.set_ylabel("Sample crop")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _classify_root_cause(rows: list[dict[str, Any]], target_roi: Roi, candidate_roi: Roi) -> dict[str, Any]:
    by_id = {row["experiment_id"]: row for row in rows}
    raw = by_id["no_zero_time_no_dewow"]
    safe = by_id["safe_default_zero_time_only"]
    dewow_rows = [row for row in rows if row["experiment_id"].startswith("zero0_dewow_window_")]
    best_dewow = max(dewow_rows, key=lambda item: float(item["roi_energy_ratio"])) if dewow_rows else None

    gx003_contrast = float(raw["gx003_roi_output"]["roi_to_local_background_contrast"])
    candidate_contrast = float(raw["candidate_roi_output"]["roi_to_local_background_contrast"])
    safe_global_ratio = float(safe["global_energy_ratio"])
    dewow_best_ratio = float(best_dewow["roi_energy_ratio"]) if best_dewow else 1.0

    findings: list[str] = []
    if gx003_contrast < 1.0 and candidate_contrast > gx003_contrast * 2.0:
        roi_status = "suspect"
        findings.append("GX-003 ROI is weaker than local background; candidate ROI has materially higher contrast.")
    else:
        roi_status = "usable_but_visual_review_required"
        findings.append("GX-003 ROI is not clearly invalid by local contrast alone.")

    zero_status = "problem" if safe_global_ratio < 0.25 else "not_primary"
    if zero_status == "problem":
        findings.append("Registry default zero-time shifts a large sample interval and collapses global energy.")

    dewow_status = "problem" if dewow_best_ratio < 0.25 else "parameter_domain_can_preserve_roi"
    if dewow_status == "problem":
        findings.append("All tested dewow windows strongly reduce tracked ROI energy.")
    else:
        findings.append("At least one dewow window preserves enough tracked ROI energy for further tuning.")

    if zero_status == "problem" and dewow_status == "problem" and roi_status == "suspect":
        next_fix = "benchmark_roi_then_zero_time_defaults_then_dewow_domain"
    elif roi_status == "suspect":
        next_fix = "roi_definition_or_benchmark_design"
    elif zero_status == "problem":
        next_fix = "zero_time_defaults"
    elif dewow_status == "problem":
        next_fix = "dewow_parameter_domain_or_sanity_policy"
    else:
        next_fix = "sanity_threshold_policy"

    return {
        "roi_status": roi_status,
        "zero_time_status": zero_status,
        "dewow_status": dewow_status,
        "recommended_next_fix": next_fix,
        "corrected_roi_recommended": roi_status == "suspect",
        "candidate_roi": candidate_roi.as_dict(),
        "best_dewow_experiment": best_dewow["experiment_id"] if best_dewow else None,
        "best_dewow_roi_energy_ratio": dewow_best_ratio,
        "safe_default_global_energy_ratio": safe_global_ratio,
        "findings": findings,
        "at002_conclusion": "inconclusive",
    }


def _render_report(summary: dict[str, Any]) -> str:
    root = summary["root_cause_classification"]
    lines = [
        "# AT-004 ROI / Zero-Time / Dewow Triage",
        "",
        "## Scope",
        "- AT-002 remains `inconclusive`.",
        "- This run does not change AutoTune scoring, motion compensation, or processing_engine behavior.",
        "- Candidate ROI is diagnostic only and does not silently replace GX-003 ground truth.",
        "",
        "## Dataset",
        f"- Scenario: `{summary['dataset']['name']}`",
        f"- Shape: `{summary['dataset']['shape']}`",
        f"- Source commit: `{summary['source_commit']}`",
        f"- GX-003 ROI: `{summary['gx003_roi']}`",
        f"- Candidate ROI: `{summary['candidate_roi']}`",
        "",
        "## ROI Review",
        f"- GX-003 ROI local-background contrast: `{summary['roi_review']['gx003']['roi_to_local_background_contrast']:.4g}`",
        f"- Candidate ROI local-background contrast: `{summary['roi_review']['candidate']['roi_to_local_background_contrast']:.4g}`",
        f"- Corrected ROI recommended: `{summary['roi_review']['candidate_is_recommended']}`",
        "- Overlay: `figures/input_bscan_roi_overlay.png`",
        "",
        "## Root-Cause Classification",
        f"- ROI status: `{root['roi_status']}`",
        f"- Zero-time status: `{root['zero_time_status']}`",
        f"- Dewow status: `{root['dewow_status']}`",
        f"- Recommended next fix: `{root['recommended_next_fix']}`",
        f"- Best dewow experiment: `{root['best_dewow_experiment']}`",
        f"- Best dewow ROI energy ratio: `{root['best_dewow_roi_energy_ratio']:.4g}`",
        f"- Safe default global energy ratio: `{root['safe_default_global_energy_ratio']:.4g}`",
        "",
        "Findings:",
    ]
    lines.extend(f"- {item}" for item in root["findings"])
    lines.extend(
        [
            "",
            "## Experiment Table",
            "| Experiment | Shift samples | ROI energy ratio | Global energy ratio | ROI local contrast | Candidate local contrast | Overlay |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["experiments"]:
        lines.append(
            "| {exp} | {shift} | {roi_ratio:.4g} | {global_ratio:.4g} | {roi_contrast:.4g} | {candidate_contrast:.4g} | `{overlay}` |".format(
                exp=row["experiment_id"],
                shift=row["zero_time_shift_samples"],
                roi_ratio=float(row["roi_energy_ratio"]),
                global_ratio=float(row["global_energy_ratio"]),
                roi_contrast=float(row["roi_local_background_contrast"]),
                candidate_contrast=float(row["candidate_roi_local_background_contrast"]),
                overlay=row["overlay_png"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "- Do not claim AutoTune improvement from this artifact.",
            "- Do not replace GX-003 ROI without a separate benchmark/ground-truth revision.",
            "- If fixing next, prioritize the classified target above, then rerun AT-002/AT-003.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "experiment_id",
        "description",
        "steps",
        "zero_time_shift_samples",
        "zero_time_shift_ns",
        "tracked_roi_after",
        "roi_energy_ratio",
        "global_energy_ratio",
        "roi_local_background_contrast",
        "candidate_roi_local_background_contrast",
        "overlay_png",
        "roi_crop_png",
        "candidate_roi_crop_png",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(row.get(key)), ensure_ascii=False) for key in fields})


if __name__ == "__main__":
    raise SystemExit(main())

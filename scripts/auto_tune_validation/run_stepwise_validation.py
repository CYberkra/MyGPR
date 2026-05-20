#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate stepwise AutoTune research validation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.auto_tune import AutoTuneError, auto_tune_method
from core.gprmax_truth_metrics import compute_ground_truth_metrics
from core.processing_engine import (
    clone_header_info,
    clone_trace_metadata,
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.quality_metrics import compute_benchmark_metrics
from core.runtime_warnings import merge_runtime_warnings


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "sample_data" / "gprmax_benchmarks" / "cylinder_single_v1"
DEFAULT_PIPELINE = [
    "set_zero_time",
    "dewow",
    "frequency_filter_1d",
    "subtracting_average_2D",
    "energy_decay_gain",
]
MANUAL_EXPERT_PARAMS = {
    "set_zero_time": {"new_zero_time": 0.0},
    "dewow": {"window": 23},
    "frequency_filter_1d": {
        "filter_type": "bandpass",
        "low_freq_mhz": 300.0,
        "high_freq_mhz": 2500.0,
        "taper_ratio": 0.08,
    },
    "subtracting_average_2D": {"ntraces": 41},
    "energy_decay_gain": {
        "strength": 0.8,
        "smoothing_samples": 31,
        "min_gain": 0.6,
        "max_gain": 5.0,
        "floor_ratio": 0.05,
    },
}
AUTO_TUNE_SEARCH_MODE = {"smoke": "fast", "normal": "standard"}
HEURISTIC_KEYS = {
    "baseline_bias_before",
    "baseline_bias_after",
    "baseline_bias_reduction",
    "low_freq_energy_ratio_before",
    "low_freq_energy_ratio_after",
    "low_freq_energy_reduction",
    "horizontal_coherence_before",
    "horizontal_coherence_after",
    "horizontal_coherence_reduction",
    "target_band_energy_ratio",
    "local_saliency_preservation",
    "edge_preservation",
    "deep_zone_contrast_before",
    "deep_zone_contrast_after",
    "deep_zone_contrast_gain",
    "clipping_ratio_after",
    "hot_pixel_ratio_after",
    "kurtosis_or_spikiness_after",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="AT-001 evidence output directory")
    parser.add_argument("--dataset", default="cylinder_single_v1", help="Fixture name, dataset dir, or CSV path")
    parser.add_argument("--mode", choices=["smoke", "normal"], default="smoke")
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)

    result = run_validation(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        mode=args.mode,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_validation(
    *,
    evidence_root: Path,
    dataset: str,
    mode: str,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Run AT-001 stepwise manual-vs-auto validation."""
    source_commit = source_commit or _git_rev_parse(ROOT)
    package = _load_dataset(dataset)
    evidence_root.mkdir(parents=True, exist_ok=True)
    figures_dir = evidence_root / "figures"
    tables_dir = evidence_root / "tables"
    reports_dir = evidence_root / "reports"
    manifests_dir = evidence_root / "manifests"
    for directory in (figures_dir, tables_dir, reports_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    impact_scan = _build_impact_scan(
        dataset_package=package,
        evidence_root=evidence_root,
        mode=mode,
    )
    raw = package["data"]
    header_info = package["header_info"]
    trace_metadata = package["trace_metadata"]
    ground_truth = package.get("ground_truth")
    metric_type = "ground_truth" if ground_truth else "heuristic_qc"
    search_mode = AUTO_TUNE_SEARCH_MODE[mode]

    input_png = figures_dir / "step_00_input.png"
    _save_bscan_png(raw, input_png, title="Input B-scan")

    manual = _run_branch(
        branch="manual",
        raw=raw,
        header_info=header_info,
        trace_metadata=trace_metadata,
        ground_truth=ground_truth,
        figures_dir=figures_dir,
        auto_tune=False,
        search_mode=search_mode,
    )
    automatic = _run_branch(
        branch="auto",
        raw=raw,
        header_info=header_info,
        trace_metadata=trace_metadata,
        ground_truth=ground_truth,
        figures_dir=figures_dir,
        auto_tune=True,
        search_mode=search_mode,
    )

    manual_png = figures_dir / "manual_bscan.png"
    auto_png = figures_dir / "auto_bscan.png"
    side_by_side_png = figures_dir / "side_by_side.png"
    _save_bscan_png(manual["result"], manual_png, title="Manual expert baseline")
    _save_bscan_png(automatic["result"], auto_png, title="Auto-tuned branch")
    _save_side_by_side(raw, manual["result"], automatic["result"], side_by_side_png)

    summary = _build_summary(
        package=package,
        mode=mode,
        metric_type=metric_type,
        ground_truth=ground_truth,
        manual=manual,
        automatic=automatic,
        impact_scan=impact_scan,
        source_repo=source_repo,
        source_branch=source_branch,
        source_commit=source_commit,
    )
    stepwise_report = {
        "schema": "mygpr_stepwise_report_v1",
        "step_index_convention": "one-based for processing steps; input preview is step_00",
        "metric_type": metric_type,
        "ground_truth_available": bool(ground_truth),
        "input_preview_png": _rel(input_png, evidence_root),
        "steps": manual["steps"] + automatic["steps"],
    }
    trial_table = _build_trial_table(automatic)
    evidence_manifest = _build_manifest(
        package=package,
        mode=mode,
        metric_type=metric_type,
        source_repo=source_repo,
        source_branch=source_branch,
        source_commit=source_commit,
        evidence_root=evidence_root,
        command=" ".join(["python", "scripts/auto_tune_validation/run_stepwise_validation.py"]),
        artifacts={
            "comparison_report": "reports/comparison_report.md",
            "comparison_summary": "manifests/comparison_summary.json",
            "stepwise_report": "manifests/stepwise_report.json",
            "trial_table_csv": "tables/trial_table.csv",
            "trial_table_json": "tables/trial_table.json",
            "manual_bscan": "figures/manual_bscan.png",
            "auto_bscan": "figures/auto_bscan.png",
            "side_by_side": "figures/side_by_side.png",
        },
        limitations=summary["limitations"],
        known_risks=summary["known_risks"],
    )

    _write_json(manifests_dir / "comparison_summary.json", summary)
    _write_json(manifests_dir / "stepwise_report.json", stepwise_report)
    _write_json(manifests_dir / "evidence_manifest.json", evidence_manifest)
    _write_json(tables_dir / "trial_table.json", trial_table)
    _write_trial_csv(tables_dir / "trial_table.csv", trial_table)
    (reports_dir / "comparison_report.md").write_text(
        _render_report(summary, stepwise_report),
        encoding="utf-8",
    )

    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "dataset_name": package["dataset_name"],
        "ground_truth_available": bool(ground_truth),
        "metric_type": metric_type,
        "manual_branch_invalid_reason": manual["branch_invalid_reason"],
        "auto_branch_invalid_reason": automatic["branch_invalid_reason"],
        "comparison_report": str((reports_dir / "comparison_report.md").resolve()),
    }


def _load_dataset(dataset: str) -> dict[str, Any]:
    path = DEFAULT_FIXTURE if dataset == "cylinder_single_v1" else Path(dataset)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if path.is_dir():
        csv_path = path / "mygpr_bscan.csv"
        scenario_path = path / "scenario.json"
        ground_truth_path = path / "ground_truth.json"
    else:
        csv_path = path
        scenario_path = path.with_name("scenario.json")
        ground_truth_path = path.with_name("ground_truth.json")
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    scenario = _read_json_optional(scenario_path)
    ground_truth = _read_json_optional(ground_truth_path)
    ground_truth = _normalize_ground_truth(ground_truth, data.shape)
    sim = scenario.get("simulation", {}) if isinstance(scenario, dict) else {}
    header_info = {
        "source": "auto_tune_validation_fixture",
        "data_context": "gprmax_impulse",
        "total_time_ns": float(sim.get("total_time_ns", data.shape[0])),
        "time_step_s": float(sim.get("time_step_s", 1.0e-9)),
        "trace_interval_m": float(sim.get("trace_step_m", 1.0)),
        "frequency_filter_policy": "model_or_auto_tune_only",
    }
    trace_distance = np.arange(data.shape[1], dtype=np.float32) * float(header_info["trace_interval_m"])
    return {
        "dataset_name": str((scenario or {}).get("scenario_id") or path.stem),
        "dataset_path": str(csv_path),
        "dataset_shape": [int(data.shape[0]), int(data.shape[1])],
        "dataset_hash": _sha256_file(csv_path),
        "data": data,
        "scenario": scenario,
        "ground_truth": ground_truth,
        "ground_truth_path": str(ground_truth_path) if ground_truth_path.exists() else None,
        "header_info": header_info,
        "trace_metadata": {
            "trace_index": np.arange(data.shape[1], dtype=np.int32),
            "trace_distance_m": trace_distance,
        },
    }


def _normalize_ground_truth(ground_truth: dict[str, Any] | None, shape: tuple[int, int]) -> dict[str, Any] | None:
    if not isinstance(ground_truth, dict):
        return None
    prepared = dict(ground_truth)
    prepared.setdefault(
        "analysis_roi",
        {
            "time_start_idx": 0,
            "time_end_idx": int(shape[0]),
            "dist_start_idx": 0,
            "dist_end_idx": int(shape[1]),
        },
    )
    return prepared


def _run_branch(
    *,
    branch: str,
    raw: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    ground_truth: dict[str, Any] | None,
    figures_dir: Path,
    auto_tune: bool,
    search_mode: str,
) -> dict[str, Any]:
    current = np.array(raw, copy=True)
    current_header = clone_header_info(header_info)
    current_trace_metadata = clone_trace_metadata(trace_metadata)
    steps: list[dict[str, Any]] = []
    params_by_method: dict[str, dict[str, Any]] = {}
    auto_tune_results: dict[str, dict[str, Any]] = {}
    branch_invalid_reason = ""
    all_sanity: list[str] = []
    for step_index, method_key in enumerate(DEFAULT_PIPELINE, start=1):
        before = np.array(current, copy=True)
        params = dict(MANUAL_EXPERT_PARAMS.get(method_key, {}))
        if auto_tune:
            try:
                tune_result = auto_tune_method(
                    current,
                    method_key,
                    header_info=current_header,
                    trace_metadata=current_trace_metadata,
                    base_params=params,
                    search_mode=search_mode,
                )
                recommended = tune_result.get("recommended_params") or tune_result.get("best_params") or {}
                params.update(dict(recommended))
                auto_tune_results[method_key] = _compact_auto_tune_result(tune_result)
            except AutoTuneError as exc:
                all_sanity.append(f"{method_key}: auto-tune failed and manual params were used: {exc}")
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
        heuristic_metrics = _common_heuristic_metrics(before, current)
        ground_truth_metrics = (
            compute_ground_truth_metrics(raw, current, ground_truth) if ground_truth else {}
        )
        sanity = _sanity_warnings(
            before=before,
            after=current,
            heuristic_metrics=heuristic_metrics,
            ground_truth_metrics=ground_truth_metrics,
        )
        runtime_warnings = _runtime_warning_codes(meta)
        all_sanity.extend(f"{method_key}: {item}" for item in sanity)
        if not branch_invalid_reason:
            branch_invalid_reason = _branch_invalid_reason(sanity)
        preview = figures_dir / f"step_{step_index:02d}_{branch}_{method_key}.png"
        _save_bscan_png(current, preview, title=f"{branch} step {step_index}: {method_key}")
        params_by_method[method_key] = params
        steps.append(
            {
                "branch": branch,
                "step_index": step_index,
                "method_key": method_key,
                "input_shape": [int(before.shape[0]), int(before.shape[1])],
                "output_shape": [int(current.shape[0]), int(current.shape[1])],
                "params": _json_safe(params),
                "runtime_warnings": runtime_warnings,
                "qc_metrics": {
                    "heuristic": _json_safe(heuristic_metrics),
                    "ground_truth": _json_safe(ground_truth_metrics) if ground_truth else {},
                },
                "sanity_warnings": sanity,
                "branch_invalid_reason": branch_invalid_reason,
                "preview_png": f"figures/{preview.name}",
            }
        )
    final_heuristic = _common_heuristic_metrics(raw, current)
    final_truth = compute_ground_truth_metrics(raw, current, ground_truth) if ground_truth else {}
    return {
        "branch": branch,
        "result": current,
        "steps": steps,
        "params_by_method": params_by_method,
        "auto_tune_results": auto_tune_results,
        "heuristic_metrics": final_heuristic,
        "ground_truth_metrics": final_truth,
        "sanity_warnings": sorted(set(all_sanity)),
        "branch_invalid_reason": branch_invalid_reason,
    }


def _common_heuristic_metrics(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    rows = min(before.shape[0], after.shape[0])
    cols = min(before.shape[1], after.shape[1])
    metrics = compute_benchmark_metrics(before[:rows, :cols], after[:rows, :cols])
    return {key: float(metrics[key]) for key in HEURISTIC_KEYS if key in metrics}


def _sanity_warnings(
    *,
    before: np.ndarray,
    after: np.ndarray,
    heuristic_metrics: dict[str, float],
    ground_truth_metrics: dict[str, float],
) -> list[str]:
    warnings: list[str] = []
    arr = np.asarray(after, dtype=np.float64)
    if not np.isfinite(arr).all():
        warnings.append("processed B-scan contains NaN/Inf")
    finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.std(finite)) < 1.0e-8:
        warnings.append("processed B-scan is near all-zero or constant")
    energy_ratio = _energy_ratio(before, after)
    if energy_ratio < 0.03:
        warnings.append("zero-time/effective signal energy nearly disappeared")
    if heuristic_metrics.get("target_band_energy_ratio", 1.0) < 0.25:
        warnings.append("target_band_fidelity is very low")
    if heuristic_metrics.get("local_saliency_preservation", 1.0) < 0.25:
        warnings.append("local_saliency_preservation collapsed")
    if heuristic_metrics.get("edge_preservation", 1.0) < 0.25:
        warnings.append("edge_preservation collapsed")
    if heuristic_metrics.get("clipping_ratio_after", 0.0) > 0.02:
        warnings.append("clipping_ratio is high")
    if heuristic_metrics.get("hot_pixel_ratio_after", 0.0) > 0.02:
        warnings.append("hot_pixel_ratio is high")
    if ground_truth_metrics.get("truth_target_energy_preservation", 1.0) < 0.25:
        warnings.append("ground-truth target energy preservation is very low")
    if before.shape != after.shape:
        warnings.append(f"shape changed from {before.shape} to {after.shape}; report must explain")
    return warnings


def _branch_invalid_reason(sanity: list[str]) -> str:
    severe_prefixes = (
        "processed B-scan contains",
        "processed B-scan is near",
        "zero-time/effective signal energy nearly disappeared",
        "target_band_fidelity is very low",
        "local_saliency_preservation collapsed",
        "edge_preservation collapsed",
        "ground-truth target energy preservation is very low",
    )
    for item in sanity:
        if item.startswith(severe_prefixes):
            return item
    return ""


def _build_summary(
    *,
    package: dict[str, Any],
    mode: str,
    metric_type: str,
    ground_truth: dict[str, Any] | None,
    manual: dict[str, Any],
    automatic: dict[str, Any],
    impact_scan: dict[str, Any],
    source_repo: str,
    source_branch: str,
    source_commit: str,
) -> dict[str, Any]:
    return {
        "artifact_id": "AT-001",
        "task_id": "AT-001_autotune_research_validation_baseline",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "mode": mode,
        "dataset": {
            "name": package["dataset_name"],
            "path": package["dataset_path"],
            "shape": package["dataset_shape"],
            "hash": package["dataset_hash"],
        },
        "pipeline": list(DEFAULT_PIPELINE),
        "metric_type": metric_type,
        "ground_truth_available": bool(ground_truth),
        "heuristic_qc_only": not bool(ground_truth),
        "manual": _branch_summary(manual),
        "auto": _branch_summary(automatic),
        "metric_delta_auto_minus_manual": {
            "heuristic": _metric_delta(manual["heuristic_metrics"], automatic["heuristic_metrics"]),
            "ground_truth": _metric_delta(manual["ground_truth_metrics"], automatic["ground_truth_metrics"]),
        },
        "impact_scan": impact_scan,
        "limitations": [
            "This baseline validates one small deterministic fixture first; it is not a global workflow optimum claim.",
            "AutoTune uses heuristic search during selection; ground truth is used only for validation/reporting.",
            "Manual baseline is an experience-based baseline, not a visual expert session with interactive retuning.",
        ],
        "known_risks": [
            "Single-scene evidence may not generalize to field UAV-GPR data.",
            "Heuristic QC can prefer visually cleaner images that are not geologically better.",
            "Zero-time choices can invalidate later branch comparisons if early signal is removed.",
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


def _build_manifest(
    *,
    package: dict[str, Any],
    mode: str,
    metric_type: str,
    source_repo: str,
    source_branch: str,
    source_commit: str,
    evidence_root: Path,
    command: str,
    artifacts: dict[str, str],
    limitations: list[str],
    known_risks: list[str],
) -> dict[str, Any]:
    return {
        "artifact_id": "AT-001",
        "task_id": "AT-001_autotune_research_validation_baseline",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": metric_type == "ground_truth",
        "metric_type": metric_type,
        "evidence_type": mode,
        "artifacts": artifacts,
        "limitations": limitations,
        "known_risks": known_risks,
    }


def _build_impact_scan(
    *,
    dataset_package: dict[str, Any],
    evidence_root: Path,
    mode: str,
) -> dict[str, Any]:
    return {
        "available_data": {
            "selected": dataset_package["dataset_name"],
            "path": dataset_package["dataset_path"],
            "shape": dataset_package["dataset_shape"],
            "ground_truth_available": bool(dataset_package.get("ground_truth")),
        },
        "selection_reason": "Small deterministic gprMax-style fixture with bundled B-scan and target ROI; suitable for reproducible smoke/normal evidence without large raw data.",
        "manual_baseline_source": "Reasonable experience parameters for gprMax impulse-like data: no zero-time crop by default, moderate dewow, model-frequency bandpass, trace-window background removal, and bounded energy-decay gain.",
        "validation_output_path": str(evidence_root),
        "files_expected_to_change": [
            "docs/auto_tune_research_validation.md",
            "scripts/auto_tune_validation/run_stepwise_validation.py",
            "tests/test_auto_tune_stepwise_validation_runner.py",
            "MyGPR-Evidence/autotune/AT-001_research_validation_baseline/*",
            "MyGPR-Evidence/ARTIFACT_INDEX.md",
        ],
        "modules_not_touched": [
            "PythonModule/motion_compensation_v2.py",
            "PythonModule/motion_compensation_core.py",
            "PythonModule/motion_compensation_height.py",
            "PythonModule/motion_compensation_speed.py",
            "PythonModule/motion_compensation_attitude.py",
            "PythonModule/trajectory_smoothing.py",
            "core/processing_engine.py",
            "core/auto_tune.py scoring logic",
        ],
        "mode": mode,
    }


def _build_trial_table(automatic: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_key, result in automatic.get("auto_tune_results", {}).items():
        for trial in result.get("all_trials", []) or []:
            rows.append(
                {
                    "method_key": method_key,
                    "stage": trial.get("stage", ""),
                    "score": trial.get("score"),
                    "valid": trial.get("valid", True),
                    "params": json.dumps(_json_safe(trial.get("params", {})), ensure_ascii=False, sort_keys=True),
                    "metrics": json.dumps(_json_safe(trial.get("metrics", {})), ensure_ascii=False, sort_keys=True),
                    "reason": trial.get("reason", ""),
                }
            )
    return rows


def _render_report(summary: dict[str, Any], stepwise_report: dict[str, Any]) -> str:
    manual = summary["manual"]
    auto = summary["auto"]
    lines = [
        "# AT-001 AutoTune Research Validation Baseline",
        "",
        "## Impact Scan",
        f"- Dataset: `{summary['dataset']['name']}`",
        f"- Shape: `{summary['dataset']['shape']}`",
        f"- Ground truth available: `{summary['ground_truth_available']}`",
        f"- Metric type: `{summary['metric_type']}`",
        f"- Evidence mode: `{summary['mode']}`",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Output path: `{summary['impact_scan']['validation_output_path']}`",
        "",
        "## Scope Boundary",
        "- This run compares a reasonable manual expert baseline against per-step AutoTune parameters.",
        "- It does not claim AutoTune selects the global best workflow.",
        "- It does not use intentionally bad manual parameters.",
        "- Ground truth is used only for validation/reporting, not for AutoTune search.",
        "",
        "## Manual Baseline Parameters",
        "```json",
        json.dumps(manual["params_by_method"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Auto-Tuned Parameters",
        "```json",
        json.dumps(auto["params_by_method"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Final Metrics",
        "| Branch | Branch valid | Heuristic comparison_score proxy | truth_score | Branch invalid reason |",
        "|---|---:|---:|---:|---|",
        _metric_row("manual", manual),
        _metric_row("auto", auto),
        "",
        "## Stepwise Sanity Table",
        "| Branch | Step | Method | Output shape | Runtime warnings | Sanity warnings | Invalid reason | Preview |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for step in stepwise_report["steps"]:
        lines.append(
            "| {branch} | {idx} | `{method}` | `{shape}` | {runtime} | {sanity} | {invalid} | `{preview}` |".format(
                branch=step["branch"],
                idx=step["step_index"],
                method=step["method_key"],
                shape=step["output_shape"],
                runtime="<br>".join(step["runtime_warnings"]) or "-",
                sanity="<br>".join(step["sanity_warnings"]) or "-",
                invalid=step["branch_invalid_reason"] or "-",
                preview=step["preview_png"],
            )
        )
    lines.extend(
        [
            "",
            "## Branch Validity",
            f"- Manual branch invalid reason: `{manual['branch_invalid_reason'] or 'none'}`",
            f"- Auto branch invalid reason: `{auto['branch_invalid_reason'] or 'none'}`",
            "",
            "If a branch has an invalid reason, this evidence must not be used as a fair manual-vs-auto conclusion.",
            "",
            "## Ground Truth And Metric Boundary",
            f"- Ground truth available: `{summary['ground_truth_available']}`",
            f"- Heuristic QC only: `{summary['heuristic_qc_only']}`",
            "- Heuristic QC metrics and ground-truth metrics are stored separately in `comparison_summary.json` and `stepwise_report.json`.",
            "- If ground truth is unavailable, reports must not claim the result is closer to the real subsurface structure.",
            "",
            "## Current Paper Claim Boundary",
            "- Can claim: the AT-001 runner creates reproducible stepwise evidence for manual-vs-auto validation and exposes early baseline failures.",
            "- Cannot claim: AutoTune is globally optimal, field-general, or truth-guided.",
            "",
            "## Limitations",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend(["", "## Known Risks"])
    lines.extend(f"- {item}" for item in summary["known_risks"])
    lines.append("")
    return "\n".join(lines)


def _metric_row(label: str, branch: dict[str, Any]) -> str:
    heuristic = branch.get("heuristic_metrics", {})
    truth = branch.get("ground_truth_metrics", {})
    proxy = heuristic.get("target_band_energy_ratio", "")
    truth_score = truth.get("truth_score", "")
    valid = "no" if branch.get("branch_invalid_reason") else "yes"
    return f"| {label} | {valid} | {_fmt(proxy)} | {_fmt(truth_score)} | {branch.get('branch_invalid_reason') or '-'} |"


def _save_bscan_png(data: np.ndarray, path: Path, *, title: str) -> None:
    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.percentile(np.abs(finite), 98.0)) if finite.size else 1.0
    vmax = max(vmax, 1.0e-6)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    try:
        ax.imshow(arr, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_side_by_side(raw: np.ndarray, manual: np.ndarray, auto: np.ndarray, path: Path) -> None:
    panels = [("Raw", raw), ("Manual", manual), ("AutoTune", auto)]
    finite = np.concatenate([np.ravel(np.asarray(arr, dtype=np.float32)) for _, arr in panels])
    finite = finite[np.isfinite(finite)]
    vmax = float(np.percentile(np.abs(finite), 98.0)) if finite.size else 1.0
    vmax = max(vmax, 1.0e-6)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), dpi=150)
    try:
        for ax, (title, arr) in zip(axes, panels):
            ax.imshow(arr, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Trace")
        axes[0].set_ylabel("Sample")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _write_trial_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["method_key", "stage", "score", "valid", "params", "metrics", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_warning_codes(meta: dict[str, Any]) -> list[str]:
    merged = merge_runtime_warnings(meta.get("runtime_warnings") or [])
    codes = []
    for item in merged:
        if isinstance(item, dict):
            codes.append(str(item.get("code") or item.get("message") or item))
        else:
            codes.append(str(item))
    return codes


def _compact_auto_tune_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_key": result.get("method_key"),
        "best_params": _json_safe(result.get("best_params", {})),
        "recommended_params": _json_safe(result.get("recommended_params", {})),
        "best_score": result.get("best_score"),
        "best_metrics": _json_safe(result.get("best_metrics", {})),
        "risk_flags": _json_safe(result.get("risk_flags", [])),
        "parameter_domain": _json_safe(result.get("parameter_domain", {})),
        "all_trials": _json_safe(result.get("all_trials", [])),
    }


def _metric_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(left) | set(right))
    return {
        key: float(right.get(key, 0.0)) - float(left.get(key, 0.0))
        for key in keys
        if _is_number(left.get(key, 0.0)) and _is_number(right.get(key, 0.0))
    }


def _energy_ratio(before: np.ndarray, after: np.ndarray) -> float:
    rows = min(before.shape[0], after.shape[0])
    cols = min(before.shape[1], after.shape[1])
    b = np.asarray(before[:rows, :cols], dtype=np.float64)
    a = np.asarray(after[:rows, :cols], dtype=np.float64)
    return float(np.mean(a * a) / max(np.mean(b * b), 1.0e-12))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev_parse(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}" if _is_number(value) else "-"


def _is_number(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic-only background suppression AutoTune harness utilities."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import svd
from scipy.ndimage import median_filter, uniform_filter1d


CLAIM_BOUNDARY = (
    "diagnostic-only background suppression autotune; not production scoring; "
    "not AutoTune superiority evidence; not field ground-truth correctness evidence"
)


@dataclass
class CandidateSpec:
    """Background suppression candidate specification."""

    method: str
    parameter_set: dict[str, Any]
    candidate_group: str


@dataclass
class TrialResult:
    """Single trial evaluation result."""

    trial_id: str
    artifact_id: str
    scene_id: str
    method: str
    parameter_set: dict[str, Any]
    candidate_group: str
    processed_output_path: str | None
    metrics_schema: str
    mae: float | None
    mse: float | None
    rmse: float | None
    psnr: float | None
    roi_energy_ratio: float | None
    outside_roi_clutter_proxy: float | None
    target_distortion_warning: bool
    false_enhancement_warning: bool
    runtime_seconds: float
    warnings: list[dict[str, str]]
    selected: bool
    recommendation_label: str
    claim_boundary: str
    sort_key: list[float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("sort_key", None)
        return payload


def load_csv_2d(path: str | Path) -> np.ndarray:
    """Load CSV as a 2D float64 array, preserving single-column shape."""
    arr = np.genfromtxt(Path(path), delimiter=",", dtype=np.float64, ndmin=2)
    if arr.ndim != 2:
        raise ValueError(f"CSV must resolve to 2D array: path={path}, shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"CSV array is empty: path={path}")
    return arr


def load_roi_json(path: str | Path) -> dict[str, Any]:
    """Load optional ROI json payload."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ROI JSON must be object: {path}")
    return payload


def default_candidate_grid() -> list[CandidateSpec]:
    """Return AT-BG-001 v1 reproducible candidate grid."""
    windows = [5, 9, 15, 21, 31, 41]
    candidates: list[CandidateSpec] = [
        CandidateSpec(
            method="mean_background_subtraction",
            parameter_set={"mode": "global_mean", "axis": "trace"},
            candidate_group="mean",
        ),
        CandidateSpec(
            method="median_background_subtraction",
            parameter_set={"mode": "global_median", "axis": "trace"},
            candidate_group="median",
        ),
    ]
    candidates.extend(
        CandidateSpec(
            method="mean_background_subtraction",
            parameter_set={
                "mode": "moving_window_mean",
                "window_size": int(window_size),
                "axis": "trace",
            },
            candidate_group="mean",
        )
        for window_size in windows
    )
    candidates.extend(
        CandidateSpec(
            method="median_background_subtraction",
            parameter_set={
                "mode": "moving_window_median",
                "window_size": int(window_size),
                "axis": "trace",
            },
            candidate_group="median",
        )
        for window_size in windows
    )
    candidates.extend(
        CandidateSpec(
            method="svd_background_suppression",
            parameter_set={"remove_rank": int(rank)},
            candidate_group="svd",
        )
        for rank in (1, 2, 3)
    )
    return candidates


def parse_candidate_config(path: str | Path | None) -> list[CandidateSpec]:
    """Parse optional candidate config JSON."""
    if not path:
        return default_candidate_grid()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate-config JSON must be an object")
    entries = payload.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ValueError("candidate-config must contain non-empty candidates list")
    candidates: list[CandidateSpec] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"candidate[{idx}] must be object")
        method = str(item.get("method") or "").strip()
        if not method:
            raise ValueError(f"candidate[{idx}] missing method")
        params = item.get("parameter_set", {})
        if not isinstance(params, dict):
            raise ValueError(f"candidate[{idx}].parameter_set must be object")
        group = str(item.get("candidate_group") or method).strip()
        candidates.append(
            CandidateSpec(method=method, parameter_set=params, candidate_group=group)
        )
    return candidates


def run_background_suppression_diagnostic(
    *,
    raw: np.ndarray,
    target_response: np.ndarray,
    output_dir: str | Path,
    artifact_id: str = "unknown_artifact",
    scene_id: str = "unknown_scene",
    roi: dict[str, Any] | None = None,
    candidate_specs: list[CandidateSpec] | None = None,
    write_arrays: bool = False,
    max_preview_candidates: int | None = None,
    input_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run diagnostic-only AT-BG candidate trials and write report artifacts."""
    raw = np.asarray(raw, dtype=np.float64)
    target_response = np.asarray(target_response, dtype=np.float64)
    if raw.ndim != 2 or target_response.ndim != 2:
        raise ValueError("raw and target_response must be 2D arrays")
    if raw.shape != target_response.shape:
        raise ValueError(
            f"raw and target_response shape mismatch: raw={raw.shape}, target_response={target_response.shape}"
        )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    candidates = candidate_specs or default_candidate_grid()

    trial_results: list[TrialResult] = []
    for idx, candidate in enumerate(candidates, start=1):
        trial_id = f"trial_{idx:03d}"
        trial = _evaluate_candidate(
            trial_id=trial_id,
            candidate=candidate,
            raw=raw,
            target_response=target_response,
            roi=roi,
            artifact_id=artifact_id,
            scene_id=scene_id,
            output_dir=output_path,
            write_arrays=write_arrays,
        )
        trial_results.append(trial)

    ranked = sorted(
        trial_results,
        key=lambda item: (0 if item.recommendation_label.startswith("rejected") else 1),
        reverse=True,
    )
    ranked = sorted(
        ranked,
        key=lambda item: item.sort_key,
    )
    _assign_labels_and_selection(ranked)

    table_json = output_path / "trial_table.json"
    table_csv = output_path / "trial_table.csv"
    selected_json = output_path / "selected_parameters.json"
    report_md = output_path / "background_suppression_autotune_report.md"
    manifest_json = output_path / "background_suppression_autotune_manifest.json"

    table_payload = [item.to_dict() for item in ranked]
    table_json.write_text(json.dumps(table_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_trial_table_csv(table_csv, table_payload)

    selected_trial = next((item for item in ranked if item.selected), None)
    selected_payload = {
        "artifact_id": artifact_id,
        "scene_id": scene_id,
        "selected": selected_trial.to_dict() if selected_trial else None,
        "scoring_rule": (
            "primary mae asc, tie rmse asc, then psnr desc, with warning penalties and optional ROI preservation penalty"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    selected_json.write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_md.write_text(
        _build_markdown_report(
            artifact_id=artifact_id,
            scene_id=scene_id,
            input_paths=input_paths or {},
            candidates=candidates,
            ranked=ranked,
            selected=selected_trial,
            scoring_rule=selected_payload["scoring_rule"],
        ),
        encoding="utf-8",
    )

    manifest_payload = {
        "source_commit": _get_source_commit(),
        "script_name": "scripts/autotune_background_suppression_diagnostic.py",
        "artifact_id": artifact_id,
        "scene_id": scene_id,
        "input_paths": input_paths or {},
        "output_files": [
            str(table_json),
            str(table_csv),
            str(selected_json),
            str(report_md),
            str(manifest_json),
        ],
        "candidate_grid": [asdict(item) for item in candidates],
        "scoring_rule": selected_payload["scoring_rule"],
        "claim_boundary": CLAIM_BOUNDARY,
        "diagnostic_only": True,
        "write_arrays": bool(write_arrays),
        "max_preview_candidates": max_preview_candidates,
    }
    manifest_json.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "artifact_id": artifact_id,
        "scene_id": scene_id,
        "trial_count": len(ranked),
        "selected_parameters_path": str(selected_json),
        "trial_table_json_path": str(table_json),
        "trial_table_csv_path": str(table_csv),
        "report_path": str(report_md),
        "manifest_path": str(manifest_json),
        "selected_trial_id": selected_trial.trial_id if selected_trial else None,
    }


def _evaluate_candidate(
    *,
    trial_id: str,
    candidate: CandidateSpec,
    raw: np.ndarray,
    target_response: np.ndarray,
    roi: dict[str, Any] | None,
    artifact_id: str,
    scene_id: str,
    output_dir: Path,
    write_arrays: bool,
) -> TrialResult:
    start = time.perf_counter()
    warnings: list[dict[str, str]] = []
    processed_path: str | None = None
    label = "manual_review_recommended"
    mae = mse = rmse = psnr = None
    roi_ratio = None
    outside_proxy = None
    target_distortion_warning = False
    false_enhancement_warning = False
    sort_key = [math.inf, math.inf, math.inf, 1.0]

    try:
        processed = _apply_candidate(raw, candidate)
        if processed.shape != target_response.shape:
            raise ValueError(
                f"processed shape mismatch: processed={processed.shape}, target_response={target_response.shape}"
            )
        _append_nan_inf_warnings("processed_bscan", processed, warnings)
        _append_nan_inf_warnings("target_response", target_response, warnings)

        diff = processed - target_response
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(np.square(diff)))
        rmse = float(np.sqrt(mse))
        psnr = _compute_psnr(processed, target_response, mse, warnings)

        processed_energy = float(np.sum(np.square(processed)))
        target_energy = float(np.sum(np.square(target_response)))
        target_to_processed = _safe_ratio(
            target_energy, processed_energy, "target_to_processed_energy_ratio", warnings
        )

        if roi is not None:
            roi_ratio, outside_proxy = _compute_roi_metrics(
                processed=processed,
                target_response=target_response,
                raw=raw,
                roi=roi,
                warnings=warnings,
            )
            if roi_ratio is not None and roi_ratio < 0.5:
                target_distortion_warning = True
                warnings.append(
                    {
                        "code": "roi_preservation_low",
                        "message": f"ROI preservation ratio is low: {roi_ratio:.4f}",
                    }
                )
            if outside_proxy is not None and outside_proxy > 1.05:
                false_enhancement_warning = True
                warnings.append(
                    {
                        "code": "outside_roi_energy_increase",
                        "message": f"Outside-ROI energy increased: ratio={outside_proxy:.4f}",
                    }
                )

        if target_to_processed is not None and target_to_processed < 0.1:
            target_distortion_warning = True
            warnings.append(
                {
                    "code": "target_energy_loss_risk",
                    "message": "Target energy ratio is low after suppression.",
                }
            )

        if write_arrays:
            arr_path = output_dir / f"{trial_id}_processed.csv"
            np.savetxt(arr_path, processed, delimiter=",")
            processed_path = str(arr_path)

        warning_penalty = float(len(warnings)) * 1.0e-6
        if target_distortion_warning:
            warning_penalty += 1.0e-4
        if false_enhancement_warning:
            warning_penalty += 1.0e-4
        psnr_term = -float(psnr) if psnr is not None else 0.0
        sort_key = [mae + warning_penalty, rmse + warning_penalty, psnr_term, warning_penalty]
        label = "acceptable_alternative"
    except Exception as exc:
        warnings.append({"code": "trial_runtime_failure", "message": str(exc)})
        label = "rejected_shape_or_runtime_failure"
        target_distortion_warning = True

    runtime_seconds = float(time.perf_counter() - start)
    return TrialResult(
        trial_id=trial_id,
        artifact_id=artifact_id,
        scene_id=scene_id,
        method=candidate.method,
        parameter_set=candidate.parameter_set,
        candidate_group=candidate.candidate_group,
        processed_output_path=processed_path,
        metrics_schema="synthetic_paired_metrics_v1",
        mae=mae,
        mse=mse,
        rmse=rmse,
        psnr=psnr,
        roi_energy_ratio=roi_ratio,
        outside_roi_clutter_proxy=outside_proxy,
        target_distortion_warning=target_distortion_warning,
        false_enhancement_warning=false_enhancement_warning,
        runtime_seconds=runtime_seconds,
        warnings=warnings,
        selected=False,
        recommendation_label=label,
        claim_boundary=CLAIM_BOUNDARY,
        sort_key=sort_key,
    )


def _apply_candidate(raw: np.ndarray, candidate: CandidateSpec) -> np.ndarray:
    method = candidate.method.strip().lower()
    params = candidate.parameter_set
    if method == "mean_background_subtraction":
        mode = str(params.get("mode", "global_mean")).strip().lower()
        if mode == "global_mean":
            background = np.mean(raw, axis=1, keepdims=True)
        elif mode == "moving_window_mean":
            window = _normalize_window(params.get("window_size", 5))
            background = uniform_filter1d(raw, size=window, axis=1, mode="nearest")
        else:
            raise ValueError(f"Unsupported mean mode: {mode}")
        return (raw - background).astype(np.float64, copy=False)
    if method == "median_background_subtraction":
        mode = str(params.get("mode", "global_median")).strip().lower()
        if mode == "global_median":
            background = np.median(raw, axis=1, keepdims=True)
        elif mode == "moving_window_median":
            window = _normalize_window(params.get("window_size", 5))
            background = median_filter(raw, size=(1, window), mode="nearest")
        else:
            raise ValueError(f"Unsupported median mode: {mode}")
        return (raw - background).astype(np.float64, copy=False)
    if method == "svd_background_suppression":
        rank = int(params.get("remove_rank", 1))
        rank = max(1, min(rank, min(raw.shape)))
        u, s, vt = svd(raw, full_matrices=False, check_finite=False)
        s_bg = np.zeros_like(s)
        s_bg[:rank] = s[:rank]
        background = (u * s_bg) @ vt
        return (raw - background).astype(np.float64, copy=False)
    raise ValueError(f"Unsupported candidate method: {candidate.method}")


def _normalize_window(value: Any) -> int:
    window = max(1, int(value))
    if window % 2 == 0:
        window += 1
    return window


def _append_nan_inf_warnings(name: str, arr: np.ndarray, warnings: list[dict[str, str]]) -> None:
    if not np.isfinite(arr).all():
        warnings.append(
            {
                "code": f"{name}_nan_or_inf",
                "message": f"{name} contains NaN or Inf values.",
            }
        )


def _safe_ratio(
    numerator: float, denominator: float, code: str, warnings: list[dict[str, str]]
) -> float | None:
    if denominator == 0.0:
        warnings.append(
            {
                "code": f"{code}_denominator_zero",
                "message": f"{code} denominator is zero.",
            }
        )
        return None
    return float(numerator / denominator)


def _compute_psnr(
    processed: np.ndarray,
    target: np.ndarray,
    mse: float,
    warnings: list[dict[str, str]],
) -> float | None:
    if mse == 0.0:
        warnings.append(
            {
                "code": "processed_target_psnr_mse_zero",
                "message": "processed vs target MSE is zero; PSNR is mathematically infinite.",
            }
        )
        return None
    peak = float(max(np.max(np.abs(processed)), np.max(np.abs(target))))
    if peak == 0.0:
        warnings.append(
            {
                "code": "processed_target_psnr_peak_zero",
                "message": "processed/target peak is zero; PSNR undefined.",
            }
        )
        return None
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))


def _compute_roi_metrics(
    *,
    processed: np.ndarray,
    target_response: np.ndarray,
    raw: np.ndarray,
    roi: dict[str, Any],
    warnings: list[dict[str, str]],
) -> tuple[float | None, float | None]:
    try:
        sample_range = roi.get("sample_range")
        trace_range = roi.get("trace_range")
        if not (
            isinstance(sample_range, list)
            and len(sample_range) == 2
            and isinstance(trace_range, list)
            and len(trace_range) == 2
        ):
            raise ValueError("ROI must include sample_range and trace_range")
        s0, s1 = int(sample_range[0]), int(sample_range[1])
        t0, t1 = int(trace_range[0]), int(trace_range[1])
        n_samples, n_traces = target_response.shape
        if not (0 <= s0 < s1 <= n_samples and 0 <= t0 < t1 <= n_traces):
            raise ValueError(
                f"ROI out of bounds for shape {target_response.shape}: sample={sample_range}, trace={trace_range}"
            )
    except Exception as exc:
        warnings.append({"code": "roi_invalid", "message": str(exc)})
        return None, None

    roi_processed_energy = float(np.sum(np.square(processed[s0:s1, t0:t1])))
    roi_target_energy = float(np.sum(np.square(target_response[s0:s1, t0:t1])))
    roi_ratio = _safe_ratio(
        roi_processed_energy, roi_target_energy, "roi_energy_ratio", warnings
    )

    mask = np.ones(target_response.shape, dtype=bool)
    mask[s0:s1, t0:t1] = False
    outside_processed = float(np.sum(np.square(processed[mask])))
    outside_raw = float(np.sum(np.square(raw[mask])))
    outside_ratio = _safe_ratio(
        outside_processed,
        outside_raw,
        "outside_roi_clutter_proxy",
        warnings,
    )
    return roi_ratio, outside_ratio


def _assign_labels_and_selection(ranked: list[TrialResult]) -> None:
    if not ranked:
        return
    accepted = [
        item
        for item in ranked
        if item.recommendation_label != "rejected_shape_or_runtime_failure"
    ]
    if not accepted:
        return
    best = accepted[0]
    best.selected = True
    if best.target_distortion_warning:
        best.recommendation_label = "manual_review_recommended"
    else:
        best.recommendation_label = "recommended"
    for item in accepted[1:]:
        if item.target_distortion_warning:
            item.recommendation_label = "rejected_over_suppression"
        else:
            item.recommendation_label = "acceptable_alternative"


def _write_trial_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _build_markdown_report(
    *,
    artifact_id: str,
    scene_id: str,
    input_paths: dict[str, str],
    candidates: list[CandidateSpec],
    ranked: list[TrialResult],
    selected: TrialResult | None,
    scoring_rule: str,
) -> str:
    top = ranked[:5]
    lines = [
        "# Background Suppression AutoTune Diagnostic Report",
        "",
        f"- artifact_id: `{artifact_id}`",
        f"- scene_id: `{scene_id}`",
        f"- trial_count: `{len(ranked)}`",
        f"- scoring_rule: {scoring_rule}",
        "- diagnostic_scope: background suppression only",
        "",
        "## Input Paths",
        "",
        f"- raw: `{input_paths.get('raw', '')}`",
        f"- target_response: `{input_paths.get('target_response', '')}`",
        f"- roi_json: `{input_paths.get('roi_json', '')}`",
        "",
        "## Candidate Grid",
        "",
    ]
    for cand in candidates:
        lines.append(
            f"- {cand.method} | group={cand.candidate_group} | params={json.dumps(cand.parameter_set, ensure_ascii=False)}"
        )
    lines.extend(
        [
            "",
            "## Top Ranked Candidates",
            "",
        ]
    )
    for item in top:
        lines.append(
            f"- {item.trial_id}: method={item.method}, params={json.dumps(item.parameter_set, ensure_ascii=False)}, "
            f"label={item.recommendation_label}, mae={item.mae}, rmse={item.rmse}, psnr={item.psnr}"
        )
    lines.extend(["", "## Selected Parameter", ""])
    if selected is None:
        lines.append("- No valid candidate selected.")
    else:
        lines.append(
            f"- selected_trial: `{selected.trial_id}` ({selected.method}) {json.dumps(selected.parameter_set, ensure_ascii=False)}"
        )
    lines.extend(
        [
            "",
            "## Warnings Summary",
            "",
        ]
    )
    warning_count = sum(len(item.warnings) for item in ranked)
    lines.append(f"- total_warnings: `{warning_count}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This harness is diagnostic-only and not production AutoTune scoring.",
            "- This output is not AutoTune superiority evidence.",
            "- This output is not field no-prior underground correctness evidence.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _get_source_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    commit = proc.stdout.strip()
    return commit or None


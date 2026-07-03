#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research artifact export for manual-baseline vs auto-tune comparisons."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.auto_tune_comparison import AutoTuneComparisonRun, to_summary_dict
from core.scalar_utils import to_float


def export_auto_tune_comparison_artifacts(
    result: AutoTuneComparisonRun,
    *,
    out_dir: str | Path,
    bundle_name: str | None = None,
    input_ref: str | None = None,
    notes: list[str] | None = None,
    cmap: str = "gray",
) -> dict[str, Any]:
    """Export a complete manual-vs-auto research evidence bundle."""
    output_root = Path(out_dir) / _safe_bundle_name(bundle_name)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_bundle_name(bundle_name)

    paths = {
        "summary_json": output_root / "comparison_summary.json",
        "evidence_manifest_json": output_root / "evidence_manifest.json",
        "converted_ground_truth_json": output_root / "converted_ground_truth.json",
        "raw_ground_truth_json": output_root / "raw_ground_truth.json",
        "truth_metrics_json": output_root / "truth_metrics.json",
        "workflow_params_json": output_root / "workflow_params.json",
        "trial_table_csv": output_root / "trial_table.csv",
        "trial_table_json": output_root / "trial_table.json",
        "manual_png": output_root / "manual_bscan.png",
        "auto_png": output_root / "auto_bscan.png",
        "side_by_side_png": output_root / "side_by_side.png",
        "params_csv": output_root / "params_table.csv",
        "metrics_csv": output_root / "metrics_table.csv",
        "report_md": output_root / "comparison_report.md",
        "evidence_zip": output_root / "evidence_bundle.zip",
    }

    manual_arr = np.asarray(result.manual.result, dtype=np.float32)
    auto_arr = np.asarray(result.automatic.result, dtype=np.float32)
    display_spec = _locked_display_spec(
        manual_arr,
        auto_arr,
        result.display_spec,
        cmap=cmap,
    )

    _save_single_bscan(
        manual_arr,
        paths["manual_png"],
        title="Manual baseline",
        display_spec=display_spec,
    )
    _save_single_bscan(
        auto_arr,
        paths["auto_png"],
        title="Auto-tuned",
        display_spec=display_spec,
    )
    _save_side_by_side(
        manual_arr,
        auto_arr,
        paths["side_by_side_png"],
        display_spec=display_spec,
    )

    summary = to_summary_dict(result)
    summary["input_ref"] = input_ref
    summary["notes"] = list(notes or [])
    summary["display_spec"] = {
        **dict(summary.get("display_spec") or {}),
        **display_spec,
    }
    summary["exported_at"] = datetime.now().isoformat(timespec="seconds")
    summary["artifacts"] = {
        key: str(path.resolve())
        for key, path in paths.items()
        if key not in {"evidence_manifest_json", "evidence_zip"}
    }

    _write_csv_rows(
        paths["params_csv"],
        _build_params_rows(summary),
        fieldnames=[
            "candidate",
            "source",
            "method_key",
            "stage_index",
            "param_name",
            "param_value",
        ],
    )
    _write_csv_rows(
        paths["metrics_csv"],
        _build_metrics_rows(summary),
        fieldnames=[
            "metric",
            "manual_value",
            "auto_value",
            "delta",
        ],
    )
    trial_rows, trial_payload, trial_warnings = _build_trial_table(summary)
    _write_csv_rows(
        paths["trial_table_csv"],
        trial_rows,
        fieldnames=[
            "branch",
            "method_key",
            "trial_index",
            "selected",
            "params_json",
            "score",
            "comparison_score",
            "truth_score",
            "truth_target_energy_preservation",
            "truth_target_saliency_gain",
            "truth_background_energy_reduction",
            "truth_false_positive_ratio",
            "candidate_space_hash",
            "candidate_space_profile_id",
            "candidate_space_config_version",
            "candidate_space_recipe_ids_json",
            "candidate_id",
            "candidate_source",
            "candidate_group",
            "candidate_parameters_json",
            "scoring_boundary",
            "manual_review_required",
            "score_version",
            "reason",
            "warnings_json",
        ],
    )
    paths["trial_table_json"].write_text(
        json.dumps(_json_safe(trial_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary.setdefault("evidence_warnings", [])
    summary["evidence_warnings"].extend(trial_warnings)
    paths["report_md"].write_text(
        _build_report_markdown(summary),
        encoding="utf-8",
    )
    ground_truth = getattr(result, "ground_truth", None)
    if isinstance(ground_truth, dict):
        paths["converted_ground_truth_json"].write_text(
            json.dumps(_json_safe(ground_truth), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raw_sidecar = ground_truth.get("raw_sidecar")
        if isinstance(raw_sidecar, dict):
            paths["raw_ground_truth_json"].write_text(
                json.dumps(_json_safe(raw_sidecar), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    paths["truth_metrics_json"].write_text(
        json.dumps(_build_truth_metrics(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["workflow_params_json"].write_text(
        json.dumps(_build_workflow_params(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["summary_json"].write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["evidence_zip"].touch()
    manifest = _build_evidence_manifest(
        summary,
        paths,
        output_root=output_root,
        input_ref=input_ref,
        notes=notes or [],
    )
    paths["evidence_manifest_json"].write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_evidence_zip(paths["evidence_zip"], paths, output_root)

    return _json_safe(
        {
            "bundle_name": safe_name,
            "output_dir": str(output_root.resolve()),
            "artifacts": {
                key: str(path.resolve()) for key, path in paths.items() if path.exists()
            },
            "summary": summary,
        }
    )


def _safe_bundle_name(bundle_name: str | None) -> str:
    raw = str(bundle_name or datetime.now().strftime("auto_tune_comparison_%Y%m%d_%H%M%S"))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw).strip("._-")
    return safe or "auto_tune_comparison"


def _locked_display_spec(
    manual_arr: np.ndarray,
    auto_arr: np.ndarray,
    source_spec: dict[str, Any] | None,
    *,
    cmap: str,
) -> dict[str, Any]:
    source = dict(source_spec or {})
    clip = source.get("percentile_clip")
    finite_abs = _finite_abs_values(manual_arr, auto_arr)
    if finite_abs.size == 0:
        limit = 1.0
    elif clip is not None:
        percentile = max(0.0, min(to_float(clip, default=100.0), 100.0))
        limit = float(np.percentile(finite_abs, percentile))
    else:
        limit = float(np.max(finite_abs))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    return {
        "locked_scale": True,
        "lock_color_scale": True,
        "normalize": False,
        "percentile_clip": clip,
        "cmap": str(cmap or "gray"),
        "vmin": -limit,
        "vmax": limit,
    }


def _finite_abs_values(*arrays: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for arr in arrays:
        values = np.ravel(np.asarray(arr, dtype=np.float32))
        if values.size == 0:
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(np.abs(finite.astype(np.float64, copy=False)))
    if not chunks:
        return np.asarray([], dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _save_single_bscan(
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    display_spec: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=150)
    try:
        image = ax.imshow(
            np.asarray(data, dtype=np.float32),
            cmap=str(display_spec["cmap"]),
            aspect="auto",
            vmin=float(display_spec["vmin"]),
            vmax=float(display_spec["vmax"]),
        )
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        fig.colorbar(image, ax=ax, shrink=0.82)
        fig.tight_layout()
        fig.savefig(out_path)
    finally:
        plt.close(fig)


def _save_side_by_side(
    manual_data: np.ndarray,
    auto_data: np.ndarray,
    out_path: Path,
    *,
    display_spec: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=150, constrained_layout=True)
    try:
        for ax, arr, title in [
            (axes[0], manual_data, "Manual baseline"),
            (axes[1], auto_data, "Auto-tuned"),
        ]:
            image = ax.imshow(
                np.asarray(arr, dtype=np.float32),
                cmap=str(display_spec["cmap"]),
                aspect="auto",
                vmin=float(display_spec["vmin"]),
                vmax=float(display_spec["vmax"]),
            )
            ax.set_title(title)
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
        fig.savefig(out_path)
    finally:
        plt.close(fig)


def _build_params_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_key, candidate_label in [
        ("manual", "manual"),
        ("automatic", "automatic"),
    ]:
        candidate = summary.get(candidate_key) or {}
        params_by_method = candidate.get("params_by_method") or {}
        pipeline = list(candidate.get("pipeline") or params_by_method.keys())
        source = str(candidate.get("source") or candidate_label)
        for stage_index, method_key in enumerate(pipeline, start=1):
            params = params_by_method.get(method_key) or {}
            if not params:
                rows.append(
                    {
                        "candidate": candidate_label,
                        "source": source,
                        "method_key": method_key,
                        "stage_index": stage_index,
                        "param_name": "",
                        "param_value": "",
                    }
                )
                continue
            for param_name, value in sorted(params.items()):
                rows.append(
                    {
                        "candidate": candidate_label,
                        "source": source,
                        "method_key": method_key,
                        "stage_index": stage_index,
                        "param_name": param_name,
                        "param_value": _csv_value(value),
                    }
                )
    return rows


def _build_metrics_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    manual_metrics = ((summary.get("manual") or {}).get("metrics") or {})
    auto_metrics = ((summary.get("automatic") or {}).get("metrics") or {})
    metric_delta = summary.get("metric_delta") or {}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(manual_metrics) | set(auto_metrics)):
        rows.append(
            {
                "metric": str(key),
                "manual_value": _csv_value(manual_metrics.get(key)),
                "auto_value": _csv_value(auto_metrics.get(key)),
                "delta": _csv_value(metric_delta.get(key)),
            }
        )
    return rows


def _build_trial_table(
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings_list: list[str] = []
    evidence_context = _build_autotune_v1_evidence_context(summary)
    payload: dict[str, Any] = {
        "schema": "mygpr_autotune_trial_table_v1",
        "methods": {},
        "autotune_v1_evidence": evidence_context,
    }
    automatic = summary.get("automatic") or {}
    manual = summary.get("manual") or {}
    manual_params = manual.get("params_by_method") or {}
    auto_params = automatic.get("params_by_method") or {}
    auto_results = automatic.get("auto_tune_results") or {}
    pipeline = list(manual.get("pipeline") or automatic.get("pipeline") or [])
    if not pipeline:
        pipeline = sorted(set(manual_params) | set(auto_params) | set(auto_results))

    for method_key in pipeline:
        method_key = str(method_key)
        method_payload: dict[str, Any] = {
            "manual_params": _json_safe(manual_params.get(method_key, {})),
            "automatic_params": _json_safe(auto_params.get(method_key, {})),
            "trials": [],
        }
        rows.append(
            _trial_row(
                branch="manual",
                method_key=method_key,
                trial_index=0,
                selected=True,
                params=manual_params.get(method_key, {}),
                score=None,
                comparison_score=(manual.get("metrics") or {}).get("comparison_score"),
                truth_metrics=manual.get("metrics") or {},
                reason="manual baseline parameters",
                warnings=manual.get("warnings") or [],
                evidence_context=evidence_context,
            )
        )
        result = auto_results.get(method_key) or {}
        trials = list(result.get("all_trials") or [])
        selected_params = auto_params.get(method_key, {})
        if not trials:
            warning = f"trial data unavailable for method {method_key}; exported selected parameters only"
            warnings_list.append(warning)
            trials = [
                {
                    "trial_index": 0,
                    "params": selected_params,
                    "score": result.get("best_score"),
                    "reason": result.get("best_reason") or warning,
                    "warnings": [warning],
                    "valid": True,
                }
            ]
        method_payload["trials"] = _json_safe(trials)
        for index, trial in enumerate(trials):
            params = trial.get("params") or {}
            rows.append(
                _trial_row(
                    branch="automatic",
                    method_key=method_key,
                    trial_index=int(trial.get("trial_index", index)),
                    selected=_same_params(params, selected_params),
                    params=params,
                    score=trial.get("score"),
                    comparison_score=trial.get("comparison_score"),
                    truth_metrics=trial,
                    reason=trial.get("reason") or result.get("best_reason") or "",
                    warnings=trial.get("warnings") or [],
                    evidence_context=evidence_context,
                )
            )
        payload["methods"][method_key] = method_payload
    payload["warnings"] = list(warnings_list)
    return rows, payload, warnings_list


def _trial_row(
    *,
    branch: str,
    method_key: str,
    trial_index: int,
    selected: bool,
    params: dict[str, Any],
    score: Any,
    comparison_score: Any,
    truth_metrics: dict[str, Any],
    reason: str,
    warnings: list[Any],
    evidence_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trial = dict(truth_metrics or {})
    context = dict(evidence_context or {})
    boundary = _trial_scoring_boundary(trial, context)
    manual_review_required = _trial_manual_review_required(branch, trial, context)
    return {
        "branch": branch,
        "method_key": method_key,
        "trial_index": int(trial_index),
        "selected": bool(selected),
        "params_json": json.dumps(_json_safe(params or {}), ensure_ascii=False, sort_keys=True),
        "score": _csv_value(score),
        "comparison_score": _csv_value(comparison_score),
        "truth_score": _csv_value(trial.get("truth_score")),
        "truth_target_energy_preservation": _csv_value(
            trial.get("truth_target_energy_preservation")
        ),
        "truth_target_saliency_gain": _csv_value(
            trial.get("truth_target_saliency_gain")
        ),
        "truth_background_energy_reduction": _csv_value(
            trial.get("truth_background_energy_reduction")
        ),
        "truth_false_positive_ratio": _csv_value(
            trial.get("truth_false_positive_ratio")
        ),
        "candidate_space_hash": _csv_value(trial.get("candidate_space_hash") or _first_value(context.get("candidate_space_hashes"))),
        "candidate_space_profile_id": _csv_value(trial.get("candidate_space_profile_id") or trial.get("candidate_space_profile") or _first_value(context.get("profile_ids"))),
        "candidate_space_config_version": _csv_value(trial.get("candidate_space_config_version") or _first_value(context.get("config_versions"))),
        "candidate_space_recipe_ids_json": json.dumps(_json_safe(trial.get("candidate_space_recipe_ids") or context.get("recipe_ids") or []), ensure_ascii=False),
        "candidate_id": _csv_value(trial.get("candidate_id")),
        "candidate_source": _csv_value(trial.get("candidate_source")),
        "candidate_group": _csv_value(trial.get("candidate_group")),
        "candidate_parameters_json": json.dumps(_json_safe(trial.get("candidate_parameters") or {}), ensure_ascii=False, sort_keys=True),
        "scoring_boundary": boundary,
        "manual_review_required": bool(manual_review_required),
        "score_version": _csv_value(trial.get("score_version") or trial.get("autotune_scoring_version") or context.get("score_version") or ""),
        "reason": str(reason or ""),
        "warnings_json": json.dumps(_json_safe(warnings or trial.get("candidate_warnings") or []), ensure_ascii=False),
    }



def _first_value(values: Any) -> Any:
    if isinstance(values, (list, tuple)) and values:
        return values[0]
    return None


def _unique_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        safe = _json_safe(value)
        key = json.dumps(safe, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(safe)
    return out


def _iter_auto_tune_trials(summary: dict[str, Any]):
    automatic = summary.get("automatic") or {}
    auto_results = automatic.get("auto_tune_results") or {}
    if not isinstance(auto_results, dict):
        return
    for method_key, result in auto_results.items():
        if not isinstance(result, dict):
            continue
        for trial in result.get("all_trials") or []:
            if isinstance(trial, dict):
                yield str(method_key), trial


def _build_autotune_v1_evidence_context(summary: dict[str, Any]) -> dict[str, Any]:
    """Collect candidate-space and scoring-boundary metadata for exports.

    The context is deliberately compact and JSON-safe. It does not claim that
    real no-prior metrics are ground truth; it records the boundary under which
    the exported AutoTune trial table may be interpreted.
    """
    ground_truth = summary.get("ground_truth_info") or {}
    roi = summary.get("roi_info") or {}
    has_truth = bool(ground_truth.get("enabled"))
    hashes: list[Any] = []
    profiles: list[Any] = []
    config_versions: list[Any] = []
    recipe_ids: list[Any] = []
    candidate_ids: list[Any] = []
    score_versions: list[Any] = []
    display_only = False
    experimental = False
    for _method_key, trial in _iter_auto_tune_trials(summary):
        hashes.append(trial.get("candidate_space_hash"))
        profiles.append(trial.get("candidate_space_profile_id") or trial.get("candidate_space_profile"))
        config_versions.append(trial.get("candidate_space_config_version"))
        recipe_ids.extend(trial.get("candidate_space_recipe_ids") or [])
        candidate_ids.append(trial.get("candidate_id"))
        score_versions.append(trial.get("score_version") or trial.get("autotune_scoring_version"))
        if trial.get("display_only") or trial.get("metric_safe") is False:
            display_only = True
        params = trial.get("candidate_parameters") or {}
        if isinstance(params, dict) and params.get("experimental"):
            experimental = True
    scoring_boundary = "synthetic_supervised" if has_truth else "field_no_prior_proxy"
    manual_review_required = bool(
        not has_truth
        or str(roi.get("source") or "").lower() == "manual"
        or display_only
        or experimental
    )
    forbidden_metrics = [] if has_truth else ["MAE", "MSE", "RMSE", "PSNR", "SSIM", "MS-SSIM", "target_response_similarity"]
    allowed_metrics = (
        ["MAE", "MSE", "RMSE", "PSNR", "SSIM-like", "target_roi_preservation", "background_roi_suppression", "false_positive_risk"]
        if has_truth
        else ["SCR/CNR proxy", "contrast", "entropy", "continuity", "texture/coherence", "artifact risk"]
    )
    claim_boundary = (
        "Synthetic supervised benchmark: candidate ranking may be interpreted against the provided target/background reference only for this controlled input."
        if has_truth
        else "Real no-prior proxy: candidate ranking is heuristic and must not be interpreted as closer to true subsurface structure; manual review is required."
    )
    return {
        "enabled": bool(hashes or profiles or candidate_ids),
        "schema": "mygpr_autotune_v1_evidence_context",
        "candidate_space_hashes": _unique_values(hashes),
        "profile_ids": _unique_values(profiles),
        "config_versions": _unique_values(config_versions),
        "recipe_ids": _unique_values(recipe_ids),
        "candidate_ids": _unique_values(candidate_ids),
        "score_version": _first_value(_unique_values(score_versions)) or "autotune_scoring_v2",
        "scoring_boundary": scoring_boundary,
        "target_response_available": has_truth,
        "allowed_metrics": allowed_metrics,
        "forbidden_metrics": forbidden_metrics,
        "manual_review_required": manual_review_required,
        "claim_boundary": claim_boundary,
        "roi_mode": roi.get("source") or "full",
        "display_only_candidates_present": bool(display_only),
        "experimental_candidates_present": bool(experimental),
    }


def _trial_scoring_boundary(trial: dict[str, Any], context: dict[str, Any]) -> str:
    return str(
        trial.get("scoring_boundary")
        or trial.get("claim_boundary_mode")
        or context.get("scoring_boundary")
        or "field_no_prior_proxy"
    )


def _trial_manual_review_required(branch: str, trial: dict[str, Any], context: dict[str, Any]) -> bool:
    if branch == "manual":
        return bool(context.get("manual_review_required", True))
    if "manual_review_required" in trial:
        return bool(trial.get("manual_review_required"))
    if trial.get("display_only") or trial.get("metric_safe") is False:
        return True
    return bool(context.get("manual_review_required", True))

def _same_params(lhs: Any, rhs: Any) -> bool:
    return _json_safe(lhs or {}) == _json_safe(rhs or {})


def _build_truth_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    info = summary.get("ground_truth_info") or {}
    if not info.get("enabled"):
        return {"enabled": False, "reason": "ground truth unavailable"}
    manual_metrics = ((summary.get("manual") or {}).get("metrics") or {})
    auto_metrics = ((summary.get("automatic") or {}).get("metrics") or {})
    delta = summary.get("metric_delta") or {}
    keys = [
        "truth_score",
        "truth_target_energy_preservation",
        "truth_target_saliency_gain",
        "truth_background_energy_reduction",
        "truth_false_positive_ratio",
        "truth_target_count",
    ]
    return {
        "enabled": True,
        "manual": {key: _json_safe(manual_metrics.get(key)) for key in keys},
        "automatic": {key: _json_safe(auto_metrics.get(key)) for key in keys},
        "delta": {key: _json_safe(delta.get(key)) for key in keys},
    }


def _build_workflow_params(summary: dict[str, Any]) -> dict[str, Any]:
    manual = summary.get("manual") or {}
    automatic = summary.get("automatic") or {}
    auto_results = automatic.get("auto_tune_results") or {}
    recommendation_reason: dict[str, Any] = {}
    parameter_domain: dict[str, Any] = {}
    for method_key, result in auto_results.items():
        if not isinstance(result, dict):
            continue
        recommendation_reason[str(method_key)] = (
            result.get("best_reason")
            or result.get("selection_recommendation")
            or result.get("risk_reason")
        )
        parameter_domain[str(method_key)] = result.get("parameter_domain")
    return {
        "pipeline": list(manual.get("pipeline") or automatic.get("pipeline") or []),
        "manual_params_by_method": _json_safe(manual.get("params_by_method") or {}),
        "automatic_params_by_method": _json_safe(
            automatic.get("params_by_method") or {}
        ),
        "auto_tune_recommendation_reason": _json_safe(recommendation_reason),
        "parameter_domain": _json_safe(parameter_domain),
        "baseline_profile_key": summary.get("baseline_profile_key"),
        "roi_info": _json_safe(summary.get("roi_info") or {}),
        "autotune_v1_candidate_space": _json_safe(_build_autotune_v1_evidence_context(summary)),
    }


def _build_evidence_manifest(
    summary: dict[str, Any],
    paths: dict[str, Path],
    *,
    output_root: Path,
    input_ref: str | None,
    notes: list[str],
) -> dict[str, Any]:
    ground_truth = summary.get("ground_truth_info") or {"enabled": False}
    source_paths = ground_truth.get("source_paths") or {}
    artifacts = {
        key: {
            "path": _relative_artifact_path(path, output_root),
            "status": "available" if path.exists() else "missing",
        }
        for key, path in paths.items()
    }
    warnings_list = _summary_warnings(summary)
    conversion_warnings = ground_truth.get("conversion_warnings") or []
    warnings_list.extend(str(item) for item in conversion_warnings)
    autotune_v1_context = _build_autotune_v1_evidence_context(summary)
    if autotune_v1_context.get("manual_review_required"):
        warnings_list.append("AutoTune V1 export requires manual review under the recorded scoring boundary.")
    return {
        "schema": "mygpr_autotune_evidence_v1",
        "exported_at": summary.get("exported_at"),
        "project": "MyGPR",
        "git_commit": _safe_git_commit(warnings_list),
        "input": {
            "input_file": input_ref,
            "manifest_file": source_paths.get("manifest_file"),
            "ground_truth_file": source_paths.get("ground_truth_file"),
        },
        "ground_truth": {
            "enabled": bool(ground_truth.get("enabled")),
            "scenario_id": ground_truth.get("scenario_id"),
            "target_count": int(ground_truth.get("target_count") or 0),
            "has_background_rois": bool(ground_truth.get("has_background_rois")),
            "conversion_warnings": _json_safe(conversion_warnings),
        },
        "workflow": {
            "pipeline": list(
                ((summary.get("manual") or {}).get("pipeline"))
                or ((summary.get("automatic") or {}).get("pipeline"))
                or []
            ),
            "baseline_profile_key": summary.get("baseline_profile_key"),
            "roi_info": _json_safe(summary.get("roi_info") or {}),
        },
        "autotune_v1": _json_safe(autotune_v1_context),
        "artifacts": artifacts,
        "warnings": _json_safe(warnings_list),
        "notes": [str(item) for item in notes],
    }


def _summary_warnings(summary: dict[str, Any]) -> list[str]:
    warnings_list: list[str] = []
    for branch in ("manual", "automatic"):
        payload = summary.get(branch) or {}
        warnings_list.extend(str(item) for item in payload.get("warnings", []) or [])
    return warnings_list


def _safe_git_commit(warnings_list: list[str]) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        warnings_list.append(f"git commit unavailable: {exc}")
        return None
    if result.returncode != 0:
        warnings_list.append(
            "git commit unavailable: " + (result.stderr.strip() or str(result.returncode))
        )
        return None
    commit = result.stdout.strip()
    return commit or None


def _relative_artifact_path(path: Path, output_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_evidence_zip(
    zip_path: Path,
    paths: dict[str, Path],
    output_root: Path,
) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, path in paths.items():
            if key == "evidence_zip" or not path.exists():
                continue
            zf.write(path, _relative_artifact_path(path, output_root))


def _write_csv_rows(
    out_path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def _artifact_name(path: Any) -> str:
    if not path:
        return ""
    return Path(str(path)).name


def _csv_value(value: Any) -> str:
    safe = _json_safe(value)
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, ensure_ascii=False, sort_keys=True)
    if safe is None:
        return ""
    return str(safe)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, int):
        return int(value)
    return str(value)


__all__ = ["export_auto_tune_comparison_artifacts"]

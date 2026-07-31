#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized parameter, metric, trial and ground-truth evidence tables."""
from __future__ import annotations

from typing import Any

import json

from core.auto_tune_comparison_export_common import (
    _csv_value, _first_value, _json_safe, _same_params, _unique_values,
)

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

__all__ = ['_build_params_rows', '_build_metrics_rows', '_build_trial_table', '_trial_row', '_iter_auto_tune_trials', '_build_autotune_v1_evidence_context', '_trial_scoring_boundary', '_trial_manual_review_required', '_build_truth_metrics', '_build_workflow_params']

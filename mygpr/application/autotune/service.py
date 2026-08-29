#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-injected AutoTune application orchestration.

This module contains no Qt, persistence, processing-registry or concrete algorithm
dependency.  Concrete services are supplied through ``AutoTuneDependencies``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mygpr.application.autotune.ports import AutoTuneDependencies
from mygpr.application.jobs.cancellation import CancellationToken
from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.domain.processing.warnings import merge_runtime_warnings
from mygpr.application.autotune.candidate_planner import _build_candidate_trials
from mygpr.application.autotune.context import _build_auto_tune_context, _get_search_plan
from mygpr.application.autotune.diagnostics import (
    _selection_stability,
    _summarize_failed_trials,
    _summarize_parameter_domain,
)
from mygpr.application.autotune.errors import AutoTuneCancelled, AutoTuneError
from mygpr.application.autotune.evaluation import (
    _compute_stage_outer_score,
    _evaluate_trial_candidates,
    _select_seed_trials,
)
from mygpr.application.autotune.refinement import _refine_candidate_trials
from mygpr.application.autotune.scoring import _SCORE_FUNCTIONS
from mygpr.application.autotune.utils import _merge_trials, _public_params
from mygpr.domain.autotune.models import (
    AutoTuneContext,
    FAILURE_PENALTY,
    INVALID_TRIAL_SCORE,
    OuterSelectionScore,
    PROFILE_LABELS,
    TrialScore,
)
from mygpr.domain.autotune.selection import _build_profiles, _compute_pareto_front

__all__ = [
    "AutoTuneCancelled",
    "AutoTuneContext",
    "AutoTuneError",
    "FAILURE_PENALTY",
    "INVALID_TRIAL_SCORE",
    "OuterSelectionScore",
    "PROFILE_LABELS",
    "TrialScore",
    "AutoTuneService",
    "auto_select_method_group_with_dependencies",
    "auto_tune_method_with_dependencies",
]


def _resolve_execution_context(
    context: ExecutionContext | None,
    *,
    progress_callback: Callable[[int, int, str], None] | None,
    cancel_checker: Callable[[], bool] | None,
) -> ExecutionContext:
    if context is not None and cancel_checker is None and progress_callback is None:
        return context

    def combined_cancel() -> bool:
        return bool(
            (context is not None and context.is_cancelled())
            or (cancel_checker is not None and cancel_checker())
        )

    return ExecutionContext(
        cancellation_token=CancellationToken(checker=combined_cancel),
        progress_callback=(
            progress_callback
            or (context.progress_callback if context is not None else (lambda *_args: None))
        ),
        warning_callback=(
            context.warning_callback if context is not None else (lambda _payload: None)
        ),
        artifact_callback=(
            context.artifact_callback if context is not None else (lambda _payload: None)
        ),
        metadata=dict(context.metadata) if context is not None else {},
    )


def auto_tune_method_with_dependencies(
    dependencies: AutoTuneDependencies,
    data: np.ndarray,
    method_key: str,
    candidate_params: list[dict[str, Any]] | None = None,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    base_params: dict[str, Any] | None = None,
    roi_spec: dict[str, Any] | None = None,
    search_mode: str = "standard",
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    execution_context: ExecutionContext | None = None,
) -> dict[str, Any]:
    """Auto-tune one method using injected catalogue and execution services."""
    descriptor = dependencies.catalog.get(method_key)
    method_info = dependencies.catalog.raw_metadata(method_key)
    if descriptor is None or not method_info:
        raise AutoTuneError(f"未知方法: {method_key}")
    if not descriptor.auto_tune_enabled:
        raise AutoTuneError(f"方法暂不支持参数推荐: {method_key}")

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise AutoTuneError("参数推荐需要二维非空数据")

    family = str(descriptor.auto_tune_family or method_info.get("auto_tune_family") or "")
    if not family:
        raise AutoTuneError(f"方法缺少 auto_tune_family: {method_key}")

    base_params = dict(base_params or {})
    runtime_context = _resolve_execution_context(
        execution_context,
        progress_callback=progress_callback,
        cancel_checker=cancel_checker,
    )
    effective_cancel_checker = runtime_context.is_cancelled
    effective_progress_callback = progress_callback or runtime_context.report_progress
    context = _build_auto_tune_context(
        arr,
        header_info or {},
        trace_metadata or {},
        roi_spec or {},
        search_mode,
    )
    plan = _get_search_plan(context.search_mode)

    score_func = _SCORE_FUNCTIONS.get(family)
    if score_func is None:
        raise AutoTuneError(f"不支持的 auto-tune family: {family}")

    coarse_trials = candidate_params or _build_candidate_trials(
        method_key,
        arr,
        base_params,
        header_info or {},
        trace_metadata or {},
        context,
        method_info,
        stage="coarse",
        budget=plan["coarse_budget"],
    )
    if not coarse_trials:
        raise AutoTuneError(f"方法未生成候选参数: {method_key}")

    evaluated_cache: dict[str, dict[str, Any]] = {}
    scored_coarse = _evaluate_trial_candidates(
        arr,
        method_key,
        family,
        coarse_trials,
        base_params,
        header_info or {},
        trace_metadata or {},
        context,
        score_func,
        stage="coarse",
        progress_callback=effective_progress_callback,
        cancel_checker=effective_cancel_checker,
        stage_message=f"粗筛 {descriptor.name}",
        dependencies=dependencies,
        execution_context=runtime_context,
        evaluated_cache=evaluated_cache,
    )

    valid_coarse = [trial for trial in scored_coarse if trial.get("valid", True)]
    if not valid_coarse:
        raise AutoTuneError(_summarize_failed_trials(scored_coarse, method_key))

    seed_trials = _select_seed_trials(context, valid_coarse, family)
    fine_trials = _refine_candidate_trials(
        method_key,
        arr,
        base_params,
        header_info or {},
        trace_metadata or {},
        context,
        seed_trials,
        method_info,
    )
    scored_fine: list[dict[str, Any]] = []
    if fine_trials:
        scored_fine = _evaluate_trial_candidates(
            arr,
            method_key,
            family,
            fine_trials,
            base_params,
            header_info or {},
            trace_metadata or {},
            context,
            score_func,
            stage="fine",
            progress_callback=effective_progress_callback,
            cancel_checker=effective_cancel_checker,
            stage_message=f"细化 {descriptor.name}",
            dependencies=dependencies,
            execution_context=runtime_context,
            evaluated_cache=evaluated_cache,
        )

    scored_trials = _merge_trials(scored_coarse, scored_fine)
    valid_trials = [trial for trial in scored_trials if trial.get("valid", True)]
    if not valid_trials:
        raise AutoTuneError(_summarize_failed_trials(scored_trials, method_key))

    best_trial = max(valid_trials, key=lambda item: float(item.get("score", 0.0)))

    if effective_progress_callback is not None:
        effective_progress_callback(
            len(scored_trials),
            max(1, len(scored_trials)),
            f"参数推荐完成: {descriptor.name}",
        )

    pareto_trials = _compute_pareto_front(family, valid_trials)
    profiles = _build_profiles(family, valid_trials, pareto_trials)
    best_params = _public_params(best_trial["params"])
    selection_margin, selection_confidence = _selection_stability(valid_trials)
    failed_trials = [trial for trial in scored_trials if not trial.get("valid", True)]
    constraint_warnings = merge_runtime_warnings(
        *[trial.get("constraint_warnings", []) for trial in scored_trials]
    )
    best_constraint_warnings = list(best_trial.get("constraint_warnings", []) or [])
    constraint_adjustment_count = sum(
        1 for trial in scored_trials if trial.get("constraint_adjusted")
    )
    parameter_domain = _summarize_parameter_domain(
        method_key,
        method_info,
        context,
        base_params,
        scored_trials,
        best_trial,
        plan,
        selection_margin=selection_margin,
        selection_confidence=selection_confidence,
        constraint_adjustment_count=constraint_adjustment_count,
    )
    return {
        "method_key": method_key,
        "method_name": descriptor.name,
        "family": family,
        "best_params": best_params,
        "best_score": float(best_trial["score"]),
        "best_metrics": dict(best_trial["metrics"]),
        "best_penalties": dict(best_trial["penalties"]),
        "best_reason": best_trial["reason"],
        "all_trials": scored_trials,
        "coarse_trials": scored_coarse,
        "fine_trials": scored_fine,
        "pareto_trials": pareto_trials,
        "profiles": profiles,
        "recommended_profile": "balanced",
        "recommended_params": dict(
            profiles.get("balanced", {}).get("params", best_params)
        ),
        "roi_info": {
            "source": context.roi_source,
            "label": context.roi_label,
            "bounds": context.roi_bounds,
            "search_mode": context.search_mode,
        },
        "selection_margin": float(selection_margin),
        "selection_confidence": float(selection_confidence),
        "failed_trials": failed_trials,
        "constraint_warnings": constraint_warnings,
        "best_constraint_warnings": best_constraint_warnings,
        "parameter_domain": parameter_domain,
        "risk_flags": list(parameter_domain.get("risk_flags", [])),
        "risk_level": parameter_domain.get("risk_level", "low"),
        "risk_reason": parameter_domain.get("risk_reason", ""),
        "selection_recommendation": parameter_domain.get(
            "selection_recommendation", "review"
        ),
        "search_plan": dict(plan),
        "execution_stats": {
            "coarse_trial_count": int(len(scored_coarse)),
            "fine_trial_count": int(len(scored_fine)),
            "total_trial_count": int(len(scored_trials)),
            "valid_trial_count": int(len(valid_trials)),
            "failed_trial_count": int(len(failed_trials)),
            "cache_hit_count": int(
                sum(1 for trial in scored_trials if trial.get("cached"))
            ),
            "constraint_adjustment_count": int(constraint_adjustment_count),
        },
    }


def auto_select_method_group_with_dependencies(
    dependencies: AutoTuneDependencies,
    data: np.ndarray,
    method_keys: list[str],
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    base_params_map: dict[str, dict[str, Any]] | None = None,
    roi_spec: dict[str, Any] | None = None,
    search_mode: str = "standard",
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    execution_context: ExecutionContext | None = None,
) -> dict[str, Any]:
    """Compare methods in one stage using injected backend services."""
    if not method_keys:
        raise AutoTuneError("未提供可比较的方法列表")

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise AutoTuneError("参数推荐需要二维非空数据")

    stage_names = {dependencies.catalog.auto_tune_stage(method_key) for method_key in method_keys}
    if len(stage_names) != 1 or "" in stage_names:
        raise AutoTuneError("同组方法必须属于同一 auto-tune stage")

    stage_name = next(iter(stage_names))
    runtime_context = _resolve_execution_context(
        execution_context,
        progress_callback=progress_callback,
        cancel_checker=cancel_checker,
    )
    effective_cancel_checker = runtime_context.is_cancelled
    effective_progress_callback = progress_callback or runtime_context.report_progress

    context = _build_auto_tune_context(
        arr,
        header_info or {},
        trace_metadata or {},
        roi_spec or {},
        search_mode,
    )

    group_results = []
    total = len(method_keys)
    for idx, method_key in enumerate(method_keys, start=1):
        if effective_cancel_checker():
            raise AutoTuneCancelled("用户已取消参数推荐")
        if effective_progress_callback is not None:
            effective_progress_callback(idx - 1, total, f"比较方法 {idx}/{total}: {method_key}")

        method_context = runtime_context.child(idx - 1, total)
        tune_result = auto_tune_method_with_dependencies(
            dependencies,
            data=arr,
            method_key=method_key,
            header_info=header_info,
            trace_metadata=trace_metadata,
            base_params=(base_params_map or {}).get(method_key, {}),
            roi_spec=roi_spec,
            search_mode=search_mode,
            execution_context=method_context,
        )

        balanced = (tune_result.get("profiles", {}) or {}).get("balanced", {})
        champion_params = dict(
            balanced.get("params") or tune_result.get("best_params") or {}
        )
        champion_result = dependencies.executor.execute(
            ProcessingRequest(
                data=arr,
                method_id=method_key,
                params=champion_params,
                header_info=header_info or {},
                trace_metadata=trace_metadata or {},
            ),
            method_context,
        )
        after = champion_result.data
        outer = _compute_stage_outer_score(
            stage_name,
            before=arr,
            after=np.asarray(after, dtype=np.float32),
            context=context,
        )
        group_results.append(
            {
                "method_key": method_key,
                "method_name": tune_result.get("method_name", method_key),
                "stage": stage_name,
                "auto_tune_result": tune_result,
                "champion_profile": balanced.get("label", "平衡档"),
                "champion_params": champion_params,
                "outer_score": float(outer.score),
                "outer_metrics": dict(outer.metrics),
                "outer_reason": outer.reason,
            }
        )

    best = max(group_results, key=lambda item: float(item.get("outer_score", 0.0)))
    if effective_progress_callback is not None:
        effective_progress_callback(total, total, f"方法比较完成: {best['method_name']}")

    return {
        "stage": stage_name,
        "best_method_key": best["method_key"],
        "best_method_name": best["method_name"],
        "best_params": best["champion_params"],
        "best_auto_tune_result": best["auto_tune_result"],
        "outer_score": best["outer_score"],
        "outer_metrics": best["outer_metrics"],
        "outer_reason": best["outer_reason"],
        "candidates": group_results,
        "roi_info": {
            "source": context.roi_source,
            "label": context.roi_label,
            "bounds": context.roi_bounds,
            "search_mode": context.search_mode,
        },
    }


_TARGET_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {},
    "structure_preservation": {
        "edge_preservation": 1.0,
        "local_saliency_preservation": 1.0,
        "target_band_fidelity": 0.8,
        "peak_fidelity": 0.4,
    },
    "background_suppression": {
        "horizontal_coherence": -1.0,
        "horizontal_coherence_drop": 1.0,
        "low_freq_drop": 0.6,
        "edge_preservation": 0.5,
    },
    "deep_target": {
        "deep_zone_contrast": 1.0,
        "target_band_fidelity": 0.8,
        "local_saliency_preservation": 0.5,
    },
}


def _minmax(value: float, minimum: float, maximum: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if maximum <= minimum + 1.0e-12:
        return 0.5
    return float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))




def _parse_metric_weights(
    target: str,
    metric_weights: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, str]]:
    preset = dict(_TARGET_WEIGHT_PRESETS.get(str(target or "balanced"), {}))
    invalid: dict[str, str] = {}
    for key, value in dict(metric_weights or {}).items():
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            invalid[str(key)] = f"{type(exc).__name__}: {value!r}"
            continue
        if np.isfinite(weight) and abs(weight) > 1.0e-12:
            preset[str(key)] = weight
    return preset, invalid

def _apply_preference_ranking(
    result: dict[str, Any],
    *,
    target: str = "balanced",
    metric_weights: dict[str, float] | None = None,
    selection_profile: str = "balanced",
) -> dict[str, Any]:
    """Attach deterministic target/weight ranking without replacing scientific scores."""
    payload = dict(result)
    trials = [dict(item) for item in payload.get("all_trials", []) if item.get("valid", True)]
    preset, invalid_metric_weights = _parse_metric_weights(target, metric_weights)
    score_values = [float(item.get("score", 0.0)) for item in trials]
    score_min = min(score_values, default=0.0)
    score_max = max(score_values, default=0.0)
    metric_ranges: dict[str, tuple[float, float]] = {}
    for metric in preset:
        values = [
            float(item.get("metrics", {}).get(metric))
            for item in trials
            if item.get("metrics", {}).get(metric) is not None
            and np.isfinite(float(item.get("metrics", {}).get(metric)))
        ]
        if values:
            metric_ranges[metric] = (min(values), max(values))

    ranked: list[dict[str, Any]] = []
    for item in trials:
        base = _minmax(float(item.get("score", 0.0)), score_min, score_max)
        weighted_total = base
        weight_total = 1.0
        contributions: dict[str, float] = {}
        metrics = dict(item.get("metrics") or {})
        for metric, weight in preset.items():
            if metric not in metric_ranges or metric not in metrics:
                continue
            minimum, maximum = metric_ranges[metric]
            normalized = _minmax(float(metrics[metric]), minimum, maximum)
            preferred = normalized if weight >= 0 else 1.0 - normalized
            contribution = abs(weight) * preferred
            weighted_total += contribution
            weight_total += abs(weight)
            contributions[metric] = contribution
        preference_score = weighted_total / max(weight_total, 1.0e-12)
        enriched = dict(item)
        enriched["preference_score"] = float(preference_score)
        enriched["preference_contributions"] = contributions
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (
            float(item.get("preference_score", 0.0)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )
    top_candidates = []
    for rank, item in enumerate(ranked[:3], start=1):
        top_candidates.append({
            "rank": rank,
            "params": _public_params(dict(item.get("params") or {})),
            "score": float(item.get("score", 0.0)),
            "preference_score": float(item.get("preference_score", 0.0)),
            "metrics": dict(item.get("metrics") or {}),
            "penalties": dict(item.get("penalties") or {}),
            "reason": str(item.get("reason") or ""),
            "stage": str(item.get("stage") or ""),
        })
    profiles = dict(payload.get("profiles") or {})
    requested_profile = str(selection_profile or "balanced")
    profile = profiles.get(requested_profile) or profiles.get("balanced") or {}
    if preset and ranked:
        recommended_params = _public_params(dict(ranked[0].get("params") or {}))
        recommendation_source = "weighted_top_candidate"
    else:
        recommended_params = dict(
            profile.get("params")
            or payload.get("recommended_params")
            or payload.get("best_params")
            or {}
        )
        recommendation_source = f"profile:{requested_profile}"
    payload["recommended_profile"] = requested_profile
    payload["recommended_params"] = recommended_params
    payload["top_candidates"] = top_candidates
    payload["preference_audit"] = {
        "schema": "mygpr.autotune.preference_audit.v1",
        "target": str(target or "balanced"),
        "selection_profile": requested_profile,
        "metric_weights": preset,
        "invalid_metric_weights": invalid_metric_weights,
        "recommendation_source": recommendation_source,
        "ranked_candidate_count": len(ranked),
    }
    return payload


class AutoTuneService:
    """Dependency-injected object API for AutoTune callers."""

    def __init__(self, dependencies: AutoTuneDependencies) -> None:
        self._dependencies = dependencies

    @property
    def dependencies(self) -> AutoTuneDependencies:
        return self._dependencies

    def tune_method(self, data: np.ndarray, method_key: str, **kwargs: Any) -> dict[str, Any]:
        target = str(kwargs.pop("target", "balanced") or "balanced")
        selection_profile = str(kwargs.pop("selection_profile", "balanced") or "balanced")
        metric_weights = dict(kwargs.pop("metric_weights", {}) or {})
        result = auto_tune_method_with_dependencies(
            self._dependencies,
            data,
            method_key,
            **kwargs,
        )
        return _apply_preference_ranking(
            result,
            target=target,
            metric_weights=metric_weights,
            selection_profile=selection_profile,
        )

    def select_method_group(
        self,
        data: np.ndarray,
        method_keys: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return auto_select_method_group_with_dependencies(
            self._dependencies,
            data,
            method_keys,
            **kwargs,
        )

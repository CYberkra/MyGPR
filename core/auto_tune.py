#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-step rule-based auto-tuning for GPR preprocessing methods.

Validated round-2 drop-in version:
- keeps the lightweight rule-based / ROI-aware architecture
- adds robust budgeting, caching, failure tolerance and stable scoring
- preserves GUI result schema while exposing extra execution stats
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.methods_registry import PROCESSING_METHODS, get_auto_tune_stage
from core.processing_engine import (
    clone_header_info,
    clone_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.quality_metrics import (
    auto_roi_bounds,
    baseline_bias,
    clipping_ratio,
    deep_zone_contrast,
    depth_rms_cv,
    detect_first_break_indices,
    edge_preservation,
    estimate_depth_attenuation_curve,
    estimate_lateral_correlation_length,
    estimate_singular_elbow_rank,
    extract_roi_and_context,
    first_break_sharpness,
    first_break_std,
    horizontal_coherence,
    hot_pixel_ratio,
    kurtosis_or_spikiness,
    local_saliency_preservation,
    low_freq_energy_ratio,
    median_first_break,
    pre_zero_energy_ratio,
    ratio_fidelity,
    relative_reduction,
    target_band_energy_ratio,
    weighted_score_parts,
)


class AutoTuneError(RuntimeError):
    """Raised when auto-tune configuration or execution fails."""


class AutoTuneCancelled(RuntimeError):
    """Raised when auto-tune is cancelled by the user."""


@dataclass
class TrialScore:
    score: float
    metrics: dict[str, float]
    penalties: dict[str, float]
    reason: str


@dataclass
class AutoTuneContext:
    full_data: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    roi_source: str
    roi_label: str
    roi_bounds: dict[str, int] | None
    roi_data: np.ndarray | None
    context_bounds: dict[str, int] | None
    context_data: np.ndarray
    features: dict[str, Any]
    search_mode: str


@dataclass
class OuterSelectionScore:
    score: float
    metrics: dict[str, float]
    reason: str


PROFILE_LABELS = {
    "conservative": "保守档",
    "balanced": "平衡档",
    "aggressive": "增强档",
}

INVALID_TRIAL_SCORE = -1.0e9
FAILURE_PENALTY = 999.0


def auto_tune_method(
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
) -> dict[str, Any]:
    """Auto-tune a single currently selected method using rule-based scoring."""
    method_info = PROCESSING_METHODS.get(method_key)
    if not method_info:
        raise AutoTuneError(f"未知方法: {method_key}")
    if not method_info.get("auto_tune_enabled"):
        raise AutoTuneError(f"方法暂不支持自动选参: {method_key}")

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise AutoTuneError("自动选参需要二维非空数据")

    family = str(method_info.get("auto_tune_family") or "")
    if not family:
        raise AutoTuneError(f"方法缺少 auto_tune_family: {method_key}")

    base_params = dict(base_params or {})
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
        raise AutoTuneError(f"未实现的 auto-tune family: {family}")

    coarse_trials = candidate_params or _build_candidate_trials(
        method_key,
        arr,
        base_params,
        header_info or {},
        trace_metadata or {},
        context,
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
        progress_callback=progress_callback,
        cancel_checker=cancel_checker,
        stage_message=f"粗筛 {method_info['name']}",
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
            progress_callback=progress_callback,
            cancel_checker=cancel_checker,
            stage_message=f"细化 {method_info['name']}",
            evaluated_cache=evaluated_cache,
        )

    scored_trials = _merge_trials(scored_coarse, scored_fine)
    valid_trials = [trial for trial in scored_trials if trial.get("valid", True)]
    if not valid_trials:
        raise AutoTuneError(_summarize_failed_trials(scored_trials, method_key))

    best_trial = max(valid_trials, key=lambda item: float(item.get("score", 0.0)))

    if progress_callback is not None:
        progress_callback(
            len(scored_trials),
            max(1, len(scored_trials)),
            f"自动选参完成: {method_info['name']}",
        )

    pareto_trials = _compute_pareto_front(family, valid_trials)
    profiles = _build_profiles(family, valid_trials, pareto_trials)
    best_params = _public_params(best_trial["params"])
    selection_margin, selection_confidence = _selection_stability(valid_trials)
    failed_trials = [trial for trial in scored_trials if not trial.get("valid", True)]
    return {
        "method_key": method_key,
        "method_name": method_info.get("name", method_key),
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
        },
    }


def auto_select_method_group(
    data: np.ndarray,
    method_keys: list[str],
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    base_params_map: dict[str, dict[str, Any]] | None = None,
    roi_spec: dict[str, Any] | None = None,
    search_mode: str = "standard",
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Compare multiple methods within the same stage and pick the best champion."""
    if not method_keys:
        raise AutoTuneError("未提供可比较的方法列表")

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise AutoTuneError("自动选参需要二维非空数据")

    stage_names = {get_auto_tune_stage(method_key) for method_key in method_keys}
    if len(stage_names) != 1 or "" in stage_names:
        raise AutoTuneError("同组方法必须属于同一 auto-tune stage")

    stage_name = next(iter(stage_names))
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
        if cancel_checker and bool(cancel_checker()):
            raise AutoTuneCancelled("用户已取消自动选参")
        if progress_callback is not None:
            progress_callback(idx - 1, total, f"比较方法 {idx}/{total}: {method_key}")

        tune_result = auto_tune_method(
            data=arr,
            method_key=method_key,
            header_info=header_info,
            trace_metadata=trace_metadata,
            base_params=(base_params_map or {}).get(method_key, {}),
            roi_spec=roi_spec,
            search_mode=search_mode,
            cancel_checker=cancel_checker,
        )

        balanced = (tune_result.get("profiles", {}) or {}).get("balanced", {})
        champion_params = dict(
            balanced.get("params") or tune_result.get("best_params") or {}
        )
        runtime_params = prepare_runtime_params(
            method_key,
            champion_params,
            clone_header_info(header_info or {}),
            clone_trace_metadata(trace_metadata or {}),
            arr.shape,
        )
        after, _meta = run_processing_method(
            arr,
            method_key,
            runtime_params,
            cancel_checker=cancel_checker,
        )
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
    if progress_callback is not None:
        progress_callback(total, total, f"方法比较完成: {best['method_name']}")

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


def _build_candidate_trials(
    method_key: str,
    data: np.ndarray,
    base_params: dict[str, Any],
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    context: AutoTuneContext,
    stage: str = "coarse",
    budget: int | None = None,
) -> list[dict[str, Any]]:
    method_info = PROCESSING_METHODS[method_key]
    family = method_info["auto_tune_family"]
    config = method_info.get("auto_tune_candidates", {})
    plan = _get_search_plan(context.search_mode)
    stage_budget = int(
        budget or (plan["coarse_budget"] if stage == "coarse" else plan["fine_budget"])
    )

    if family == "zero_time":
        return _build_zero_time_candidates(
            data,
            config,
            base_params,
            header_info,
            context,
            stage=stage,
            budget=stage_budget,
        )
    if family == "drift":
        values = _build_drift_windows(
            data.shape[0], context, config, stage=stage, budget=stage_budget
        )
        return [{"window": value} for value in values]
    if family == "background":
        if method_key in {"subtracting_average_2D", "median_background_2D"}:
            values = _build_background_windows(
                data.shape[1],
                context,
                config,
                base_value=base_params.get("ntraces"),
                stage=stage,
                budget=stage_budget,
            )
            return [{"ntraces": value} for value in values]
        if method_key == "svd_bg":
            values = _build_background_rank_candidates(
                data, context, config, stage=stage, budget=stage_budget
            )
            return [{"rank": value} for value in values]
    if family == "fk":
        return _build_fk_filter_trials(
            base_params,
            config,
            stage=stage,
            budget=stage_budget,
        )
    if family == "denoise":
        if method_key == "svd_subspace":
            rank_start_default = config.get("rank_start", [1])
            if isinstance(rank_start_default, list):
                rank_start_default = rank_start_default[0] if rank_start_default else 1
            rank_start = int(base_params.get("rank_start", rank_start_default))
            rank_end_values = _build_subspace_rank_end_candidates(
                data,
                context,
                config,
                base_value=base_params.get("rank_end"),
                stage=stage,
                budget=stage_budget,
            )
            trials = []
            for rank_end in rank_end_values:
                if int(rank_end) >= rank_start:
                    trials.append({"rank_start": rank_start, "rank_end": int(rank_end)})
            return _dedupe_candidates(trials)
        if method_key == "wavelet_svd":
            rank_start_default = config.get("rank_start", [1])
            if isinstance(rank_start_default, list):
                rank_start_default = rank_start_default[0] if rank_start_default else 1
            rank_start = int(base_params.get("rank_start", rank_start_default))
            rank_end_values = _build_subspace_rank_end_candidates(
                data,
                context,
                config,
                base_value=base_params.get("rank_end"),
                stage=stage,
                budget=max(2, stage_budget // 2),
            )
            threshold_default = float(base_params.get("threshold", 0.05))
            threshold_values = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    list(config.get("threshold", []))
                    + [
                        threshold_default * 0.7,
                        threshold_default,
                        threshold_default * 1.3,
                    ],
                    minimum=0.01,
                ),
                budget=max(2, min(3, stage_budget)),
                center=threshold_default,
            )
            levels_default = int(base_params.get("levels", 2))
            levels_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    list(config.get("levels", []))
                    + [levels_default - 1, levels_default, levels_default + 1],
                    data.shape[0],
                    minimum=1,
                    upper=8,
                ),
                budget=max(1, min(3, stage_budget)),
                center=levels_default,
            )
            wavelet_name = str(base_params.get("wavelet", "db4"))
            trials = []
            for rank_end, threshold, levels in itertools.product(
                rank_end_values, threshold_values, levels_values
            ):
                if int(rank_end) >= rank_start:
                    trials.append(
                        {
                            "wavelet": wavelet_name,
                            "levels": int(levels),
                            "threshold": float(threshold),
                            "rank_start": rank_start,
                            "rank_end": int(rank_end),
                        }
                    )
            return _dedupe_candidates(trials)
    if family == "gain":
        if method_key == "sec_gain":
            gain_min_default = config.get("gain_min", 1.0)
            if isinstance(gain_min_default, list):
                gain_min_default = gain_min_default[0] if gain_min_default else 1.0
            gain_min = float(base_params.get("gain_min", gain_min_default))
            gain_max_values, power_values = _build_sec_gain_candidates(
                context,
                config,
                gain_min=gain_min,
                stage=stage,
                budget=stage_budget,
            )
            trials = []
            for gain_max, power in itertools.product(gain_max_values, power_values):
                if gain_max > gain_min:
                    trials.append(
                        {"gain_min": gain_min, "gain_max": gain_max, "power": power}
                    )
            return _dedupe_candidates(trials)
        if method_key == "agcGain":
            values = _build_agc_windows(
                data.shape[0], context, config, stage=stage, budget=stage_budget
            )
            return [{"window": value} for value in values]
        if method_key == "compensatingGain":
            gain_min_values, gain_max_values = _build_compensating_gain_candidates(
                context, config, stage=stage, budget=stage_budget
            )
            trials = []
            for gain_min, gain_max in itertools.product(
                gain_min_values, gain_max_values
            ):
                if gain_max > gain_min:
                    trials.append({"gain_min": gain_min, "gain_max": gain_max})
            return _dedupe_candidates(trials)
    if family == "impulse":
        values = _build_impulse_windows(
            data.shape[1], context, config, stage=stage, budget=stage_budget
        )
        return [{"ntraces": value} for value in values]

    # Generic Cartesian-product candidate builder for methods with explicit candidate lists
    if config:
        keys = list(config.keys())
        values_lists = [config[k] for k in keys]
        trials = []
        for combo in itertools.product(*values_lists):
            trial = dict(base_params)
            trial.update({k: v for k, v in zip(keys, combo)})
            trials.append(trial)
        return _dedupe_candidates(trials)

    return []


def _build_auto_tune_context(
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    roi_spec: dict[str, Any],
    search_mode: str,
) -> AutoTuneContext:
    roi_mode = str((roi_spec or {}).get("mode") or "auto")
    if roi_mode == "full":
        roi_source = "full"
        roi_label = str((roi_spec or {}).get("label") or "全图")
        roi_bounds = None
        roi_data = None
        context_bounds = None
        context_data = np.asarray(data, dtype=np.float32)
    else:
        bounds = (
            (roi_spec or {}).get("bounds") if roi_mode in {"crop", "manual"} else None
        )
        roi_payload = extract_roi_and_context(data, bounds)
        roi_bounds = dict(roi_payload.get("bounds") or {})
        context_bounds = dict(roi_payload.get("context_bounds") or {})
        roi_data = np.asarray(roi_payload.get("roi_data"), dtype=np.float32)
        context_data = np.asarray(roi_payload.get("context_data"), dtype=np.float32)
        roi_source = roi_mode if bounds else "auto"
        roi_label = str(
            (roi_spec or {}).get("label")
            or (
                "当前裁剪区"
                if roi_mode == "crop" and bounds
                else "手动框选 ROI"
                if roi_mode == "manual" and bounds
                else "自动 ROI"
            )
        )
        if roi_data.size < 64 or roi_data.shape[0] < 8 or roi_data.shape[1] < 4:
            roi_source = "full"
            roi_label = "全图"
            roi_bounds = None
            roi_data = None
            context_bounds = None
            context_data = np.asarray(data, dtype=np.float32)

    features = _extract_auto_tune_features(
        np.asarray(data, dtype=np.float32), roi_data, context_data
    )
    return AutoTuneContext(
        full_data=np.asarray(data, dtype=np.float32),
        header_info=dict(header_info or {}),
        trace_metadata=dict(trace_metadata or {}),
        roi_source=roi_source,
        roi_label=roi_label,
        roi_bounds=roi_bounds,
        roi_data=roi_data,
        context_bounds=context_bounds,
        context_data=context_data,
        features=features,
        search_mode=str(search_mode or "standard"),
    )


def _extract_auto_tune_features(
    full_data: np.ndarray, roi_data: np.ndarray | None, context_data: np.ndarray
) -> dict[str, Any]:
    arr = np.asarray(full_data, dtype=np.float64)
    context = np.asarray(context_data, dtype=np.float64)
    roi = np.asarray(roi_data, dtype=np.float64) if roi_data is not None else context
    attenuation = estimate_depth_attenuation_curve(context)
    shallow = (
        float(np.mean(attenuation[: max(4, len(attenuation) // 5)]))
        if attenuation.size
        else 0.0
    )
    deep = (
        float(np.mean(attenuation[max(0, len(attenuation) * 3 // 5) :]))
        if attenuation.size
        else 0.0
    )
    fb_idx = detect_first_break_indices(context, method="threshold", threshold=0.05)
    return {
        "shape": arr.shape,
        "roi_shape": roi.shape,
        "low_freq_ratio": float(low_freq_energy_ratio(context)),
        "lateral_corr_length": int(estimate_lateral_correlation_length(context)),
        "singular_elbow_rank": int(estimate_singular_elbow_rank(context)),
        "shallow_rms": shallow,
        "deep_rms": deep,
        "attenuation_ratio": float(shallow / max(deep, 1.0e-6)) if deep > 0 else 1.0,
        "hot_pixel_ratio": float(hot_pixel_ratio(context)),
        "spikiness": float(kurtosis_or_spikiness(context)),
        "first_break_std": float(np.std(fb_idx)) if fb_idx.size else 0.0,
        "first_break_median": int(median_first_break(fb_idx)),
    }


def _get_search_plan(search_mode: str) -> dict[str, Any]:
    plans = {
        "fast": {"coarse_budget": 6, "refine_top_k": 1, "fine_budget": 4},
        "standard": {"coarse_budget": 8, "refine_top_k": 2, "fine_budget": 6},
        "thorough": {"coarse_budget": 12, "refine_top_k": 3, "fine_budget": 8},
    }
    return plans.get(str(search_mode or "standard"), plans["standard"])


def _evaluate_trial_candidates(
    data: np.ndarray,
    method_key: str,
    family: str,
    trial_params_list: list[dict[str, Any]],
    base_params: dict[str, Any],
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    context: AutoTuneContext,
    score_func: Callable[
        [np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]], TrialScore
    ],
    stage: str,
    progress_callback: Callable[[int, int, str], None] | None,
    cancel_checker: Callable[[], bool] | None,
    stage_message: str,
    evaluated_cache: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(trial_params_list)
    before_arr = np.asarray(data, dtype=np.float32)
    for idx, trial_params in enumerate(trial_params_list, start=1):
        if cancel_checker and bool(cancel_checker()):
            raise AutoTuneCancelled("用户已取消自动选参")
        if progress_callback is not None:
            progress_callback(idx - 1, total, f"{stage_message} {idx}/{total}")

        signature = _trial_signature(trial_params)
        cached = evaluated_cache.get(signature) if evaluated_cache else None
        if cached is not None:
            record = dict(cached)
            record["stage"] = stage
            record["cached"] = True
            results.append(record)
            continue

        runtime_params = dict(base_params)
        runtime_params.update(trial_params)
        try:
            prepared_params = prepare_runtime_params(
                method_key,
                runtime_params,
                clone_header_info(header_info),
                clone_trace_metadata(trace_metadata),
                data.shape,
            )
            result, _ = run_processing_method(
                data,
                method_key,
                prepared_params,
                cancel_checker=cancel_checker,
            )
            record = _score_trial_with_context(
                context,
                family,
                score_func,
                before_arr,
                np.asarray(result, dtype=np.float32),
                dict(trial_params),
                dict(header_info or {}),
                stage=stage,
            )
            record["cached"] = False
            record["valid"] = bool(np.isfinite(record.get("score", 0.0)))
            if not record["valid"]:
                raise ValueError("候选得分出现 NaN/Inf")
        except AutoTuneCancelled:
            raise
        except Exception as exc:
            record = _build_trial_failure_record(context, trial_params, stage, exc)

        if evaluated_cache is not None:
            evaluated_cache[signature] = dict(record)
        results.append(record)
    return results


def _score_trial_with_context(
    context: AutoTuneContext,
    family: str,
    score_func: Callable[
        [np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]], TrialScore
    ],
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    full_trial = score_func(before, after, params, header_info)
    roi_trial = full_trial
    context_trial = None
    guard_score = -_penalty_sum_from_dict(full_trial.penalties)
    total_score = full_trial.score
    roi_used = False
    score_region_source = context.roi_source

    if family == "zero_time":
        shallow_end = max(
            24,
            min(
                before.shape[0],
                int(
                    max(
                        context.features.get("first_break_median", 0) * 2.5,
                        before.shape[0] * 0.25,
                    )
                ),
            ),
        )
        roi_trial = score_func(
            before[:shallow_end, :], after[:shallow_end, :], params, header_info
        )
        total_score = 0.75 * roi_trial.score + 0.25 * full_trial.score
        roi_used = True
        score_region_source = "shallow_first_break"
    elif family == "drift":
        shallow_end = max(32, min(before.shape[0], int(before.shape[0] * 0.35)))
        roi_trial = score_func(
            before[:shallow_end, :], after[:shallow_end, :], params, header_info
        )
        total_score = 0.35 * roi_trial.score + 0.65 * full_trial.score
        roi_used = True
        score_region_source = "shallow_context"
    elif family == "gain":
        deep_before = _slice_depth_band(before, 0.55, 1.0)
        deep_after = _slice_depth_band(after, 0.55, 1.0)
        roi_trial = score_func(deep_before, deep_after, params, header_info)
        total_score = 0.45 * roi_trial.score + 0.55 * full_trial.score
        roi_used = True
        score_region_source = "deep_zone"
    elif (
        family == "background"
        and context.roi_source != "full"
        and context.roi_data is not None
        and context.roi_bounds is not None
    ):
        roi_before = _slice_bounds(before, context.roi_bounds)
        roi_after = _slice_bounds(after, context.roi_bounds)
        roi_trial = score_func(roi_before, roi_after, params, header_info)
        total_score = weighted_score_parts(
            roi_trial.score,
            full_trial.score,
            guard_score,
            use_roi=True,
        )
        roi_used = True
        score_region_source = context.roi_source
    elif (
        family in {"fk", "denoise", "impulse"}
        and context.roi_source != "full"
        and context.roi_bounds is not None
        and context.context_bounds is not None
    ):
        roi_before = _slice_bounds(before, context.roi_bounds)
        roi_after = _slice_bounds(after, context.roi_bounds)
        context_before = _slice_bounds(before, context.context_bounds)
        context_after = _slice_bounds(after, context.context_bounds)
        roi_trial = score_func(roi_before, roi_after, params, header_info)
        context_trial = score_func(context_before, context_after, params, header_info)
        if family == "impulse":
            total_score = (
                0.55 * roi_trial.score
                + 0.25 * context_trial.score
                + 0.20 * full_trial.score
            )
        else:
            total_score = (
                0.45 * roi_trial.score
                + 0.35 * context_trial.score
                + 0.20 * full_trial.score
            )
        roi_used = True
        score_region_source = context.roi_source

    return {
        "params": dict(params),
        "score": float(total_score),
        "metrics": dict(full_trial.metrics),
        "roi_metrics": dict(roi_trial.metrics),
        "context_metrics": dict(context_trial.metrics)
        if context_trial is not None
        else {},
        "penalties": dict(full_trial.penalties),
        "reason": roi_trial.reason if roi_used else full_trial.reason,
        "stage": stage,
        "roi_score": float(roi_trial.score),
        "context_score": float(context_trial.score)
        if context_trial is not None
        else float(full_trial.score),
        "full_score": float(full_trial.score),
        "guard_score": float(guard_score),
        "roi_used": bool(roi_used),
        "roi_source": score_region_source,
    }


def _compute_stage_outer_score(
    stage_name: str,
    before: np.ndarray,
    after: np.ndarray,
    context: AutoTuneContext,
) -> OuterSelectionScore:
    if stage_name == "background":
        coherence = horizontal_coherence(after)
        saliency = local_saliency_preservation(before, after)
        edge = edge_preservation(before, after)
        peak_ratio = float(
            np.percentile(np.abs(after), 99.0)
            / max(np.percentile(np.abs(before), 99.0), 1.0e-6)
        )
        penalties = {
            "edge_loss": max(0.0, 0.72 - edge) * 3.0,
            "target_drop": max(0.0, 0.60 - peak_ratio) * 2.5,
        }
        score = -3.0 * coherence + 2.2 * saliency + 1.2 * edge - sum(penalties.values())
        metrics = {
            "horizontal_coherence": float(coherence),
            "local_saliency_preservation": float(saliency),
            "edge_preservation": float(edge),
            "peak_ratio": float(peak_ratio),
        }
        reason = "优先比较背景一致性下降与显著结构保留。"
        return OuterSelectionScore(float(score), metrics, reason)

    if stage_name == "denoise":
        hot_drop = relative_reduction(hot_pixel_ratio(before), hot_pixel_ratio(after))
        spiky_drop = relative_reduction(
            kurtosis_or_spikiness(before), kurtosis_or_spikiness(after)
        )
        saliency_fid = ratio_fidelity(
            local_saliency_preservation(before, after), 1.0, 0.18
        )
        edge_fid = ratio_fidelity(edge_preservation(before, after), 1.0, 0.18)
        band_fid = ratio_fidelity(target_band_energy_ratio(before, after), 1.0, 0.20)
        score = (
            2.2 * hot_drop
            + 1.8 * spiky_drop
            + 1.3 * saliency_fid
            + 1.1 * edge_fid
            + 1.0 * band_fid
        )
        metrics = {
            "hot_drop": float(hot_drop),
            "spiky_drop": float(spiky_drop),
            "saliency_fidelity": float(saliency_fid),
            "edge_fidelity": float(edge_fid),
            "band_fidelity": float(band_fid),
        }
        reason = "优先比较噪声改善、边缘保真和频带保真。"
        return OuterSelectionScore(float(score), metrics, reason)

    family = stage_name
    score_func = _SCORE_FUNCTIONS.get(family)
    if score_func is None:
        return OuterSelectionScore(0.0, {}, f"未定义 stage outer score: {stage_name}")
    trial = score_func(before, after, {}, context.header_info)
    return OuterSelectionScore(float(trial.score), dict(trial.metrics), trial.reason)


def _select_seed_trials(
    context: AutoTuneContext,
    coarse_trials: list[dict[str, Any]],
    family: str | None = None,
) -> list[dict[str, Any]]:
    plan = _get_search_plan(context.search_mode)
    valid_trials = sorted(
        [trial for trial in coarse_trials if trial.get("valid", True)],
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    if not valid_trials:
        return []

    target_k = max(1, int(plan["refine_top_k"]))
    if len(valid_trials) <= target_k:
        return valid_trials

    seeds = [valid_trials[0]]
    pool = valid_trials[1 : min(len(valid_trials), max(8, target_k * 4))]
    best_score = max(float(valid_trials[0].get("score", 0.0)), 1.0)

    while len(seeds) < target_k and pool:
        candidate = max(
            pool,
            key=lambda trial: (
                0.72 * (float(trial.get("score", 0.0)) / best_score)
                + 0.28 * _min_param_distance(trial, seeds)
            ),
        )
        seeds.append(candidate)
        pool.remove(candidate)

    if len(seeds) < target_k:
        for trial in valid_trials:
            if trial not in seeds:
                seeds.append(trial)
            if len(seeds) >= target_k:
                break
    return seeds


def _refine_candidate_trials(
    method_key: str,
    data: np.ndarray,
    base_params: dict[str, Any],
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    context: AutoTuneContext,
    seed_trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not seed_trials:
        return []
    method_info = PROCESSING_METHODS[method_key]
    family = method_info["auto_tune_family"]
    plan = _get_search_plan(context.search_mode)
    refined: list[dict[str, Any]] = []
    n_samples, n_traces = int(data.shape[0]), int(data.shape[1])

    for seed_rank, trial in enumerate(seed_trials, start=1):
        params = trial.get("params", {})
        if family == "background" and "ntraces" in params:
            center = int(params["ntraces"])
            values = _sanitize_int_candidates(
                [
                    int(round(center * 0.70)),
                    int(round(center * 0.85)),
                    center,
                    int(round(center * 1.15)),
                    int(round(center * 1.35)),
                    center - 4,
                    center + 4,
                ],
                n_traces,
                minimum=3,
                upper=max(3, n_traces),
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                refined.append({"ntraces": int(value), "_seed_rank": seed_rank})
        elif family == "drift" and "window" in params:
            center = int(params["window"])
            values = _sanitize_int_candidates(
                [
                    int(round(center * 0.70)),
                    int(round(center * 0.85)),
                    center,
                    int(round(center * 1.15)),
                    int(round(center * 1.30)),
                ],
                n_samples,
                minimum=8,
                upper=max(16, n_samples // 2),
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                refined.append({"window": int(value), "_seed_rank": seed_rank})
        elif family == "background" and method_key == "svd_bg" and "rank" in params:
            center = int(params["rank"])
            values = _sanitize_int_candidates(
                [center - 2, center - 1, center, center + 1, center + 2],
                max(1, min(data.shape) - 1),
                minimum=1,
                upper=max(1, min(data.shape) - 1),
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                refined.append({"rank": int(value), "_seed_rank": seed_rank})
        elif family == "fk" and "angle_low" in params and "angle_high" in params:
            center_low = int(params["angle_low"])
            center_high = int(params["angle_high"])
            center_taper = int(params.get("taper_width", 4))
            low_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    [
                        center_low - 4,
                        center_low - 2,
                        center_low,
                        center_low + 2,
                        center_low + 4,
                    ],
                    90,
                    minimum=0,
                    upper=80,
                ),
                budget=max(2, plan["fine_budget"] // 2),
                center=center_low,
            )
            high_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    [
                        center_high - 6,
                        center_high - 3,
                        center_high,
                        center_high + 3,
                        center_high + 6,
                    ],
                    90,
                    minimum=10,
                    upper=90,
                ),
                budget=max(2, plan["fine_budget"] // 2),
                center=center_high,
            )
            taper_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    [
                        center_taper - 2,
                        center_taper - 1,
                        center_taper,
                        center_taper + 1,
                        center_taper + 2,
                    ],
                    20,
                    minimum=0,
                    upper=20,
                ),
                budget=max(2, min(3, plan["fine_budget"])),
                center=center_taper,
            )
            trials = []
            for angle_low, angle_high, taper_width in itertools.product(
                low_values, high_values, taper_values
            ):
                if int(angle_high) - int(angle_low) >= 8:
                    trials.append(
                        {
                            "angle_low": int(angle_low),
                            "angle_high": int(angle_high),
                            "taper_width": int(taper_width),
                            "_seed_rank": seed_rank,
                        }
                    )
            refined.extend(
                _trim_trial_candidates(
                    trials,
                    budget=plan["fine_budget"],
                    center_params={
                        "angle_low": center_low,
                        "angle_high": center_high,
                        "taper_width": center_taper,
                    },
                )
            )
        elif (
            family == "denoise"
            and method_key == "svd_subspace"
            and "rank_end" in params
        ):
            center = int(params["rank_end"])
            rank_start = int(params.get("rank_start", 1))
            rank_limit = max(rank_start, min(data.shape))
            values = _sanitize_int_candidates(
                [
                    int(round(center * 0.75)),
                    int(round(center * 0.90)),
                    center,
                    int(round(center * 1.10)),
                    int(round(center * 1.25)),
                ],
                rank_limit,
                minimum=rank_start,
                upper=rank_limit,
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                if int(value) >= rank_start:
                    refined.append(
                        {
                            "rank_start": rank_start,
                            "rank_end": int(value),
                            "_seed_rank": seed_rank,
                        }
                    )
        elif (
            family == "denoise" and method_key == "wavelet_svd" and "rank_end" in params
        ):
            center = int(params["rank_end"])
            rank_start = int(params.get("rank_start", 1))
            rank_limit = max(rank_start, min(data.shape))
            rank_values = _sanitize_int_candidates(
                [
                    int(round(center * 0.80)),
                    int(round(center * 0.90)),
                    center,
                    int(round(center * 1.10)),
                    int(round(center * 1.20)),
                ],
                rank_limit,
                minimum=rank_start,
                upper=rank_limit,
            )
            rank_values = _trim_numeric_candidates(
                rank_values, budget=max(2, plan["fine_budget"] // 2), center=center
            )
            threshold_center = float(params.get("threshold", 0.05))
            threshold_values = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [
                        threshold_center * 0.8,
                        threshold_center * 0.95,
                        threshold_center,
                        threshold_center * 1.1,
                        threshold_center * 1.25,
                    ],
                    minimum=0.01,
                ),
                budget=max(2, min(3, plan["fine_budget"])),
                center=threshold_center,
            )
            levels_center = int(params.get("levels", 2))
            levels_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    [levels_center - 1, levels_center, levels_center + 1],
                    data.shape[0],
                    minimum=1,
                    upper=8,
                ),
                budget=max(1, min(3, plan["fine_budget"])),
                center=levels_center,
            )
            wavelet_name = str(params.get("wavelet", "db4"))
            for rank_end, threshold, levels in itertools.product(
                rank_values, threshold_values, levels_values
            ):
                if int(rank_end) >= rank_start:
                    refined.append(
                        {
                            "wavelet": wavelet_name,
                            "levels": int(levels),
                            "threshold": float(threshold),
                            "rank_start": rank_start,
                            "rank_end": int(rank_end),
                            "_seed_rank": seed_rank,
                        }
                    )
        elif family == "gain" and method_key == "sec_gain":
            center_gain = float(params.get("gain_max", 5.0))
            center_power = float(params.get("power", 1.0))
            gain_span = max(0.40, center_gain * 0.18)
            power_span = max(0.06, center_power * 0.15)
            dim_budget = max(
                3,
                min(4, int(np.ceil(np.sqrt(max(4, plan["fine_budget"] * 2))))),
            )
            gain_candidates = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [
                        center_gain - gain_span,
                        center_gain - gain_span * 0.5,
                        center_gain,
                        center_gain + gain_span * 0.5,
                        center_gain + gain_span,
                    ],
                    minimum=float(
                        base_params.get("gain_min", params.get("gain_min", 1.0))
                    ),
                ),
                budget=dim_budget,
                center=center_gain,
            )
            power_candidates = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [
                        center_power - power_span,
                        center_power - power_span * 0.5,
                        center_power,
                        center_power + power_span * 0.5,
                        center_power + power_span,
                    ],
                    minimum=0.2,
                ),
                budget=dim_budget,
                center=center_power,
            )
            for gain_value, power_value in itertools.product(
                gain_candidates, power_candidates
            ):
                refined.append(
                    {
                        "gain_min": float(
                            base_params.get("gain_min", params.get("gain_min", 1.0))
                        ),
                        "gain_max": max(1.0, float(gain_value)),
                        "power": max(0.2, float(power_value)),
                        "_seed_rank": seed_rank,
                    }
                )
        elif family == "gain" and method_key == "agcGain" and "window" in params:
            center = int(params["window"])
            values = _sanitize_int_candidates(
                [
                    int(round(center * 0.80)),
                    int(round(center * 0.90)),
                    center,
                    int(round(center * 1.10)),
                    int(round(center * 1.25)),
                ],
                n_samples,
                minimum=3,
                upper=n_samples,
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                refined.append({"window": int(value), "_seed_rank": seed_rank})
        elif family == "gain" and method_key == "compensatingGain":
            center_min = float(params.get("gain_min", 1.0))
            center_max = float(params.get("gain_max", 5.0))
            min_values = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [center_min - 0.2, center_min, center_min + 0.2], minimum=0.1
                ),
                budget=3,
                center=center_min,
            )
            max_span = max(0.4, center_max * 0.12)
            max_values = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [
                        center_max - max_span,
                        center_max - max_span * 0.5,
                        center_max,
                        center_max + max_span * 0.5,
                        center_max + max_span,
                    ],
                    minimum=0.2,
                ),
                budget=max(3, min(4, plan["fine_budget"])),
                center=center_max,
            )
            for gain_min, gain_max in itertools.product(min_values, max_values):
                if float(gain_max) > float(gain_min):
                    refined.append(
                        {
                            "gain_min": float(gain_min),
                            "gain_max": float(gain_max),
                            "_seed_rank": seed_rank,
                        }
                    )
        elif family == "zero_time":
            time_step = _resolve_time_step_ns(data.shape[0], header_info)
            center_idx = int(params.get("_zero_idx", 0))
            detector = str(params.get("_detector", "threshold"))
            threshold = float(params.get("_threshold", 0.05) or 0.05)
            backup = int(params.get("_backup_samples", 0))
            for delta in [0, -1, 1, -2, 2, -4, 4][: plan["fine_budget"]]:
                zero_idx = max(0, center_idx + delta)
                refined.append(
                    {
                        "new_zero_time": float(zero_idx) * time_step,
                        "_detector": detector,
                        "_threshold": max(0.001, threshold),
                        "_backup_samples": backup,
                        "_zero_idx": zero_idx,
                        "_seed_rank": seed_rank,
                    }
                )
        elif family == "impulse" and "ntraces" in params:
            center = int(params["ntraces"])
            values = _sanitize_int_candidates(
                [center - 2, center - 1, center, center + 1, center + 2],
                n_traces,
                minimum=3,
                upper=max(3, n_traces),
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                refined.append({"ntraces": int(value), "_seed_rank": seed_rank})

    return _dedupe_candidates(refined)


def _build_zero_time_candidates(
    data: np.ndarray,
    config: dict[str, Any],
    base_params: dict[str, Any],
    header_info: dict[str, Any],
    context: AutoTuneContext,
    stage: str = "coarse",
    budget: int = 8,
) -> list[dict[str, Any]]:
    n_samples = int(data.shape[0])
    time_step_ns = _resolve_time_step_ns(n_samples, header_info)
    search_ratio = float(config.get("search_ratio", 0.35))
    detectors = config.get("detectors", ["threshold", "peak", "first_break"])
    base_threshold = float(
        np.clip(0.02 + 0.02 * context.features.get("first_break_std", 0.0), 0.02, 0.14)
    )
    thresholds = _sanitize_float_candidates(
        list(config.get("thresholds", []))
        + [base_threshold * s for s in [0.75, 1.0, 1.25, 1.5]],
        minimum=0.001,
    )[: max(3, min(len(config.get("thresholds", [])) + 4, budget))]
    base_backup = max(1, int(round(context.features.get("first_break_std", 0.0) / 2.0)))
    backups = _sanitize_int_candidates(
        list(config.get("backup_samples", []))
        + [base_backup, base_backup + 2, base_backup + 4],
        n_samples,
        minimum=0,
        upper=max(1, n_samples - 1),
    )
    if stage == "coarse":
        detectors = list(detectors)[: min(len(detectors), 3)]
        thresholds = thresholds[: min(len(thresholds), 4)]
        backups = backups[: min(len(backups), 3)]
    else:
        thresholds = thresholds[: min(len(thresholds), 5)]
        backups = backups[: min(len(backups), 4)]

    trials: list[dict[str, Any]] = []
    seen: set[tuple[float, str, int, float]] = set()
    for detector, threshold, backup in itertools.product(
        detectors, thresholds, backups
    ):
        fb_idx = detect_first_break_indices(
            data,
            method=str(detector),
            threshold=float(threshold),
            search_ratio=search_ratio,
        )
        zero_idx = max(0, median_first_break(fb_idx) - int(backup))
        new_zero_time = float(zero_idx) * time_step_ns
        key = (
            round(new_zero_time, 6),
            str(detector),
            int(backup),
            round(float(threshold), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        trials.append(
            {
                "new_zero_time": new_zero_time,
                "_detector": str(detector),
                "_threshold": float(threshold),
                "_backup_samples": int(backup),
                "_zero_idx": int(zero_idx),
                "_first_break_std_before": float(np.std(fb_idx)),
            }
        )

    fallback = float(base_params.get("new_zero_time", 5.0))
    fallback_key = (round(fallback, 6), "manual", 0, 0.0)
    if fallback_key not in seen:
        trials.append(
            {
                "new_zero_time": fallback,
                "_detector": "manual",
                "_threshold": 0.0,
                "_backup_samples": 0,
                "_zero_idx": int(round(fallback / max(time_step_ns, 1.0e-6))),
            }
        )
    return _dedupe_candidates(trials)


def _resolve_time_step_ns(n_samples: int, header_info: dict[str, Any]) -> float:
    total_time_ns = header_info.get("total_time_ns") if header_info else None
    if total_time_ns and float(total_time_ns) > 0:
        return float(total_time_ns) / max(1, int(n_samples))
    return 48.0 / max(1, int(n_samples))


def _sanitize_int_candidates(
    values: list[Any],
    data_limit: int,
    minimum: int,
    upper: int,
) -> list[int]:
    cleaned: list[int] = []
    for value in values:
        try:
            current = int(round(float(value)))
        except Exception:
            continue
        current = max(int(minimum), min(int(upper), current))
        if current not in cleaned:
            cleaned.append(current)
    if not cleaned:
        cleaned = [max(int(minimum), min(int(upper), max(1, data_limit // 8 or 1)))]
    return cleaned


def _adaptive_trace_windows(
    n_traces: int,
    configured_values: list[Any],
    base_value: Any | None,
    minimum: int,
    upper: int,
) -> list[int]:
    """Build trace-window candidates using configured values plus data-adaptive ratios."""
    ratio_values = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5, 0.8]
    adaptive = [max(minimum, int(round(n_traces * ratio))) for ratio in ratio_values]
    if base_value is not None:
        try:
            base_int = int(round(float(base_value)))
        except Exception:
            base_int = None
        if base_int is not None:
            adaptive.extend(
                [max(minimum, base_int - 20), base_int, min(upper, base_int + 20)]
            )

    values = _sanitize_int_candidates(
        list(configured_values) + adaptive,
        n_traces,
        minimum=minimum,
        upper=upper,
    )
    return values


def _sanitize_float_candidates(values: list[Any], minimum: float) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        try:
            current = float(value)
        except Exception:
            continue
        current = max(float(minimum), current)
        if current not in cleaned:
            cleaned.append(current)
    return cleaned or [float(minimum)]


def _build_drift_windows(
    n_samples: int,
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[int]:
    low_freq = float(context.features.get("low_freq_ratio", 0.1))
    base_window = int(round(n_samples * (0.05 + 0.30 * low_freq)))
    base_window = max(8, min(max(16, n_samples // 2), base_window))
    multipliers = (
        [0.55, 0.8, 1.0, 1.25, 1.6, 2.0]
        if stage == "coarse"
        else [0.7, 0.85, 1.0, 1.15, 1.3]
    )
    values = [int(round(base_window * scale)) for scale in multipliers]
    values = _sanitize_int_candidates(
        list(config.get("window", [])) + values,
        n_samples,
        minimum=8,
        upper=max(16, n_samples // 2),
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_window)
    ]


def _build_background_windows(
    n_traces: int,
    context: AutoTuneContext,
    config: dict[str, Any],
    base_value: Any | None,
    stage: str,
    budget: int | None = None,
) -> list[int]:
    corr_length = max(2, int(context.features.get("lateral_corr_length", 6)))
    base_window = max(5, min(n_traces, int(round(corr_length * 4.0))))
    multipliers = (
        [0.6, 1.0, 1.5, 2.0, 3.0, 4.0]
        if stage == "coarse"
        else [0.75, 0.9, 1.0, 1.1, 1.25]
    )
    adaptive = [int(round(base_window * scale)) for scale in multipliers]
    values = _sanitize_int_candidates(
        list(config.get("ntraces", []))
        + adaptive
        + ([base_value] if base_value is not None else []),
        n_traces,
        minimum=3,
        upper=max(3, n_traces),
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_window)
    ]


def _build_background_rank_candidates(
    data: np.ndarray,
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[int]:
    rank_limit = max(1, min(data.shape) - 1)
    elbow = max(1, min(rank_limit, int(context.features.get("singular_elbow_rank", 2))))
    values = [
        1,
        max(1, elbow - 1),
        elbow,
        min(rank_limit, elbow + 1),
        min(rank_limit, elbow + 2),
    ]
    if stage == "coarse":
        values.append(min(rank_limit, elbow * 2))
    values = _sanitize_int_candidates(
        list(config.get("rank", [])) + values,
        rank_limit,
        minimum=1,
        upper=rank_limit,
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=elbow)
    ]


def _build_fk_filter_trials(
    base_params: dict[str, Any],
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    low_default = int(base_params.get("angle_low", 12))
    high_default = int(base_params.get("angle_high", 55))
    taper_default = int(base_params.get("taper_width", 4))

    if stage == "coarse":
        low_values = _sanitize_int_candidates(
            list(config.get("angle_low", [])) + [low_default],
            90,
            minimum=0,
            upper=80,
        )
        high_values = _sanitize_int_candidates(
            list(config.get("angle_high", [])) + [high_default],
            90,
            minimum=10,
            upper=90,
        )
        taper_values = _sanitize_int_candidates(
            list(config.get("taper_width", [])) + [taper_default],
            20,
            minimum=0,
            upper=20,
        )
    else:
        low_values = [low_default]
        high_values = [high_default]
        taper_values = [taper_default]

    trials = []
    for angle_low, angle_high, taper_width in itertools.product(
        low_values, high_values, taper_values
    ):
        if int(angle_high) - int(angle_low) >= 8:
            trials.append(
                {
                    "angle_low": int(angle_low),
                    "angle_high": int(angle_high),
                    "taper_width": int(taper_width),
                }
            )

    return _trim_trial_candidates(
        trials,
        budget=budget,
        center_params={
            "angle_low": low_default,
            "angle_high": high_default,
            "taper_width": taper_default,
        },
    )


def _build_subspace_rank_end_candidates(
    data: np.ndarray,
    context: AutoTuneContext,
    config: dict[str, Any],
    base_value: Any | None,
    stage: str,
    budget: int | None = None,
) -> list[int]:
    rank_limit = max(2, min(data.shape))
    elbow = max(2, min(rank_limit, int(context.features.get("singular_elbow_rank", 4))))
    base_rank = max(
        4, min(rank_limit, int(base_value) if base_value is not None else elbow * 3)
    )
    values = [
        max(4, elbow),
        max(6, elbow + 2),
        max(8, elbow * 2),
        base_rank,
        min(rank_limit, max(base_rank + 4, elbow * 3)),
        min(rank_limit, max(base_rank + 8, elbow * 4)),
    ]
    if stage == "fine":
        values.extend(
            [
                int(round(base_rank * 0.85)),
                int(round(base_rank * 0.95)),
                int(round(base_rank * 1.05)),
                int(round(base_rank * 1.15)),
            ]
        )
    values = _sanitize_int_candidates(
        list(config.get("rank_end", [])) + values,
        rank_limit,
        minimum=2,
        upper=rank_limit,
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_rank)
    ]


def _build_sec_gain_candidates(
    context: AutoTuneContext,
    config: dict[str, Any],
    gain_min: float,
    stage: str,
    budget: int | None = None,
) -> tuple[list[float], list[float]]:
    attenuation_ratio = float(context.features.get("attenuation_ratio", 1.8))
    base_gain_max = np.clip(
        2.2 + 1.6 * np.log1p(max(0.0, attenuation_ratio - 1.0)), 2.5, 12.0
    )
    base_power = np.clip(
        0.55 + 0.32 * np.log1p(max(0.0, attenuation_ratio - 1.0)), 0.5, 2.2
    )
    gain_scales = (
        [0.65, 0.85, 1.0, 1.2, 1.45]
        if stage == "coarse"
        else [0.85, 0.95, 1.0, 1.08, 1.18]
    )
    power_scales = (
        [0.7, 0.9, 1.0, 1.15, 1.35]
        if stage == "coarse"
        else [0.85, 0.95, 1.0, 1.08, 1.18]
    )
    dim_budget = max(
        3,
        min(
            5,
            int(np.ceil(np.sqrt(max(4, float(budget or 8) * 1.6)))),
        ),
    )
    gain_values = _trim_numeric_candidates(
        _sanitize_float_candidates(
            list(config.get("gain_max", [])) + [base_gain_max * s for s in gain_scales],
            minimum=gain_min,
        ),
        budget=dim_budget,
        center=base_gain_max,
    )
    power_values = _trim_numeric_candidates(
        _sanitize_float_candidates(
            list(config.get("power", [])) + [base_power * s for s in power_scales],
            minimum=0.2,
        ),
        budget=dim_budget,
        center=base_power,
    )
    return [float(value) for value in gain_values], [
        float(value) for value in power_values
    ]


def _build_agc_windows(
    n_samples: int,
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[int]:
    attenuation_ratio = float(context.features.get("attenuation_ratio", 1.5))
    base_window = int(
        round(n_samples * np.clip(0.035 + 0.015 * attenuation_ratio, 0.03, 0.18))
    )
    values = [
        int(round(base_window * scale))
        for scale in (
            [0.6, 0.85, 1.0, 1.25, 1.6]
            if stage == "coarse"
            else [0.8, 0.9, 1.0, 1.1, 1.25]
        )
    ]
    values = _sanitize_int_candidates(
        list(config.get("window", [])) + values,
        n_samples,
        minimum=3,
        upper=n_samples,
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_window)
    ]


def _build_compensating_gain_candidates(
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> tuple[list[float], list[float]]:
    attenuation_ratio = float(context.features.get("attenuation_ratio", 1.6))
    base_max = np.clip(
        2.5 + 1.4 * np.log1p(max(0.0, attenuation_ratio - 1.0)), 2.0, 10.0
    )
    max_budget = 4 if stage == "coarse" else 3
    min_budget = 3
    max_values = _trim_numeric_candidates(
        _sanitize_float_candidates(
            list(config.get("gain_max", []))
            + [
                base_max * s
                for s in (
                    [0.7, 0.9, 1.0, 1.2, 1.4]
                    if stage == "coarse"
                    else [0.85, 0.95, 1.0, 1.1, 1.2]
                )
            ],
            minimum=0.2,
        ),
        budget=max_budget,
        center=base_max,
    )
    min_values = _trim_numeric_candidates(
        _sanitize_float_candidates(
            list(config.get("gain_min", [])) + [0.8, 1.0, 1.2], minimum=0.1
        ),
        budget=min_budget,
        center=1.0,
    )
    return [float(value) for value in min_values], [
        float(value) for value in max_values
    ]


def _build_impulse_windows(
    n_traces: int,
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[int]:
    spiky = float(context.features.get("spikiness", 0.0))
    hot = float(context.features.get("hot_pixel_ratio", 0.0))
    severity = spiky + 8.0 * hot
    base_window = (
        3 if severity < 0.5 else 5 if severity < 1.5 else 7 if severity < 3.0 else 9
    )
    values = (
        [base_window - 2, base_window, base_window + 2]
        if stage == "coarse"
        else [
            base_window - 2,
            base_window - 1,
            base_window,
            base_window + 1,
            base_window + 2,
        ]
    )
    values = _sanitize_int_candidates(
        list(config.get("ntraces", [])) + values,
        n_traces,
        minimum=3,
        upper=max(3, n_traces),
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_window)
    ]


def _trial_signature(params: dict[str, Any]) -> str:
    return str(sorted(_public_params(params).items()))


def _trim_numeric_candidates(
    values: list[Any],
    budget: int | None,
    center: float | int | None = None,
) -> list[Any]:
    cleaned: list[Any] = []
    for value in values:
        if value not in cleaned:
            cleaned.append(value)
    if budget is None or budget <= 0 or len(cleaned) <= int(budget):
        return cleaned

    ordered = sorted(cleaned, key=float)
    budget = int(max(1, budget))
    if center is None:
        if budget >= len(ordered):
            return ordered
        positions = np.linspace(0, len(ordered) - 1, num=budget)
        selected = []
        for pos in positions:
            value = ordered[int(round(float(pos)))]
            if value not in selected:
                selected.append(value)
        return sorted(selected, key=float)

    center_value = float(center)
    selected: list[Any] = []
    closest = min(ordered, key=lambda item: abs(float(item) - center_value))
    selected.append(closest)
    if budget > 1 and ordered[0] not in selected:
        selected.append(ordered[0])
    if budget > 2 and ordered[-1] not in selected:
        selected.append(ordered[-1])

    remaining = [item for item in ordered if item not in selected]
    while len(selected) < budget and remaining:
        best = max(
            remaining,
            key=lambda item: (
                min(abs(float(item) - float(chosen)) for chosen in selected),
                -abs(float(item) - center_value),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return sorted(selected, key=float)


def _safe_ratio(numerator: float, denominator: float, floor: float = 1.0e-6) -> float:
    return float(numerator) / max(abs(float(denominator)), float(floor))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _param_distance(lhs: dict[str, Any], rhs: dict[str, Any]) -> float:
    a = _public_params(lhs.get("params", {}))
    b = _public_params(rhs.get("params", {}))
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    parts: list[float] = []
    for key in keys:
        va = a.get(key)
        vb = b.get(key)
        if _is_number(va) and _is_number(vb):
            scale = max(abs(float(va)), abs(float(vb)), 1.0)
            parts.append(min(1.5, abs(float(va) - float(vb)) / scale))
        else:
            parts.append(0.0 if va == vb else 1.0)
    return float(np.mean(parts)) if parts else 0.0


def _min_param_distance(trial: dict[str, Any], seeds: list[dict[str, Any]]) -> float:
    if not seeds:
        return 1.0
    return float(min(_param_distance(trial, seed) for seed in seeds))


def _trim_trial_candidates(
    trials: list[dict[str, Any]],
    budget: int | None,
    center_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """按“中心值 + 两端 + 分散覆盖”裁剪 trial 候选。"""
    unique_trials = _dedupe_candidates(trials)
    if budget is None or budget <= 0 or len(unique_trials) <= int(budget):
        return unique_trials

    budget = int(max(1, budget))
    center_trial = min(
        unique_trials,
        key=lambda trial: _param_distance(
            {"params": trial}, {"params": dict(center_params)}
        ),
    )
    selected = [center_trial]
    remaining = [trial for trial in unique_trials if trial is not center_trial]

    while len(selected) < budget and remaining:
        candidate = max(
            remaining,
            key=lambda trial: _min_param_distance(
                {"params": trial}, [{"params": item} for item in selected]
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)

    return selected


def _build_trial_failure_record(
    context: AutoTuneContext,
    params: dict[str, Any],
    stage: str,
    exc: Exception,
) -> dict[str, Any]:
    message = f"候选执行失败: {type(exc).__name__}: {exc}"
    roi_used = bool(context.roi_source != "full" and context.roi_bounds is not None)
    return {
        "params": dict(params),
        "score": float(INVALID_TRIAL_SCORE),
        "metrics": {},
        "roi_metrics": {},
        "penalties": {"execution_failure": float(FAILURE_PENALTY)},
        "reason": message,
        "stage": stage,
        "roi_score": float(INVALID_TRIAL_SCORE),
        "full_score": float(INVALID_TRIAL_SCORE),
        "guard_score": -float(FAILURE_PENALTY),
        "roi_used": roi_used,
        "roi_source": context.roi_source,
        "valid": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "cached": False,
    }


def _summarize_failed_trials(trials: list[dict[str, Any]], method_key: str) -> str:
    errors = [
        str(trial.get("error") or trial.get("reason") or "未知错误")
        for trial in trials
        if not trial.get("valid", True)
    ]
    if not errors:
        return f"自动选参失败：{method_key} 没有可用候选。"
    preview = "；".join(errors[:3])
    return f"自动选参失败：{method_key} 所有候选均未成功执行。示例错误：{preview}"


def _selection_stability(trials: list[dict[str, Any]]) -> tuple[float, float]:
    ordered = sorted(
        [trial for trial in trials if trial.get("valid", True)],
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    if not ordered:
        return 0.0, 0.0
    if len(ordered) == 1:
        return 1.0, 1.0
    best = float(ordered[0].get("score", 0.0))
    second = float(ordered[1].get("score", 0.0))
    scale = max(1.0, abs(best))
    margin = max(0.0, (best - second) / scale)
    confidence = float(np.clip(0.35 + 2.0 * margin, 0.0, 1.0))
    return float(margin), confidence


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        signature = _trial_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique


def _merge_trials(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for trial in group:
            signature = _trial_signature(trial.get("params", {}))
            current = merged.get(signature)
            if current is None:
                merged[signature] = trial
                continue
            current_valid = bool(current.get("valid", True))
            trial_valid = bool(trial.get("valid", True))
            if trial_valid and not current_valid:
                merged[signature] = trial
                continue
            if trial_valid == current_valid and float(trial.get("score", 0.0)) > float(
                current.get("score", 0.0)
            ):
                merged[signature] = trial
    return sorted(
        merged.values(),
        key=lambda item: (
            1 if item.get("valid", True) else 0,
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )


def _score_zero_time(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    zero_idx = int(params.get("_zero_idx", params.get("_backup_samples", 0)))
    detector = str(params.get("_detector", "threshold"))
    threshold = float(params.get("_threshold", 0.05) or 0.05)
    before_pre = pre_zero_energy_ratio(before, zero_idx)
    after_pre = pre_zero_energy_ratio(after, zero_idx)
    after_std = first_break_std(
        after,
        method=detector if detector != "manual" else "threshold",
        threshold=max(threshold, 0.03),
    )
    sharpness = first_break_sharpness(after, max(1, zero_idx))
    sharp_norm = sharpness / max(float(np.mean(np.abs(after))), 1.0e-6)
    std_norm = after_std / max(float(before.shape[0]), 1.0)

    penalties = {
        "pre_zero_regression": max(0.0, after_pre - before_pre) * 4.0,
        "large_shift": max(
            0.0,
            params.get("new_zero_time", 0.0)
            - _resolve_time_step_ns(before.shape[0], header_info)
            * before.shape[0]
            * 0.2,
        )
        / max(
            _resolve_time_step_ns(before.shape[0], header_info) * before.shape[0], 1.0
        ),
    }
    score = (
        -3.2 * after_pre - 1.8 * std_norm + 1.6 * sharp_norm - sum(penalties.values())
    )
    metrics = {
        "pre_zero_energy_ratio": float(after_pre),
        "first_break_std": float(after_std),
        "first_break_sharpness": float(sharp_norm),
    }
    reason = (
        f"零时前能量={after_pre:.4f}，首波离散度={after_std:.2f}，锐度={sharp_norm:.3f}；"
        f"检测={detector}，回退样本={zero_idx}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_drift(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    baseline_before = baseline_bias(before)
    baseline_after = baseline_bias(after)
    baseline_drop = relative_reduction(baseline_before, baseline_after)

    low_freq_before = low_freq_energy_ratio(before)
    low_freq_after = low_freq_energy_ratio(after)
    low_freq_drop = relative_reduction(low_freq_before, low_freq_after)

    band_ratio_raw = target_band_energy_ratio(before, after)
    band_keep = float(np.clip(band_ratio_raw, 0.0, 1.25))
    band_fidelity = ratio_fidelity(band_ratio_raw, target=1.0, tol=0.18)
    peak_ratio_raw = _safe_ratio(
        float(np.percentile(np.abs(after), 99.0)),
        float(np.percentile(np.abs(before), 99.0)),
    )
    peak_ratio = float(np.clip(peak_ratio_raw, 0.0, 1.35))
    peak_fidelity = ratio_fidelity(peak_ratio_raw, target=1.0, tol=0.22)
    penalties = {
        "baseline_regression": max(0.0, -baseline_drop) * 2.5,
        "low_freq_regression": max(0.0, -low_freq_drop) * 3.0,
        "band_distortion": max(0.0, 0.72 - band_fidelity) * 2.5,
        "peak_distortion": max(0.0, 0.72 - peak_fidelity) * 1.5,
    }
    score = (
        2.4 * baseline_drop
        + 2.8 * low_freq_drop
        + 1.6 * band_fidelity
        + 0.6 * peak_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "baseline_bias_before": float(baseline_before),
        "baseline_bias_after": float(baseline_after),
        "baseline_drop": float(baseline_drop),
        "low_freq_energy_ratio_before": float(low_freq_before),
        "low_freq_energy_ratio_after": float(low_freq_after),
        "low_freq_drop": float(low_freq_drop),
        "target_band_energy_ratio": float(band_ratio_raw),
        "target_band_keep": float(band_keep),
        "target_band_fidelity": float(band_fidelity),
        "peak_ratio": float(peak_ratio),
        "peak_ratio_raw": float(peak_ratio_raw),
        "peak_fidelity": float(peak_fidelity),
    }
    reason = (
        f"基线改善={baseline_drop:.3f}，低频改善={low_freq_drop:.3f}，"
        f"目标频带保真={band_fidelity:.3f}，峰值保真={peak_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_background(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    coherence = horizontal_coherence(after)
    saliency = local_saliency_preservation(before, after)
    edge = edge_preservation(before, after)
    peak_ratio = float(
        np.percentile(np.abs(after), 99.0)
        / max(np.percentile(np.abs(before), 99.0), 1.0e-6)
    )
    penalties = {
        "edge_loss": max(0.0, 0.72 - edge) * 3.0,
        "target_drop": max(0.0, 0.60 - peak_ratio) * 2.5,
    }
    score = -3.0 * coherence + 2.2 * saliency + 1.2 * edge - sum(penalties.values())
    metrics = {
        "horizontal_coherence": float(coherence),
        "local_saliency_preservation": float(saliency),
        "edge_preservation": float(edge),
        "peak_ratio": float(peak_ratio),
    }
    reason = f"背景一致性={coherence:.4f}，显著结构保留={saliency:.3f}，边缘保留={edge:.3f}。"
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_fk_filter(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    coherence_before = horizontal_coherence(before)
    coherence_after = horizontal_coherence(after)
    coherence_drop = relative_reduction(coherence_before, coherence_after)
    saliency_ratio = local_saliency_preservation(before, after)
    saliency_fidelity = ratio_fidelity(saliency_ratio, target=1.0, tol=0.18)
    edge_ratio = edge_preservation(before, after)
    edge_fidelity = ratio_fidelity(edge_ratio, target=1.0, tol=0.18)
    band_ratio_raw = target_band_energy_ratio(before, after)
    band_keep = float(np.clip(band_ratio_raw, 0.0, 1.25))
    band_fidelity = ratio_fidelity(band_ratio_raw, target=1.0, tol=0.20)
    peak_ratio_raw = float(
        np.percentile(np.abs(after), 99.0)
        / max(np.percentile(np.abs(before), 99.0), 1.0e-6)
    )
    peak_fidelity = ratio_fidelity(peak_ratio_raw, target=1.0, tol=0.25)
    penalties = {
        "coherence_regression": max(0.0, -coherence_drop) * 2.6,
        "saliency_distortion": max(0.0, 0.72 - saliency_fidelity) * 2.2,
        "edge_distortion": max(0.0, 0.75 - edge_fidelity) * 2.2,
        "band_distortion": max(0.0, 0.72 - band_fidelity) * 2.8,
        "peak_distortion": max(0.0, 0.70 - peak_fidelity) * 1.8,
    }
    score = (
        2.5 * coherence_drop
        + 1.4 * saliency_fidelity
        + 1.3 * edge_fidelity
        + 1.8 * band_fidelity
        + 0.5 * peak_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "horizontal_coherence_before": float(coherence_before),
        "horizontal_coherence_after": float(coherence_after),
        "horizontal_coherence_drop": float(coherence_drop),
        "local_saliency_preservation": float(saliency_ratio),
        "local_saliency_fidelity": float(saliency_fidelity),
        "edge_preservation": float(edge_ratio),
        "edge_fidelity": float(edge_fidelity),
        "target_band_energy_ratio": float(band_ratio_raw),
        "target_band_keep": float(band_keep),
        "target_band_fidelity": float(band_fidelity),
        "peak_ratio": float(peak_ratio_raw),
        "peak_fidelity": float(peak_fidelity),
    }
    reason = (
        f"背景改善={coherence_drop:.3f}，显著结构保真={saliency_fidelity:.3f}，"
        f"边缘保真={edge_fidelity:.3f}，目标频带保真={band_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_gain(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    rms_cv = depth_rms_cv(after)
    deep_before = deep_zone_contrast(before)
    deep_after = deep_zone_contrast(after)
    deep_gain_raw = _safe_ratio(deep_after, deep_before)
    deep_gain_effective = float(np.log1p(np.clip(deep_gain_raw, 0.0, 12.0)))
    clip = clipping_ratio(after)
    hot = hot_pixel_ratio(after)
    shallow_before = float(np.std(_zone_slice(before, 0.0, 0.2)))
    shallow_after = float(np.std(_zone_slice(after, 0.0, 0.2)))
    shallow_blow_raw = _safe_ratio(shallow_after, shallow_before)
    shallow_blow = float(np.clip(shallow_blow_raw, 0.0, 4.0))
    penalties = {
        "clipping": clip * 10.0,
        "hot_pixels": hot * 6.0,
        "shallow_blowup": max(0.0, shallow_blow - 2.3) * 1.0,
    }
    score = -2.0 * rms_cv + 2.6 * deep_gain_effective - sum(penalties.values())
    metrics = {
        "depth_rms_cv": float(rms_cv),
        "deep_zone_contrast": float(deep_after),
        "deep_gain_ratio": float(deep_gain_raw),
        "deep_gain_effective": float(deep_gain_effective),
        "clipping_ratio": float(clip),
        "hot_pixel_ratio": float(hot),
        "shallow_blow_ratio": float(shallow_blow_raw),
    }
    reason = (
        f"深部对比提升={deep_gain_raw:.3f}，有效提升={deep_gain_effective:.3f}，"
        f"深浅均衡CV={rms_cv:.4f}，过曝比={clip:.4f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_impulse(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    hot_before = hot_pixel_ratio(before)
    hot_after = hot_pixel_ratio(after)
    hot_drop = relative_reduction(hot_before, hot_after)
    spiky_before = kurtosis_or_spikiness(before)
    spiky_after = kurtosis_or_spikiness(after)
    spiky_drop = relative_reduction(spiky_before, spiky_after)
    edge_ratio = edge_preservation(before, after)
    edge_fidelity = ratio_fidelity(edge_ratio, target=1.0, tol=0.18)
    penalties = {
        "hot_regression": max(0.0, -hot_drop) * 3.0,
        "spiky_regression": max(0.0, -spiky_drop) * 2.5,
        "edge_distortion": max(0.0, 0.75 - edge_fidelity) * 2.4,
    }
    score = (
        2.6 * hot_drop
        + 1.9 * spiky_drop
        + 1.5 * edge_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "hot_pixel_ratio_before": float(hot_before),
        "hot_pixel_ratio_after": float(hot_after),
        "hot_pixel_drop": float(hot_drop),
        "spikiness_before": float(spiky_before),
        "spikiness_after": float(spiky_after),
        "spikiness_drop": float(spiky_drop),
        "edge_preservation": float(edge_ratio),
        "edge_fidelity": float(edge_fidelity),
    }
    reason = (
        f"热点改善={hot_drop:.3f}，尖峰改善={spiky_drop:.3f}，"
        f"边缘保真={edge_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _score_denoise(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    hot_before = hot_pixel_ratio(before)
    hot_after = hot_pixel_ratio(after)
    hot_drop = relative_reduction(hot_before, hot_after)
    spiky_before = kurtosis_or_spikiness(before)
    spiky_after = kurtosis_or_spikiness(after)
    spiky_drop = relative_reduction(spiky_before, spiky_after)
    edge_ratio = edge_preservation(before, after)
    edge_fidelity = ratio_fidelity(edge_ratio, target=1.0, tol=0.18)
    saliency_ratio = local_saliency_preservation(before, after)
    saliency_fidelity = ratio_fidelity(saliency_ratio, target=1.0, tol=0.18)
    band_keep_raw = target_band_energy_ratio(before, after)
    band_keep = float(np.clip(band_keep_raw, 0.0, 1.25))
    band_fidelity = ratio_fidelity(band_keep_raw, target=1.0, tol=0.20)
    penalties = {
        "hot_regression": max(0.0, -hot_drop) * 2.5,
        "spiky_regression": max(0.0, -spiky_drop) * 2.5,
        "edge_distortion": max(0.0, 0.72 - edge_fidelity) * 2.4,
        "saliency_distortion": max(0.0, 0.72 - saliency_fidelity) * 2.4,
        "band_distortion": max(0.0, 0.72 - band_fidelity) * 2.4,
    }
    score = (
        2.2 * hot_drop
        + 1.8 * spiky_drop
        + 1.5 * saliency_fidelity
        + 1.2 * edge_fidelity
        + 1.1 * band_fidelity
        - sum(penalties.values())
    )
    metrics = {
        "hot_pixel_ratio_before": float(hot_before),
        "hot_pixel_ratio_after": float(hot_after),
        "hot_pixel_drop": float(hot_drop),
        "spikiness_before": float(spiky_before),
        "spikiness_after": float(spiky_after),
        "spikiness_drop": float(spiky_drop),
        "edge_preservation": float(edge_ratio),
        "edge_fidelity": float(edge_fidelity),
        "local_saliency_preservation": float(saliency_ratio),
        "local_saliency_fidelity": float(saliency_fidelity),
        "target_band_energy_ratio": float(band_keep_raw),
        "target_band_keep": float(band_keep),
        "target_band_fidelity": float(band_fidelity),
    }
    reason = (
        f"热点改善={hot_drop:.3f}，尖峰改善={spiky_drop:.3f}，边缘保真={edge_fidelity:.3f}，"
        f"显著结构保真={saliency_fidelity:.3f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


def _zone_slice(data: np.ndarray, start_ratio: float, end_ratio: float) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    start = int(np.floor(arr.shape[0] * start_ratio))
    end = int(np.ceil(arr.shape[0] * end_ratio))
    return arr[max(0, start) : max(start + 1, min(arr.shape[0], end)), :]


def _slice_depth_band(
    data: np.ndarray, start_ratio: float, end_ratio: float
) -> np.ndarray:
    """Slice a depth band using normalized start/end ratios."""
    return _zone_slice(data, start_ratio, end_ratio)


def _slice_bounds(data: np.ndarray, bounds: dict[str, int]) -> np.ndarray:
    """Slice a 2D array with validated ROI/context bounds."""
    arr = np.asarray(data, dtype=np.float64)
    t0 = max(0, min(int(bounds.get("time_start_idx", 0)), arr.shape[0] - 1))
    t1 = max(t0 + 1, min(int(bounds.get("time_end_idx", arr.shape[0])), arr.shape[0]))
    d0 = max(0, min(int(bounds.get("dist_start_idx", 0)), arr.shape[1] - 1))
    d1 = max(d0 + 1, min(int(bounds.get("dist_end_idx", arr.shape[1])), arr.shape[1]))
    return arr[t0:t1, d0:d1]


def _score_motion_comp(
    before: np.ndarray,
    after: np.ndarray,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> TrialScore:
    """Score motion compensation height normalization.

    Rewards:
    - Reduced depth RMS CV (more consistent trace amplitudes)
    - Improved horizontal coherence
    Penalizes:
    - Clipping artifacts
    - Regression in amplitude consistency
    """
    from core.quality_metrics import depth_rms_cv, horizontal_coherence, clipping_ratio

    rms_cv_before = depth_rms_cv(before)
    rms_cv_after = depth_rms_cv(after)
    rms_cv_drop = relative_reduction(rms_cv_before, rms_cv_after)

    coh_before = horizontal_coherence(before)
    coh_after = horizontal_coherence(after)
    coh_gain = _safe_ratio(coh_after, coh_before) - 1.0

    clip = clipping_ratio(after)

    penalties = {
        "clipping": clip * 8.0,
        "rms_cv_regression": max(0.0, -rms_cv_drop) * 4.0,
    }

    score = 2.5 * rms_cv_drop + 1.8 * max(0.0, coh_gain) - sum(penalties.values())

    metrics = {
        "depth_rms_cv_before": float(rms_cv_before),
        "depth_rms_cv_after": float(rms_cv_after),
        "depth_rms_cv_drop": float(rms_cv_drop),
        "horizontal_coherence_before": float(coh_before),
        "horizontal_coherence_after": float(coh_after),
        "coherence_gain": float(coh_gain),
        "clipping_ratio": float(clip),
    }

    reason = (
        f"深浅均衡CV改善={rms_cv_drop:.3f}，横向相干增益={coh_gain:.3f}，"
        f"过曝比={clip:.4f}。"
    )
    return TrialScore(
        score=float(score), metrics=metrics, penalties=penalties, reason=reason
    )


_SCORE_FUNCTIONS: dict[
    str, Callable[[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]], TrialScore]
] = {
    "zero_time": _score_zero_time,
    "drift": _score_drift,
    "background": _score_background,
    "fk": _score_fk_filter,
    "denoise": _score_denoise,
    "gain": _score_gain,
    "impulse": _score_impulse,
    "motion_comp": _score_motion_comp,
}


def _public_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if not str(k).startswith("_")}


def _penalty_sum(trial: dict[str, Any]) -> float:
    penalties = trial.get("penalties", {}) or {}
    return _penalty_sum_from_dict(penalties)


def _penalty_sum_from_dict(penalties: dict[str, Any]) -> float:
    return float(sum(float(value) for value in (penalties or {}).values()))


def _effective_metrics(trial: dict[str, Any]) -> dict[str, Any]:
    if trial.get("roi_used") and trial.get("roi_metrics"):
        return trial.get("roi_metrics", {}) or {}
    return trial.get("metrics", {}) or {}


def _compute_pareto_front(
    family: str, trials: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute a simple Pareto front across primary objectives and penalties."""
    if not trials:
        return []

    objectives = [_trial_objectives(family, trial) for trial in trials]
    pareto_indices: list[int] = []
    for idx, current in enumerate(objectives):
        dominated = False
        for jdx, other in enumerate(objectives):
            if idx == jdx:
                continue
            if _dominates(other, current):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(idx)

    front = [trials[idx] for idx in pareto_indices]
    front.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return front


def _trial_objectives(family: str, trial: dict[str, Any]) -> tuple[float, ...]:
    metrics = _effective_metrics(trial)
    penalty = -_penalty_sum(trial)
    if family == "zero_time":
        return (
            -float(metrics.get("pre_zero_energy_ratio", 0.0)),
            -float(metrics.get("first_break_std", 0.0)),
            float(metrics.get("first_break_sharpness", 0.0)),
            penalty,
        )
    if family == "drift":
        band = float(metrics.get("target_band_fidelity", 0.0))
        return (
            float(metrics.get("baseline_drop", 0.0)),
            float(metrics.get("low_freq_drop", 0.0)),
            band,
            penalty,
        )
    if family == "background":
        return (
            -float(metrics.get("horizontal_coherence", 0.0)),
            float(metrics.get("local_saliency_preservation", 0.0)),
            float(metrics.get("edge_preservation", 0.0)),
            penalty,
        )
    if family == "fk":
        return (
            float(metrics.get("horizontal_coherence_drop", 0.0)),
            float(metrics.get("local_saliency_fidelity", 0.0)),
            float(metrics.get("edge_fidelity", 0.0)),
            float(metrics.get("target_band_fidelity", 0.0)),
            penalty,
        )
    if family == "denoise":
        return (
            float(metrics.get("hot_pixel_drop", 0.0)),
            float(metrics.get("spikiness_drop", 0.0)),
            float(metrics.get("local_saliency_fidelity", 0.0)),
            float(metrics.get("edge_fidelity", 0.0)),
            float(metrics.get("target_band_fidelity", 0.0)),
            penalty,
        )
    if family == "gain":
        deep_effective = float(
            metrics.get("deep_gain_effective", metrics.get("deep_gain_ratio", 0.0))
        )
        return (
            -float(metrics.get("depth_rms_cv", 0.0)),
            deep_effective,
            -float(metrics.get("clipping_ratio", 0.0)),
            -float(metrics.get("hot_pixel_ratio", 0.0)),
            penalty,
        )
    if family == "impulse":
        return (
            float(metrics.get("hot_pixel_drop", 0.0)),
            float(metrics.get("spikiness_drop", 0.0)),
            float(metrics.get("edge_fidelity", 0.0)),
            penalty,
        )
    return (float(trial.get("score", 0.0)), penalty)


def _dominates(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> bool:
    return all(l >= r for l, r in zip(lhs, rhs)) and any(
        l > r for l, r in zip(lhs, rhs)
    )


def _build_profiles(
    family: str,
    trials: list[dict[str, Any]],
    pareto_trials: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    pool = (
        pareto_trials
        or sorted(trials, key=lambda item: float(item.get("score", 0.0)), reverse=True)[
            : min(8, len(trials))
        ]
    )
    used: set[int] = set()
    profiles: dict[str, dict[str, Any]] = {}
    for key in ["conservative", "balanced", "aggressive"]:
        trial = _pick_profile_trial(family, key, pool, used)
        if trial is None and trials:
            trial = max(trials, key=lambda item: float(item.get("score", 0.0)))
        if trial is None:
            continue
        used.add(id(trial))
        profiles[key] = {
            "label": PROFILE_LABELS[key],
            "params": _public_params(trial.get("params", {})),
            "score": float(trial.get("score", 0.0)),
            "metrics": dict(_effective_metrics(trial)),
            "penalties": dict(trial.get("penalties", {})),
            "reason": _profile_reason(family, key, trial),
            "stage": str(trial.get("stage", "coarse")),
        }
    return profiles


def _pick_profile_trial(
    family: str,
    profile_key: str,
    pool: list[dict[str, Any]],
    used: set[int],
) -> dict[str, Any] | None:
    available = [trial for trial in pool if id(trial) not in used] or list(pool)
    if not available:
        return None
    return max(
        available, key=lambda trial: _profile_priority(family, profile_key, trial)
    )


def _profile_priority(family: str, profile_key: str, trial: dict[str, Any]) -> float:
    score = float(trial.get("score", 0.0))
    metrics = _effective_metrics(trial)
    penalty = _penalty_sum(trial)
    if profile_key == "balanced":
        return score - 0.35 * penalty

    if family == "background":
        coherence = float(metrics.get("horizontal_coherence", 0.0))
        saliency = float(metrics.get("local_saliency_preservation", 0.0))
        edge = float(metrics.get("edge_preservation", 0.0))
        if profile_key == "conservative":
            return 2.2 * saliency + 1.8 * edge - 4.0 * penalty - coherence
        return -3.0 * coherence + 1.2 * saliency + 0.8 * edge - 1.5 * penalty

    if family == "fk":
        coherence_gain = float(metrics.get("horizontal_coherence_drop", 0.0))
        saliency_fid = float(metrics.get("local_saliency_fidelity", 0.0))
        edge_fid = float(metrics.get("edge_fidelity", 0.0))
        band_fid = float(metrics.get("target_band_fidelity", 0.0))
        if profile_key == "conservative":
            return (
                1.8 * saliency_fid
                + 1.7 * edge_fid
                + 1.6 * band_fid
                + 1.2 * coherence_gain
                - 4.0 * penalty
            )
        return (
            2.5 * coherence_gain
            + 1.2 * saliency_fid
            + 1.0 * edge_fid
            + 1.4 * band_fid
            - 1.8 * penalty
        )

    if family == "denoise":
        hot_drop = float(metrics.get("hot_pixel_drop", 0.0))
        spiky_drop = float(metrics.get("spikiness_drop", 0.0))
        edge_fid = float(metrics.get("edge_fidelity", 0.0))
        saliency_fid = float(metrics.get("local_saliency_fidelity", 0.0))
        band_fid = float(metrics.get("target_band_fidelity", 0.0))
        if profile_key == "conservative":
            return (
                1.8 * saliency_fid
                + 1.7 * edge_fid
                + 1.2 * band_fid
                + 1.5 * hot_drop
                + 1.2 * spiky_drop
                - 4.0 * penalty
            )
        return (
            2.2 * hot_drop
            + 1.8 * spiky_drop
            + 1.2 * saliency_fid
            + 0.9 * edge_fid
            + 0.9 * band_fid
            - 2.0 * penalty
        )

    if family == "gain":
        deep_effective = float(
            metrics.get("deep_gain_effective", metrics.get("deep_gain_ratio", 0.0))
        )
        clip = float(metrics.get("clipping_ratio", 0.0))
        hot = float(metrics.get("hot_pixel_ratio", 0.0))
        if profile_key == "conservative":
            return 1.6 * deep_effective - 10.0 * clip - 7.0 * hot - 4.0 * penalty
        return 3.1 * deep_effective - 4.5 * clip - 3.0 * hot - 2.0 * penalty

    if family == "drift":
        band = float(metrics.get("target_band_fidelity", 0.0))
        low_drop = float(metrics.get("low_freq_drop", 0.0))
        baseline_drop = float(metrics.get("baseline_drop", 0.0))
        if profile_key == "conservative":
            return 1.8 * band + 1.5 * low_drop + 1.4 * baseline_drop - 3.5 * penalty
        return 2.2 * low_drop + 1.8 * baseline_drop + 1.2 * band - 1.4 * penalty

    if family == "zero_time":
        sharp = float(metrics.get("first_break_sharpness", 0.0))
        pre_zero = float(metrics.get("pre_zero_energy_ratio", 0.0))
        std = float(metrics.get("first_break_std", 0.0))
        if profile_key == "conservative":
            return -pre_zero - 0.8 * std - 4.0 * penalty
        return 1.8 * sharp - pre_zero - 1.2 * penalty

    if family == "impulse":
        hot_drop = float(metrics.get("hot_pixel_drop", 0.0))
        spiky_drop = float(metrics.get("spikiness_drop", 0.0))
        edge_fid = float(metrics.get("edge_fidelity", 0.0))
        if profile_key == "conservative":
            return 1.8 * edge_fid + 1.6 * hot_drop + 1.2 * spiky_drop - 4.0 * penalty
        return 2.4 * hot_drop + 1.8 * spiky_drop + 1.1 * edge_fid - 2.0 * penalty

    return score - penalty


def _profile_reason(family: str, profile_key: str, trial: dict[str, Any]) -> str:
    base = str(trial.get("reason", ""))
    prefix = {
        "conservative": "更保守，优先压低过处理风险。",
        "balanced": "更均衡，优先综合评分与稳定性。",
        "aggressive": "更增强，优先提升主目标效果。",
    }.get(profile_key, "")
    return f"{prefix} {base}".strip()

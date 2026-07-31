#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trial execution, contextual scoring and seed selection."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mygpr.domain.autotune.quality_metrics import (
    edge_preservation,
    horizontal_coherence,
    hot_pixel_ratio,
    kurtosis_or_spikiness,
    local_saliency_preservation,
    ratio_fidelity,
    relative_reduction,
    target_band_energy_ratio,
    weighted_score_parts,
)
from mygpr.application.autotune.context import _get_search_plan
from mygpr.application.autotune.ports import AutoTuneDependencies
from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.application.autotune.diagnostics import (
    _attach_constraint_metadata,
    _build_trial_failure_record,
)
from mygpr.application.autotune.errors import AutoTuneCancelled
from mygpr.application.autotune.scoring import (
    _SCORE_FUNCTIONS,
    _slice_bounds,
    _slice_depth_band,
)
from mygpr.application.autotune.utils import (
    _min_param_distance,
    _penalty_sum_from_dict,
    _trial_signature,
)
from mygpr.domain.autotune.models import AutoTuneContext, OuterSelectionScore, TrialScore


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
    dependencies: AutoTuneDependencies,
    execution_context: ExecutionContext | None = None,
    evaluated_cache: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(trial_params_list)
    before_arr = np.asarray(data, dtype=np.float32)
    for idx, trial_params in enumerate(trial_params_list, start=1):
        if cancel_checker and bool(cancel_checker()):
            raise AutoTuneCancelled("用户已取消参数推荐")
        if progress_callback is not None:
            progress_callback(idx - 1, total, f"{stage_message} {idx}/{total}")

        constraint_result = dependencies.constraints.constrain(
            method_key,
            dict(trial_params),
            data.shape,
            header_info,
        )
        effective_trial_params = dict(constraint_result.effective_params)
        signature = _trial_signature(effective_trial_params)
        cached = evaluated_cache.get(signature) if evaluated_cache else None
        if cached is not None:
            record = dict(cached)
            record["stage"] = stage
            record["cached"] = True
            _attach_constraint_metadata(record, constraint_result)
            results.append(record)
            continue

        runtime_params = dict(base_params)
        runtime_params.update(effective_trial_params)
        try:
            result_value = dependencies.executor.execute(
                ProcessingRequest(
                    data=data,
                    method_id=method_key,
                    params=runtime_params,
                    header_info=header_info,
                    trace_metadata=trace_metadata,
                ),
                execution_context,
            )
            result = result_value.data
            result_meta = result_value.metadata
            method_runtime_warnings = list(result_value.runtime_warnings or [])
            record = _score_trial_with_context(
                context,
                family,
                score_func,
                before_arr,
                np.asarray(result, dtype=np.float32),
                effective_trial_params,
                dict(header_info or {}),
                stage=stage,
            )
            _attach_constraint_metadata(
                record,
                constraint_result,
                method_runtime_warnings=method_runtime_warnings,
            )
            record["cached"] = False
            record["valid"] = bool(np.isfinite(record.get("score", 0.0)))
            if not record["valid"]:
                raise ValueError("候选得分出现 NaN/Inf")
        except AutoTuneCancelled:
            raise
        except Exception as exc:
            record = _build_trial_failure_record(
                context,
                effective_trial_params,
                stage,
                exc,
            )
            _attach_constraint_metadata(record, constraint_result)

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
        family in {"fk", "frequency", "denoise", "impulse"}
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline-level auto-tuning with per-step scoring and rollback support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from core.auto_tune import auto_tune_method
from core.gprmax_truth_metrics import compute_ground_truth_metrics
from core.methods_registry import PROCESSING_METHODS
from core.preset_profiles import GUI_PRESETS_V1, RECOMMENDED_RUN_PROFILES
from core.processing_engine import (
    clone_header_info,
    clone_trace_metadata,
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.quality_metrics import (
    auto_roi_bounds,
    compute_benchmark_metrics,
    ratio_fidelity,
)
from core.scalar_utils import to_int
from core.app_errors import MyGPRError


ProgressCallback = Callable[[int, int, str], None]
CancelChecker = Callable[[], bool]

SCORE_REJECT_DELTA = -0.02
SCORE_REVIEW_DELTA = 0.02
LOW_CONFIDENCE_THRESHOLD = 0.45
LOW_MARGIN_THRESHOLD = 0.03


class AutoTunePipelineError(MyGPRError):
    """Raised when a pipeline-level auto-tune run cannot be executed."""


@dataclass
class PipelineCandidate:
    """One final branch produced by a pipeline-level auto-tune run."""

    name: str
    source: str
    pipeline: list[str]
    params_by_method: dict[str, dict[str, Any]]
    result: np.ndarray
    metadata: dict[str, Any]
    metrics: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    auto_tune_results: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class PipelineStepRecord:
    """Per-step evidence for reports and research comparison."""

    index: int
    method_key: str
    method_name: str
    manual_params: dict[str, Any]
    auto_params: dict[str, Any]
    manual_before: np.ndarray
    manual_after: np.ndarray
    auto_before: np.ndarray
    auto_after: np.ndarray
    manual_metrics: dict[str, float]
    auto_metrics: dict[str, float]
    metric_delta: dict[str, float]
    auto_tune_result: dict[str, Any] | None
    manual_roi_before: dict[str, int]
    manual_roi_after: dict[str, int]
    auto_roi_before: dict[str, int]
    auto_roi_after: dict[str, int]
    warnings: dict[str, list[str]]
    risk_flags: list[str]
    recommendation: str
    reason: str
    rolled_back_to_manual: bool = False


@dataclass
class AutoTunePipelineRun:
    """Full pipeline-level auto-tune result."""

    input_shape: tuple[int, int]
    pipeline: list[str]
    baseline_profile_key: str | None
    manual_source: str
    roi_info: dict[str, Any]
    ground_truth_info: dict[str, Any]
    steps: list[PipelineStepRecord]
    manual: PipelineCandidate
    automatic: PipelineCandidate
    metric_delta: dict[str, float]
    overall_recommendation: str
    risk_flags: list[str]


@dataclass
class _BranchState:
    current: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    params_by_method: dict[str, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    auto_tune_results: dict[str, dict[str, Any]] = field(default_factory=dict)


def run_auto_tune_pipeline(
    data: np.ndarray,
    *,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    pipeline: list[str] | None = None,
    manual_params_by_method: dict[str, dict[str, Any]] | None = None,
    locked_params_by_method: dict[str, dict[str, Any]] | None = None,
    baseline_profile_key: str | None = None,
    roi_spec: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
    search_mode: str = "standard",
    rollback_on_reject: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_checker: CancelChecker | None = None,
) -> AutoTunePipelineRun:
    """Run a manual baseline and an auto-tuned pipeline step by step."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise AutoTunePipelineError("流程级自动选参需要二维非空 B-scan 数据")
    if ground_truth is None and isinstance(header_info, dict):
        embedded_ground_truth = header_info.get("ground_truth")
        if isinstance(embedded_ground_truth, dict):
            ground_truth = embedded_ground_truth

    profile_key = baseline_profile_key or (
        None if pipeline is not None else "uav_gpr_experience_baseline_v1"
    )
    pipeline_order = _resolve_pipeline(pipeline, profile_key)
    if not pipeline_order:
        raise AutoTunePipelineError("流程级自动选参 pipeline 不能为空")
    _validate_pipeline(pipeline_order)

    roi_info = _resolve_roi_info(arr, roi_spec, ground_truth)
    manual_source = "current_ui_params" if manual_params_by_method else "experience_profile"
    manual_params = _resolve_manual_params(
        pipeline_order,
        profile_key,
        manual_params_by_method or {},
    )
    locked_params = {
        str(key): dict(value) for key, value in (locked_params_by_method or {}).items()
    }
    for method_key, params in locked_params.items():
        if method_key in pipeline_order:
            manual_params[method_key] = dict(params)

    manual_state = _BranchState(
        current=np.array(arr, dtype=np.float32, copy=True),
        header_info=clone_header_info(header_info),
        trace_metadata=clone_trace_metadata(trace_metadata),
        params_by_method={key: dict(value) for key, value in manual_params.items()},
    )
    auto_state = _BranchState(
        current=np.array(arr, dtype=np.float32, copy=True),
        header_info=clone_header_info(header_info),
        trace_metadata=clone_trace_metadata(trace_metadata),
        params_by_method={key: dict(value) for key, value in manual_params.items()},
    )

    manual_roi = dict(roi_info["bounds"])
    auto_roi = dict(roi_info["bounds"])
    steps: list[PipelineStepRecord] = []

    for idx, method_key in enumerate(pipeline_order, start=1):
        if cancel_checker and bool(cancel_checker()):
            raise AutoTunePipelineError("用户已取消流程级自动选参")
        if progress_callback is not None:
            progress_callback(idx - 1, len(pipeline_order), f"自动选参流程: {method_key}")

        step, manual_roi, auto_roi = _run_pipeline_step(
            index=idx,
            method_key=method_key,
            manual_state=manual_state,
            auto_state=auto_state,
            manual_roi=manual_roi,
            auto_roi=auto_roi,
            locked_params_by_method=locked_params,
            ground_truth=ground_truth,
            search_mode=search_mode,
            rollback_on_reject=rollback_on_reject,
            cancel_checker=cancel_checker,
        )
        steps.append(step)

    if progress_callback is not None:
        progress_callback(len(pipeline_order), len(pipeline_order), "自动选参流程完成")

    manual_metrics = _compute_branch_metrics(
        arr,
        manual_state.current,
        dict(roi_info["bounds"]),
        manual_roi,
        ground_truth,
    )
    auto_metrics = _compute_branch_metrics(
        arr,
        auto_state.current,
        dict(roi_info["bounds"]),
        auto_roi,
        ground_truth,
    )
    metric_delta = _compute_metric_delta(manual_metrics, auto_metrics)
    overall_risk_flags = _dedupe_flags(
        flag for step in steps for flag in step.risk_flags
    )
    overall_recommendation = _overall_recommendation(
        steps,
        metric_delta,
        overall_risk_flags,
    )

    manual_candidate = PipelineCandidate(
        name="人工 baseline",
        source=manual_source,
        pipeline=list(pipeline_order),
        params_by_method=manual_state.params_by_method,
        result=manual_state.current,
        metadata={
            "header_info": manual_state.header_info,
            "trace_metadata": manual_state.trace_metadata,
        },
        metrics=manual_metrics,
        warnings=manual_state.warnings,
    )
    auto_candidate = PipelineCandidate(
        name="自动选参",
        source="auto_tune_pipeline",
        pipeline=list(pipeline_order),
        params_by_method=auto_state.params_by_method,
        result=auto_state.current,
        metadata={
            "header_info": auto_state.header_info,
            "trace_metadata": auto_state.trace_metadata,
        },
        metrics=auto_metrics,
        warnings=auto_state.warnings,
        auto_tune_results=auto_state.auto_tune_results,
    )

    return AutoTunePipelineRun(
        input_shape=(int(arr.shape[0]), int(arr.shape[1])),
        pipeline=list(pipeline_order),
        baseline_profile_key=profile_key,
        manual_source=manual_source,
        roi_info=roi_info,
        ground_truth_info=_ground_truth_info(ground_truth),
        steps=steps,
        manual=manual_candidate,
        automatic=auto_candidate,
        metric_delta=metric_delta,
        overall_recommendation=overall_recommendation,
        risk_flags=overall_risk_flags,
    )


def to_summary_dict(result: AutoTunePipelineRun) -> dict[str, Any]:
    """Return a JSON-safe summary without raw B-scan arrays."""
    return {
        "input_shape": list(result.input_shape),
        "pipeline": list(result.pipeline),
        "baseline_profile_key": result.baseline_profile_key,
        "manual_source": result.manual_source,
        "roi_info": _json_safe(result.roi_info),
        "ground_truth_info": _json_safe(result.ground_truth_info),
        "overall_recommendation": result.overall_recommendation,
        "risk_flags": list(result.risk_flags),
        "metric_delta": _json_safe(result.metric_delta),
        "manual": _candidate_summary(result.manual),
        "automatic": _candidate_summary(result.automatic),
        "steps": [_step_summary(step) for step in result.steps],
    }


def _run_pipeline_step(
    *,
    index: int,
    method_key: str,
    manual_state: _BranchState,
    auto_state: _BranchState,
    manual_roi: dict[str, int],
    auto_roi: dict[str, int],
    locked_params_by_method: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any] | None,
    search_mode: str,
    rollback_on_reject: bool,
    cancel_checker: CancelChecker | None,
) -> tuple[PipelineStepRecord, dict[str, int], dict[str, int]]:
    method_info = PROCESSING_METHODS[method_key]
    method_name = str(method_info.get("name") or method_key)
    manual_before = np.array(manual_state.current, copy=True)
    auto_before = np.array(auto_state.current, copy=True)
    manual_roi_before = _clamp_bounds(manual_before.shape, manual_roi)
    auto_roi_before = _clamp_bounds(auto_before.shape, auto_roi)

    manual_params = dict(manual_state.params_by_method.get(method_key, {}))
    auto_params, tune_result = _resolve_auto_params(
        method_key=method_key,
        auto_state=auto_state,
        base_params=manual_params,
        roi_bounds=auto_roi_before,
        locked_params_by_method=locked_params_by_method,
        search_mode=search_mode,
        cancel_checker=cancel_checker,
    )

    manual_after, manual_meta = _execute_method(
        manual_before,
        method_key,
        manual_params,
        manual_state.header_info,
        manual_state.trace_metadata,
        cancel_checker,
    )
    auto_after, auto_meta = _execute_method(
        auto_before,
        method_key,
        auto_params,
        auto_state.header_info,
        auto_state.trace_metadata,
        cancel_checker,
    )

    manual_header_after = merge_result_header_info(
        manual_state.header_info,
        manual_meta,
        manual_after.shape,
    )
    manual_trace_after = merge_result_trace_metadata(
        manual_state.trace_metadata,
        manual_meta,
    )
    auto_header_after = merge_result_header_info(
        auto_state.header_info,
        auto_meta,
        auto_after.shape,
    )
    auto_trace_after = merge_result_trace_metadata(
        auto_state.trace_metadata,
        auto_meta,
    )

    manual_roi_after = _map_roi_after_step(
        manual_roi_before,
        manual_before.shape,
        manual_after.shape,
    )
    auto_roi_after = _map_roi_after_step(
        auto_roi_before,
        auto_before.shape,
        auto_after.shape,
    )
    manual_metrics = _compute_branch_metrics(
        manual_before,
        manual_after,
        manual_roi_before,
        manual_roi_after,
        ground_truth,
    )
    auto_metrics = _compute_branch_metrics(
        auto_before,
        auto_after,
        auto_roi_before,
        auto_roi_after,
        ground_truth,
    )
    metric_delta = _compute_metric_delta(manual_metrics, auto_metrics)
    risk_flags, recommendation, reason = _assess_step_risk(
        manual_metrics,
        auto_metrics,
        tune_result,
    )
    rolled_back = bool(rollback_on_reject and recommendation == "keep_manual")

    manual_state.current = manual_after
    manual_state.header_info = manual_header_after
    manual_state.trace_metadata = manual_trace_after
    manual_state.params_by_method[method_key] = dict(manual_params)
    manual_warnings = _extract_warning_messages(manual_meta)
    manual_state.warnings.extend(manual_warnings)

    auto_state.params_by_method[method_key] = dict(auto_params)
    auto_warnings = _extract_warning_messages(auto_meta)
    auto_state.warnings.extend(auto_warnings)
    if tune_result is not None:
        auto_state.auto_tune_results[method_key] = _compact_auto_tune_result(
            tune_result
        )

    if rolled_back:
        auto_state.current = np.array(manual_after, copy=True)
        auto_state.header_info = clone_header_info(manual_header_after)
        auto_state.trace_metadata = clone_trace_metadata(manual_trace_after)
        next_auto_roi = dict(manual_roi_after)
    else:
        auto_state.current = auto_after
        auto_state.header_info = auto_header_after
        auto_state.trace_metadata = auto_trace_after
        next_auto_roi = dict(auto_roi_after)

    step = PipelineStepRecord(
        index=index,
        method_key=method_key,
        method_name=method_name,
        manual_params=manual_params,
        auto_params=auto_params,
        manual_before=manual_before,
        manual_after=manual_after,
        auto_before=auto_before,
        auto_after=auto_after,
        manual_metrics=manual_metrics,
        auto_metrics=auto_metrics,
        metric_delta=metric_delta,
        auto_tune_result=_compact_auto_tune_result(tune_result)
        if tune_result is not None
        else None,
        manual_roi_before=manual_roi_before,
        manual_roi_after=manual_roi_after,
        auto_roi_before=auto_roi_before,
        auto_roi_after=auto_roi_after,
        warnings={"manual": manual_warnings, "automatic": auto_warnings},
        risk_flags=risk_flags,
        recommendation=recommendation,
        reason=reason,
        rolled_back_to_manual=rolled_back,
    )
    return step, dict(manual_roi_after), next_auto_roi


def _resolve_auto_params(
    *,
    method_key: str,
    auto_state: _BranchState,
    base_params: dict[str, Any],
    roi_bounds: dict[str, int],
    locked_params_by_method: dict[str, dict[str, Any]],
    search_mode: str,
    cancel_checker: CancelChecker | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if method_key in locked_params_by_method:
        return dict(locked_params_by_method[method_key]), None

    method_info = PROCESSING_METHODS[method_key]
    if not method_info.get("auto_tune_enabled"):
        return dict(base_params), None

    tune_result = auto_tune_method(
        auto_state.current,
        method_key,
        header_info=auto_state.header_info,
        trace_metadata=auto_state.trace_metadata,
        base_params=base_params,
        roi_spec={
            "mode": "manual",
            "bounds": dict(roi_bounds),
            "label": "流程当前 ROI",
        },
        search_mode=search_mode,
        cancel_checker=cancel_checker,
    )
    tuned_params = dict(base_params)
    tuned_params.update(tune_result.get("recommended_params", {}) or {})
    return tuned_params, tune_result


def _execute_method(
    current: np.ndarray,
    method_key: str,
    params: dict[str, Any],
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    cancel_checker: CancelChecker | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime_params = prepare_runtime_params(
        method_key,
        params,
        header_info,
        trace_metadata,
        current.shape,
    )
    result, meta = run_processing_method(
        current,
        method_key,
        runtime_params,
        cancel_checker=cancel_checker,
    )
    return np.asarray(result, dtype=np.float32), dict(meta or {})


def _resolve_pipeline(
    pipeline: list[str] | None,
    baseline_profile_key: str | None,
) -> list[str]:
    if pipeline is not None:
        return [str(method_key) for method_key in pipeline]
    if not baseline_profile_key:
        return []
    profile = RECOMMENDED_RUN_PROFILES.get(baseline_profile_key)
    if not profile:
        raise AutoTunePipelineError(f"未知经验 baseline profile: {baseline_profile_key}")
    return [str(method_key) for method_key in profile.get("order", [])]


def _validate_pipeline(pipeline: list[str]) -> None:
    unknown = [method_key for method_key in pipeline if method_key not in PROCESSING_METHODS]
    if unknown:
        raise AutoTunePipelineError(f"pipeline 包含未知方法: {', '.join(unknown)}")


def _resolve_manual_params(
    pipeline: list[str],
    baseline_profile_key: str | None,
    manual_params_by_method: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    defaults = _profile_method_params(baseline_profile_key)
    resolved: dict[str, dict[str, Any]] = {}
    for method_key in pipeline:
        params = dict(defaults.get(method_key, {}))
        params.update(manual_params_by_method.get(method_key, {}))
        resolved[method_key] = params
    return resolved


def _profile_method_params(
    baseline_profile_key: str | None,
) -> dict[str, dict[str, Any]]:
    if not baseline_profile_key:
        return {}
    profile = RECOMMENDED_RUN_PROFILES.get(baseline_profile_key, {})
    preset_key = profile.get("preset_key")
    params: dict[str, dict[str, Any]] = {}
    if preset_key and preset_key in GUI_PRESETS_V1:
        for method_key, method_params in GUI_PRESETS_V1[preset_key].get(
            "method_params",
            {},
        ).items():
            params[str(method_key)] = dict(method_params)
    for method_key, method_params in profile.get("method_params", {}).items():
        params[str(method_key)] = dict(method_params)
    return params


def _resolve_roi_info(
    data: np.ndarray,
    roi_spec: dict[str, Any] | None,
    ground_truth: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = dict(roi_spec or {})
    mode = str(spec.get("mode") or "")
    if mode in {"manual", "crop"} and isinstance(spec.get("bounds"), dict):
        bounds = _clamp_bounds(data.shape, spec["bounds"])
        return {
            "source": mode,
            "label": str(spec.get("label") or ("手动 ROI" if mode == "manual" else "裁剪区")),
            "bounds": bounds,
        }
    if mode == "auto":
        return {
            "source": "auto",
            "label": str(spec.get("label") or "自动 ROI"),
            "bounds": _clamp_bounds(data.shape, auto_roi_bounds(data)),
        }
    if ground_truth and isinstance(ground_truth.get("analysis_roi"), dict):
        scenario_id = ground_truth.get("scenario_id") or "ground_truth"
        return {
            "source": "ground_truth",
            "label": f"{scenario_id} analysis ROI",
            "bounds": _clamp_bounds(data.shape, ground_truth["analysis_roi"]),
        }
    return {
        "source": "full",
        "label": "全图",
        "bounds": {
            "time_start_idx": 0,
            "time_end_idx": int(data.shape[0]),
            "dist_start_idx": 0,
            "dist_end_idx": int(data.shape[1]),
        },
    }


def _clamp_bounds(shape: tuple[int, int], bounds: dict[str, Any]) -> dict[str, int]:
    samples, traces = int(shape[0]), int(shape[1])
    t0 = max(
        0,
        min(to_int(bounds.get("time_start_idx"), default=0), max(samples - 1, 0)),
    )
    t1 = max(
        t0 + 1,
        min(to_int(bounds.get("time_end_idx"), default=samples), samples),
    )
    d0 = max(
        0,
        min(to_int(bounds.get("dist_start_idx"), default=0), max(traces - 1, 0)),
    )
    d1 = max(
        d0 + 1,
        min(to_int(bounds.get("dist_end_idx"), default=traces), traces),
    )
    return {
        "time_start_idx": int(t0),
        "time_end_idx": int(t1),
        "dist_start_idx": int(d0),
        "dist_end_idx": int(d1),
    }


def _map_roi_after_step(
    bounds: dict[str, int],
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
) -> dict[str, int]:
    row_shift = max(0, int(before_shape[0]) - int(after_shape[0]))
    col_shift = max(0, int(before_shape[1]) - int(after_shape[1]))
    shifted = {
        "time_start_idx": int(bounds["time_start_idx"]) - row_shift,
        "time_end_idx": int(bounds["time_end_idx"]) - row_shift,
        "dist_start_idx": int(bounds["dist_start_idx"]) - col_shift,
        "dist_end_idx": int(bounds["dist_end_idx"]) - col_shift,
    }
    return _clamp_bounds(after_shape, shifted)


def _compute_branch_metrics(
    reference: np.ndarray,
    processed: np.ndarray,
    reference_roi: dict[str, int],
    processed_roi: dict[str, int],
    ground_truth: dict[str, Any] | None,
) -> dict[str, float]:
    before_roi, after_roi = _slice_roi_pair(
        reference,
        processed,
        reference_roi,
        processed_roi,
    )
    metrics = compute_benchmark_metrics(before_roi, after_roi)
    if ground_truth:
        metrics.update(
            compute_ground_truth_metrics(
                reference,
                processed,
                ground_truth,
                reference_roi=reference_roi,
                processed_roi=processed_roi,
            )
        )
    metrics["pipeline_score"] = _pipeline_score(metrics)
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
    }


def _slice_roi_pair(
    reference: np.ndarray,
    processed: np.ndarray,
    reference_roi: dict[str, int],
    processed_roi: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float32)
    proc = np.asarray(processed, dtype=np.float32)
    ref_bounds = _clamp_bounds(ref.shape, reference_roi)
    proc_bounds = _clamp_bounds(proc.shape, processed_roi)
    ref_roi = ref[
        ref_bounds["time_start_idx"] : ref_bounds["time_end_idx"],
        ref_bounds["dist_start_idx"] : ref_bounds["dist_end_idx"],
    ]
    proc_roi = proc[
        proc_bounds["time_start_idx"] : proc_bounds["time_end_idx"],
        proc_bounds["dist_start_idx"] : proc_bounds["dist_end_idx"],
    ]
    rows = max(1, min(ref_roi.shape[0], proc_roi.shape[0]))
    cols = max(1, min(ref_roi.shape[1], proc_roi.shape[1]))
    return ref_roi[:rows, :cols], proc_roi[:rows, :cols]


def _pipeline_score(metrics: dict[str, float]) -> float:
    band_fidelity = ratio_fidelity(metrics["target_band_energy_ratio"], tol=0.35)
    saliency_fidelity = ratio_fidelity(
        metrics["local_saliency_preservation"],
        tol=0.35,
    )
    edge_fidelity = ratio_fidelity(metrics["edge_preservation"], tol=0.35)
    deep_gain = max(0.0, float(metrics["deep_zone_contrast_gain"]) - 1.0)
    target_loss_penalty = (
        max(0.0, 0.55 - float(metrics["target_band_energy_ratio"])) * 3.0
        + max(0.0, 0.55 - float(metrics["local_saliency_preservation"])) * 4.0
        + max(0.0, 0.55 - float(metrics["edge_preservation"])) * 3.0
    )
    artifact_penalty = (
        6.0 * float(metrics["clipping_ratio_after"])
        + 4.0 * float(metrics["hot_pixel_ratio_after"])
        + 0.08 * float(metrics["kurtosis_or_spikiness_after"])
    )
    score = (
        1.2 * float(metrics["baseline_bias_reduction"])
        + 1.4 * float(metrics["low_freq_energy_reduction"])
        + 0.8 * float(metrics["horizontal_coherence_reduction"])
        + 1.8 * band_fidelity
        + 2.0 * saliency_fidelity
        + 1.4 * edge_fidelity
        + 0.4 * np.log1p(deep_gain)
        - target_loss_penalty
        - artifact_penalty
    )
    if "truth_score" in metrics:
        score = 0.65 * score + 2.2 * float(metrics["truth_score"])
    return float(score)


def _compute_metric_delta(
    manual_metrics: dict[str, float],
    auto_metrics: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(manual_metrics) & set(auto_metrics))
    return {
        key: float(auto_metrics[key] - manual_metrics[key])
        for key in keys
        if np.isfinite(manual_metrics[key]) and np.isfinite(auto_metrics[key])
    }


def _assess_step_risk(
    manual_metrics: dict[str, float],
    auto_metrics: dict[str, float],
    tune_result: dict[str, Any] | None,
) -> tuple[list[str], str, str]:
    flags: list[str] = []
    score_delta = float(auto_metrics.get("pipeline_score", 0.0)) - float(
        manual_metrics.get("pipeline_score", 0.0)
    )
    if score_delta < SCORE_REJECT_DELTA:
        flags.append("auto_worse_than_manual")
    elif abs(score_delta) <= SCORE_REVIEW_DELTA:
        flags.append("near_tie")

    if tune_result is not None:
        confidence = float(tune_result.get("selection_confidence", 1.0))
        margin = float(tune_result.get("selection_margin", 1.0))
        stats = tune_result.get("execution_stats", {}) or {}
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            flags.append("low_selection_confidence")
        if margin < LOW_MARGIN_THRESHOLD:
            flags.append("multiple_near_optima")
        if int(stats.get("constraint_adjustment_count", 0) or 0) > 0:
            flags.append("constraint_adjusted")
        if tune_result.get("constraint_warnings"):
            flags.append("constraint_adjusted")

    truth_count = float(auto_metrics.get("truth_target_count", -1.0))
    if truth_count > 0.0:
        manual_preserve = float(
            manual_metrics.get("truth_target_energy_preservation", 1.0)
        )
        auto_preserve = float(auto_metrics.get("truth_target_energy_preservation", 1.0))
        manual_truth_score = float(manual_metrics.get("truth_score", 0.0))
        auto_truth_score = float(auto_metrics.get("truth_score", 0.0))
        if auto_preserve < manual_preserve - 0.08:
            flags.append("target_truth_degraded")
        elif auto_preserve < 0.55:
            flags.append("low_truth_target_preservation")
        if auto_truth_score < manual_truth_score - 0.05:
            flags.append("target_truth_degraded")
    elif truth_count == 0.0:
        manual_fp = float(manual_metrics.get("truth_false_positive_ratio", 0.0))
        auto_fp = float(auto_metrics.get("truth_false_positive_ratio", 0.0))
        if auto_fp > manual_fp + max(0.10, abs(manual_fp) * 0.20):
            flags.append("false_positive_risk")

    if (
        float(auto_metrics.get("clipping_ratio_after", 0.0))
        > float(manual_metrics.get("clipping_ratio_after", 0.0)) + 0.01
    ):
        flags.append("overexposure_risk")
    if (
        float(auto_metrics.get("hot_pixel_ratio_after", 0.0))
        > float(manual_metrics.get("hot_pixel_ratio_after", 0.0)) + 0.01
    ):
        flags.append("overexposure_risk")

    flags = _dedupe_flags(flags)
    severe = {
        "auto_worse_than_manual",
        "target_truth_degraded",
        "false_positive_risk",
        "overexposure_risk",
    }
    caution = {
        "near_tie",
        "low_selection_confidence",
        "multiple_near_optima",
        "constraint_adjusted",
        "low_truth_target_preservation",
    }
    if any(flag in severe for flag in flags):
        return flags, "keep_manual", _risk_reason(flags, score_delta)
    if any(flag in caution for flag in flags):
        return flags, "review", _risk_reason(flags, score_delta)
    return flags, "adopt_auto", f"auto pipeline score delta={score_delta:.4f}"


def _risk_reason(flags: list[str], score_delta: float) -> str:
    if not flags:
        return f"auto pipeline score delta={score_delta:.4f}"
    return f"risk={', '.join(flags)}; auto pipeline score delta={score_delta:.4f}"


def _overall_recommendation(
    steps: list[PipelineStepRecord],
    metric_delta: dict[str, float],
    risk_flags: list[str],
) -> str:
    if any(step.recommendation == "keep_manual" for step in steps):
        return "keep_manual"
    final_delta = float(metric_delta.get("pipeline_score", 0.0))
    if final_delta < SCORE_REJECT_DELTA:
        return "keep_manual"
    if any(step.recommendation == "review" for step in steps):
        return "review"
    if risk_flags or abs(final_delta) <= SCORE_REVIEW_DELTA:
        return "review"
    return "adopt_auto"


def _dedupe_flags(flags: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for flag in flags:
        text = str(flag)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _extract_warning_messages(meta: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    for warning in meta.get("runtime_warnings", []) or []:
        if isinstance(warning, dict):
            messages.append(str(warning.get("message") or warning.get("code") or warning))
        else:
            messages.append(str(warning))
    for warning in meta.get("warnings", []) or []:
        messages.append(str(warning))
    if meta.get("skipped"):
        reason = str(meta.get("reason") or "method skipped")
        messages.append(reason)
    return messages


def _compact_auto_tune_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    return {
        "method_key": result.get("method_key"),
        "method_name": result.get("method_name"),
        "family": result.get("family"),
        "recommended_profile": result.get("recommended_profile"),
        "recommended_params": _json_safe(result.get("recommended_params", {})),
        "best_params": _json_safe(result.get("best_params", {})),
        "best_score": _json_safe(result.get("best_score")),
        "best_reason": result.get("best_reason"),
        "roi_info": _json_safe(result.get("roi_info", {})),
        "parameter_domain": _json_safe(result.get("parameter_domain", {})),
        "risk_flags": _json_safe(result.get("risk_flags", [])),
        "risk_level": _json_safe(result.get("risk_level")),
        "risk_reason": result.get("risk_reason"),
        "selection_recommendation": result.get("selection_recommendation"),
        "selection_confidence": _json_safe(result.get("selection_confidence")),
        "selection_margin": _json_safe(result.get("selection_margin")),
        "execution_stats": _json_safe(result.get("execution_stats", {})),
        "constraint_warnings": _json_safe(result.get("constraint_warnings", [])),
        "best_constraint_warnings": _json_safe(
            result.get("best_constraint_warnings", [])
        ),
    }


def _candidate_summary(candidate: PipelineCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "source": candidate.source,
        "pipeline": list(candidate.pipeline),
        "params_by_method": _json_safe(candidate.params_by_method),
        "shape": [int(candidate.result.shape[0]), int(candidate.result.shape[1])],
        "metrics": _json_safe(candidate.metrics),
        "warnings": list(candidate.warnings),
        "auto_tune_results": _json_safe(candidate.auto_tune_results),
    }


def _step_summary(step: PipelineStepRecord) -> dict[str, Any]:
    return {
        "index": int(step.index),
        "method_key": step.method_key,
        "method_name": step.method_name,
        "manual_params": _json_safe(step.manual_params),
        "auto_params": _json_safe(step.auto_params),
        "manual_shape_after": [
            int(step.manual_after.shape[0]),
            int(step.manual_after.shape[1]),
        ],
        "auto_shape_after": [
            int(step.auto_after.shape[0]),
            int(step.auto_after.shape[1]),
        ],
        "manual_metrics": _json_safe(step.manual_metrics),
        "auto_metrics": _json_safe(step.auto_metrics),
        "metric_delta": _json_safe(step.metric_delta),
        "auto_tune_result": _json_safe(step.auto_tune_result),
        "manual_roi_before": _json_safe(step.manual_roi_before),
        "manual_roi_after": _json_safe(step.manual_roi_after),
        "auto_roi_before": _json_safe(step.auto_roi_before),
        "auto_roi_after": _json_safe(step.auto_roi_after),
        "warnings": _json_safe(step.warnings),
        "risk_flags": list(step.risk_flags),
        "recommendation": step.recommendation,
        "reason": step.reason,
        "rolled_back_to_manual": bool(step.rolled_back_to_manual),
    }


def _ground_truth_info(ground_truth: dict[str, Any] | None) -> dict[str, Any]:
    if not ground_truth:
        return {"enabled": False}
    targets = ground_truth.get("targets", []) or []
    preserved = [
        target for target in targets if isinstance(target, dict) and target.get("must_preserve") is not False
    ]
    return {
        "enabled": True,
        "schema": ground_truth.get("schema"),
        "scenario_id": ground_truth.get("scenario_id"),
        "target_count": int(len(preserved)),
        "analysis_roi": _json_safe(ground_truth.get("analysis_roi", {})),
        "targets": _json_safe([_ground_truth_target_info(target) for target in preserved]),
        "background_rois": _json_safe(ground_truth.get("background_rois", []) or []),
    }


def _ground_truth_target_info(target: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "id",
        "target_id",
        "type",
        "material",
        "depth_m",
        "center_x_m",
        "center_y_m",
        "radius_m",
        "must_preserve",
        "roi",
    ):
        if key in target:
            summary[key] = target[key]
    return summary


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


__all__ = [
    "AutoTunePipelineError",
    "AutoTunePipelineRun",
    "PipelineCandidate",
    "PipelineStepRecord",
    "run_auto_tune_pipeline",
    "to_summary_dict",
]

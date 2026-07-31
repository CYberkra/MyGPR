#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline-level auto-tuning orchestration with rollback support."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mygpr.application.autotune.use_case import auto_tune_method
from core.auto_tune_pipeline_geometry import _clamp_bounds
from core.auto_tune_pipeline_evaluation import (
    _assess_step_risk, _compute_branch_metrics, _compute_metric_delta,
    _dedupe_flags, _extract_warning_messages, _overall_recommendation,
)
from core.auto_tune_pipeline_models import (
    AutoTunePipelineError, AutoTunePipelineRun, PipelineCandidate,
    PipelineStepRecord, _BranchState,
)
from core.auto_tune_pipeline_summary import (
    _candidate_summary, _compact_auto_tune_result, _ground_truth_info,
    _json_safe, _step_summary,
)
from core.methods_registry import PROCESSING_METHODS
from core.preset_profiles import GUI_PRESETS_V1, RECOMMENDED_RUN_PROFILES
from core.processing_engine import (
    clone_header_info, clone_trace_metadata, merge_result_header_info,
    merge_result_trace_metadata, prepare_runtime_params, run_processing_method,
)
from core.quality_metrics import auto_roi_bounds

ProgressCallback = Callable[[int, int, str], None]
CancelChecker = Callable[[], bool]

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

__all__ = [
    "AutoTunePipelineError", "AutoTunePipelineRun", "PipelineCandidate",
    "PipelineStepRecord", "run_auto_tune_pipeline", "to_summary_dict",
]

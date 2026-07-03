#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research comparison backend for manual baseline vs auto-tuned processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from core.auto_tune import auto_tune_method
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
from core.gprmax_truth_metrics import compute_ground_truth_metrics
from core.scalar_utils import to_int
from core.app_errors import MyGPRError


ProgressCallback = Callable[[int, int, str], None]
CancelChecker = Callable[[], bool]


class AutoTuneComparisonError(MyGPRError):
    """Raised when a manual-vs-auto comparison cannot be executed."""


@dataclass
class ComparisonCandidate:
    """One processed branch in a manual-vs-auto comparison."""

    name: str
    source: str
    pipeline: list[str]
    params_by_method: dict[str, dict[str, Any]]
    result: np.ndarray
    metadata: dict[str, Any]
    metrics: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    auto_tune_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    step_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AutoTuneComparisonRun:
    """Full comparison result for one input dataset and one pipeline."""

    input_shape: tuple[int, int]
    baseline_profile_key: str | None
    roi_info: dict[str, Any]
    display_spec: dict[str, Any]
    manual: ComparisonCandidate
    automatic: ComparisonCandidate
    metric_delta: dict[str, float]
    verdict: str
    ground_truth_info: dict[str, Any]
    ground_truth: dict[str, Any] | None = None


def run_auto_tune_comparison(
    data: np.ndarray,
    *,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    pipeline: list[str] | None = None,
    manual_params_by_method: dict[str, dict[str, Any]] | None = None,
    baseline_profile_key: str | None = None,
    roi_spec: dict[str, Any] | None = None,
    search_mode: str = "standard",
    display_spec: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_checker: CancelChecker | None = None,
) -> AutoTuneComparisonRun:
    """Run manual baseline and auto-tuned branches on the same input."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise AutoTuneComparisonError("参数对比需要二维非空 B-scan 数据")
    if ground_truth is None and isinstance(header_info, dict):
        embedded_ground_truth = header_info.get("ground_truth")
        if isinstance(embedded_ground_truth, dict):
            ground_truth = embedded_ground_truth
    runtime_header_info = _header_without_ground_truth(header_info)

    profile_key = baseline_profile_key or (
        None if pipeline is not None else "uav_gpr_experience_baseline_v1"
    )
    pipeline_order = _resolve_pipeline(pipeline, profile_key)
    if not pipeline_order:
        raise AutoTuneComparisonError("参数对比流程不能为空")

    _validate_pipeline(pipeline_order)
    roi_info = _resolve_roi_info(arr, roi_spec)
    explicit_manual_params = bool(manual_params_by_method)
    manual_source = "current_ui_params" if explicit_manual_params else "experience_profile"
    manual_params = _resolve_manual_params(
        pipeline_order,
        profile_key,
        manual_params_by_method or {},
    )

    manual_candidate = _execute_candidate(
        name="人工 baseline",
        source=manual_source,
        data=arr,
        header_info=runtime_header_info,
        trace_metadata=trace_metadata,
        pipeline=pipeline_order,
        params_by_method=manual_params,
        auto_tune=False,
        roi_spec=roi_spec,
        search_mode=search_mode,
        progress_callback=progress_callback,
        cancel_checker=cancel_checker,
        progress_offset=0,
        progress_total=max(1, len(pipeline_order) * 2),
    )

    automatic_candidate = _execute_candidate(
        name="自动选参",
        source="auto_tune",
        data=arr,
        header_info=runtime_header_info,
        trace_metadata=trace_metadata,
        pipeline=pipeline_order,
        params_by_method=manual_params,
        auto_tune=True,
        roi_spec=roi_spec,
        search_mode=search_mode,
        progress_callback=progress_callback,
        cancel_checker=cancel_checker,
        progress_offset=len(pipeline_order),
        progress_total=max(1, len(pipeline_order) * 2),
    )

    manual_candidate.metrics = _compute_candidate_metrics(
        arr,
        manual_candidate.result,
        roi_info,
        ground_truth,
    )
    automatic_candidate.metrics = _compute_candidate_metrics(
        arr,
        automatic_candidate.result,
        roi_info,
        ground_truth,
    )
    metric_delta = _compute_metric_delta(
        manual_candidate.metrics, automatic_candidate.metrics
    )

    return AutoTuneComparisonRun(
        input_shape=(int(arr.shape[0]), int(arr.shape[1])),
        baseline_profile_key=profile_key,
        roi_info=roi_info,
        display_spec=dict(display_spec or _default_display_spec()),
        manual=manual_candidate,
        automatic=automatic_candidate,
        metric_delta=metric_delta,
        verdict=_comparison_verdict(metric_delta),
        ground_truth_info=_ground_truth_info(ground_truth),
        ground_truth=ground_truth,
    )


def to_summary_dict(result: AutoTuneComparisonRun) -> dict[str, Any]:
    """Return a JSON-safe summary without raw image arrays."""
    return {
        "input_shape": list(result.input_shape),
        "baseline_profile_key": result.baseline_profile_key,
        "roi_info": _json_safe(result.roi_info),
        "display_spec": _json_safe(result.display_spec),
        "verdict": result.verdict,
        "metric_delta": _json_safe(result.metric_delta),
        "ground_truth_info": _json_safe(result.ground_truth_info),
        "manual": _candidate_summary(result.manual),
        "automatic": _candidate_summary(result.automatic),
    }


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
        raise AutoTuneComparisonError(
            f"未知经验 baseline profile: {baseline_profile_key}"
        )
    return [str(method_key) for method_key in profile.get("order", [])]


def _header_without_ground_truth(header_info: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(header_info, dict):
        return header_info
    sanitized = dict(header_info)
    sanitized.pop("ground_truth", None)
    return sanitized


def _validate_pipeline(pipeline: list[str]) -> None:
    unknown = [method_key for method_key in pipeline if method_key not in PROCESSING_METHODS]
    if unknown:
        raise AutoTuneComparisonError(f"pipeline 包含未知方法: {', '.join(unknown)}")


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
            "method_params", {}
        ).items():
            params[str(method_key)] = dict(method_params)
    for method_key, method_params in profile.get("method_params", {}).items():
        params[str(method_key)] = dict(method_params)
    return params


def _execute_candidate(
    *,
    name: str,
    source: str,
    data: np.ndarray,
    header_info: dict[str, Any] | None,
    trace_metadata: dict[str, np.ndarray] | None,
    pipeline: list[str],
    params_by_method: dict[str, dict[str, Any]],
    auto_tune: bool,
    roi_spec: dict[str, Any] | None,
    search_mode: str,
    progress_callback: ProgressCallback | None,
    cancel_checker: CancelChecker | None,
    progress_offset: int,
    progress_total: int,
) -> ComparisonCandidate:
    current = np.array(data, dtype=np.float32, copy=True)
    current_header = clone_header_info(header_info)
    current_trace_metadata = clone_trace_metadata(trace_metadata)
    resolved_params = {
        method_key: dict(params_by_method.get(method_key, {})) for method_key in pipeline
    }
    warnings: list[str] = []
    step_records: list[dict[str, Any]] = []
    auto_tune_results: dict[str, dict[str, Any]] = {}

    for idx, method_key in enumerate(pipeline, start=1):
        if cancel_checker and bool(cancel_checker()):
            raise AutoTuneComparisonError("用户已取消参数对比")
        params = dict(resolved_params.get(method_key, {}))
        if auto_tune and PROCESSING_METHODS[method_key].get("auto_tune_enabled"):
            tune_result = auto_tune_method(
                current,
                method_key,
                header_info=current_header,
                trace_metadata=current_trace_metadata,
                base_params=params,
                roi_spec=roi_spec,
                search_mode=search_mode,
                cancel_checker=cancel_checker,
            )
            recommended = dict(
                tune_result.get("recommended_params")
                or tune_result.get("best_params")
                or {}
            )
            params.update(recommended)
            resolved_params[method_key] = params
            auto_tune_results[method_key] = _compact_auto_tune_result(tune_result)

        if progress_callback is not None:
            progress_callback(
                progress_offset + idx - 1,
                progress_total,
                f"{name}: {method_key}",
            )

        runtime_params = prepare_runtime_params(
            method_key,
            params,
            current_header,
            current_trace_metadata,
            current.shape,
        )
        current, meta = run_processing_method(
            current,
            method_key,
            runtime_params,
            cancel_checker=cancel_checker,
        )
        current_header = merge_result_header_info(current_header, meta, current.shape)
        current_trace_metadata = merge_result_trace_metadata(current_trace_metadata, meta)
        warnings.extend(_extract_warning_messages(meta))
        step_records.append(
            {
                "method_key": method_key,
                "params": dict(params),
                "shape": [int(current.shape[0]), int(current.shape[1])],
                "warnings": _extract_warning_messages(meta),
                "meta": _compact_step_meta(meta),
            }
        )

    if progress_callback is not None:
        progress_callback(progress_offset + len(pipeline), progress_total, f"{name}: 完成")

    return ComparisonCandidate(
        name=name,
        source=source,
        pipeline=list(pipeline),
        params_by_method=resolved_params,
        result=current,
        metadata={
            "header_info": current_header,
            "trace_metadata": current_trace_metadata,
        },
        metrics={},
        warnings=warnings,
        auto_tune_results=auto_tune_results,
        step_records=step_records,
    )


def _resolve_roi_info(data: np.ndarray, roi_spec: dict[str, Any] | None) -> dict[str, Any]:
    spec = dict(roi_spec or {})
    mode = str(spec.get("mode") or "full")
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


def _clamp_bounds(
    shape: tuple[int, int],
    bounds: dict[str, Any],
) -> dict[str, int]:
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
        "time_start_idx": t0,
        "time_end_idx": t1,
        "dist_start_idx": d0,
        "dist_end_idx": d1,
    }


def _compute_candidate_metrics(
    raw: np.ndarray,
    processed: np.ndarray,
    roi_info: dict[str, Any],
    ground_truth: dict[str, Any] | None,
) -> dict[str, float]:
    before, after = _slice_common_roi(raw, processed, roi_info.get("bounds"))
    metrics = compute_benchmark_metrics(before, after)
    if ground_truth:
        metrics.update(
            compute_ground_truth_metrics(
                raw,
                processed,
                ground_truth,
                reference_roi=roi_info.get("bounds"),
                processed_roi=roi_info.get("bounds"),
            )
        )
    metrics["comparison_score"] = _comparison_score(metrics)
    return {key: float(value) for key, value in metrics.items()}


def _slice_common_roi(
    before: np.ndarray,
    after: np.ndarray,
    bounds: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    before_arr = np.asarray(before, dtype=np.float32)
    after_arr = np.asarray(after, dtype=np.float32)
    before_bounds = _clamp_bounds(before_arr.shape, bounds or {})
    after_bounds = _clamp_bounds(after_arr.shape, bounds or {})
    before_roi = before_arr[
        before_bounds["time_start_idx"] : before_bounds["time_end_idx"],
        before_bounds["dist_start_idx"] : before_bounds["dist_end_idx"],
    ]
    after_roi = after_arr[
        after_bounds["time_start_idx"] : after_bounds["time_end_idx"],
        after_bounds["dist_start_idx"] : after_bounds["dist_end_idx"],
    ]
    rows = max(1, min(before_roi.shape[0], after_roi.shape[0]))
    cols = max(1, min(before_roi.shape[1], after_roi.shape[1]))
    return before_roi[:rows, :cols], after_roi[:rows, :cols]


def _comparison_score(metrics: dict[str, float]) -> float:
    band_fidelity = ratio_fidelity(metrics["target_band_energy_ratio"], tol=0.35)
    saliency_fidelity = ratio_fidelity(
        metrics["local_saliency_preservation"], tol=0.35
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
    return float(score)


def _compute_metric_delta(
    manual_metrics: dict[str, float],
    automatic_metrics: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(manual_metrics) & set(automatic_metrics))
    return {
        key: float(automatic_metrics[key] - manual_metrics[key])
        for key in keys
        if np.isfinite(manual_metrics[key]) and np.isfinite(automatic_metrics[key])
    }


def _comparison_verdict(metric_delta: dict[str, float]) -> str:
    score_delta = float(metric_delta.get("comparison_score", 0.0))
    if score_delta > 0.02:
        return "auto_better"
    if score_delta < -0.02:
        return "manual_better"
    return "tie"


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


def _compact_step_meta(meta: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "method",
        "method_id",
        "skipped",
        "reason",
        "input_height_valid",
        "time_shift_correction_applied",
        "amplitude_correction_applied",
        "selection_confidence",
    }
    return {key: _json_safe(value) for key, value in meta.items() if key in keep}


def _compact_auto_tune_result(result: dict[str, Any]) -> dict[str, Any]:
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
        "execution_stats": _json_safe(result.get("execution_stats", {})),
        "all_trials": _json_safe(_compact_trials(result.get("all_trials", []) or [])),
        "coarse_trials": _json_safe(
            _compact_trials(result.get("coarse_trials", []) or [])
        ),
        "fine_trials": _json_safe(_compact_trials(result.get("fine_trials", []) or [])),
        "failed_trials": _json_safe(
            _compact_trials(result.get("failed_trials", []) or [])
        ),
    }


def _compact_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            continue
        metrics = trial.get("roi_metrics") or trial.get("metrics") or {}
        compact.append(
            {
                "trial_index": int(trial.get("trial_index", index)),
                "stage": trial.get("stage"),
                "params": _json_safe(trial.get("params", {})),
                "requested_params": _json_safe(trial.get("requested_params", {})),
                "effective_params": _json_safe(trial.get("effective_params", {})),
                "score": _json_safe(trial.get("score")),
                "comparison_score": _json_safe(metrics.get("comparison_score")),
                "truth_score": _json_safe(metrics.get("truth_score")),
                "truth_target_energy_preservation": _json_safe(
                    metrics.get("truth_target_energy_preservation")
                ),
                "truth_target_saliency_gain": _json_safe(
                    metrics.get("truth_target_saliency_gain")
                ),
                "truth_background_energy_reduction": _json_safe(
                    metrics.get("truth_background_energy_reduction")
                ),
                "truth_false_positive_ratio": _json_safe(
                    metrics.get("truth_false_positive_ratio")
                ),
                "reason": trial.get("reason") or trial.get("error"),
                "warnings": _json_safe(
                    trial.get("constraint_warnings")
                    or trial.get("warnings")
                    or trial.get("runtime_warnings")
                    or []
                ),
                "valid": bool(trial.get("valid", True)),
                "candidate_space_hash": _json_safe(trial.get("candidate_space_hash")),
                "candidate_space_profile_id": _json_safe(trial.get("candidate_space_profile_id")),
                "candidate_space_config_version": _json_safe(trial.get("candidate_space_config_version")),
                "candidate_space_recipe_ids": _json_safe(trial.get("candidate_space_recipe_ids", [])),
                "candidate_id": _json_safe(trial.get("candidate_id")),
                "candidate_source": _json_safe(trial.get("candidate_source")),
                "candidate_group": _json_safe(trial.get("candidate_group")),
                "candidate_parameters": _json_safe(trial.get("candidate_parameters", {})),
                "candidate_warnings": _json_safe(trial.get("candidate_warnings", [])),
                "score_version": _json_safe(trial.get("score_version")),
                "scoring_boundary": _json_safe(trial.get("scoring_boundary")),
                "manual_review_required": _json_safe(trial.get("manual_review_required")),
                "display_only": _json_safe(trial.get("display_only")),
                "metric_safe": _json_safe(trial.get("metric_safe")),
            }
        )
    return compact


def _candidate_summary(candidate: ComparisonCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "source": candidate.source,
        "pipeline": list(candidate.pipeline),
        "params_by_method": _json_safe(candidate.params_by_method),
        "shape": [int(candidate.result.shape[0]), int(candidate.result.shape[1])],
        "metrics": _json_safe(candidate.metrics),
        "warnings": list(candidate.warnings),
        "auto_tune_results": _json_safe(candidate.auto_tune_results),
        "step_records": _json_safe(candidate.step_records),
    }


def _ground_truth_info(ground_truth: dict[str, Any] | None) -> dict[str, Any]:
    if not ground_truth:
        return {"enabled": False}
    targets = ground_truth.get("targets", []) or []
    preserved = [
        target
        for target in targets
        if isinstance(target, dict) and target.get("must_preserve") is not False
    ]
    source_paths = ground_truth.get("source_paths") or {}
    return {
        "enabled": True,
        "schema": ground_truth.get("schema"),
        "scenario_id": ground_truth.get("scenario_id"),
        "target_count": int(len(preserved)),
        "has_background_rois": bool(ground_truth.get("background_rois")),
        "analysis_roi": _json_safe(ground_truth.get("analysis_roi", {})),
        "targets": _json_safe([_ground_truth_target_info(target) for target in preserved]),
        "background_rois": _json_safe(ground_truth.get("background_rois", []) or []),
        "source_paths": _json_safe(source_paths if isinstance(source_paths, dict) else {}),
        "conversion_warnings": _json_safe(
            ground_truth.get("conversion_warnings", []) or []
        ),
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


def _default_display_spec() -> dict[str, Any]:
    return {
        "lock_color_scale": True,
        "normalize": False,
        "percentile_clip": None,
    }


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
    "AutoTuneComparisonError",
    "AutoTuneComparisonRun",
    "ComparisonCandidate",
    "run_auto_tune_comparison",
    "to_summary_dict",
]

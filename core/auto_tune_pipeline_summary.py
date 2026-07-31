#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON-safe summary projection for pipeline-level automatic tuning."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.auto_tune_pipeline_models import PipelineCandidate, PipelineStepRecord

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

__all__ = ['_compact_auto_tune_result', '_candidate_summary', '_step_summary', '_ground_truth_info', '_ground_truth_target_info', '_json_safe']

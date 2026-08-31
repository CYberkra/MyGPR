#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failure records, stability summaries and parameter-domain diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.domain.autotune.constraints import ParameterConstraintResult
from mygpr.domain.processing.warnings import merge_runtime_warnings
from mygpr.domain.common.scalars import to_float_or_none
from mygpr.application.autotune.utils import _is_number, _public_params, _trial_signature
from mygpr.domain.autotune.models import (
    AutoTuneContext,
    FAILURE_PENALTY,
    INVALID_TRIAL_SCORE,
)


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


def _attach_constraint_metadata(
    record: dict[str, Any],
    constraint_result: ParameterConstraintResult,
    *,
    method_runtime_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach requested/effective params and warnings to one trial record."""
    constraint_warnings = list(constraint_result.warnings or [])
    if method_runtime_warnings is None:
        method_runtime_warnings = list(record.get("method_runtime_warnings", []) or [])
    record["requested_params"] = _public_params(constraint_result.requested_params)
    record["effective_params"] = _public_params(constraint_result.effective_params)
    record["constraint_warnings"] = constraint_warnings
    record["constraint_adjusted"] = bool(constraint_result.adjusted)
    record["method_runtime_warnings"] = list(method_runtime_warnings or [])
    record["runtime_warnings"] = merge_runtime_warnings(
        constraint_warnings,
        method_runtime_warnings,
    )
    return record


def _summarize_failed_trials(trials: list[dict[str, Any]], method_key: str) -> str:
    errors = [
        str(trial.get("error") or trial.get("reason") or "未知错误")
        for trial in trials
        if not trial.get("valid", True)
    ]
    if not errors:
        return f"参数推荐失败：{method_key} 没有可用候选。"
    preview = "；".join(errors[:3])
    return f"参数推荐失败：{method_key} 所有候选均未成功执行。示例错误：{preview}"


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


def _summarize_parameter_domain(
    method_key: str,
    method_info: dict[str, Any],
    context: AutoTuneContext,
    base_params: dict[str, Any],
    trials: list[dict[str, Any]],
    best_trial: dict[str, Any],
    search_plan: dict[str, Any],
    *,
    selection_margin: float,
    selection_confidence: float,
    constraint_adjustment_count: int,
) -> dict[str, Any]:
    requested_values: dict[str, list[Any]] = {}
    effective_values: dict[str, list[Any]] = {}
    clamped_counts: dict[str, int] = {}
    unique_effective_signatures: set[str] = set()

    for trial in trials:
        requested = _public_params(
            trial.get("requested_params", trial.get("params", {})) or {}
        )
        effective = _public_params(
            trial.get("effective_params", trial.get("params", {})) or {}
        )
        unique_effective_signatures.add(_trial_signature(effective))
        for name in sorted(set(requested) | set(effective)):
            requested_value = _normalize_summary_value(requested.get(name))
            effective_value = _normalize_summary_value(effective.get(name))
            requested_values.setdefault(name, []).append(requested_value)
            effective_values.setdefault(name, []).append(effective_value)
            if requested_value != effective_value:
                clamped_counts[name] = clamped_counts.get(name, 0) + 1

    best_params = _public_params(best_trial.get("params", {}))
    parameter_summaries: dict[str, dict[str, Any]] = {}
    best_params_at_edge = False
    parameter_names = sorted(
        set(base_params)
        | set(best_params)
        | set(requested_values)
        | set(effective_values)
    )
    for name in parameter_names:
        requested_summary = _summarize_domain_values(requested_values.get(name, []))
        effective_summary = _summarize_domain_values(effective_values.get(name, []))
        best_value = _normalize_summary_value(best_params.get(name))
        best_value_at_edge = _value_at_domain_edge(best_value, effective_summary)
        best_params_at_edge = best_params_at_edge or bool(best_value_at_edge)
        parameter_summaries[name] = {
            "requested": requested_summary,
            "effective": effective_summary,
            "best_value": best_value,
            "best_value_at_edge": bool(best_value_at_edge),
            "constraint_adjusted_trial_count": int(clamped_counts.get(name, 0)),
        }

    notes: list[str] = []
    if constraint_adjustment_count > 0:
        notes.append("部分候选参数被按数据尺度收缩，实际搜索域小于原始候选列表。")
    if best_params_at_edge:
        notes.append("最优参数贴近有效搜索域边界，建议结合人工参数再核查。")
    if len(unique_effective_signatures) <= 2 and len(trials) > 1:
        notes.append("有效搜索域较窄，当前数据对该方法的可辨识度偏低。")

    risk_flags: list[str] = []
    if selection_confidence < 0.45:
        risk_flags.append("low_selection_confidence")
    if selection_margin < 0.03:
        risk_flags.append("multiple_near_optima")
    if constraint_adjustment_count > 0:
        risk_flags.append("constraint_adjusted")
    if best_params_at_edge:
        risk_flags.append("best_params_at_edge")
    if len(unique_effective_signatures) <= 2 and len(trials) > 1:
        risk_flags.append("narrow_effective_domain")

    severe = {"low_selection_confidence", "best_params_at_edge", "narrow_effective_domain"}
    if any(flag in severe for flag in risk_flags):
        risk_level = "high"
    elif risk_flags:
        risk_level = "medium"
    else:
        risk_level = "low"

    if risk_level == "low":
        selection_recommendation = "adopt_auto"
        risk_reason = "自动推荐稳定，参数域与评分差距都较健康。"
    else:
        selection_recommendation = "review"
        risk_reason = "；".join(notes) if notes else "当前结果建议人工复核。"

    return {
        "method_key": method_key,
        "method_name": str(method_info.get("name") or method_key),
        "family": str(method_info.get("auto_tune_family") or ""),
        "stage": str(method_info.get("auto_tune_stage") or ""),
        "search_mode": context.search_mode,
        "data_shape": [
            int(context.full_data.shape[0]),
            int(context.full_data.shape[1]),
        ],
        "roi_source": context.roi_source,
        "roi_label": context.roi_label,
        "base_params": _public_params(base_params),
        "search_plan": dict(search_plan),
        "trial_counts": {
            "coarse": int(
                sum(1 for trial in trials if str(trial.get("stage")) == "coarse")
            ),
            "fine": int(
                sum(1 for trial in trials if str(trial.get("stage")) == "fine")
            ),
            "total": int(len(trials)),
            "valid": int(sum(1 for trial in trials if trial.get("valid", True))),
            "failed": int(sum(1 for trial in trials if not trial.get("valid", True))),
            "cached": int(sum(1 for trial in trials if trial.get("cached"))),
            "constraint_adjusted": int(constraint_adjustment_count),
            "unique_effective": int(len(unique_effective_signatures)),
        },
        "parameters": parameter_summaries,
        "best_params": best_params,
        "best_params_at_edge": bool(best_params_at_edge),
        "selection_margin": float(selection_margin),
        "selection_confidence": float(selection_confidence),
        "notes": notes,
        "risk_flags": risk_flags,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "selection_recommendation": selection_recommendation,
    }


def _summarize_domain_values(values: list[Any]) -> dict[str, Any]:
    cleaned: list[Any] = []
    for value in values:
        normalized = _normalize_summary_value(value)
        if normalized not in cleaned:
            cleaned.append(normalized)

    if not cleaned:
        return {
            "kind": "empty",
            "values": [],
            "count": 0,
            "min": None,
            "max": None,
        }

    if all(_is_number(value) for value in cleaned):
        ordered = sorted(cleaned, key=float)
        return {
            "kind": "numeric",
            "values": [_normalize_summary_value(value) for value in ordered],
            "count": int(len(ordered)),
            "min": _normalize_summary_value(min(ordered, key=float)),
            "max": _normalize_summary_value(max(ordered, key=float)),
        }

    kind = "categorical"
    if any(_is_number(value) for value in cleaned):
        kind = "mixed"
    return {
        "kind": kind,
        "values": cleaned,
        "count": int(len(cleaned)),
        "min": None,
        "max": None,
    }


def _value_at_domain_edge(value: Any, summary: dict[str, Any]) -> bool:
    if value is None or summary.get("kind") != "numeric":
        return False
    if int(summary.get("count", 0) or 0) <= 1:
        return False
    current = to_float_or_none(value)
    minimum = to_float_or_none(summary.get("min"))
    maximum = to_float_or_none(summary.get("max"))
    if current is None or minimum is None or maximum is None:
        return False
    return abs(current - minimum) <= 1.0e-9 or abs(current - maximum) <= 1.0e-9


def _normalize_summary_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-blocking recommendation labeling for AutoTune diagnostic outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


BACKGROUND_METHOD_KEYS = {
    "subtracting_average_2D",
    "median_background_2D",
    "running_average_2D",
    "svd_bg",
    "fk_filter",
    "ccbs",
}

CLAIM_BOUNDARY_TEXT = (
    "diagnostic-only recommendation; no target detection; no underground correctness; "
    "no preset promotion; no amplitude-preserving claim after display gain"
)


@dataclass(frozen=True)
class AutoTuneRecommendationLabel:
    recommendation_label: str
    severity: str
    manual_review_recommended: bool
    no_prior_background_warning: bool
    risk_flags: list[str]
    risk_basis: dict[str, Any]
    user_log_messages: list[str]
    claim_boundary: str


def assess_auto_tune_recommendation_label(
    *,
    method_key: str,
    selected_params: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    score: float | None,
    no_prior_policy: dict[str, Any] | None,
    trace_count: int | None,
) -> AutoTuneRecommendationLabel:
    """Assess recommendation risk labels without changing score or candidate selection."""
    params = dict(selected_params or {})
    metric_dict = dict(metrics or {})
    policy = dict(no_prior_policy or {})
    method = str(method_key or "")

    no_prior_level = str(policy.get("no_prior_level") or "ok").strip().lower()
    target_prior_available = bool(policy.get("target_prior_available"))
    roi_available = bool(policy.get("roi_available"))
    no_prior_missing = (not target_prior_available) or (not roi_available)
    high_risk_no_prior = no_prior_level == "high_risk" or no_prior_missing

    is_background = method in BACKGROUND_METHOD_KEYS
    risk_flags: list[str] = []
    log_messages: list[str] = []
    risk_basis: dict[str, Any] = {
        "method_key": method,
        "is_background_stage": is_background,
        "no_prior_level": no_prior_level,
    }

    ntraces_ratio = None
    if trace_count and trace_count > 0 and "ntraces" in params:
        try:
            ntraces_ratio = float(params.get("ntraces", 0)) / float(trace_count)
            risk_basis["ntraces_ratio"] = float(ntraces_ratio)
        except (TypeError, ValueError):
            ntraces_ratio = None

    if is_background and ntraces_ratio is not None:
        if ntraces_ratio > 0.20:
            risk_flags.append("background_large_window_risk")
            log_messages.append(
                "背景抑制候选窗口较大，可能压制层状结构或目标响应；本结果不代表目标识别或地下结构正确性。"
            )
        elif ntraces_ratio > 0.10:
            risk_flags.append("background_medium_window_caution")
        elif ntraces_ratio > 0.05:
            risk_flags.append("background_local_window_manual_review")

    metric_thresholds = {
        "local_saliency_preservation": 0.50,
        "edge_preservation": 0.50,
        "peak_ratio": 0.40,
    }
    for key, threshold in metric_thresholds.items():
        value = metric_dict.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        risk_basis[key] = numeric
        if numeric < threshold:
            risk_flags.append(f"{key}_low")

    numeric_score = None
    if score is not None:
        try:
            numeric_score = float(score)
            risk_basis["score"] = numeric_score
        except (TypeError, ValueError):
            numeric_score = None
    if numeric_score is not None and numeric_score < 0:
        risk_flags.append("negative_score_risk")
        log_messages.append(
            "当前背景抑制候选分数为负，说明部分保真/风险指标不支持将其作为安全默认推荐。"
        )

    no_prior_background_warning = bool(is_background and high_risk_no_prior and risk_flags)
    manual_review_recommended = bool(no_prior_background_warning)

    if no_prior_background_warning:
        log_messages.insert(
            0,
            "自动选参提示：当前数据缺少目标先验/ROI，背景抑制候选仅作为诊断建议。该候选可能存在过抑制风险，建议结合原始剖面与人工复核后使用。",
        )

    has_severe = any(
        flag in risk_flags
        for flag in {"background_large_window_risk", "negative_score_risk"}
    )
    if no_prior_background_warning:
        recommendation_label = "manual_review_recommended" if has_severe else "diagnostic_candidate"
        severity = "high_risk" if has_severe else "warning"
    elif risk_flags:
        recommendation_label = "caution"
        severity = "warning"
    else:
        recommendation_label = "normal"
        severity = "info"

    return AutoTuneRecommendationLabel(
        recommendation_label=recommendation_label,
        severity=severity,
        manual_review_recommended=manual_review_recommended,
        no_prior_background_warning=no_prior_background_warning,
        risk_flags=risk_flags,
        risk_basis=risk_basis,
        user_log_messages=log_messages,
        claim_boundary=CLAIM_BOUNDARY_TEXT,
    )


def recommendation_label_to_dict(
    label: AutoTuneRecommendationLabel,
) -> dict[str, Any]:
    """Convert label dataclass into a JSON-safe dict."""
    return asdict(label)


__all__ = [
    "AutoTuneRecommendationLabel",
    "BACKGROUND_METHOD_KEYS",
    "CLAIM_BOUNDARY_TEXT",
    "assess_auto_tune_recommendation_label",
    "recommendation_label_to_dict",
]

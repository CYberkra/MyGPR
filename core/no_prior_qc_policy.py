#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-prior QC warning policy helpers for UI/spec-facing safety decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


NO_PRIOR_WARNING_TEMPLATES: list[str] = [
    "当前数据触发高风险质控告警，建议先查看原始剖面并进行人工复核。",
    "系统不会自动推荐激进背景抑制或默认参数。",
    "以下显示增强仅用于可视化，不代表幅值保真或目标识别。",
    "未提供目标区域或先验信息，因此本结果不构成地下目标判断。",
    "建议由有经验人员复核后再尝试任何参数化处理。",
]

NO_PRIOR_CLAIM_BOUNDARY = (
    "no target detection; no underground correctness; no AutoTune claim; "
    "no preset promotion; display transforms are not amplitude-preserving claims; "
    "heuristic thresholds only"
)

NO_PRIOR_ACTION_POLICY: list[dict[str, str]] = [
    {
        "action": "raw_preview",
        "policy": "allowed",
        "reason": "Baseline non-invasive view.",
        "claim_boundary": "No target detection claim.",
    },
    {
        "action": "contrast_clip_display",
        "policy": "caution",
        "reason": "Display-only contrast adjustment allowed.",
        "claim_boundary": "Display-only, not amplitude-preserving.",
    },
    {
        "action": "conservative_energy_decay_gain_display",
        "policy": "caution",
        "reason": "Only as labeled display transform.",
        "claim_boundary": "No amplitude-preserving claim.",
    },
    {
        "action": "AGC_display_only",
        "policy": "caution",
        "reason": "Display aid with explicit warning labels.",
        "claim_boundary": "Not amplitude-preserving; no target claim.",
    },
    {
        "action": "background_suppression_conservative",
        "policy": "caution",
        "reason": "Only after warnings acknowledged and manual review.",
        "claim_boundary": "No automatic recommendation.",
    },
    {
        "action": "background_suppression_aggressive",
        "policy": "blocked",
        "reason": "High-risk no-prior baseline blocks aggressive auto path.",
        "claim_boundary": "Blocked in no-prior safety pilot.",
    },
    {
        "action": "dewow",
        "policy": "blocked",
        "reason": "Not validated for no-prior safety auto path here.",
        "claim_boundary": "Blocked in no-prior safety pilot.",
    },
    {
        "action": "migration",
        "policy": "blocked",
        "reason": "Out of no-prior safety pilot scope.",
        "claim_boundary": "Blocked in no-prior safety pilot.",
    },
    {
        "action": "AutoTune",
        "policy": "blocked",
        "reason": "AutoTune is out of scope for no-prior safety pilot.",
        "claim_boundary": "No AutoTune claim.",
    },
    {
        "action": "preset_recommendation",
        "policy": "blocked",
        "reason": "Preset promotion forbidden under no-prior high-risk warnings.",
        "claim_boundary": "No preset promotion claim.",
    },
]


@dataclass(frozen=True)
class NoPriorQcPolicy:
    no_prior_level: str
    target_prior_available: bool
    roi_available: bool
    safe_auto_recommendation_allowed: bool
    aggressive_background_suppression_allowed: bool
    amplitude_claim_allowed: bool
    auto_gain_allowed: bool
    manual_review_required: bool
    recommended_initial_policy: str
    claim_boundary: str
    user_facing_warnings: list[str]
    allowed_actions: list[str]
    caution_actions: list[str]
    blocked_actions: list[str]
    action_policy: list[dict[str, str]]


def derive_no_prior_level(
    *,
    quality_metrics: dict[str, Any] | None,
    metric_alerts: dict[str, bool] | None,
    airborne_qc: dict[str, Any] | None,
    runtime_warnings: list[dict[str, Any]] | None,
    target_prior_available: bool,
    roi_available: bool,
) -> str:
    """Infer a conservative no-prior level for UI warning display."""
    if target_prior_available or roi_available:
        return "ok"

    risk_score = 0
    if quality_metrics:
        active_metric_alerts = [
            key for key, enabled in (metric_alerts or {}).items() if bool(enabled)
        ]
        if active_metric_alerts:
            risk_score += 2
    if airborne_qc and (airborne_qc.get("alerts") or []):
        risk_score += 1
    if runtime_warnings:
        risk_score += 1

    if risk_score >= 2:
        return "high_risk"
    if risk_score == 1:
        return "caution"
    return "ok"


def build_no_prior_qc_policy(
    *,
    no_prior_level: str,
    target_prior_available: bool,
    roi_available: bool,
) -> NoPriorQcPolicy:
    """Build UI/spec-facing no-prior policy decisions."""
    normalized_level = str(no_prior_level or "ok").strip().lower()
    if normalized_level not in {"ok", "caution", "high_risk"}:
        normalized_level = "caution"

    if normalized_level == "high_risk" and not target_prior_available and not roi_available:
        safe_auto = False
        aggressive_bg = False
        amplitude_claim = False
        auto_gain = False
        manual_review = True
        initial_policy = "conservative_display_only"
        warnings = list(NO_PRIOR_WARNING_TEMPLATES)
    elif normalized_level == "caution" and not target_prior_available and not roi_available:
        safe_auto = False
        aggressive_bg = False
        amplitude_claim = False
        auto_gain = False
        manual_review = True
        initial_policy = "conservative_display_only"
        warnings = [
            "当前数据缺少目标先验/ROI，建议采用保守可视化流程并人工复核。",
            "系统不会自动推荐激进背景抑制或默认参数。",
            "以下显示增强仅用于可视化，不代表幅值保真或目标识别。",
        ]
    else:
        safe_auto = True
        aggressive_bg = True
        amplitude_claim = False
        auto_gain = True
        manual_review = False
        initial_policy = "normal_guided_mode"
        warnings = [
            "当前 no-prior 风险较低，仍建议在关键结论前进行人工复核。",
            "显示增强仅用于可视化，不代表幅值保真或目标识别。",
        ]

    allowed_actions = [
        item["action"] for item in NO_PRIOR_ACTION_POLICY if item["policy"] == "allowed"
    ]
    caution_actions = [
        item["action"] for item in NO_PRIOR_ACTION_POLICY if item["policy"] == "caution"
    ]
    blocked_actions = [
        item["action"] for item in NO_PRIOR_ACTION_POLICY if item["policy"] == "blocked"
    ]

    return NoPriorQcPolicy(
        no_prior_level=normalized_level,
        target_prior_available=bool(target_prior_available),
        roi_available=bool(roi_available),
        safe_auto_recommendation_allowed=safe_auto,
        aggressive_background_suppression_allowed=aggressive_bg,
        amplitude_claim_allowed=amplitude_claim,
        auto_gain_allowed=auto_gain,
        manual_review_required=manual_review,
        recommended_initial_policy=initial_policy,
        claim_boundary=NO_PRIOR_CLAIM_BOUNDARY,
        user_facing_warnings=warnings,
        allowed_actions=allowed_actions,
        caution_actions=caution_actions,
        blocked_actions=blocked_actions,
        action_policy=[dict(item) for item in NO_PRIOR_ACTION_POLICY],
    )


def policy_to_dict(policy: NoPriorQcPolicy) -> dict[str, Any]:
    """Convert dataclass policy payload to JSON-serializable dict."""
    return asdict(policy)


__all__ = [
    "NO_PRIOR_ACTION_POLICY",
    "NO_PRIOR_CLAIM_BOUNDARY",
    "NO_PRIOR_WARNING_TEMPLATES",
    "NoPriorQcPolicy",
    "build_no_prior_qc_policy",
    "derive_no_prior_level",
    "policy_to_dict",
]

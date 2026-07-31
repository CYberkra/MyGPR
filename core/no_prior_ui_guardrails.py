#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI guardrail decision helpers for no-prior high-risk mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class NoPriorGuardDecision:
    action_id: str
    decision: str
    reason: str
    warning_text: str
    claim_boundary: str
    log_event: bool


def evaluate_no_prior_action(
    action_id: str,
    no_prior_policy: dict[str, Any] | None,
    *,
    allow_override: bool = True,
) -> NoPriorGuardDecision:
    """Evaluate one UI action under current no-prior policy."""
    policy = dict(no_prior_policy or {})
    action = str(action_id or "").strip()
    level = str(policy.get("no_prior_level") or "ok")
    target_prior_available = bool(policy.get("target_prior_available"))
    roi_available = bool(policy.get("roi_available"))
    claim_boundary = str(policy.get("claim_boundary") or "")
    warning_lines = list(policy.get("user_facing_warnings") or [])
    if action == "raw_preview":
        return NoPriorGuardDecision(
            action_id=action,
            decision="allowed",
            reason="Raw preview remains allowed.",
            warning_text="",
            claim_boundary=claim_boundary,
            log_event=False,
        )
    if level != "high_risk" or target_prior_available or roi_available:
        return NoPriorGuardDecision(
            action_id=action,
            decision="allowed",
            reason="No-prior guardrail not in high-risk blocking mode.",
            warning_text="",
            claim_boundary=claim_boundary,
            log_event=False,
        )

    action_policy = {str(item.get("action")): dict(item) for item in policy.get("action_policy", [])}
    policy_item = action_policy.get(action)
    if action == "workflow_run":
        base_warning = warning_lines[0] if warning_lines else "当前数据触发高风险质控告警。"
        return NoPriorGuardDecision(
            action_id=action,
            decision="requires_confirmation",
            reason="Workflow run in high-risk no-prior mode requires manual confirmation.",
            warning_text=(
                base_warning
                + "\n建议先执行原始剖面检查；若继续，请确认仅用于保守人工复核。"
            ),
            claim_boundary=claim_boundary,
            log_event=True,
        )

    if not policy_item:
        return NoPriorGuardDecision(
            action_id=action,
            decision="allowed",
            reason="Action is not explicitly constrained by no-prior policy.",
            warning_text="",
            claim_boundary=claim_boundary,
            log_event=False,
        )

    policy_state = str(policy_item.get("policy") or "allowed")
    reason = str(policy_item.get("reason") or "")
    claim = str(policy_item.get("claim_boundary") or claim_boundary)
    if policy_state == "blocked":
        warning_text = "\n".join(warning_lines[:3]) if warning_lines else "当前操作已被无先验高风险策略阻断。"
        return NoPriorGuardDecision(
            action_id=action,
            decision="blocked",
            reason=reason or "Blocked by no-prior high-risk policy.",
            warning_text=warning_text,
            claim_boundary=claim,
            log_event=True,
        )
    if policy_state == "caution":
        warning_text = "\n".join(warning_lines) if warning_lines else "当前操作仅限谨慎可视化用途。"
        if allow_override:
            return NoPriorGuardDecision(
                action_id=action,
                decision="requires_confirmation",
                reason=reason or "Caution action requires user confirmation.",
                warning_text=warning_text,
                claim_boundary=claim,
                log_event=True,
            )
        return NoPriorGuardDecision(
            action_id=action,
            decision="caution",
            reason=reason or "Caution action.",
            warning_text=warning_text,
            claim_boundary=claim,
            log_event=True,
        )

    return NoPriorGuardDecision(
        action_id=action,
        decision="allowed",
        reason=reason or "Allowed by no-prior policy.",
        warning_text="",
        claim_boundary=claim,
        log_event=False,
    )


def build_no_prior_guard_event(
    action_id: str,
    decision: NoPriorGuardDecision,
    no_prior_policy: dict[str, Any] | None,
    *,
    override_used: bool = False,
) -> dict[str, Any]:
    """Build one lightweight guardrail event payload."""
    policy = dict(no_prior_policy or {})
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action_id": str(action_id),
        "decision": str(decision.decision),
        "no_prior_level": str(policy.get("no_prior_level") or "ok"),
        "reason": str(decision.reason),
        "manual_review_required": bool(policy.get("manual_review_required")),
        "override_used": bool(override_used),
        "claim_boundary": str(decision.claim_boundary),
    }
    return event


def guard_decision_to_dict(decision: NoPriorGuardDecision) -> dict[str, Any]:
    """Convert dataclass decision to dictionary."""
    return asdict(decision)


__all__ = [
    "NoPriorGuardDecision",
    "build_no_prior_guard_event",
    "evaluate_no_prior_action",
    "guard_decision_to_dict",
]

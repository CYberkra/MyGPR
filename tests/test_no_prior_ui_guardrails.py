#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-prior UI guardrail decision tests."""

from __future__ import annotations

import inspect

from core.no_prior_qc_policy import build_no_prior_qc_policy, policy_to_dict
from core.no_prior_ui_guardrails import (
    build_no_prior_guard_event,
    evaluate_no_prior_action,
)


def _high_risk_no_prior_policy() -> dict:
    return policy_to_dict(
        build_no_prior_qc_policy(
            no_prior_level="high_risk",
            target_prior_available=False,
            roi_available=False,
        )
    )


def test_high_risk_blocks_autotune():
    policy = _high_risk_no_prior_policy()
    decision = evaluate_no_prior_action("AutoTune", policy, allow_override=False)
    assert decision.decision == "blocked"
    assert "高风险" in decision.warning_text


def test_high_risk_blocks_preset_recommendation():
    policy = _high_risk_no_prior_policy()
    decision = evaluate_no_prior_action(
        "preset_recommendation", policy, allow_override=False
    )
    assert decision.decision == "blocked"


def test_high_risk_blocks_aggressive_background_suppression():
    policy = _high_risk_no_prior_policy()
    decision = evaluate_no_prior_action(
        "background_suppression_aggressive", policy, allow_override=False
    )
    assert decision.decision == "blocked"


def test_raw_preview_remains_allowed():
    policy = _high_risk_no_prior_policy()
    decision = evaluate_no_prior_action("raw_preview", policy, allow_override=False)
    assert decision.decision == "allowed"


def test_agc_display_only_requires_confirmation():
    policy = _high_risk_no_prior_policy()
    decision = evaluate_no_prior_action("AGC_display_only", policy, allow_override=True)
    assert decision.decision == "requires_confirmation"
    assert "可视化" in decision.warning_text
    assert "幅值保真" in decision.warning_text


def test_guard_event_payload_contains_required_fields():
    policy = _high_risk_no_prior_policy()
    decision = evaluate_no_prior_action("AutoTune", policy, allow_override=False)
    event = build_no_prior_guard_event(
        "AutoTune",
        decision,
        policy,
        override_used=False,
    )
    assert event["action_id"] == "AutoTune"
    assert event["decision"] == "blocked"
    assert event["no_prior_level"] == "high_risk"
    assert "reason" in event and event["reason"]
    assert event["manual_review_required"] is True


def test_guardrail_module_does_not_import_processing_engine():
    import core.no_prior_ui_guardrails as module

    source = inspect.getsource(module)
    assert "processing_engine" not in source

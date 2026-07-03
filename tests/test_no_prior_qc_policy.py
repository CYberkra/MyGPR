#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-prior QC policy mapping tests."""

from __future__ import annotations

import inspect

from core.no_prior_qc_policy import (
    NO_PRIOR_WARNING_TEMPLATES,
    build_no_prior_qc_policy,
    derive_no_prior_level,
    policy_to_dict,
)


def test_high_risk_no_prior_maps_to_conservative_display_only():
    policy = build_no_prior_qc_policy(
        no_prior_level="high_risk",
        target_prior_available=False,
        roi_available=False,
    )
    payload = policy_to_dict(policy)
    assert payload["no_prior_level"] == "high_risk"
    assert payload["safe_auto_recommendation_allowed"] is False
    assert payload["aggressive_background_suppression_allowed"] is False
    assert payload["amplitude_claim_allowed"] is False
    assert payload["auto_gain_allowed"] is False
    assert payload["manual_review_required"] is True
    assert payload["recommended_initial_policy"] == "conservative_display_only"
    assert "AutoTune" in payload["blocked_actions"]
    assert "preset_recommendation" in payload["blocked_actions"]
    assert "background_suppression_aggressive" in payload["blocked_actions"]


def test_warning_templates_and_claim_boundary_are_present():
    policy = build_no_prior_qc_policy(
        no_prior_level="high_risk",
        target_prior_available=False,
        roi_available=False,
    )
    payload = policy_to_dict(policy)
    assert len(payload["user_facing_warnings"]) >= 5
    assert "高风险质控告警" in payload["user_facing_warnings"][0]
    assert "可视化" in " ".join(payload["user_facing_warnings"])
    assert "no target detection" in payload["claim_boundary"]
    assert "no underground correctness" in payload["claim_boundary"]
    assert NO_PRIOR_WARNING_TEMPLATES[0] == payload["user_facing_warnings"][0]


def test_derive_level_becomes_high_risk_without_prior_roi_when_alerts_present():
    level = derive_no_prior_level(
        quality_metrics={"focus_ratio": 0.1},
        metric_alerts={"focus_ratio": True, "time_ms": False},
        airborne_qc={"alerts": ["spacing_cv_high"]},
        runtime_warnings=[{"code": "data_sanitized"}],
        target_prior_available=False,
        roi_available=False,
    )
    assert level == "high_risk"


def test_derive_level_is_ok_when_target_prior_available():
    level = derive_no_prior_level(
        quality_metrics={"focus_ratio": 0.1},
        metric_alerts={"focus_ratio": True},
        airborne_qc={"alerts": ["spacing_cv_high"]},
        runtime_warnings=[{"code": "data_sanitized"}],
        target_prior_available=True,
        roi_available=False,
    )
    assert level == "ok"


def test_policy_module_does_not_import_processing_engine():
    import core.no_prior_qc_policy as policy_module

    source = inspect.getsource(policy_module)
    assert "processing_engine" not in source

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for non-blocking AutoTune recommendation risk labels."""

from __future__ import annotations

import inspect

from core.auto_tune_recommendation_labels import (
    assess_auto_tune_recommendation_label,
)


def _high_risk_no_prior_policy() -> dict:
    return {
        "no_prior_level": "high_risk",
        "target_prior_available": False,
        "roi_available": False,
    }


def test_high_risk_large_window_is_manual_review_recommended():
    label = assess_auto_tune_recommendation_label(
        method_key="subtracting_average_2D",
        selected_params={"ntraces": 676},
        metrics={},
        score=-0.12,
        no_prior_policy=_high_risk_no_prior_policy(),
        trace_count=2378,
    )
    assert label.recommendation_label == "manual_review_recommended"
    assert label.manual_review_recommended is True
    assert "background_large_window_risk" in label.risk_flags
    assert "negative_score_risk" in label.risk_flags


def test_high_risk_negative_score_adds_negative_risk_flag():
    label = assess_auto_tune_recommendation_label(
        method_key="median_background_2D",
        selected_params={"ntraces": 120},
        metrics={},
        score=-0.01,
        no_prior_policy=_high_risk_no_prior_policy(),
        trace_count=2378,
    )
    assert "negative_score_risk" in label.risk_flags


def test_high_risk_low_preservation_adds_preservation_flags():
    label = assess_auto_tune_recommendation_label(
        method_key="running_average_2D",
        selected_params={"ntraces": 120},
        metrics={
            "local_saliency_preservation": 0.32,
            "edge_preservation": 0.36,
            "peak_ratio": 0.14,
        },
        score=0.1,
        no_prior_policy=_high_risk_no_prior_policy(),
        trace_count=2378,
    )
    assert "local_saliency_preservation_low" in label.risk_flags
    assert "edge_preservation_low" in label.risk_flags
    assert "peak_ratio_low" in label.risk_flags


def test_roi_available_low_risk_can_remain_normal():
    label = assess_auto_tune_recommendation_label(
        method_key="subtracting_average_2D",
        selected_params={"ntraces": 24},
        metrics={
            "local_saliency_preservation": 0.85,
            "edge_preservation": 0.88,
            "peak_ratio": 0.72,
        },
        score=0.25,
        no_prior_policy={
            "no_prior_level": "ok",
            "target_prior_available": True,
            "roi_available": True,
        },
        trace_count=2378,
    )
    assert label.recommendation_label == "normal"
    assert label.manual_review_recommended is False


def test_helper_is_non_blocking_and_score_passthrough():
    score = -0.2
    label = assess_auto_tune_recommendation_label(
        method_key="subtracting_average_2D",
        selected_params={"ntraces": 100},
        metrics={},
        score=score,
        no_prior_policy=_high_risk_no_prior_policy(),
        trace_count=2378,
    )
    assert isinstance(label.recommendation_label, str)
    assert label.risk_basis.get("score") == score


def test_module_does_not_import_processing_engine():
    import core.auto_tune_recommendation_labels as module

    source = inspect.getsource(module)
    assert "processing_engine" not in source


def test_contains_chinese_warning_text():
    label = assess_auto_tune_recommendation_label(
        method_key="subtracting_average_2D",
        selected_params={"ntraces": 676},
        metrics={},
        score=-0.4,
        no_prior_policy=_high_risk_no_prior_policy(),
        trace_count=2378,
    )
    joined = "\n".join(label.user_log_messages)
    assert "自动选参提示" in joined
    assert "人工复核" in joined

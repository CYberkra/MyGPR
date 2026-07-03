#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded workflow planner contracts for AutoTune phase 2."""

from __future__ import annotations

import numpy as np

from core.autotune_workflow_planner import diagnose_bscan, plan_workflow_recipes


def _demo_bscan(samples: int = 96, traces: int = 48) -> np.ndarray:
    y = np.linspace(0.0, 1.0, samples)[:, None]
    x = np.linspace(0.0, 1.0, traces)[None, :]
    arr = 0.10 * np.sin(10 * y) + 0.04 * np.cos(8 * x)
    interface = np.exp(-((y - (0.55 + 0.04 * np.sin(2 * np.pi * x))) ** 2) / 0.002)
    arr += 0.35 * interface
    arr += 0.08 * np.exp(-((x - 0.52) ** 2 + (y - 0.35) ** 2) / 0.006)
    return arr.astype(float)


def test_diagnose_bscan_returns_bounded_descriptors():
    diag = diagnose_bscan(_demo_bscan(), target_response_available=True)
    assert diag.samples == 96
    assert diag.traces == 48
    assert 0.0 <= diag.drift_strength <= 1.0
    assert 0.0 <= diag.stripe_strength <= 1.0
    assert 0.0 <= diag.continuity <= 1.0
    assert diag.target_response_available is True


def test_plan_workflow_recipes_returns_ranked_target_aware_recipes():
    raw = _demo_bscan()
    bg_results = [
        {"name": "SVD 背景抑制 rank=2", "method": "svd", "params": "rank=2", "score": 0.82, "status": "可用", "residual_ratio": 0.42, "cnr_gain": 0.15},
        {"name": "中位数背景扣除", "method": "median", "params": "method=median", "score": 0.74, "status": "可用", "residual_ratio": 0.51, "cnr_gain": 0.10},
    ]
    recipes = plan_workflow_recipes(
        raw,
        target_goal="landslide_interface",
        roi_mode="none",
        scoring_metrics=["roi_retention", "residual", "shape", "cnr"],
        target_response=raw * 0.4,
        background_results=bg_results,
        max_candidates=6,
    )
    assert 2 <= len(recipes) <= 6
    assert recipes[0]["score"] >= recipes[-1]["score"]
    assert recipes[0]["target_goal"] == "滑坡基覆界面 / 潜在滑移面"
    assert recipes[0]["method"] == "workflow_recipe"
    assert recipes[0]["recipe_steps"]
    assert any(step["key"] == "background" for step in recipes[0]["recipe_steps"])
    assert "深部" in recipes[0]["workflow_flow"] or "界面" in recipes[0]["name"]


def test_workflow_planner_keeps_baseline_as_reference_not_recommended_background():
    raw = _demo_bscan()
    recipes = plan_workflow_recipes(
        raw,
        target_goal="balanced",
        background_results=[
            {"name": "不处理基线", "method": "baseline", "params": "method=none", "score": 0.95, "status": "基线"},
            {"name": "中位数背景扣除", "method": "median", "params": "method=median", "score": 0.55, "status": "可用"},
        ],
        max_candidates=4,
    )
    assert recipes
    top = recipes[0]
    bg_steps = [step for step in top["recipe_steps"] if step["key"] == "background"]
    assert bg_steps
    assert bg_steps[0]["enabled"] is True
    assert "跳过" not in bg_steps[0]["method"]
    assert top["background_candidate"]["method"] != "baseline"
    assert top["background_low_benefit"] is True
    assert "背景抑制收益较弱" in top["warning"]


def test_workflow_planner_falls_back_to_real_background_when_only_baseline_exists():
    raw = _demo_bscan()
    recipes = plan_workflow_recipes(
        raw,
        target_goal="balanced",
        background_results=[
            {"name": "不处理基线", "method": "baseline", "params": "method=none", "score": 0.80, "status": "基线"},
        ],
        max_candidates=3,
    )
    assert recipes
    top = recipes[0]
    bg = top["background_candidate"]
    assert bg["method"] in {"median", "mean"}
    assert top["background_low_benefit"] is True
    assert any(
        step["key"] == "background" and step["enabled"] and "跳过" not in step["method"]
        for step in top["recipe_steps"]
    )

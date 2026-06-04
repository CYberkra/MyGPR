#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contracts for AutoTune scoring v2 rule reports and breakdowns."""

from __future__ import annotations

import numpy as np

from core.autotune_background_runner import run_background_candidates
from core.autotune_goal_profiles import goal_profile_table, resolve_goal_profile
from core.autotune_workflow_planner import plan_workflow_recipes


def _demo_bscan(samples: int = 80, traces: int = 40) -> np.ndarray:
    y = np.linspace(0.0, 1.0, samples)[:, None]
    x = np.linspace(0.0, 1.0, traces)[None, :]
    arr = 0.08 * np.sin(12 * y) + 0.05 * np.cos(5 * x)
    arr += 0.35 * np.exp(-((y - (0.58 + 0.03 * np.sin(2 * np.pi * x))) ** 2) / 0.002)
    return arr.astype(float)


def test_goal_profiles_are_normalized_and_target_specific():
    table = goal_profile_table()
    assert "均衡推荐" in table
    for row in table.values():
        assert abs(sum(row["weights"].values()) - 1.0) < 1e-9

    landslide = resolve_goal_profile("landslide_interface")
    assert landslide.label == "滑坡基覆界面 / 潜在滑移面"
    assert landslide.weights["continuity"] > landslide.weights["background_suppression"]
    assert landslide.weights["deep_balance"] > landslide.weights["background_suppression"]


def test_background_candidates_emit_scoring_v2_breakdown():
    rows = run_background_candidates(
        _demo_bscan(),
        candidate_methods=["baseline", "mean", "median"],
        target_goal="balanced",
        scoring_metrics=["roi_retention", "residual", "shape", "cnr"],
    )
    assert rows
    top = rows[0]
    assert top["score_version"] == "autotune_scoring_v2"
    assert top["selection_rule"] == "goal_profile_weighted_background_score"
    assert "background_suppression" in top["goal_profile_weights"]
    assert "v2_background_suppression" in top["scoring_terms"]
    assert "score" in top["score_breakdown"]


def test_workflow_recipes_emit_scoring_v2_terms_and_weights():
    raw = _demo_bscan()
    background_rows = run_background_candidates(
        raw,
        candidate_methods=["baseline", "mean", "median"],
        target_goal="landslide_interface",
    )
    recipes = plan_workflow_recipes(
        raw,
        target_goal="landslide_interface",
        background_results=background_rows,
        max_candidates=4,
    )
    assert recipes
    top = recipes[0]
    assert top["score_version"] == "autotune_scoring_v2"
    assert "background_candidate_score" in top["workflow_score_weights"]
    assert "workflow_fit" in top["scoring_terms"]
    assert any(step["key"] == "background" and step["enabled"] for step in top["recipe_steps"])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune scoring v2 output closure contracts."""

from __future__ import annotations

import numpy as np

from core.autotune_background_runner import run_background_candidates
from core.autotune_scoring_record import build_scoring_v2_record, summarize_record
from core.autotune_workflow_planner import plan_workflow_recipes


def _demo_bscan(samples: int = 72, traces: int = 36) -> np.ndarray:
    y = np.linspace(0.0, 1.0, samples)[:, None]
    x = np.linspace(0.0, 1.0, traces)[None, :]
    arr = 0.07 * np.sin(11 * y) + 0.04 * np.cos(5 * x)
    arr += 0.28 * np.exp(-((y - (0.60 + 0.03 * np.sin(2 * np.pi * x))) ** 2) / 0.0025)
    return arr.astype(float)


def test_workflow_recipe_contains_complete_scoring_v2_record():
    raw = _demo_bscan()
    bg = run_background_candidates(raw, candidate_methods=["baseline", "median", "mean"], target_goal="landslide_interface")
    recipes = plan_workflow_recipes(raw, target_goal="landslide_interface", background_results=bg, max_candidates=3)
    assert recipes
    top = recipes[0]
    record = top["autotune_scoring_record"]
    assert record["autotune_scoring_version"] == "autotune_scoring_v2"
    assert record["target_goal"] == "滑坡基覆界面 / 潜在滑移面"
    assert record["data_mode"] == "无参考标签"
    assert "goal_weights" in record and record["goal_weights"]["continuity"] > 0
    assert "workflow_score" in record and "terms" in record["workflow_score"]
    assert "background_score" in record and "terms" in record["background_score"]
    assert record["background_score"]["method"] in {"mean", "median", "svd", "sliding"}
    assert "diagnostics" in record
    assert summarize_record(record).startswith("scoring v2")


def test_build_record_can_recover_from_legacy_like_recipe_row():
    row = {
        "name": "测试流程",
        "score": 0.71,
        "target_goal": "balanced",
        "roi_mode": "none",
        "scoring_terms": {"workflow_fit": 0.6, "v2_continuity": 0.8},
        "workflow_score_weights": {"workflow_fit": 0.36},
        "background_candidate": {
            "method": "median",
            "name": "中位数背景扣除",
            "params": "method=median",
            "score": 0.62,
            "scoring_terms": {"v2_background_suppression": 0.5, "v2_response_preservation": 0.7},
        },
    }
    record = build_scoring_v2_record(row, target_goal="balanced", roi_mode="none", target_response_available=False)
    assert record["final_score"] == 0.71
    assert record["workflow_score"]["terms"]["continuity"] == 0.8
    assert record["background_score"]["terms"]["background_suppression"] == 0.5

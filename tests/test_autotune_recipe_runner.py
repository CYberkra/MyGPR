#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executable recipe runner contracts for AutoTune workflow recommendations."""

from __future__ import annotations

import numpy as np

from core.autotune_recipe_runner import build_recipe_execution_plan, build_recipe_processing_tasks
from core.autotune_workflow_planner import plan_workflow_recipes
from core.workflow_executor import WorkflowExecutor


def _demo_bscan(samples: int = 64, traces: int = 24) -> np.ndarray:
    y = np.linspace(0.0, 1.0, samples)[:, None]
    x = np.linspace(0.0, 1.0, traces)[None, :]
    return (0.05 * np.sin(10 * y) + 0.02 * np.cos(8 * x) + 0.25 * np.exp(-((y - 0.52) ** 2) / 0.004)).astype(float)


def test_recipe_runner_maps_planner_recipe_to_existing_processing_methods():
    raw = _demo_bscan()
    rows = plan_workflow_recipes(
        raw,
        target_goal="landslide_interface",
        background_results=[
            {"name": "SVD 背景抑制 rank=2", "method": "svd", "params": "rank=2", "score": 0.82, "status": "可用"}
        ],
        max_candidates=3,
    )
    tasks, plan = build_recipe_processing_tasks(rows[0], out_dir=".")

    method_ids = [task["method_key"] for task in tasks]
    assert "dewow" in method_ids
    assert "frequency_filter_1d" in method_ids
    assert "svd_bg" in method_ids
    assert "agcGain" in method_ids
    assert all(task["method"] for task in tasks)
    assert plan.executable_steps
    assert any(step.recipe_key == "zero_time" for step in plan.skipped_steps)


def test_recipe_execution_plan_can_run_through_workflow_executor_without_shape_change():
    raw = _demo_bscan()
    recipe = {
        "name": "单元测试推荐流程",
        "target_goal": "均衡推荐",
        "roi_mode": "none",
        "score": 0.8,
        "recipe_steps": [
            {"key": "zero_time", "label": "零时校正", "method": "保持当前校正", "params": "使用当前设置"},
            {"key": "dewow", "label": "Dewow", "method": "移动窗口去低频漂移", "params": "window=11"},
            {"key": "background", "label": "背景抑制", "method": "SVD 背景抑制 rank=1", "params": "rank=1"},
            {"key": "gain", "label": "增益", "method": "AGC / 温和增益", "params": "window=21"},
        ],
        "background_candidate": {"name": "SVD 背景抑制 rank=1", "method": "svd", "params": "rank=1"},
    }
    plan = build_recipe_execution_plan(recipe)
    result = WorkflowExecutor().execute_all(raw, plan.to_workflow_methods())

    assert result.shape == raw.shape
    assert np.isfinite(result).all()
    assert len(plan.to_workflow_methods()) == 3


def test_recipe_runner_converts_legacy_baseline_background_to_mild_real_method():
    recipe = {
        "name": "legacy baseline recipe",
        "recipe_steps": [
            {"key": "zero_time", "label": "零时校正", "method": "保持当前校正", "params": "使用当前设置"},
            {"key": "background", "label": "背景抑制", "method": "跳过", "params": "保留当前数据"},
        ],
        "background_candidate": {"name": "不处理基线", "method": "baseline", "params": "method=none"},
    }
    tasks, plan = build_recipe_processing_tasks(recipe, out_dir=".")
    method_ids = [task["method_key"] for task in tasks]
    assert "median_background_2D" in method_ids
    assert not any(step.recipe_key == "background" for step in plan.skipped_steps)

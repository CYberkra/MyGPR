#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V0.8.46 regression tests for scoring v2 lineage/evidence closure."""

from __future__ import annotations

import numpy as np

from core.autotune_recipe_runner import build_recipe_processing_tasks
from core.shared_data_model import SharedDataModel
from ui.processing_lineage_controller import ProcessingLineageController


class _Host:
    def __init__(self, shared):
        self.shared_data = shared
        self._compare_selected_indices = set()


def _recipe_row() -> dict:
    scoring_record = {
        "autotune_scoring_version": "autotune_scoring_v2",
        "final_score": 0.81,
        "target_goal": "均衡推荐",
        "roi_mode": "full",
        "workflow_score": {"terms": {"workflow_fit": 0.8}},
        "background_score": {"terms": {"background_suppression": 0.7}},
    }
    return {
        "name": "推荐流程",
        "target_goal": "balanced",
        "roi_mode": "full",
        "score": 0.81,
        "recipe_steps": [
            {"key": "dewow", "label": "Dewow", "method": "auto", "params": "window=23", "enabled": True},
            {"key": "background", "label": "背景抑制", "method": "median", "params": "ntraces=9", "enabled": True},
        ],
        "background_candidate": {"method": "median", "name": "中位数背景扣除", "params": "ntraces=9"},
        "autotune_scoring_record": scoring_record,
    }


def test_recipe_processing_tasks_carry_scoring_v2_record():
    tasks, plan = build_recipe_processing_tasks(_recipe_row())

    assert plan.scoring_record["autotune_scoring_version"] == "autotune_scoring_v2"
    assert tasks
    assert all(task.get("autotune_scoring_record") for task in tasks)
    assert tasks[0]["autotune_recipe_plan"]["autotune_scoring_record"]["final_score"] == 0.81


def test_processing_lineage_export_preserves_scoring_v2_header_metadata():
    shared = SharedDataModel()
    raw = np.zeros((4, 3), dtype=np.float32)
    processed = np.ones((4, 3), dtype=np.float32)
    scoring_record = _recipe_row()["autotune_scoring_record"]
    shared.load_data(raw, path="demo.csv")
    shared.apply_current_data(
        processed,
        label="中位数背景扣除",
        header_info={
            "method_key": "median_background_2D",
            "params": {"ntraces": 9},
            "autotune_scoring_record": scoring_record,
        },
    )

    controller = ProcessingLineageController(_Host(shared))
    steps = controller.build_export_steps()

    assert steps[-1]["label"] == "中位数背景扣除"
    assert steps[-1]["autotune_scoring_record"]["autotune_scoring_version"] == "autotune_scoring_v2"

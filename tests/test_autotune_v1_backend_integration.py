#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration contracts for AutoTune V1 candidate-space backend hookup."""

from __future__ import annotations

import numpy as np

from core.autotune_background_runner import run_background_candidates
from core.autotune_workflow_planner import plan_workflow_recipes


def _demo_bscan(samples: int = 96, traces: int = 48) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    x = np.linspace(0.0, 1.0, traces)[None, :]
    layer = 0.35 * np.exp(-((t - (0.55 + 0.05 * np.sin(2.0 * np.pi * x))) ** 2) / 0.002)
    stripe = 0.10 * np.sin(7.0 * t) + 0.04 * np.cos(5.0 * x)
    anomaly = 0.45 * np.exp(-((t - 0.35) ** 2) / 0.003 - ((x - 0.62) ** 2) / 0.010)
    return (stripe + layer + anomaly).astype(float)


def test_background_runner_can_use_v1_candidate_space_and_records_hash():
    raw = _demo_bscan()
    rows = run_background_candidates(
        raw,
        candidate_methods=["baseline", "mean", "median", "sliding", "svd"],
        svd_ranks=[1, 2, 3],
        target_goal="landslide_interface",
        metadata={"dt_ns": 0.1, "center_frequency": 250},
        use_v1_candidate_space=True,
    )

    assert rows
    assert all(row.get("candidate_space_hash") for row in rows)
    assert rows[0]["selection_rule"] == "v1_candidate_space_hash_ranked_background_score"
    assert rows[0]["candidate_space_profile_id"] == "landslide_bedrock_sliding_surface"
    svd_rows = [row for row in rows if row["method"] == "svd" and row["status"] != "已跳过"]
    assert svd_rows
    assert max(int(row["candidate_parameters"].get("remove_rank", 0)) for row in svd_rows) <= 1
    assert all("candidate_space_context" in row for row in rows)


def test_workflow_planner_propagates_v1_candidate_space_into_scoring_record():
    raw = _demo_bscan()
    background_rows = run_background_candidates(
        raw,
        candidate_methods=["baseline", "mean", "median", "sliding", "svd"],
        target_goal="deep_weak",
        metadata={"dt_ns": 0.1, "center_frequency_hz": 300_000_000},
        use_v1_candidate_space=True,
    )
    recipes = plan_workflow_recipes(
        raw,
        target_goal="deep_weak",
        background_results=background_rows,
        max_candidates=4,
    )

    assert recipes
    top = recipes[0]
    assert top["candidate_space_hash"] == background_rows[0]["candidate_space_hash"]
    assert top["candidate_space_profile_id"] == "deep_weak_reflector"
    record = top["autotune_scoring_record"]
    assert record["candidate_space"]["candidate_space_hash"] == top["candidate_space_hash"]
    assert record["candidate_space"]["profile_id"] == "deep_weak_reflector"

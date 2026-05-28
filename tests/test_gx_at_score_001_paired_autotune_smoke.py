#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for GX-AT-SCORE-001 paired autotune smoke inventory/scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.autotune_paired_scoring_smoke import (
    build_inventory,
    score_scene_candidates,
    select_stable_pairs,
)


def _make_task(
    root: Path,
    name: str,
    *,
    expected: int = 80,
    raw_count: int = 80,
    bg_count: int = 80,
    include_target: bool = True,
) -> Path:
    task = root / name
    model = task / "1_模型输入"
    raw_root = task / "2_gprMax原始输出" / "raw_含目标"
    bg_root = task / "2_gprMax原始输出" / "background_纯背景"
    read_root = task / "3_MyGPR读取文件"
    log_root = task / "4_日志与报告"
    model.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    bg_root.mkdir(parents=True, exist_ok=True)
    read_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scene_id": f"scene_{name}",
        "stages": [{"stage": "run_plan", "trace_count": expected, "status": "planned"}],
    }
    (task / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    for i in range(1, raw_count + 1):
        (raw_root / f"raw_with_target{i}.out").write_bytes(b"")
    for i in range(1, bg_count + 1):
        (bg_root / f"background_only{i}.out").write_bytes(b"")

    arr = np.ones((32, expected), dtype=np.float64)
    np.save(read_root / "raw_Ey.npy", arr)
    if bg_count > 0:
        np.save(read_root / "background_Ey.npy", arr * 0.5)
    if include_target and bg_count > 0 and raw_count == bg_count == expected:
        np.save(read_root / "target_response_Ey.npy", arr * 0.25)
    return task


def test_inventory_recognizes_output_v5_single_directory(tmp_path: Path) -> None:
    single = tmp_path / "01_单次仿真"
    _make_task(single, "task_single_ok")
    inventory = build_inventory([single], component_preference="Ey")
    assert len(inventory) == 1
    assert inventory[0].root_kind == "single"
    assert inventory[0].status == "stable_completed"


def test_inventory_recognizes_output_v5_batch_directory(tmp_path: Path) -> None:
    batch = tmp_path / "02_批量仿真" / "Batch_A" / "T1_单目标"
    _make_task(batch, "task_batch_ok")
    inventory = build_inventory([tmp_path / "02_批量仿真"], component_preference="Ey")
    assert len(inventory) == 1
    assert inventory[0].root_kind == "batch"
    assert inventory[0].status == "stable_completed"


def test_79_of_80_not_selected_as_stable_pair(tmp_path: Path) -> None:
    root = tmp_path / "01_单次仿真"
    _make_task(root, "task_79of80", expected=80, raw_count=79, bg_count=79)
    inventory = build_inventory([root], component_preference="Ey")
    stable = select_stable_pairs(inventory)
    assert len(stable) == 0


def test_raw_only_not_selected_for_scoring(tmp_path: Path) -> None:
    root = tmp_path / "01_单次仿真"
    _make_task(root, "task_raw_only", expected=80, raw_count=80, bg_count=0, include_target=False)
    inventory = build_inventory([root], component_preference="Ey")
    stable = select_stable_pairs(inventory)
    assert len(stable) == 0


def test_background_only_not_selected_for_scoring(tmp_path: Path) -> None:
    root = tmp_path / "01_单次仿真"
    _make_task(root, "task_bg_only", expected=80, raw_count=0, bg_count=80, include_target=False)
    inventory = build_inventory([root], component_preference="Ey")
    stable = select_stable_pairs(inventory)
    assert len(stable) == 0


def test_metrics_compute_for_same_shape_arrays() -> None:
    raw = np.random.default_rng(0).normal(size=(64, 31))
    target = raw - np.mean(raw, axis=1, keepdims=True)
    rows = score_scene_candidates(
        scene_id="scene_test",
        task_dir=Path("D:/dummy"),
        raw=raw,
        target_response=target,
        roi={"sample_range": [10, 40], "trace_range": [8, 20]},
        component="Ey",
    )
    assert rows
    first = rows[0]
    assert first.mae >= 0
    assert first.rmse >= 0
    assert first.mse >= 0


def test_svd_rank_sweep_returns_trial_rows() -> None:
    raw = np.random.default_rng(1).normal(size=(48, 25))
    target = raw * 0.8
    rows = score_scene_candidates(
        scene_id="scene_svd",
        task_dir=Path("D:/dummy"),
        raw=raw,
        target_response=target,
        roi=None,
        component="Ey",
    )
    names = {row.candidate for row in rows}
    for rank in (1, 2, 3, 5, 8, 10):
        assert f"svd_rank_{rank}" in names
    assert len(rows) == 9  # baseline + mean + median + 6 svd

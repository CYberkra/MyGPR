#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for GX-AT-SCORE-002 scoring hardening behaviors."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from core.autotune_paired_scoring_smoke import (
    _build_array_from_out_dir,
    build_inventory,
    ensure_scene_arrays,
    score_scene_candidates,
    select_scoreable_pairs,
    summarize_candidate_aggregates,
    top_k_candidates_for_scene,
)


def _write_out(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("rxs")
        rx1 = g.create_group("rx1")
        rx1.create_dataset("Ey", data=data)


def _make_task(
    root: Path,
    name: str,
    *,
    expected: int = 80,
    raw_count: int = 80,
    bg_count: int = 80,
    include_target_npy: bool = True,
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
    arr = np.ones((16,), dtype=np.float64)
    for i in range(1, raw_count + 1):
        _write_out(raw_root / f"raw_with_target{i}.out", arr * i)
    for i in range(1, bg_count + 1):
        _write_out(bg_root / f"background_only{i}.out", arr * (i * 0.5))

    if raw_count == expected:
        np.save(read_root / "raw_Ey.npy", np.ones((16, expected), dtype=np.float64))
    if bg_count == expected:
        np.save(read_root / "background_Ey.npy", np.ones((16, expected), dtype=np.float64) * 0.5)
    if include_target_npy and raw_count == expected and bg_count == expected:
        np.save(read_root / "target_response_Ey.npy", np.ones((16, expected), dtype=np.float64) * 0.5)
    return task


def test_inventory_marks_convertible_pair_when_target_missing(tmp_path: Path) -> None:
    root = tmp_path / "01_单次仿真"
    _make_task(root, "convertible_ok", include_target_npy=False)
    inventory = build_inventory([root], component_preference="Ey")
    assert len(inventory) == 1
    assert inventory[0].status == "convertible_pair"
    scoreable = select_scoreable_pairs(inventory)
    assert len(scoreable) == 1


def test_convertible_pair_generates_target_response(tmp_path: Path) -> None:
    root = tmp_path / "01_单次仿真"
    _make_task(root, "convertible_build", include_target_npy=False)
    item = build_inventory([root], component_preference="Ey")[0]
    raw, bg, tr, summary = ensure_scene_arrays(item, component="Ey")
    assert raw.shape == bg.shape == tr.shape
    assert np.allclose(tr, raw - bg)
    assert summary["target_response_source"] == "computed_raw_minus_background"
    tr_path = Path(item.paths["mygpr_read_dir"]) / "target_response_Ey.npy"
    assert tr_path.exists()


def test_smoke_out_file_without_index_is_excluded(tmp_path: Path) -> None:
    out_dir = tmp_path / "raw_含目标"
    arr = np.arange(8, dtype=np.float64)
    _write_out(out_dir / "raw_with_target.out", np.full((8,), 999.0))
    for i in range(1, 81):
        _write_out(out_dir / f"raw_with_target{i}.out", arr + i)
    built = _build_array_from_out_dir(out_dir, "Ey")
    assert built.shape == (8, 80)
    assert np.allclose(built[:, 0], arr + 1)
    assert np.allclose(built[:, -1], arr + 80)


def test_non_continuous_trace_indices_raise(tmp_path: Path) -> None:
    out_dir = tmp_path / "background_纯背景"
    arr = np.arange(6, dtype=np.float64)
    _write_out(out_dir / "background_only1.out", arr)
    _write_out(out_dir / "background_only3.out", arr + 2)
    try:
        _build_array_from_out_dir(out_dir, "Ey")
    except ValueError as exc:
        assert "not continuous" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-continuous indices")


def test_scoreable_excludes_raw_only_background_only_and_79_of_80(tmp_path: Path) -> None:
    root = tmp_path / "01_单次仿真"
    _make_task(root, "ok", expected=80, raw_count=80, bg_count=80, include_target_npy=False)
    _make_task(root, "raw_only", expected=80, raw_count=80, bg_count=0, include_target_npy=False)
    _make_task(root, "bg_only", expected=80, raw_count=0, bg_count=80, include_target_npy=False)
    _make_task(root, "partial", expected=80, raw_count=79, bg_count=79, include_target_npy=False)
    inv = build_inventory([root], component_preference="Ey")
    ids = {item.scene_id_guess: item for item in inv}
    assert ids["scene_ok"].status == "convertible_pair"
    assert ids["scene_raw_only"].status in {"incomplete", "unknown", "running_or_unstable"}
    assert ids["scene_bg_only"].status in {"incomplete", "unknown", "running_or_unstable"}
    assert ids["scene_partial"].status in {"incomplete", "unknown", "running_or_unstable"}
    scoreable = select_scoreable_pairs(inv)
    assert [x.scene_id_guess for x in scoreable] == ["scene_ok"]


def test_svd_rank_sweep_rows_and_aggregate_and_top3() -> None:
    raw = np.random.default_rng(42).normal(size=(48, 31))
    target = raw - np.mean(raw, axis=1, keepdims=True)
    rows = score_scene_candidates(
        scene_id="scene_demo",
        task_dir=Path("D:/dummy"),
        raw=raw,
        target_response=target,
        roi=None,
        component="Ey",
    )
    names = {r.candidate for r in rows}
    for rank in (1, 2, 3, 5, 8, 10):
        assert f"svd_rank_{rank}" in names
    agg = summarize_candidate_aggregates(rows)
    assert "median_background" in agg
    assert agg["median_background"]["scene_count"] == 1
    top3 = top_k_candidates_for_scene(rows, k=3)
    assert "scene_demo" in top3
    assert len(top3["scene_demo"]) == 3
    assert top3["scene_demo"][0].mae <= top3["scene_demo"][1].mae

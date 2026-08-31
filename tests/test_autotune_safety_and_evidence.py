#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-4 AutoTune 安全边界、候选空间与证据导出的直接覆盖。

此前 constraints/candidate_planner/candidate_generators（约 1800 行生产代码）
没有任何直接测试——它们是 AutoTune 参数搜索的生产安全防线，本文件钉住：
1. 越界参数被夹紧到数据形状派生的安全域，并产生约束警告；
2. 非数值参数保持原样（设计契约：交给失败试验而非静默转换）；
3. 候选生成确定性且全部落在约束域内；
4. 证据导出的载荷结构、SHA-256 完整性与往返稳定。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mygpr.application.autotune.candidate_planner import _build_candidate_trials
from mygpr.domain.autotune.models import AutoTuneContext  # noqa: F401 (kept for typing)
from mygpr.application.autotune.evidence import (
    AUTOTUNE_EVIDENCE_SCHEMA,
    build_autotune_evidence,
    export_autotune_evidence,
)
from mygpr.domain.autotune.constraints import constrain_auto_tune_params

SHAPE = (256, 128)
HEADER = {"total_time_ns": 512.0}


def _constrain(method_key: str, params: dict, header: dict | None = None):
    return constrain_auto_tune_params(method_key, dict(params), SHAPE, header if header is not None else dict(HEADER or {}))


def _warnings(result) -> list[dict]:
    return list(getattr(result, "warnings", []) or [])


# ---------------------------------------------------------------------------
# 1) 安全边界：越界夹紧
# ---------------------------------------------------------------------------


def test_dewow_window_clamped_to_half_samples() -> None:
    result = _constrain("dewow", {"window": 10_000})
    assert result.effective_params["window"] <= SHAPE[0] // 2
    assert result.adjusted
    assert any(
        w.get("details", {}).get("parameter") == "window" for w in _warnings(result)
    )


def test_agc_window_clamped_into_safe_band() -> None:
    low = _constrain("agcGain", {"window": 0})
    assert low.effective_params["window"] >= 1
    high = _constrain("agcGain", {"window": 1_000_000})
    assert high.effective_params["window"] <= SHAPE[0]
    assert high.adjusted


def test_set_zero_time_respects_time_window() -> None:
    result = _constrain("set_zero_time", {"new_zero_time": 1.0e9})
    assert float(result.effective_params["new_zero_time"]) <= 512.0
    assert result.adjusted


def test_non_numeric_params_left_unchanged() -> None:
    """非数值保持原样——契约要求它们产生失败试验而非被静默转成合法值。"""
    result = _constrain("dewow", {"window": "not-a-number"})
    assert result.effective_params["window"] == "not-a-number"


def test_in_range_params_pass_through_without_warning() -> None:
    result = _constrain("dewow", {"window": 23})
    assert result.effective_params["window"] == 23
    assert not result.adjusted
    assert _warnings(result) == []


# ---------------------------------------------------------------------------
# 2) 候选空间：确定性 + 域内
# ---------------------------------------------------------------------------


def _planner_context(method_key: str) -> AutoTuneContext:
    from mygpr.application.autotune.context import _build_auto_tune_context

    data = np.zeros((48, 32), dtype=np.float32)
    return _build_auto_tune_context(
        data, dict(HEADER), {}, {"mode": "full"}, "standard"
    )


def _planner_candidates(method_key: str, family: str, base: dict) -> list[dict]:
    data = np.zeros((48, 32), dtype=np.float32)
    context = _planner_context(method_key)
    method_info = {
        "auto_tune_family": family,
        "auto_tune_candidates": {"window": [3, 5, 7]},
    }
    trials = _build_candidate_trials(
        method_key,
        data,
        dict(base),
        dict(HEADER),
        {},
        context,
        method_info,
        stage="coarse",
    )
    assert trials, "候选空间不应为空"
    assert all(isinstance(item, dict) for item in trials)
    return trials


def test_candidate_generation_is_deterministic() -> None:
    a = _planner_candidates("dewow", "drift", {})
    b = _planner_candidates("dewow", "drift", {})
    assert [sorted(map(str, t.items())) for t in a] == [sorted(map(str, t.items())) for t in b]


def test_generated_candidates_stay_within_constrained_domain() -> None:
    """生成的候选经安全约束后不应再被调整——生成器与约束层一致。"""
    for params in _planner_candidates("dewow", "drift", {}):
        result = constrain_auto_tune_params("dewow", params, SHAPE, dict(HEADER))
        assert result.effective_params == params, f"候选越界: {params} -> {result.effective_params}"


# ---------------------------------------------------------------------------
# 3) 证据导出
# ---------------------------------------------------------------------------


def _sample_result() -> dict:
    return {
        "method": "dewow",
        "method_name": "低频漂移矫正（Dewow）",
        "recommended_params": {"window": 23},
        "best_params": {"window": 23},
        "best_score": 0.87,
        "trial_count": 6,
        "valid_trial_count": 5,
        "all_trials": [
            {"params": {"window": 23}, "score": 0.87, "valid": True, "metrics": {"snr": 3.2}},
            {"params": {"window": 999}, "score": 0.0, "valid": False, "reason": "约束失败"},
        ],
        "top_candidates": [{"rank": 1, "params": {"window": 23}, "score": 0.87}],
        "runtime_warnings": [{"code": "sample", "message": "演示"}],
    }


def test_evidence_payload_structure_and_digest(tmp_path: Path) -> None:
    result = _sample_result()
    evidence = build_autotune_evidence(result, method_key="dewow", data_shape=SHAPE)
    assert evidence["schema"] == AUTOTUNE_EVIDENCE_SCHEMA
    assert evidence["method_key"] == "dewow"
    assert evidence["data_shape"] == list(SHAPE)
    body = evidence["body"]
    assert body["recommended_params"] == {"window": 23}
    assert len(body["all_trials"]) == 2

    expected_digest = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert evidence["body_sha256"] == expected_digest

    out = export_autotune_evidence(result, tmp_path / "evidence" / "dewow.json", method_key="dewow", data_shape=SHAPE)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["body_sha256"] == expected_digest
    assert reloaded["body"]["all_trials"][1]["valid"] is False


def test_evidence_digest_changes_when_trials_change() -> None:
    a = build_autotune_evidence(_sample_result(), method_key="dewow")
    mutated = _sample_result()
    mutated["all_trials"][0]["score"] = 0.99
    b = build_autotune_evidence(mutated, method_key="dewow")
    assert a["body_sha256"] != b["body_sha256"]

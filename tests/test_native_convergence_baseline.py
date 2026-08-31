#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务 F 候选 2（双执行器收敛）阶段 0：等价性基线矩阵。

本文件是收敛工作的验收基准，分两部分：

1. 数值等价矩阵 —— 每个已迁移方法都必须有一种等价证据：
   ``direct_comparison``（native 与 legacy 直接对比）、``golden_digest``
   （native 输出被 SHA-256 摘要钉死）、``bitwise_kernel``（与历史 CPU kernel
   逐位一致）或 ``determinism_contract``（实验性方法，仅要求确定性）。
   ``test_equivalence_evidence_covers_all_native_methods`` 保证 36 个方法
   无一遗漏；本文件同时为证据最少的 wavelet_2d / wavelet_svd 补上直接对比。

2. 描述符等价矩阵 —— ``descriptor_baseline.json`` 钉死 Composite 目录当前
   对全部 36 个方法的输出（中文 display_name、category、visibility、
   auto_tune_stage、parameter_schema 合并结果等）。阶段 1 元数据迁移后，
   新目录实现必须逐字段复现这份基线。

详见 ``_handoff_20260830/任务F候选2_双执行器收敛实施计划.md``。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
from mygpr.infrastructure.processing.native_adapter import (
    NativeProcessingCatalog,
    NativeProcessingExecutor,
)

from core.processing_engine import (
    clone_header_info,
    clone_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "processing_convergence" / "descriptor_baseline.json"
)


def _json_normalize(value: object) -> object:
    """把 tuple 等 JSON 可序列化但类型不稳的值归一到 fixture 的 JSON 形态。"""
    return json.loads(json.dumps(value, ensure_ascii=False))


def _legacy_kernel(request: ProcessingRequest) -> np.ndarray:
    """复刻旧 LegacyProcessingExecutor 的执行路径（core kernel 直调，等价证据用）。

    自任务 F 候选 2 拆除 legacy 适配器后，历史对照统一走
    ``core.processing_engine`` 原始 kernel。
    """
    prepared = prepare_runtime_params(
        request.method_id,
        request.params,
        clone_header_info(request.header_info),
        clone_trace_metadata(request.trace_metadata),
        request.data.shape,
    )
    output, _ = run_processing_method(request.data, request.method_id, prepared)
    return np.asarray(output)

# 数值等价证据矩阵：method_id -> (证据类型, 所在测试文件)
# direct_comparison = 测试内同时跑 native 执行器与 core kernel（旧 Legacy 执行器路径）并 assert_allclose
# golden_digest      = native 输出被 SHA-256 摘要钉死（摘要于迁移验证时捕获）
# bitwise_kernel     = 与历史 CPU kernel 逐位一致
# determinism_contract = 实验性方法，仅要求确定性/预算契约（无 legacy 对照）
EQUIVALENCE_EVIDENCE: dict[str, tuple[str, str]] = {
    # tests/test_native_processing_backend.py — 直接对比
    "compensatingGain": ("direct_comparison", "test_native_processing_backend.py"),
    "dewow": ("direct_comparison", "test_native_processing_backend.py"),
    "set_zero_time": ("direct_comparison", "test_native_processing_backend.py"),
    "agcGain": ("direct_comparison", "test_native_processing_backend.py"),
    "sec_gain": ("direct_comparison", "test_native_processing_backend.py"),
    "subtracting_average_2D": ("direct_comparison", "test_native_processing_backend.py"),
    "running_average_2D": ("direct_comparison", "test_native_processing_backend.py"),
    "sliding_avg": ("direct_comparison", "test_native_processing_backend.py"),
    "frequency_filter_1d": ("direct_comparison", "test_native_processing_backend.py"),
    "trace_median_filter": ("direct_comparison", "test_native_processing_backend.py"),
    "trace_savgol_filter": ("direct_comparison", "test_native_processing_backend.py"),
    # tests/test_native_global_processing.py — 直接对比
    "svd_bg": ("direct_comparison", "test_native_global_processing.py"),
    "svd_subspace": ("direct_comparison", "test_native_global_processing.py"),
    "fk_filter": ("direct_comparison", "test_native_global_processing.py"),
    "stolt_migration": ("direct_comparison", "test_native_global_processing.py"),
    "rpca_background": ("direct_comparison", "test_native_global_processing.py"),
    "hankel_svd": ("direct_comparison", "test_native_global_processing.py"),
    # 本文件 —— 直接对比（阶段 0 补齐的缺口）
    "wavelet_2d": ("direct_comparison", "test_native_convergence_baseline.py"),
    "wavelet_svd": ("direct_comparison", "test_native_convergence_baseline.py"),
    # tests/test_native_extended_processing.py — golden 摘要
    "time_cut": ("golden_digest", "test_native_extended_processing.py"),
    "trace_qc": ("golden_digest", "test_native_extended_processing.py"),
    "equidistant_trace_resample": ("golden_digest", "test_native_extended_processing.py"),
    "energy_decay_gain": ("golden_digest", "test_native_extended_processing.py"),
    "amplitude_scale": ("golden_digest", "test_native_extended_processing.py"),
    "median_background_2D": ("golden_digest", "test_native_extended_processing.py"),
    "hilbert_envelope": ("golden_digest", "test_native_extended_processing.py"),
    "ccbs": ("golden_digest", "test_native_extended_processing.py"),
    "time_to_depth": ("golden_digest", "test_native_extended_processing.py"),
    # tests/test_native_motion_processing.py — golden 摘要
    "motion_compensation_height": ("golden_digest", "test_native_motion_processing.py"),
    "motion_compensation_speed": ("golden_digest", "test_native_motion_processing.py"),
    "trajectory_smoothing": ("golden_digest", "test_native_motion_processing.py"),
    "motion_compensation_attitude": ("golden_digest", "test_native_motion_processing.py"),
    "motion_compensation_vibration": ("golden_digest", "test_native_motion_processing.py"),
    "motion_compensation_v2": ("golden_digest", "test_native_motion_processing.py"),
    # tests/test_native_migration_imaging.py
    "kirchhoff_migration": ("bitwise_kernel", "test_native_migration_imaging.py"),
    "rtm_migration": ("determinism_contract", "test_native_migration_imaging.py"),
}


def _matrix(rows: int = 64, cols: int = 48) -> np.ndarray:
    rng = np.random.default_rng(20260831)
    t = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, cols, dtype=np.float32)[None, :]
    base = 0.4 * np.sin(4.0 * t) @ np.ones((1, cols), dtype=np.float32)
    reflector = 0.25 * np.exp(-((t - (0.5 + 0.06 * x**2)) ** 2) / 0.003)
    return np.asarray(base + reflector + rng.normal(0.0, 0.03, (rows, cols)), dtype=np.float32)


def test_equivalence_evidence_covers_all_native_methods() -> None:
    """证据矩阵必须与 NATIVE_ALGORITHMS 精确同构——不多不少。"""
    assert set(EQUIVALENCE_EVIDENCE) == set(NATIVE_ALGORITHMS)


@pytest.mark.parametrize(
    ("method_id", "params"),
    [
        # 显式指定 threshold_strategy，避免默认值漂移影响对比
        (
            "wavelet_2d",
            {
                "wavelet": "db4",
                "levels": 2,
                "threshold": 0.12,
                "threshold_strategy": "mad_universal",
                "threshold_mode": "soft",
            },
        ),
        (
            "wavelet_svd",
            {
                "wavelet": "db4",
                "levels": 3,
                "threshold": 0.08,
                "rank_start": 1,
                "rank_end": 6,
                "threshold_strategy": "mad_universal",
                "threshold_mode": "soft",
            },
        ),
    ],
)
def test_wavelet_native_matches_legacy_kernel(method_id: str, params: dict) -> None:
    """阶段 0 缺口补齐：wavelet 两方法此前只有 kernel 契约测试，无执行器级对比。

    两条路径最终调同一实现（PythonModule 已是兼容门面），预期逐位一致。
    """
    request = ProcessingRequest(
        data=_matrix(),
        method_id=method_id,
        params=params,
        header_info={"total_time_ns": 128.0},
    )
    native = NativeProcessingExecutor().execute(request)
    legacy = _legacy_kernel(request)
    np.testing.assert_allclose(native.data, legacy, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("method_id", sorted(NATIVE_ALGORITHMS))
def test_descriptor_baseline_matches_native_catalog(method_id: str) -> None:
    """合并后的 NativeProcessingCatalog 必须逐字段复现基线 fixture（阶段 1 回归断言）。"""
    baseline = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert method_id in baseline, f"baseline fixture 缺少 {method_id}，需重新生成"
    catalog = NativeProcessingCatalog()
    descriptor = catalog.get(method_id)
    assert descriptor is not None
    expected = baseline[method_id]
    actual = {
        "name": descriptor.name,
        "category": descriptor.category,
        "auto_tune_enabled": descriptor.auto_tune_enabled,
        "auto_tune_family": descriptor.auto_tune_family,
        "auto_tune_stage": descriptor.auto_tune_stage,
        "visibility": descriptor.visibility,
        "parameter_schema": dict(descriptor.parameter_schema),
        "capabilities": sorted(descriptor.capabilities),
        "implementation_version": descriptor.implementation_version,
    }
    assert _json_normalize(actual) == expected


def test_native_catalog_list_order_and_public_filter() -> None:
    """目录遍历顺序与 public_only 过滤契约（收敛后自足断言，不再依赖已删的 Composite）。"""
    catalog = NativeProcessingCatalog()
    ids = [d.method_id for d in catalog.list()]
    assert ids == list(NATIVE_ALGORITHMS)
    public_ids = [d.method_id for d in catalog.list(public_only=True)]
    assert len(public_ids) < len(ids)
    assert all(catalog.get(mid).visibility == "public" for mid in public_ids)  # type: ignore[union-attr]
    hidden = [mid for mid in ids if catalog.get(mid).visibility == "hidden"]  # type: ignore[union-attr]
    assert hidden and all(mid not in public_ids for mid in hidden)


def test_descriptor_baseline_fixture_is_complete() -> None:
    """fixture 覆盖且仅覆盖全部 native 方法。"""
    baseline = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert set(baseline) == set(NATIVE_ALGORITHMS)

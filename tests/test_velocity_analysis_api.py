#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.1 无界面验收：拾取点 → 双曲线拟合 → 速度模型写回 → 证据链。

全部经 ``MyGPRBackend`` 公共 API（job 系统）完成，不 import 任何 Qt 模块；
覆盖 hybrid（默认）与 legacy npz 两种持久化存储模式的 ε 写回与深度轴重算。
"""
from __future__ import annotations


import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mygpr.application.jobs.models import JobStatus
from mygpr.interfaces.backend import MyGPRBackend
from mygpr.domain.velocity.models import (
    VELOCITY_ANALYSIS_EVIDENCE_SCHEMA,
    HyperbolaFit,
    VelocityPick,
)
from mygpr.domain.velocity.fitting import fit_hyperbola
from mygpr.infrastructure.persistence.velocity_adapter import VelocityPersistenceMixin
from mygpr.application.velocity.evidence import (
    build_velocity_evidence,
    compute_velocity_body_digest,
)

pytestmark = [
    pytest.mark.integration,
]

C_M_PER_NS = 0.299792458


def _write_hyperbola_csv(path: Path, *, traces: int = 41, samples: int = 256,
                         v_m_ns: float = 0.1, x0_m: float = 2.0,
                         z0_m: float = 0.5, trace_step_m: float = 0.1) -> None:
    """合成含单一绕射双曲线的 B-scan CSV（MyGPR 侧车格式）。"""
    dt = 512.0 / samples  # ns，Time windows (ns) = 512
    amp = np.zeros((samples, traces), dtype=np.float32)
    for tr in range(traces):
        x = tr * trace_step_m
        two_way = 2.0 * np.sqrt(max((x - x0_m) ** 2 + z0_m ** 2, 1e-9)) / v_m_ns
        center = two_way / dt
        width = 2.5
        s0 = max(int(center - 3 * width), 0)
        s1 = min(int(center + 3 * width) + 1, samples)
        for s in range(s0, s1):
            amp[s, tr] = float(np.exp(-((s - center) ** 2) / (2 * width ** 2)))
    rng = np.random.default_rng(20260905)
    lines = [
        f"Number of Samples = {samples},,,\n",
        "Time windows (ns) = 512,,\n",
        f"Number of Traces = {traces},,\n",
        f"Trace interval (m) = {trace_step_m},,\n",
    ]
    for tr in range(traces):
        lon = 106.80 + tr * 1e-5
        lat = 31.26 + tr * 1e-5
        for s in range(samples):
            noise = float(rng.normal(0.0, 0.005))
            lines.append(f"{lon:.8f},{lat:.8f},441.7,{amp[s, tr] + noise:.6f},1.2000\n")
    path.write_text("".join(lines), encoding="utf-8")


def _exact_picks(*, x0_m: float = 2.0, z0_m: float = 0.5,
                 v_m_ns: float = 0.1, trace_step_m: float = 0.1,
                 window_ns: float = 512.0, samples: int = 256,
                 traces: int = 41) -> list[VelocityPick]:
    """与 CSV 合成参数一致的理想拾取（t 无噪声，fit 应精确还原）。"""
    dt = window_ns / samples
    picks: list[VelocityPick] = []
    for tr in (0, 10, 20, 30, 40):
        tr = min(tr, traces - 1)
        x = tr * trace_step_m
        t_ns = 2.0 * np.sqrt((x - x0_m) ** 2 + z0_m ** 2) / v_m_ns
        picks.append(VelocityPick(
            trace_index=tr, sample_index=int(round(t_ns / dt)),
            x_m=x, t_ns=t_ns,
        ))
    return picks


def _create_project_with_line(backend: MyGPRBackend, tmp_path: Path, *,
                              name: str) -> str:
    project = backend.projects.create_project(
        tmp_path / name, name=name,
        coordinate_system="CGCS2000 / 3-degree GK zone 36",
    )
    csv_path = tmp_path / f"{name}.csv"
    _write_hyperbola_csv(csv_path)
    line = backend.projects.import_line_source(
        project.project_id, "L01", csv_path, name="Line01",
    )
    assert line.trace_count == 41 and line.sample_count == 256
    return project.project_id


def test_velocity_analysis_fits_exact_hyperbola_domain() -> None:
    """无数据依赖：精确双曲线拾取应还原 v=0.1, x0=2.0, z0=0.5。"""
    picks = _exact_picks()
    fit = fit_hyperbola(picks)
    assert isinstance(fit, HyperbolaFit)
    assert fit.v_m_ns == pytest.approx(0.1, abs=1e-6)
    assert fit.x0_m == pytest.approx(2.0, abs=1e-6)
    assert fit.z0_m == pytest.approx(0.5, abs=1e-6)
    assert fit.rmse_ns < 1e-6
    assert fit.pick_count == 5


def test_velocity_evidence_payload_and_digest() -> None:
    picks = _exact_picks()
    fit = fit_hyperbola(picks)
    evidence = build_velocity_evidence(line_id="L01", fit=fit, picks=picks,
                                       dataset_shape=(256, 41))
    assert evidence["schema"] == VELOCITY_ANALYSIS_EVIDENCE_SCHEMA
    assert evidence["line_id"] == "L01"
    assert evidence["data_shape"] == [256, 41]
    body = evidence["body"]
    assert set(body) == {
        "v_m_ns", "x0_m", "z0_m", "rmse_ns", "r_squared",
        "pick_count", "picks", "dielectric_constant",
    }
    assert body["dielectric_constant"] == pytest.approx((C_M_PER_NS / 0.1) ** 2)
    assert evidence["body_sha256"] == compute_velocity_body_digest(body)
    # 摘要可独立重算（消费者不信任载荷内嵌值）
    recomputed = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert evidence["body_sha256"] == recomputed
    # 篡改 body 任一字段 → 摘要变化
    tampered = json.loads(json.dumps(evidence))
    tampered["body"]["v_m_ns"] = 0.2
    assert compute_velocity_body_digest(tampered["body"]) != evidence["body_sha256"]


def test_velocity_analysis_api_npz_writeback(tmp_path: Path) -> None:
    """无头链路（默认 npz 存储）：job → 证据 → ε/深度轴写回 → 回读一致。"""
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id = _create_project_with_line(backend, tmp_path, name="vel-npz")
        before = backend.projects.get_dataset_info(project_id, "L01")
        assert before.dielectric_constant == pytest.approx(9.0)

        job_id = backend.submit_velocity_analysis(
            project_id, "L01",
            [{"trace_index": p.trace_index, "sample_index": p.sample_index}
             for p in _exact_picks()],
        )
        snapshot = backend.jobs.wait(job_id, timeout=60)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        result = snapshot.result
        assert isinstance(result, dict)
        evidence = result["evidence"]
        assert evidence["schema"] == VELOCITY_ANALYSIS_EVIDENCE_SCHEMA
        assert result["applied"] is True
        # 拾取经时间轴量化（dt=2ns）→ 拟合速度有界偏差；断言内部一致性 ε=(c/v)²
        v_fit = float(evidence["body"]["v_m_ns"])
        assert 0.09 < v_fit < 0.11
        assert result["dielectric_constant"] == pytest.approx((C_M_PER_NS / v_fit) ** 2)

        # 证据摘要一致性
        assert evidence["body_sha256"] == compute_velocity_body_digest(evidence["body"])

        # 持久层回读：证据 + ε + 深度轴
        stored = backend.projects.load_velocity_model(project_id, "L01")
        assert stored is not None
        assert stored["schema"] == VELOCITY_ANALYSIS_EVIDENCE_SCHEMA
        assert stored["body_sha256"] == compute_velocity_body_digest(stored["body"])
        assert stored["body"]["v_m_ns"] == pytest.approx(0.1, abs=0.01)
        assert stored["body"]["x0_m"] == pytest.approx(2.0, abs=0.05)
        assert stored["body"]["z0_m"] == pytest.approx(0.5, abs=0.1)
        assert stored["body"]["pick_count"] == 5

        after = backend.projects.get_dataset_info(project_id, "L01")
        # 深度轴与 ε 同源重算：z = t·c/(2√ε)
        dataset = backend.projects.read_dataset(project_id, "L01")
        depth = np.asarray(dataset.header_info["depth_axis_m"], dtype=np.float64)
        time_ns = np.asarray(dataset.header_info["time_axis_ns"], dtype=np.float64)
        expected = time_ns * C_M_PER_NS / (2.0 * np.sqrt((C_M_PER_NS / v_fit) ** 2))
        np.testing.assert_allclose(depth, expected, rtol=1e-5)

        # 二次加载（重新打开 store 读取 npz）证据不丢
        again = backend.projects.load_velocity_model(project_id, "L01")
        assert again is not None and again["body"] == stored["body"]
    finally:
        backend.shutdown()



    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id = _create_project_with_line(backend, tmp_path, name="vel-hybrid")
        # 默认 manifest 即 hybrid_hdf5_sqlite_v1；经 job 全链路验证 hybrid 路径
        job_id = backend.submit_velocity_analysis(
            project_id, "L01",
            [{"trace_index": p.trace_index, "sample_index": p.sample_index}
             for p in _exact_picks()],
        )
        snapshot = backend.jobs.wait(job_id, timeout=60)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        assert snapshot.result["applied"] is True
        stored = backend.projects.load_velocity_model(project_id, "L01")
        assert stored is not None
        assert stored["schema"] == VELOCITY_ANALYSIS_EVIDENCE_SCHEMA
        assert stored["body_sha256"] == compute_velocity_body_digest(stored["body"])
        v_fit = float(stored["body"]["v_m_ns"])
        assert 0.09 < v_fit < 0.11
        # h5 容器内 metadata_json 携带证据（hybrid 布局硬契约）
        import h5py
        repo = backend.projects._repository  # noqa: SLF001 - 测试装配钩子
        session = repo.open(tmp_path / "vel-hybrid", read_only=True, recover_stale_lock=False)
        try:
            assert session._store.storage.is_hybrid  # noqa: SLF001
            container = session._store.storage.line_container_path("L01")
            with h5py.File(container, "r") as handle:
                metadata = json.loads(handle["raw"].attrs["metadata_json"])
            assert "velocity_analysis" in metadata["metadata"]
        finally:
            session.close()
    finally:
        backend.shutdown()


class _VelocitySessionShim:
    """直接装配 VelocityPersistenceMixin（绕过 facade），锁语义与 LegacyFieldProjectSession 一致。"""

    def __init__(self, store: Any) -> None:
        import threading

        self._store = store
        self._lock = threading.RLock()

    save_velocity_model = VelocityPersistenceMixin.save_velocity_model
    load_velocity_model = VelocityPersistenceMixin.load_velocity_model



def test_velocity_persistence_legacy_npz_layout(tmp_path: Path) -> None:
    from core.project_storage_backend import LegacyProjectStorageBackend
    from tests.field_project_test_utils import create_test_project

    store = create_test_project(tmp_path / "legacy-project", line_ids=("L01",))
    store.storage = LegacyProjectStorageBackend(store.root, store.manifest)
    from core.gpr_data_model import GPRDataSet
    store.save_gpr_dataset("L01", GPRDataSet.synthetic("L01", rows=64, cols=32))
    session = _VelocitySessionShim(store)
    session.save_velocity_model("L01", 0.1)
    stored = session.load_velocity_model("L01")
    assert stored is not None
    assert stored["schema"] == VELOCITY_ANALYSIS_EVIDENCE_SCHEMA
    assert stored["body"]["v_m_ns"] == pytest.approx(0.1, abs=1e-9)
    assert stored["body"]["dielectric_constant"] == pytest.approx((C_M_PER_NS / 0.1) ** 2)
    # 深度轴与 ε 同源
    dataset = store.load_gpr_dataset("L01")
    depth = np.asarray(dataset.depth_axis_m, dtype=np.float64)
    time_ns = np.asarray(dataset.time_axis_ns, dtype=np.float64)
    epsilon = (C_M_PER_NS / 0.1) ** 2
    np.testing.assert_allclose(depth, time_ns * C_M_PER_NS / (2.0 * np.sqrt(epsilon)), rtol=1e-5)


def test_velocity_analysis_rejects_bad_picks(tmp_path: Path) -> None:
    """越界拾取与共线拾取必须显式报错（VelocityAnalysisError）。"""
    from mygpr.domain.velocity.errors import VelocityAnalysisError

    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id = _create_project_with_line(backend, tmp_path, name="vel-bad")
        # 越界 trace_index
        with pytest.raises(VelocityAnalysisError):
            backend.velocity.analyze(
                project_id, "L01",
                [{"trace_index": 0, "sample_index": 10},
                 {"trace_index": 9999, "sample_index": 20},
                 {"trace_index": 5, "sample_index": 30}],
            )
        # 共线（秩亏）拾取：三点同一直线 → 拟合非物理
        collinear = [
            VelocityPick(trace_index=0, sample_index=0, x_m=0.0, t_ns=100.0),
            VelocityPick(trace_index=1, sample_index=1, x_m=1.0, t_ns=100.0),
            VelocityPick(trace_index=2, sample_index=2, x_m=2.0, t_ns=100.0),
        ]
        with pytest.raises(VelocityAnalysisError):
            fit_hyperbola(collinear)
    finally:
        backend.shutdown()

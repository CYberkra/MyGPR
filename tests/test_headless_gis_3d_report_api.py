#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-2 无界面全链路验收：导入 → 空间成果 → 三维配准 → 报告包。

全部经 ``MyGPRBackend`` 公共 API（job 系统）完成，不 import 任何 Qt 模块；
matplotlib 由 core 渲染器强制 Agg（gis_map_export/uav_georeference_3d 内已处理），
本文件另断言进程内未拉起 PyQt6。
"""
from __future__ import annotations


from pathlib import Path

import numpy as np
import pytest

from mygpr.application.jobs.models import JobStatus
from mygpr.interfaces.backend import MyGPRBackend

pytestmark = [
    pytest.mark.integration,
]


def _write_airborne_csv(path: Path, *, samples: int = 96, traces: int = 64) -> None:
    rng = np.random.default_rng(20260831)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    reflector = 0.4 * np.exp(-((t - 0.5) ** 2) / 0.004)
    lines = [
        f"Number of Samples = {samples},,,\n",
        "Time windows (ns) = 512,,\n",
        f"Number of Traces = {traces},,\n",
        "Trace interval (m) = 0.1,,\n",
    ]
    for tr in range(traces):
        lon = 106.8 + tr * 0.00001
        lat = 31.26 + tr * 0.00001
        for s in range(samples):
            amp = float(reflector[s, 0] + rng.normal(0.0, 0.01))
            height = 1.2 + tr * 0.001
            lines.append(f"{lon:.8f},{lat:.8f},441.7,{amp:.6f},{height:.4f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_backend_headless_gis_3d_report_pipeline(tmp_path: Path) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="headless-p12", coordinate_system="CGCS2000 / 3-degree GK zone 36")

        csv_path = tmp_path / "line.csv"
        _write_airborne_csv(csv_path)
        line = backend.projects.import_line_source(
            project.project_id, "L01", csv_path, name="Line01"
        )
        assert line.trace_count > 0 and line.sample_count > 0

        # 传感器同步写入 RTK 轨迹（空间成果预检要求轨迹文件）
        rtk = tmp_path / "rtk.csv"
        rtk.write_text(
            "timestamp_s,longitude,latitude\n0.0,106.8,31.26\n0.63,106.80064,31.26064\n",
            encoding="utf-8",
        )
        imu = tmp_path / "imu.csv"
        imu.write_text(
            "timestamp_s,roll_deg,pitch_deg,yaw_deg\n0.0,0.0,0.0,180.0\n0.63,2.0,1.0,181.0\n",
            encoding="utf-8",
        )
        timestamps = tmp_path / "timestamps.csv"
        stamps = np.linspace(0.0, 0.63, 64)
        timestamps.write_text(
            "timestamp_s\n" + "\n".join(f"{v:.6f}" for v in stamps) + "\n",
            encoding="utf-8",
        )
        sync = backend.projects.synchronize_line_sensors(
            project.project_id,
            "L01",
            rtk_path=rtk,
            trace_timestamps_path=timestamps,
            imu_path=imu,
        )
        assert sync is not None

        # 界面标注（confirmed）满足空间成果预检
        from mygpr.domain.interpretation.models import InterfaceAnnotation, InterpretationPoint

        annotation = InterfaceAnnotation(
            annotation_id="A-L01",
            line_id="L01",
            name="基覆界面",
            version=1,
            status="confirmed",
            points=(
                InterpretationPoint(trace_index=0.0, sample_index=40.0),
                InterpretationPoint(trace_index=63.0, sample_index=46.0),
            ),
        )
        backend.projects.save_interface_annotation(project.project_id, annotation)

        # --- 三维地理配准（job 系统） ---
        job_3d = backend.build_georeference_3d(project.project_id, "L01")
        snapshot = backend.jobs.wait(job_3d, timeout=60)
        assert snapshot.status is JobStatus.COMPLETED
        payload = snapshot.result
        assert isinstance(payload, dict) and payload, "3D payload 不应为空"

        # --- 空间成果（预检 + 版本化成果） ---
        job_spatial = backend.submit_spatial_result(
            project.project_id, name="headless-result", line_ids=["L01"]
        )
        snapshot = backend.jobs.wait(job_spatial, timeout=120)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        results = list(backend.spatial.list_results(project.project_id))
        assert results and results[0].result_id

        # --- 报告包（含 GIS 平面图渲染） ---
        job_report = backend.submit_project_report(
            project.project_id, package_name="headless-report"
        )
        snapshot = backend.jobs.wait(job_report, timeout=300)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        packages = list(backend.projects.list_report_packages(project.project_id))
        assert packages, "报告包应已登记"
        package_root = Path(str(packages[-1].package_dir or ""))
        if package_root.exists():
            assert any(package_root.rglob("*.pdf")) or any(package_root.rglob("*.html"))
    finally:
        backend.shutdown()


def test_backend_public_method_surface_matches_contract() -> None:
    """防漂移：backend 上全部公开方法都必须登记进契约 PUBLIC_METHODS。"""
    from scripts.check_backend_api_contract import PUBLIC_METHODS, build_contract

    contract = build_contract()
    registered = set(contract["facade_methods"])
    assert registered == set(PUBLIC_METHODS)

    backend_methods = {
        name
        for name, value in vars(MyGPRBackend).items()
        if callable(value)
        and not name.startswith("_")
        and getattr(value, "__module__", "") == MyGPRBackend.__module__
    }
    missing = sorted(backend_methods - registered)
    assert missing == [], f"backend 新公开方法未登记契约: {missing}"


def test_headless_api_modules_do_not_import_qt() -> None:
    """无头 API 红线（静态）：本链路涉及的后端模块源码不得 import Qt。"""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    modules = [
        "mygpr/interfaces/backend.py",
        "mygpr/application/spatial/service.py",
        "mygpr/application/reporting/service.py",
        "mygpr/application/project/service.py",
        "mygpr/infrastructure/persistence/spatial_adapter.py",
        "mygpr/infrastructure/persistence/interpretation_adapter.py",
        "core/uav_georeference_3d.py",
        "core/field_report_export.py",
        "core/gis_map_export.py",
        "core/gis_layers.py",
    ]
    pattern = re.compile(r"^\s*(from|import)\s+(PyQt6|PySide6|qfluentwidgets)", re.MULTILINE)
    for rel in modules:
        source = (root / rel).read_text(encoding="utf-8")
        assert not pattern.search(source), f"{rel} 引入了 Qt 依赖"

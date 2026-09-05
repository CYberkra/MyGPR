"""Phase 3.1 界面深度切片 API 集成测试。

覆盖：
1. ``interface_depth_preview`` 预览 payload 契约（形状 / NaN 布局 / 深度范围）。
2. ``submit_grid_layer`` job 链：提交 → 等待完成 → ``list_gis_layers`` 回读。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mygpr.application.jobs.models import JobStatus
from mygpr.domain.grid.errors import GridAnalysisError
from mygpr.domain.interpretation.models import InterfaceAnnotation, InterpretationPoint
from mygpr.interfaces.backend import MyGPRBackend

from tests.test_grid_clustering import _create_grouped_project

pytestmark = [
    pytest.mark.integration,
]


def _seed_interface_annotations(backend: MyGPRBackend, project_id: str,
                                line_ids: list[str]) -> None:
    """每条测线 3 个界面点（trace 2/5/8，样本 20）→ 深度约 12.8 m。"""
    for line_id in line_ids:
        points = tuple(
            InterpretationPoint(trace_index=float(tr), sample_index=20.0)
            for tr in (2, 5, 8)
        )
        annotation = InterfaceAnnotation(
            annotation_id=f"B-{line_id}", line_id=line_id,
            name=f"{line_id} 基覆界面", version=1, status="confirmed",
            points=points,
        )
        backend.projects.save_interface_annotation(project_id, annotation)


def test_interface_depth_preview_payload_contract(tmp_path: Path) -> None:
    """预览 payload：形状、valid_count、深度范围与有限值一致。"""
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id, line_ids = _create_grouped_project(
            backend, tmp_path, "depth-preview")
        _seed_interface_annotations(backend, project_id, line_ids)

        payload = backend.interface_depth_preview(
            project_id, line_ids, cell_size_m=2.0)

        assert isinstance(payload, dict)
        assert payload["attribute"] == "interface_depth_m"
        assert payload["cell_size_m"] == 2.0
        matrix = payload["matrix"]
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (payload["nrows"], payload["ncols"])
        assert payload["valid_count"] >= 6  # 2 线 × 3 点至少各落 1 cell

        finite = matrix[np.isfinite(matrix)]
        assert finite.size == payload["valid_count"]
        assert math.isfinite(payload["x_origin_m"])
        assert math.isfinite(payload["y_origin_m"])
        # 深度范围与矩阵有限值一致
        assert payload["depth_min_m"] == pytest.approx(float(finite.min()))
        assert payload["depth_max_m"] == pytest.approx(float(finite.max()))
        # 合成数据 64 samples / 512 ns / ε=9 → 界面样点 20 →
        # depth = t·c/√ε/2 = 20·(512/63)·0.2998/3/2 ≈ 8.12 m（双程换算）
        assert payload["depth_min_m"] == pytest.approx(8.12, abs=0.2)
    finally:
        backend.shutdown()


def test_interface_depth_preview_without_annotations_raises(tmp_path: Path) -> None:
    """无界面标注 → 预览路径同样报 GridAnalysisError（与 job 路径一致）。"""
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id, line_ids = _create_grouped_project(
            backend, tmp_path, "depth-empty")
        with pytest.raises(GridAnalysisError):
            backend.interface_depth_preview(project_id, line_ids, cell_size_m=1.0)
    finally:
        backend.shutdown()


def test_submit_grid_layer_job_persists_layer(tmp_path: Path) -> None:
    """存为图层 job：submit → wait → list_gis_layers 回读新图层。"""
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id, line_ids = _create_grouped_project(
            backend, tmp_path, "depth-layer")
        _seed_interface_annotations(backend, project_id, line_ids)

        before = backend.projects.list_gis_layers(project_id)
        job_id = backend.submit_grid_layer(project_id, line_ids, cell_size_m=2.0)
        assert isinstance(job_id, str) and job_id
        snapshot = backend.jobs.wait(job_id, timeout=90)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message

        after = backend.projects.list_gis_layers(project_id)
        assert len(after) == len(before) + 1
        result = snapshot.result
        assert isinstance(result, dict)
        assert result["layer"]["name"] in {
            str(item.get("name")) for item in after}
    finally:
        backend.shutdown()

# -*- coding: utf-8 -*-
"""成果预览代数独立性回归测试。

处理链运行完成后的自动成果预览曾与测线预览共享 ``_preview_generation``：
``refresh_lines`` → 测线表重建重选 → ``line_selected`` → ``preview_line``
推进共享代数，把在飞的成果预览回包作废，处理页"处理结果"分段停留在
"暂无数据——请先在项目页导入测线"（2026-09-16 真机 computer-use 复现）。

修复后成果预览使用独立的 ``_artifact_preview_generation``（沿用深度切片
预览的既有先例：不同预览域互不失效），仅在切换测线/项目上下文时显式作废。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6")

from ui.controllers.project_controller import (  # noqa: E402
    ProjectController,
    _PreviewArtifactCommand,
)


class _FakeProjectsApi:
    """最小成果数据集 API：返回确定性小窗口。"""

    def get_artifact_dataset_info(self, project_id, line_id, artifact_id):
        return types.SimpleNamespace(
            time_window_ns=250.0, length_m=100.0, shape=(501, 1950))

    def read_artifact_window(self, project_id, line_id, artifact_id,
                             *, max_samples, max_traces):
        import numpy as np

        ns = list(range(min(int(max_samples), 501)))
        ts = list(range(min(int(max_traces), 1950)))
        return np.zeros((len(ns), len(ts)), dtype=np.float32), ns, ts


def _make_controller(qapp) -> ProjectController:
    pc = ProjectController()
    pc.set_backend(types.SimpleNamespace(
        backend=types.SimpleNamespace(projects=_FakeProjectsApi())))
    pc._current = types.SimpleNamespace(project_id="P-001")
    return pc


def test_artifact_preview_survives_concurrent_line_preview(qapp):
    """回归核心：在飞成果预览不得被并发测线预览的代数推进作废。"""
    pc = _make_controller(qapp)
    # 成果预览已发起（gen=1，对应运行完成后的自动预览）
    pc._artifact_preview_generation = 1
    # 并发场景：refresh_lines → 表格重选 → preview_line 推进测线预览代数。
    # 修复前两者共享同一计数器，此推进会把成果回包丢弃。
    pc._preview_generation += 3

    got: list[str] = []
    pc.artifact_preview_ready.connect(lambda aid, _b: got.append(aid))

    _PreviewArtifactCommand(pc, "P-001", "L01", "A1", 1).execute()

    assert got == ["A1"]


def test_artifact_preview_discarded_on_line_switch(qapp):
    """切换测线必须作废在飞成果预览，防旧测线成果串台渲染。"""
    pc = _make_controller(qapp)
    pc._artifact_preview_generation = 1

    got: list[str] = []
    pc.artifact_preview_ready.connect(lambda aid, _b: got.append(aid))

    pc.invalidate_artifact_previews()  # gen → 2
    _PreviewArtifactCommand(pc, "P-001", "L01", "A1", 1).execute()

    assert got == []


def test_stale_artifact_preview_still_discarded(qapp):
    """用户又点了别的成果（代数推进）时，旧回包仍按原语义丢弃。"""
    pc = _make_controller(qapp)
    pc._artifact_preview_generation = 5

    got: list[str] = []
    pc.artifact_preview_ready.connect(lambda aid, _b: got.append(aid))

    _PreviewArtifactCommand(pc, "P-001", "L01", "A1", 4).execute()
    assert got == []


def test_project_switch_invalidates_artifact_preview(qapp):
    """项目上下文切换入口推进成果预览代数（与测线/深度预览同语义）。"""
    pc = ProjectController()  # 无后端：open_project 推进代数后即返回
    before = pc._artifact_preview_generation
    pc.open_project(r"C:\nonexistent")
    assert pc._artifact_preview_generation == before + 1

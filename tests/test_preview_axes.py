# -*- coding: utf-8 -*-
"""预览物理轴透传测试（P1-5：read_window 不再丢弃真实时窗/坐标轴）。"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计

from ui.controllers.project_controller import ProjectController  # noqa: E402


class TestBundleFromWindow:
    def test_uses_real_time_window_and_axes(self, qapp):
        matrix = np.zeros((72, 48), dtype=np.float32)
        # read_window 默认 stride=1（72 ≤ max_samples=900），全量保留
        sample_idx = np.arange(0, 72, dtype=np.int64)
        trace_idx = np.arange(0, 48, dtype=np.int64)
        bundle = ProjectController._bundle_from_window(
            matrix, 'L01', title='t',
            time_window_ns=500.0, length_m=24.0,
            sample_indices=sample_idx, trace_indices=trace_idx,
            total_samples=72, total_traces=48,
        )
        assert bundle.sample_axis is not None
        assert bundle.sample_axis.size == 72
        assert float(bundle.sample_axis[-1]) == pytest.approx(500.0, abs=1.0)
        assert float(bundle.sample_axis[1]) == pytest.approx(
            500.0 / 71, abs=2.0)  # 第二个采样约 1/71 时窗
        assert bundle.trace_axis_m is not None
        assert float(bundle.trace_axis_m[-1]) == pytest.approx(24.0, abs=0.1)

    def test_defaults_to_250_when_no_axes(self, qapp):
        matrix = np.zeros((10, 5), dtype=np.float32)
        bundle = ProjectController._bundle_from_window(matrix, 'L01', title='t')
        assert bundle.sample_axis is not None
        assert float(bundle.sample_axis[-1]) == pytest.approx(250.0, abs=1.0)

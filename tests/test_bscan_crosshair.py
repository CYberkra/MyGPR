# -*- coding: utf-8 -*-
"""B-scan 十字光标读数测试（纯函数 + offscreen 控件）。"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui.widgets.bscan_view import format_crosshair_readout  # noqa: E402


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ------------------------------------------------------------------ 纯函数
class TestFormatReadout:
    def test_index_only(self):
        text = format_crosshair_readout(2, 4, (100, 50), 1.5)
        assert text == '道 3\n采样 5\n幅值 1.5'

    def test_physical_axes(self):
        text = format_crosshair_readout(
            1, 2, (10, 5), -0.25,
            trace_axis_m=np.array([0.0, 2.5, 5.0]),
            sample_axis=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
            sample_axis_label='时间 (ns)')
        lines = text.split('\n')
        assert lines[0] == '道 2'
        assert lines[1] == '距起点 2.5 m'
        assert lines[2] == '时间 (ns) 20'
        assert lines[3] == '幅值 -0.25'

    def test_downsampled_counts_annotated(self):
        text = format_crosshair_readout(
            50, 25, (101, 51), 0.0, trace_count=1001, sample_count=501)
        assert '道 51（原始约 501）' in text
        assert '采样 26（原始约 251）' in text

    def test_axis_length_mismatch_falls_back(self):
        # 物理轴与显示矩阵不等长时退回索引显示
        text = format_crosshair_readout(
            5, 5, (10, 10), 1.0, sample_axis=np.array([1.0, 2.0]))
        assert '采样 6' in text


# ------------------------------------------------------------------ 控件行为
class TestCrosshairWidget:
    def _make_view(self, qapp):
        from ui.widgets.bscan_view import BScanView
        view = BScanView()
        matrix = np.arange(50 * 20, dtype=np.float32).reshape(50, 20)
        bundle = SimpleNamespace(
            matrix=matrix, vmin=0.0, vmax=1000.0, title='t',
            x_label='道数', y_label='采样点',
            trace_count=20, sample_count=50,
            trace_axis_m=np.arange(20, dtype=float) * 0.5,
            sample_axis=np.arange(50, dtype=float) * 2.0,
            sample_axis_label='时间 (ns)')
        view.set_bundle(bundle)
        view.resize(400, 300)
        view.show()              # offscreen 下也需 show，否则 ViewBox 尺寸为 0 映射退化
        qapp.processEvents()
        return view

    def test_mouse_move_shows_readout(self, qapp):
        from PyQt6.QtCore import QPointF
        view = self._make_view(qapp)
        pos = view._plot.vb.mapViewToScene(QPointF(10.5, 25.5))
        view._on_mouse_moved(pos)
        assert view._vline.isVisible()
        assert view._hline.isVisible()
        assert not view._readout.isHidden()
        text = view._readout.text()
        assert '道 11' in text
        assert '距起点 5 m' in text
        assert '时间 (ns) 50' in text
        assert f'幅值 {float(25 * 20 + 10):.4g}' in text

    def test_toggle_off_hides(self, qapp):
        from PyQt6.QtCore import QPointF
        view = self._make_view(qapp)
        pos = view._plot.vb.mapViewToScene(QPointF(10.5, 25.5))
        view._on_mouse_moved(pos)
        view._toggle_crosshair(False)
        assert not view._vline.isVisible()
        assert view._readout.isHidden()
        view._on_mouse_moved(pos)   # 关闭后移动不再显示
        assert view._readout.isHidden()

    def test_outside_image_hides(self, qapp):
        from PyQt6.QtCore import QPointF
        view = self._make_view(qapp)
        view.resize(400, 300)
        view._on_mouse_moved(QPointF(-100.0, -100.0))
        assert not view._vline.isVisible()
        assert view._readout.isHidden()

# -*- coding: utf-8 -*-
"""B-scan 拾取坐标契约测试（降采样预览下 sig_point_picked 必须发原始坐标）。

契约（BScanView 类 docstring）：sig_point_picked(trace, sample) 统一为
**原始数据坐标**——大测线（>900×1800）预览经 strided 降采样，点击换算
必须经 _view_to_data 映射，否则拾取点系统性偏移（双曲线拟合的输入错位）。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pytest.importorskip("PyQt6")


class TestPickedCoordinatesAreOriginalGrid:
    def _downsampled_view(self, qapp):
        from ui.widgets.bscan_view import BScanView

        view = BScanView()
        # 900×900 预览代表 1800×1800 原始 B-scan（2:1 降采样）
        bundle = SimpleNamespace(
            matrix=np.zeros((900, 900), dtype=np.float32),
            vmin=-1.0, vmax=1.0, title='t',
            x_label='道数', y_label='采样点',
            trace_count=1800, sample_count=1800,
            trace_axis_m=None, sample_axis=None, sample_axis_label='')
        view.set_bundle(bundle)
        view.set_pick_enabled(True)
        view.resize(400, 300)
        view.show()
        qapp.processEvents()
        return view

    def _click_at_view(self, view, trace: int, sample: int):
        """在显示坐标 (trace, sample) 处模拟左键点击（走真实处理器）。"""
        from PyQt6.QtCore import QPointF, Qt

        class _FakeEvent:
            def __init__(self, scene_pos):
                self._pos = scene_pos

            def button(self):
                return Qt.MouseButton.LeftButton

            def scenePos(self):
                return self._pos

        pos = view._plot.vb.mapViewToScene(QPointF(float(trace), float(sample)))
        # 经场景坐标再映射回视图坐标，与真实鼠标事件一致（含 ±1e-9 浮点抖动）
        view._on_mouse_clicked(_FakeEvent(pos))
        back = view._plot.vb.mapSceneToView(pos)
        return int(back.x()), int(back.y())

    def test_clicked_signal_emits_original_coordinates(self, qapp):
        view = self._downsampled_view(qapp)
        received = []
        view.sig_point_picked.connect(lambda t, s: received.append((t, s)))

        # 契约：信号携带的原始坐标必须等于 _view_to_data(点击处显示坐标)。
        # 旧 bug：直接发射显示坐标，降采样时拾取点系统性偏移。
        clicked_view = self._click_at_view(view, 400, 300)
        expected = view._view_to_data(*clicked_view)

        assert received == [expected]
        # 且确实发生了换算：2:1 降采样下不能等于显示坐标本身
        assert expected[0] > clicked_view[0]

    def test_clicked_signal_identity_when_not_downsampled(self, qapp):
        from ui.widgets.bscan_view import BScanView

        view = BScanView()
        bundle = SimpleNamespace(
            matrix=np.zeros((50, 20), dtype=np.float32),
            vmin=-1.0, vmax=1.0, title='t',
            x_label='道数', y_label='采样点',
            trace_count=20, sample_count=50,
            trace_axis_m=None, sample_axis=None, sample_axis_label='')
        view.set_bundle(bundle)
        view.set_pick_enabled(True)
        view.resize(400, 300)
        view.show()
        qapp.processEvents()
        received = []
        view.sig_point_picked.connect(lambda t, s: received.append((t, s)))

        clicked_view = self._click_at_view(view, 7, 25)

        # 无降采样：恒等映射（容许 ±1 浮点抖动）
        assert len(received) == 1
        assert abs(received[0][0] - clicked_view[0]) <= 1
        assert abs(received[0][1] - clicked_view[1]) <= 1

    def test_popup_follow_receives_display_trace(self, qapp):
        """跟随浮窗消费显示坐标列（与当前预览一致），原信号发原始坐标。"""
        view = self._downsampled_view(qapp)
        view.set_ascan_follow(True)
        received = []
        view.sig_point_picked.connect(lambda t, s: received.append((t, s)))

        clicked_view = self._click_at_view(view, 400, 300)

        # 原信号：原始数据坐标
        assert received == [view._view_to_data(*clicked_view)]
        # 浮窗波形列取自 900 列显示矩阵（旧 bug 会用原始坐标 800 越界/取错列）
        curve_data = view._ascan_popup._ascan_view._curve.getData()[0]
        assert len(curve_data) == 900


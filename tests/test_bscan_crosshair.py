# -*- coding: utf-8 -*-
"""B-scan 十字光标读数测试（纯函数 + offscreen 控件）。"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计

from ui.widgets.bscan_view import format_crosshair_readout  # noqa: E402


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
    def _queue_readout(self, view, trace=18, sample=8):
        from PyQt6.QtCore import QPointF
        pos = view._plot.vb.mapViewToScene(QPointF(trace + .5, sample + .5))
        view._on_mouse_moved(pos)
        view._on_mouse_moved(pos)
        assert view._pending_readout is not None
        assert view._readout_timer.isActive()

    def _bundle(self, traces, value):
        return SimpleNamespace(
            matrix=np.full((10, traces), value, dtype=np.float32), vmin=0., vmax=100.,
            trace_count=traces, sample_count=10, trace_indices=np.arange(traces),
            sample_indices=np.arange(10), trace_elevation_m=np.full(traces, 100.),
            depth_axis_m=np.arange(10, dtype=float))

    def _assert_readout_cleared(self, view):
        assert view._pending_readout is None
        assert not view._readout_timer.isActive()
        assert not view._vline.isVisible()
        assert not view._hline.isVisible()
        assert view._readout.isHidden()
        view._flush_crosshair_readout()

    @pytest.mark.parametrize('traces', [3, 20])
    def test_pending_readout_is_discarded_when_bundle_changes(self, qapp, traces):
        view = self._make_view(qapp)
        view.set_bundle(self._bundle(20, 1.))
        qapp.processEvents()
        self._queue_readout(view)
        view.set_bundle(self._bundle(traces, 2.))
        self._assert_readout_cleared(view)

    def test_pending_readout_is_discarded_on_axis_grid_switch(self, qapp):
        view = self._make_view(qapp)
        view.set_bundle(self._bundle(20, 1.))
        qapp.processEvents()
        self._queue_readout(view)
        view.set_y_axis_mode('elevation')
        self._assert_readout_cleared(view)
        self._queue_readout(view)
        view.set_y_axis_mode('sample')
        self._assert_readout_cleared(view)

    def test_clear_discards_pending_readout(self, qapp):
        view = self._make_view(qapp)
        self._queue_readout(view)
        view.clear()
        self._assert_readout_cleared(view)

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

    def test_theme_styles_the_floating_fullscreen_button(self, qapp):
        from ui.widgets.bscan_view import BScanView
        from ui.theme_helpers import control_palette

        view = BScanView()
        view.apply_theme(True)
        # 工具条退役后，图内悬浮全屏钮是唯一常驻控件，仍吃调色板 QSS
        assert control_palette(True)['button_bg'] in view._fullscreen_btn.styleSheet()
        assert control_palette(True)['button_text'] in view._fullscreen_btn.styleSheet()

        view.apply_theme(False)
        assert control_palette(False)['button_bg'] in view._fullscreen_btn.styleSheet()
        assert control_palette(False)['button_text'] in view._fullscreen_btn.styleSheet()


class TestPreviewCoordinateMapping:
    def test_downsampled_preview_maps_between_view_and_source_coordinates(self, qapp):
        """Picks and overlays share the original, rather than preview, grid."""
        from ui.widgets.bscan_view import BScanView

        view = BScanView()
        # A 900 × 900 preview represents a 1,800 × 1,800 source B-scan.
        # Check both endpoints and an interior location.  A 2:1 downsample
        # cannot make every source coordinate exactly reversible, so validate
        # that the round-trip remains within one source sample.
        view._image_shape = (900, 900)
        view._trace_count = 1800
        view._sample_count = 1800

        assert view._view_to_data(0, 0) == (0, 0)
        assert view._view_to_data(899, 899) == (1799, 1799)
        assert view._view_to_data(450, 450) == (901, 901)

        for trace, sample in ((0, 0), (1799, 1799), (901, 901)):
            preview_trace, preview_sample = view._data_to_view(trace, sample)
            restored = view._view_to_data(round(preview_trace), round(preview_sample))
            assert abs(restored[0] - trace) <= 1
            assert abs(restored[1] - sample) <= 1

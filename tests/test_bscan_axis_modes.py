# -*- coding: utf-8 -*-
"""B-Scan 轴单位（Phase1 能力包）：横轴 道↔距离、纵轴 采样轴↔海拔。

设计要点（改动即回归，本文件逐条锁定）：

1. **两种纵轴 = 两种显示网格**。采样轴/距离模式下显示坐标是索引（x=列、
   y=采样行），物理量只在 ``IndexAxis.tickStrings`` 里换算；海拔模式是
   **整幅剖面重采样到共享绝对海拔网格**（见 test_bscan_elevation_warp.py），
   显示 y 变成「海拔网格行号」。pick/overlay/降采样换算在两种模式下分别
   由 BScanView 的坐标映射兜底（test_bscan_pick_coordinates.py）。
2. **海拔 = 逐道地面高程 − 该道下方深度**，需要两份原料同时齐备：
   ``trace_elevation_m``（数据层已交付）与 ``depth_axis_m``。任一缺失 →
   海拔切不过去（右键菜单/设置页该项不可用）。
3. 切换偏好与数据能力解耦：数据不支持时切不过去，但偏好记住，
   换到支持的数据后自动生效。
4. pyqtgraph 0.14 的空刻度崩溃（轴条 2px 时 ``min(map(min, tickPositions))``
   抛 ValueError）已由 ``IndexAxis.tickValues`` 过滤空刻度级兜住——旧版
   BScanView 同样会抛，属既有问题，本文件守住不再复发。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.widgets.bscan_axes import (  # noqa: E402
    IndexAxis,
    distance_available,
    distance_tick_strings,
    elevation_available,
    elevation_tick_strings,
    sample_unit_label,
    tick_index,
)
from ui.desktop_backend_facade import compute_display_levels  # noqa: E402
from ui.widgets.bscan_view import (  # noqa: E402
    DEFAULT_P_HIGH,
    DEFAULT_P_LOW,
    BScanView,
    format_crosshair_readout,
)

_TRACES, _SAMPLES = 120, 200
_EPS = 9.0
_DEPTH = np.linspace(0.0, 250.0, _SAMPLES) * 0.299792458 / (2.0 * np.sqrt(_EPS))
_DISTANCE = np.linspace(0.0, 59.5, _TRACES)
_ELEVATION = 442.0 + 8.0 * np.sin(np.linspace(0.0, 3.1, _TRACES))


def make_view(**kwargs) -> BScanView:
    return BScanView(**kwargs)


@pytest.fixture
def view(qapp):
    widget = make_view()
    yield widget
    widget.close()


def feed(view: BScanView, *, elevation=True, distance=True, depth=True) -> None:
    """塞一份最小可用数据；三个开关分别控制三类物理轴是否齐备。"""
    from core.gui_rendering import make_preview_bundle

    matrix = np.random.RandomState(0).rand(_SAMPLES, _TRACES).astype(np.float32)
    kwargs = {'sample_axis': np.linspace(0.0, 250.0, _SAMPLES),
              'sample_axis_label': '时间 (ns)'}
    if distance:
        kwargs['trace_axis_m'] = _DISTANCE
    if depth:
        kwargs['depth_axis_m'] = _DEPTH
    if elevation:
        kwargs['trace_elevation_m'] = _ELEVATION
    view.set_bundle(make_preview_bundle(matrix, **kwargs))


# ============================================================ 纯函数
class TestTickIndex:
    """轴刻度值 → 数据索引：floor + 夹取（像素边界处的刻度不算错行）。"""

    @pytest.mark.parametrize('value,expected', [
        (0.0, 0), (0.99, 0), (1.0, 1), (5.5, 5), (-3.0, 0), (1e9, 9),
    ])
    def test_floor_and_clamp(self, value, expected):
        assert tick_index(value, 10) == expected

    def test_empty_axis(self):
        assert tick_index(3.7, 0) == 0


class TestUnitLabel:
    def test_unit_label_from_bundle_label(self):
        assert sample_unit_label('时间 (ns)') == '时间'
        assert sample_unit_label('深度 (m)') == '深度'
        assert sample_unit_label('') == '采样点'


class TestAvailability:
    def test_distance_requires_full_length(self):
        assert distance_available(np.zeros(10), 10)
        assert not distance_available(np.zeros(9), 10)
        assert not distance_available(None, 10)
        assert not distance_available(np.zeros(10), 0)

    def test_elevation_requires_both_axes(self):
        ground = np.zeros(10)
        depth = np.zeros(5)
        assert elevation_available(ground, depth, 10, 5)
        assert not elevation_available(None, depth, 10, 5)
        assert not elevation_available(ground, None, 10, 5)
        assert not elevation_available(np.zeros(4), depth, 10, 5)
        assert not elevation_available(ground, np.zeros(3), 10, 5)


class TestTickStrings:
    def test_distance_maps_to_mileage(self):
        texts = distance_tick_strings([0.0, 5.0], _DISTANCE)
        assert float(texts[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(texts[1]) == pytest.approx(float(_DISTANCE[5]), abs=0.05)

    def test_distance_none_without_axis(self):
        assert distance_tick_strings([0.0], None) is None
        assert distance_tick_strings([0.0], np.empty(0)) is None

    def test_elevation_reads_shared_axis(self):
        """海拔刻度 = 行号直查共享海拔轴（降序：行号增大海拔变小）。"""
        axis = np.array([500.0, 499.0, 498.0, 497.0])
        texts = elevation_tick_strings([0.0, 3.0], axis)
        assert float(texts[0]) == pytest.approx(500.0)
        assert float(texts[1]) == pytest.approx(497.0)

    def test_elevation_none_when_axis_missing(self):
        assert elevation_tick_strings([0.0], None) is None
        assert elevation_tick_strings([0.0], np.empty(0)) is None


# ============================================================ 视图层
class TestAxisModeAvailability:
    def test_defaults_are_sample_and_trace(self, view):
        assert view.x_axis_mode() == 'trace'
        assert view.y_axis_mode() == 'sample'

    def test_illegal_defaults_fall_back(self, qapp):
        widget = make_view(default_x_axis='bogus', default_y_axis='bogus')
        assert widget.x_axis_mode() == 'trace'
        assert widget.y_axis_mode() == 'sample'
        widget.close()

    def test_elevation_disabled_without_data(self, view):
        feed(view, elevation=False)
        assert not view.elevation_available()
        view.set_y_axis_mode('elevation', notify=True)
        assert view.effective_y_axis_mode() == 'sample'   # 切不过去

    def test_elevation_switch_rejected_without_data(self, view):
        feed(view, elevation=False)
        view.set_y_axis_mode('elevation', notify=True)
        assert view.y_axis_mode() == 'sample'

    def test_elevation_requires_depth_not_just_ground(self, view):
        """只有地面高程、没有米制深度时，海拔依然不可用（不能拿 ns 当米）。"""
        feed(view, depth=False)
        assert not view.elevation_available()

    def test_elevation_enabled_with_full_data(self, view):
        feed(view)
        assert view.elevation_available()

    def test_distance_disabled_without_mileage(self, view):
        feed(view, distance=False)
        assert not view.distance_available()
        view.set_x_axis_mode('distance', notify=True)
        assert view.x_axis_mode() == 'trace'

    def test_distance_switch_rejected_without_mileage(self, view):
        feed(view, distance=False)
        view.set_x_axis_mode('distance', notify=True)
        assert view.x_axis_mode() == 'trace'

    def test_axis_buttons_disabled_without_data(self, view):
        """空视图（未 set_bundle）不应让用户切到任何物理单位。"""
        assert not view.distance_available()
        assert not view.elevation_available()


class TestAxisModeSwitch:
    def test_distance_switch_applies(self, view):
        feed(view)
        view.set_x_axis_mode('distance', notify=True)
        assert view.x_axis_mode() == 'distance'
        assert view._plot.getAxis('bottom').labelText == '距离 (m)'

    def test_switch_back_to_trace_restores_label(self, view):
        feed(view)
        view.set_x_axis_mode('distance')
        view.set_x_axis_mode('trace')
        assert view._plot.getAxis('bottom').labelText == '道数'

    def test_elevation_switch_applies_label(self, view):
        feed(view)
        view.set_y_axis_mode('elevation')
        assert view.effective_y_axis_mode() == 'elevation'
        assert view._plot.getAxis('left').labelText == '海拔 (m)'

    def test_sample_label_follows_bundle_unit(self, view):
        feed(view)
        assert view._plot.getAxis('left').labelText == '时间 (ns)'

    def test_illegal_mode_is_noop(self, view):
        feed(view)
        view.set_x_axis_mode('bogus')
        view.set_y_axis_mode('bogus')
        assert (view.x_axis_mode(), view.y_axis_mode()) == ('trace', 'sample')

    def test_tick_conversion_actually_happens(self, view):
        """必须验证刻度文本真的换算了，而不是只换了轴标签。"""
        feed(view)
        view.set_x_axis_mode('distance')
        view.set_y_axis_mode('elevation')
        bottom = view._convert_bottom_ticks([0.0, 60.0], 1.0, 60.0)
        left = view._convert_left_ticks([0.0, 100.0], 1.0, 100.0)
        assert bottom is not None and left is not None
        assert float(bottom[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(bottom[-1]) == pytest.approx(30.0, abs=1.0)
        assert float(left[0]) > float(left[1])   # 越深海拔越低

    def test_converters_fall_back_when_disabled(self, view):
        feed(view)
        assert view._convert_bottom_ticks([0.0], 1.0, 1.0) is None
        assert view._convert_left_ticks([0.0], 1.0, 1.0) is None


class TestAxisModePreferencePersistence:
    """偏好与数据能力解耦：数据不支持时置灰，但偏好记住，换数据后自动生效。"""

    def test_preference_survives_unsupported_data(self, view):
        view.set_y_axis_mode('elevation')      # 空视图：不被接受
        assert view.y_axis_mode() == 'sample'  # 没数据时连偏好都不该改

    def test_preference_restored_when_data_supports_it(self, view):
        feed(view)
        view.set_x_axis_mode('distance')
        view.set_y_axis_mode('elevation')
        feed(view)   # 新数据同样支持
        assert view.x_axis_mode() == 'distance'
        assert view.y_axis_mode() == 'elevation'
        assert view._plot.getAxis('bottom').labelText == '距离 (m)'

    def test_switch_to_unsupported_data_falls_back_visibly(self, view):
        """用户偏好海拔，但新测线没有高程 → 显示层回落采样轴（不能画错刻度）。"""
        feed(view)
        view.set_y_axis_mode('elevation')
        feed(view, elevation=False)
        assert view.y_axis_mode() == 'elevation'          # 偏好仍在
        assert view.effective_y_axis_mode() == 'sample'   # 但实际回落
        assert view._convert_left_ticks([0.0, 10.0], 1.0, 10.0) is None


class TestAxisModeSignal:
    def test_user_switch_emits(self, view):
        feed(view)
        seen_x: list[str] = []
        seen_y: list[str] = []
        view.sig_x_axis_changed.connect(seen_x.append)
        view.sig_y_axis_changed.connect(seen_y.append)
        view.set_x_axis_mode('distance', notify=True)
        view.set_y_axis_mode('elevation', notify=True)
        assert seen_x == ['distance']
        assert seen_y == ['elevation']

    def test_data_driven_apply_does_not_emit(self, view):
        feed(view)
        seen: list[str] = []
        view.sig_x_axis_changed.connect(seen.append)
        view.sig_y_axis_changed.connect(seen.append)
        feed(view)
        assert seen == []

    def test_restore_does_not_emit(self, view):
        feed(view)
        seen: list[str] = []
        view.sig_x_axis_changed.connect(seen.append)
        view.sig_y_axis_changed.connect(seen.append)
        view.set_axis_modes('distance', 'elevation', notify=False)
        assert seen == []
        assert (view.x_axis_mode(), view.y_axis_mode()) == ('distance', 'elevation')


class TestCrosshairElevation:
    def test_readout_shows_elevation_via_axis(self):
        """海拔读数 = 行号直查海拔轴（调用方传 elev_axis + 海拔标签）。"""
        axis = np.linspace(500.0, 450.0, 51)   # 降序海拔轴
        text = format_crosshair_readout(
            0, 10, (_TRACES, len(axis)), 1.0,
            sample_axis=axis, sample_axis_label='海拔 (m)')
        assert '海拔 (m)' in text
        assert f'{float(axis[10]):.4g}' in text

    def test_readout_without_elevation_args_unchanged(self):
        """采样轴模式下读数保持原样（不出现海拔字样）。"""
        text = format_crosshair_readout(
            0, 10, (_TRACES, _SAMPLES), 1.0,
            sample_axis=np.linspace(0.0, 250.0, _SAMPLES),
            sample_axis_label='时间 (ns)')
        assert '海拔' not in text
        assert '时间' in text or '纵轴' in text

    def test_readout_nan_amplitude_shows_dash(self):
        """海拔模式地表以上是 NaN（留白），幅值显示占位符而不是 nan。"""
        text = format_crosshair_readout(0, 0, (_TRACES, 10), float('nan'))
        assert '幅值 —' in text
        assert 'nan' not in text

    def test_view_switches_readout_axis_by_mode(self, view):
        feed(view)
        view._image_shape = (_TRACES, _SAMPLES)
        view._pending_readout = (0, 10, 1.0)
        view._flush_crosshair_readout()
        assert '海拔' not in view._readout.text()
        view.set_y_axis_mode('elevation')
        view._pending_readout = (0, 10, 1.0)
        view._flush_crosshair_readout()
        assert '海拔' in view._readout.text()


class TestAutoScaleButtonHidden:
    """pyqtgraph 的「A」钮（英文语境 autoscale）由工具条/菜单取代。"""

    def test_hidden(self, view):
        assert view._plot.buttonsHidden is True


class TestLevelsPreferenceSurvivesNoData:
    """色阶偏好与渲染分离：无数据时必须「记下偏好」，有数据后自动生效。

    ``set_display_levels`` 在无矩阵时算不出 vmin/vmax，但百分位本身是纯状态。
    跨会话恢复恰好发生在主窗构造期（此时还没有数据），若整体放弃，用户存的
    5/95 会被静默丢掉，直到他手动再设一次。
    """

    def test_preference_remembered_without_data(self, view):
        assert view.display_levels() == (DEFAULT_P_LOW, DEFAULT_P_HIGH)
        applied = view.set_display_levels(5.0, 95.0, notify=True)
        assert applied is False, '无数据时不该声称已重算渲染'
        assert view.display_levels() == (5.0, 95.0), '但偏好必须记下'

    def test_preference_applied_when_data_arrives(self, view):
        import numpy as np

        view.set_display_levels(5.0, 95.0)
        matrix = np.random.default_rng(0).random((60, 40)).astype('float32')
        view.set_matrix(matrix, 0.0, 1.0)
        vmin, vmax = view._image_item.levels
        expected = compute_display_levels(matrix, p_low=5.0, p_high=95.0)
        assert (float(vmin), float(vmax)) == pytest.approx(tuple(expected))

    def test_illegal_levels_rejected_and_keep_previous(self, view):
        view.set_display_levels(5.0, 95.0)
        assert view.set_display_levels(50.0, 50.0) is False
        assert view.display_levels() == (5.0, 95.0), '非法值不得污染现有偏好'

    def test_notify_still_emitted_without_data(self, view):
        """偏好变了就是变了，宿主该写盘——与能否渲染无关。"""
        seen = []
        view.sig_levels_changed.connect(lambda low, high: seen.append((low, high)))
        view.set_display_levels(4.0, 96.0, notify=True)
        assert seen == [(4.0, 96.0)]


class TestPyqtgraphEmptyTickGuard:
    """pyqtgraph 0.14 在轴条 2px 时抛空刻度异常；IndexAxis 必须兜住。

    旧版 BScanView（无 IndexAxis）实测同样抛 1 次，属既有缺陷；
    这里锁住修复，防止后续有人「简化」掉 tickValues 覆写。
    """

    def test_tick_values_drops_empty_levels(self, qapp):
        axis = IndexAxis('left', lambda v, s, sp: None)
        axis.setRange(204.0, -4.0)
        levels = axis.tickValues(204.0, -4.0, 2.0)   # 2px：父类会给出空级
        assert levels, '空 level 必须被过滤，否则 generateDrawSpecs 会抛 ValueError'
        assert all(values for _spacing, values in levels)

    def test_no_paint_exception_during_show(self, qapp):
        """真实 show + 全屏往返一圈，不该有任何未捕获异常。"""
        import sys
        import traceback

        from PyQt6.QtWidgets import QVBoxLayout, QWidget

        seen: list[str] = []
        original = sys.excepthook
        sys.excepthook = lambda t, e, tb: seen.append(
            ''.join(traceback.format_exception(t, e, tb)))
        try:
            widget = make_view()
            feed(widget)
            host = QWidget()
            layout = QVBoxLayout(host)
            layout.addWidget(widget, 1)
            host.show()
            for _ in range(10):
                qapp.processEvents()
            widget.enter_fullscreen()
            for _ in range(10):
                qapp.processEvents()
            widget.exit_fullscreen()
            for _ in range(10):
                qapp.processEvents()
            host.close()
            widget.close()
        finally:
            sys.excepthook = original
        assert [text for text in seen if 'min() iterable' in text] == []
        assert seen == []

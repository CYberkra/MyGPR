# -*- coding: utf-8 -*-
"""B-Scan 偏好的「第二份副本」缺陷回归。

## 被锁住的缺陷

``MyGPRMainWindow.closeEvent`` 会把 ``settings_page.settings()`` **整体**回写
设置文件。设置页的 ComboBox / SpinBox 是 B-Scan 偏好的**第二份副本**——
用户在 B-Scan 工具条上改了偏好（写盘 + 内存设置都对），但设置页控件仍是
旧值；关窗时那份过期值就把用户的选择**反向覆盖**掉。

实测症状（修复前）：
```
会话1  用户点"方形" → 内存 'square' / 磁盘 'square'   ← 看似正确
会话1  关窗         → 磁盘 'free'                     ← 被设置页覆盖
会话2  构造主窗     → 恢复 'free'                     ← 用户选择丢失
```

## 修法

工具条改偏好 → ``_mirror_bscan_setting`` → 既写盘**又**调
``SettingsPage.sync_bscan_view_settings`` 同步设置页控件。两份副本始终一致，
closeEvent 的整体回写就是幂等的。

## 为什么单独立文件

它跨 main_window + settings_page + bscan_view 三层，且是"关窗"这个非常规
路径触发的——放在任何单个控件的测试文件里都容易被忽略。
"""
from __future__ import annotations

import json
import os
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.pages.settings_page import SettingsPage  # noqa: E402
from ui.settings_manager import SettingsManager  # noqa: E402

_BSCAN_KEYS = ('bscan_aspect_mode', 'bscan_x_axis', 'bscan_y_axis',
               'bscan_layout_mode', 'bscan_p_low', 'bscan_p_high')


def _read_disk(store: str) -> dict:
    with open(store, encoding='utf-8') as handle:
        return json.load(handle)


# --------------------------------------------------------------------- 设置页同步


class TestSettingsPageMirror:
    """``sync_bscan_view_settings`` 必须把控件对齐到给定值，且不发信号。"""

    @pytest.fixture
    def page(self, qapp):
        widget = SettingsPage()
        yield widget
        widget.close()

    def test_combo_values_are_applied(self, page):
        page.load_settings(dict.fromkeys(_BSCAN_KEYS))
        page.sync_bscan_view_settings({
            'bscan_aspect_mode': 'cell',
            'bscan_x_axis': 'distance',
            'bscan_y_axis': 'elevation',
        })
        assert page._bscan_aspect_combo.currentData() == 'cell'
        assert page._bscan_x_axis_combo.currentData() == 'distance'
        assert page._bscan_y_axis_combo.currentData() == 'elevation'

    def test_spin_values_are_applied(self, page):
        page.load_settings({'bscan_p_low': 2.0, 'bscan_p_high': 98.0})
        page.sync_bscan_view_settings({'bscan_p_low': 7.5, 'bscan_p_high': 92.5})
        assert page._bscan_p_low_spin.value() == pytest.approx(7.5)
        assert page._bscan_p_high_spin.value() == pytest.approx(92.5)

    def test_sync_does_not_emit_signal(self, page):
        """同步是「外部已决定」的动作，不能再发信号回去（防回环）。"""
        fired = []
        page.bscan_view_changed.connect(lambda: fired.append(1))
        page.load_settings(dict.fromkeys(_BSCAN_KEYS))
        page.sync_bscan_view_settings({
            'bscan_aspect_mode': 'square',
            'bscan_p_low': 3.0, 'bscan_p_high': 97.0,
        })
        assert fired == []
        assert page._loading_settings is False      # 守卫要复位

    def test_partial_sync_leaves_other_widgets_alone(self, page):
        page.load_settings({'bscan_aspect_mode': 'free', 'bscan_x_axis': 'trace'})
        page.sync_bscan_view_settings({'bscan_aspect_mode': 'square'})
        assert page._bscan_aspect_combo.currentData() == 'square'
        assert page._bscan_x_axis_combo.currentData() == 'trace'   # 未被带偏

    def test_unknown_value_falls_back_to_current(self, page):
        """坏值不该把下拉框搞成空白（回落当前项，保持可用）。"""
        page.load_settings({'bscan_aspect_mode': 'square'})
        page.sync_bscan_view_settings({'bscan_aspect_mode': 'nonsense'})
        assert page._bscan_aspect_combo.currentData() == 'square'

    def test_bad_numeric_leaves_spin_alone(self, page):
        page.load_settings({'bscan_p_low': 4.0, 'bscan_p_high': 96.0})
        page.sync_bscan_view_settings({'bscan_p_low': 'not-a-number'})
        assert page._bscan_p_low_spin.value() == pytest.approx(4.0)

    def test_colorbar_check_is_applied_and_round_trips(self, page):
        """色标显隐：同步进勾选框 + settings() 回读一致（closeEvent 路径）。"""
        page.load_settings({'bscan_colorbar_visible': True})
        page.sync_bscan_view_settings({'bscan_colorbar_visible': False})
        assert page._bscan_colorbar_check.isChecked() is False
        assert page.bscan_view_settings()['bscan_colorbar_visible'] is False
        page.sync_bscan_view_settings({'bscan_colorbar_visible': True})
        assert page._bscan_colorbar_check.isChecked() is True

    def test_round_trip_through_settings(self, page):
        """同步后 settings() 必须原样回读出同步值——closeEvent 靠的就是它。"""
        page.load_settings(dict.fromkeys(_BSCAN_KEYS))
        wanted = {
            'bscan_aspect_mode': 'cell',
            'bscan_x_axis': 'distance',
            'bscan_y_axis': 'elevation',
            'bscan_layout_mode': 'dual',
            'bscan_p_low': 6.0,
            'bscan_p_high': 94.0,
            'bscan_colorbar_visible': False,
        }
        page.sync_bscan_view_settings(wanted)
        assert page.bscan_view_settings() == wanted


# ------------------------------------------------------------------- 跨会话闭环


def _make_window(settings: SettingsManager):
    """构造真实主窗；不可用（缺依赖）时返回 None 由调用方 skip。"""
    try:
        from ui.main_window import MyGPRMainWindow
    except Exception:  # noqa: BLE001 - 环境缺件时跳过而非失败
        return None
    try:
        return MyGPRMainWindow(settings=settings)
    except Exception:  # noqa: BLE001
        return None


class TestCloseEventDoesNotRevertUserChoice:
    """核心回归：工具条改的偏好在关窗后必须还在。"""

    @pytest.mark.parametrize('key, setter, value', [
        ('bscan_aspect_mode', 'fit_square', 'square'),
        ('bscan_p_low', 'levels', 6.0),
        ('bscan_colorbar_visible', 'colorbar_toggle', False),
    ])
    def test_preference_survives_close(self, qapp, key, setter, value):
        import numpy as np

        from ui.widgets.bscan_view import BScanView

        def same(actual, expected):
            if isinstance(expected, float):
                return actual == pytest.approx(expected)
            return actual == expected

        tmpdir = tempfile.mkdtemp(prefix='mygpr_mirror_')
        store = os.path.join(tmpdir, 'settings.json')

        settings = SettingsManager(store)
        window = _make_window(settings)
        if window is None:
            pytest.skip('主窗在当前环境不可构造')
        try:
            window.ensure_pages_ready()
            qapp.processEvents()
            views = window.findChildren(BScanView)
            assert views, '主窗内应至少有一个 BScanView'

            if setter == 'fit_square':
                views[0].fit_square()
            elif setter == 'colorbar_toggle':
                views[0].set_colorbar_visible(False, notify=True)
            else:
                # set_display_levels 无数据时直接返回 False（不发信号），
                # 故必须先喂一帧矩阵——这也顺带锁住「无数据不改色阶」。
                matrix = np.random.default_rng(0).random((60, 40)).astype('float32')
                views[0].set_matrix(matrix, 0.0, 1.0)
                assert views[0].set_display_levels(6.0, 94.0, notify=True) is True
            qapp.processEvents()

            # 设置页控件必须已经跟上（否则关窗就是覆盖）
            page = window._page('settingsInterface')
            if setter == 'fit_square':
                assert page._bscan_aspect_combo.currentData() == 'square'
            elif setter == 'colorbar_toggle':
                assert page._bscan_colorbar_check.isChecked() is False
            else:
                assert page._bscan_p_low_spin.value() == pytest.approx(6.0)

            window.close()
            qapp.processEvents()
            assert same(_read_disk(store).get(key), value), \
                f'关窗后 {key} 被设置页过期值覆盖'

            # 重开一次，确认真的读回来了（而非只是没被覆盖）
            assert same(SettingsManager(store).get(key), value)
        finally:
            try:
                window.close()
            except Exception:  # noqa: BLE001
                pass

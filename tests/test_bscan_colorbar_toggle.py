# -*- coding: utf-8 -*-
"""B-Scan 色标显隐：右键开关 + 设置持久化（2026-09-23 用户指令）。

语义要点（探针 ``_probe_colorbar_toggle.py`` 实证）：

- ColorBarItem 插在 plot 右列，纯 ``setVisible`` 即可——隐藏后 72px 宽度
  完整归还画布（vb.width 644→716），再显示恢复，免动 GraphicsLayout；
- 偏好与色标对象解耦：``with_colorbar=False`` 的视图没有色标，设置只记
  偏好不动渲染；
- 跟随 aspect/axis 既有语义：右键改单视图不广播兄弟视图（``notify=True``
  由宿主写盘 + 镜像设置页），设置页改动才全量下发。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.settings_manager import DEFAULT_SETTINGS  # noqa: E402


# ------------------------------------------------------------------ 视图行为
class TestColorbarToggleOnView:
    """视图级：显隐切换 / 空间归还 / 信号 / 无色标视图只记偏好。"""

    def _make_view(self, qapp, **kwargs):
        from ui.widgets.bscan_view import BScanView
        view = BScanView(**kwargs)
        matrix = np.arange(60 * 40, dtype=np.float32).reshape(60, 40)
        view.set_matrix(matrix, 0.0, 1.0)
        view.resize(800, 600)
        view.show()
        qapp.processEvents()
        qapp.processEvents()      # 两轮：等 GraphicsLayout 完成二次排布
        return view

    def test_default_visible(self, qapp):
        view = self._make_view(qapp)
        try:
            assert view._colorbar is not None
            assert view.colorbar_visible() is True
            # ColorBarItem 是 QGraphicsItem（非 QWidget），用 isVisible()
            assert view._colorbar.isVisible() is True
        finally:
            view.close()

    def test_hide_returns_space_to_canvas(self, qapp):
        """探针实证的核心收益：隐藏色标后其宽度完整归还画布。"""
        view = self._make_view(qapp)
        try:
            width_before = view._plot.vb.width()
            view.set_colorbar_visible(False)
            qapp.processEvents()
            qapp.processEvents()
            assert view._colorbar.isVisible() is False
            assert view._plot.vb.width() >= width_before + 30, \
                '隐藏色标后画布应显著加宽'
            view.set_colorbar_visible(True)
            qapp.processEvents()
            qapp.processEvents()
            assert abs(view._plot.vb.width() - width_before) <= 4, \
                '恢复显示后画布应回到原宽'
        finally:
            view.close()

    def test_notify_emits_signal_once(self, qapp):
        view = self._make_view(qapp)
        try:
            got = []
            view.sig_colorbar_visible_changed.connect(
                lambda visible: got.append(bool(visible)))
            view.set_colorbar_visible(False, notify=True)
            assert got == [False]
            view.set_colorbar_visible(True)          # notify=False 不发
            assert got == [False]
        finally:
            view.close()

    def test_view_without_colorbar_records_preference_only(self, qapp):
        """with_colorbar=False 的视图无色标对象，设置只记偏好不炸。"""
        view = self._make_view(qapp, with_colorbar=False)
        try:
            assert view._colorbar is None
            view.set_colorbar_visible(False, notify=True)
            assert view.colorbar_visible() is False
            assert view._colorbar is None
        finally:
            view.close()


# ------------------------------------------------------------------ 右键入口
class TestColorbarMenuEntry:
    """右键「交互开关组」必须含 checkable「显示色标」，状态与偏好一致。"""

    def _menu_actions(self, **kwargs):
        from qfluentwidgets import RoundMenu
        from ui.widgets.bscan_view import BScanView
        view = BScanView(**kwargs)
        menu = RoundMenu(parent=view)
        view._add_menu_toggles(menu)
        mapping = {a.text(): a for a in menu.actions()}
        menu.close()
        view.close()
        return mapping

    def test_menu_action_reflects_state(self, qapp):
        actions = self._menu_actions()
        action = actions['显示色标']
        assert action.isCheckable()
        assert action.isChecked() is True

    def test_rebuilt_menu_tracks_changed_preference(self, qapp):
        """菜单每次右键都重建，重建后应反映最新偏好。"""
        from qfluentwidgets import RoundMenu
        from ui.widgets.bscan_view import BScanView
        view = BScanView()
        view.set_colorbar_visible(False)
        menu = RoundMenu(parent=view)
        view._add_menu_toggles(menu)
        action = {a.text(): a for a in menu.actions()}['显示色标']
        assert action.isChecked() is False
        menu.close()
        view.close()

    def test_menu_action_without_colorbar_is_absent(self, qapp):
        actions = self._menu_actions(with_colorbar=False)
        assert '显示色标' not in actions


# ------------------------------------------------------------------ 设置默认
def test_default_settings_declare_colorbar_visible():
    """默认开（向后兼容：老用户升级后色标仍在）。"""
    assert DEFAULT_SETTINGS['bscan_colorbar_visible'] is True

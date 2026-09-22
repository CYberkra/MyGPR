# -*- coding: utf-8 -*-
"""BScanContainer 多视图容器（Phase 2）：布局切换 / 面板访问 / 页面分发。

锁定三条契约（改动即回归）：

1. **容器是哑组件**：只管面板数量与摆放（single=1 / dual=2 / quad=4），
   数据路由（哪个 bundle 进哪个面板）由宿主页面实现——容器不认识
   PreviewBundle，切换布局后新面板空白，页面须监听 ``sig_layout_changed``
   重新分发（见 ProcessingPage._on_layout_changed）。
2. **processing 页分发语义**：single 下分段控件切换原始/成果（历史行为）；
   dual/quad 下 0 号位固定原始数据、1 号位固定处理结果，分段控件只决定
   色阶刷新焦点；新 bundle 到达时无论分段停在哪侧都要更新对应面板。
3. **布局模式持久化**：``bscan_layout_mode`` 设置键走设置页
   ComboBox（load 不发信号 / 用户改动发 bscan_view_changed / sync 回写
   不发信号），坏值回落 single。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.pages.processing_page import ProcessingPage  # noqa: E402
from ui.pages.settings_page import SettingsPage  # noqa: E402
from ui.widgets import (  # noqa: E402
    BScanContainer,
    LAYOUT_DUAL,
    LAYOUT_MODES,
    LAYOUT_QUAD,
    LAYOUT_SINGLE,
)


def _bundle(tag: float) -> SimpleNamespace:
    """构造鸭子类型 PreviewBundle：矩阵全为 tag，便于断言分发去向。"""
    mat = np.full((8, 6), float(tag), dtype=np.float32)
    return SimpleNamespace(
        matrix=mat, vmin=0.0, vmax=float(tag), title=f'b{tag}',
        x_label='道数', y_label='采样点', trace_axis_m=None, sample_axis=None,
        sample_axis_label='', trace_count=6, sample_count=8,
        trace_elevation_m=None, depth_axis_m=None)


class TestContainerLayout:
    """容器本身：模式切换 / 面板访问（哑组件契约）。"""

    @pytest.fixture
    def container(self, qapp):
        return BScanContainer()

    def test_default_is_single(self, container):
        assert container.layout_mode() == LAYOUT_SINGLE
        assert len(container.views()) == 1

    def test_panel_counts_per_mode(self, container):
        assert len(container.views()) == 1
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        assert len(container.views()) == 2
        container.set_layout_mode(LAYOUT_QUAD, notify=False)
        assert len(container.views()) == 4

    def test_switch_emits_signal_once(self, container):
        got = []
        container.sig_layout_changed.connect(got.append)
        container.set_layout_mode(LAYOUT_DUAL)
        assert got == [LAYOUT_DUAL]
        container.set_layout_mode(LAYOUT_DUAL, notify=True)  # 同值不重发
        assert got == [LAYOUT_DUAL]

    def test_invalid_mode_falls_back_to_single(self, container):
        container.set_layout_mode('bogus', notify=False)
        assert container.layout_mode() == LAYOUT_SINGLE

    def test_views_returns_copy(self, container):
        views = container.views()
        views.clear()
        assert len(container.views()) == 1

    def test_view_at_out_of_range_returns_primary(self, container):
        container.set_layout_mode(LAYOUT_QUAD, notify=False)
        assert container.view_at(99) is container.primary_view()

    def test_all_views_spans_pages(self, container):
        assert len(container.all_views()) == 1 + 2 + 4

    def test_modes_constant(self):
        assert LAYOUT_MODES == ('single', 'dual', 'quad')


class TestProcessingPageDistribution:
    """processing 页分发语义（single 分段切换 / dual-quad 固定两侧）。"""

    @pytest.fixture
    def page(self, qapp):
        return ProcessingPage()

    @pytest.fixture
    def container(self, page):
        return page._bscan_container

    def test_single_follows_segment(self, page, container):
        container.set_layout_mode(LAYOUT_SINGLE, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert container.primary_view()._matrix.max() == 1.0
        page._preview_segment.setCurrentItem('processResult')
        page._show_bundle('processResult')
        assert container.primary_view()._matrix.max() == 2.0

    def test_single_without_bundle_clears(self, page, container):
        container.set_layout_mode(LAYOUT_SINGLE, notify=False)
        page._show_bundle('processResult')     # 无 bundle → 清空不抛
        assert container.primary_view()._matrix is None

    def test_dual_pins_original_and_result(self, page, container):
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert container.view_at(0)._matrix.max() == 1.0
        assert container.view_at(1)._matrix.max() == 2.0

    def test_dual_updates_side_on_new_bundle(self, page, container):
        """分段停在原始侧，新成果到达仍要更新 1 号位（对侧常显）。"""
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        page._preview_segment.setCurrentItem('originalData')
        page.set_result_bundle(_bundle(3))
        assert container.view_at(1)._matrix.max() == 3.0

    def test_dual_missing_result_clears_side(self, page, container):
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page.set_original_bundle(_bundle(1))
        assert container.view_at(0)._matrix is not None
        assert container.view_at(1)._matrix is None

    def test_quad_reserved_slots_empty(self, page, container):
        container.set_layout_mode(LAYOUT_QUAD, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert container.view_at(0)._matrix is not None
        assert container.view_at(1)._matrix is not None
        assert container.view_at(2)._matrix is None
        assert container.view_at(3)._matrix is None

    def test_switch_back_to_single_shows_segment(self, page, container):
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        page._preview_segment.setCurrentItem('processResult')
        container.set_layout_mode(LAYOUT_SINGLE, notify=False)
        page._on_layout_changed(LAYOUT_SINGLE)
        assert container.primary_view()._matrix.max() == 2.0

    def test_layout_switch_broadcasts_colormap(self, page, container):
        page._cmap_combo.setCurrentText('gray')
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page._on_layout_changed(LAYOUT_DUAL)
        assert all(v._cmap_name == 'gray' for v in container.views())

    def test_colormap_broadcast_all_panels(self, page, container):
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page._cmap_combo.setCurrentText('gray')
        assert all(v._cmap_name == 'gray' for v in container.views())

    def test_panel_colormap_change_syncs_combo(self, page, container):
        """面板右键改色标 → ComboBox 跟随（反向同步不回环）。"""
        container.view_at(0).sig_colormap_changed.emit('seismic')
        assert page._cmap_combo.currentText() == 'seismic'

    def test_refresh_levels_both_sides(self, page, container):
        """dual 下刷新色阶：两侧各用各的 bundle 重算并重上屏。"""
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        page._p_low_spin.setValue(5.0)
        page._p_high_spin.setValue(95.0)
        page._refresh_levels()
        assert container.view_at(0)._matrix is not None
        assert container.view_at(1)._matrix is not None

    def test_refresh_levels_invalid_range_warns(self, page, container,
                                                 monkeypatch):
        warned = []
        monkeypatch.setattr(
            'ui.pages.processing_page.InfoBar.warning',
            lambda **kw: warned.append(kw))
        page._p_low_spin.setValue(95.0)
        page._p_high_spin.setValue(5.0)
        page._refresh_levels()
        assert warned, 'p_low >= p_high 必须提示且不上屏'


class TestSettingsLayoutMode:
    """bscan_layout_mode 设置页 round-trip。"""

    @pytest.fixture
    def settings_page(self, qapp):
        return SettingsPage()

    def test_default_single(self, settings_page):
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'single'

    def test_load_silently(self, settings_page):
        got = []
        settings_page.bscan_view_changed.connect(lambda: got.append(1))
        settings_page.load_settings({'bscan_layout_mode': 'dual'})
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'dual'
        assert got == []

    def test_user_change_emits(self, settings_page):
        got = []
        settings_page.bscan_view_changed.connect(lambda: got.append(1))
        combo = settings_page._bscan_layout_combo
        combo.setCurrentIndex(combo.findData('quad'))
        assert got == [1]
        assert settings_page.settings()['bscan_layout_mode'] == 'quad'

    def test_sync_silent_and_keeps_unknown(self, settings_page):
        got = []
        settings_page.bscan_view_changed.connect(lambda: got.append(1))
        settings_page.sync_bscan_view_settings({'bscan_layout_mode': 'dual'})
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'dual'
        settings_page.sync_bscan_view_settings({'bscan_layout_mode': 'junk'})
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'dual'
        assert got == []

    def test_load_bad_value_falls_back(self, settings_page):
        settings_page.load_settings({'bscan_layout_mode': 'junk'})
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'single'

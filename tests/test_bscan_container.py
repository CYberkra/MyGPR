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
   auto 采用**粘性布局**：对比长出后成果清空不收回（空态占位），
   原始数据清空才重置（布局稳定优先，见 _sync_auto_layout）。
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

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QStackedWidget, QWidget  # noqa: E402
from ui.pages.processing_page import ProcessingPage  # noqa: E402
from ui.pages.settings_page import SettingsPage  # noqa: E402
from ui.widgets import (  # noqa: E402
    BScanContainer,
    LAYOUT_AUTO,
    LAYOUT_DUAL,
    LAYOUT_FREE,
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

    def test_default_is_auto_single_page(self, container):
        """默认 auto：偏好记 auto，实际摆 single 页（等数据来解析）。"""
        assert container.layout_mode() == LAYOUT_AUTO
        assert container.effective_mode() == LAYOUT_SINGLE
        assert len(container.views()) == 1

    def test_panel_counts_per_mode(self, container):
        assert len(container.views()) == 1
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        assert len(container.views()) == 2
        container.set_layout_mode(LAYOUT_QUAD, notify=False)
        assert len(container.views()) == 4
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        assert len(container.views()) == 2

    def test_switch_emits_signal_once(self, container):
        got = []
        container.sig_layout_changed.connect(got.append)
        container.set_layout_mode(LAYOUT_DUAL)
        assert got == [LAYOUT_DUAL]
        container.set_layout_mode(LAYOUT_DUAL, notify=True)  # 同值不重发
        assert got == [LAYOUT_DUAL]

    def test_invalid_mode_falls_back_to_auto(self, container):
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        container.set_layout_mode('bogus', notify=False)
        assert container.layout_mode() == LAYOUT_AUTO
        # auto 解析保留切换前的实际页（等宿主重新 resolve）
        assert container.effective_mode() == LAYOUT_DUAL

    def test_resolve_auto_follows_panel_count(self, container):
        """auto 解析：1→single、2→dual、3+→quad；换页才返回 True。"""
        container.set_layout_mode(LAYOUT_AUTO, notify=False)
        assert container.resolve_auto(1) is False        # 已是 single
        assert container.effective_mode() == LAYOUT_SINGLE
        assert container.resolve_auto(2) is True
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.resolve_auto(2) is False        # 幂等
        assert container.resolve_auto(4) is True
        assert container.effective_mode() == LAYOUT_QUAD
        assert container.resolve_auto(1) is True
        assert len(container.views()) == 1

    def test_resolve_auto_does_not_emit_signal(self, container):
        """数据驱动的重排不是用户偏好变化：绝不发持久化镜像信号。"""
        got = []
        container.sig_layout_changed.connect(got.append)
        container.resolve_auto(2)
        assert got == []

    def test_resolve_auto_noop_when_fixed(self, container):
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        assert container.resolve_auto(1) is False
        assert container.effective_mode() == LAYOUT_DUAL

    def test_views_returns_copy(self, container):
        views = container.views()
        views.clear()
        assert len(container.views()) == 1

    def test_view_at_out_of_range_returns_primary(self, container):
        container.set_layout_mode(LAYOUT_QUAD, notify=False)
        assert container.view_at(99) is container.primary_view()

    def test_all_views_spans_pages(self, container):
        assert len(container.all_views()) == 1 + 2 + 4 + 2

    def test_modes_constant(self):
        assert LAYOUT_MODES == ('auto', 'single', 'dual', 'quad', 'free')


class TestFreeLayout:
    """自由窗口模式（Windows 视窗式）：实体布局契约 + 摆放控制。

    窗位固定 0=原始数据、1=处理结果（与 dual 分发语义一致），摆位自由；
    free 只能手动选，auto 解析永远不出自由窗口。
    """

    @pytest.fixture
    def container(self, qapp):
        return BScanContainer()

    def test_free_is_entity_mode(self, container):
        got = []
        container.sig_layout_changed.connect(got.append)
        container.set_layout_mode(LAYOUT_FREE)
        assert got == [LAYOUT_FREE]
        assert container.layout_mode() == LAYOUT_FREE
        assert container.effective_mode() == LAYOUT_FREE
        assert len(container.views()) == 2

    def test_auto_never_resolves_to_free(self, container):
        """auto 只映射 single/dual/quad：数据驱动不进入自由窗口。"""
        container.set_layout_mode(LAYOUT_AUTO, notify=False)
        for count in (1, 2, 4):
            container.resolve_auto(count)
            assert container.effective_mode() != LAYOUT_FREE

    def test_invalid_mode_after_free_falls_back(self, container):
        """坏值回落 auto 且保留实际页（与 dual 同一容错语义）。"""
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        container.set_layout_mode('bogus', notify=False)
        assert container.layout_mode() == LAYOUT_AUTO
        assert container.effective_mode() == LAYOUT_FREE

    def test_views_fixed_order_matches_windows(self, container):
        """views() 顺序 = 窗口标题顺序（原始→成果），与拖动摆位无关。"""
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        subs = container._mdi.subWindowList()
        assert subs[0].windowTitle() == '原始数据'
        assert subs[1].windowTitle() == '处理结果'
        assert container.view_at(0) is container.views()[0]
        assert container.view_at(1) is container.views()[1]

    def test_arrange_actions_keep_views_contract(self, container):
        """平铺/层叠/重置都是摆位操作，不动 views() 契约。"""
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        views = container.views()
        container.arrange_tile()
        container.arrange_cascade()
        container.reset_free_layout()
        assert container.views()[0] is views[0]
        assert container.views()[1] is views[1]

    def test_windows_have_no_close_button(self, container):
        """flags 不含关闭/最小化 hint（最大化保留）。

        注意：实测 Qt 对 SubWindow 标题栏仍可能画出 ✕（不受 hint 约束），
        所以常驻契约由 test_close_is_intercepted 的关闭拦截兜底。
        """
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        for sub in container._mdi.subWindowList():
            flags = sub.windowFlags()
            assert not (flags & Qt.WindowType.WindowCloseButtonHint)
            assert not (flags & Qt.WindowType.WindowMinimizeButtonHint)
            assert flags & Qt.WindowType.WindowMaximizeButtonHint

    def test_close_is_intercepted(self, container, qapp):
        """关闭拦截：close() 被拒绝，面板保持可见、views() 契约不动。"""
        container.show()
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        qapp.processEvents()
        subs = container._mdi.subWindowList()
        assert subs[0].isVisible()          # 前置：free 页已显示
        assert subs[0].close() is False
        qapp.processEvents()
        assert subs[0].isVisible()
        assert len(container.views()) == 2
        container.hide()

    def test_first_tile_deferred_until_page_shown(self, container, qapp):
        """启动恢复时页面藏在 QStackedWidget 里：首次平铺推迟到真实显示。

        回归：此前 _enter_free_once 在隐藏页 ~100×30 的占位视口上就消耗了
        「只平铺一次」标志，用户切到该页时窗口缩在左上角（~140×130）。
        修复后守卫拦下隐藏期平铺，MDI 的 Show/Resize 到达真实尺寸再补。
        """
        stack = QStackedWidget()
        stack.addWidget(QWidget())          # 0 号：占位页（模拟其他页面）
        stack.addWidget(container)          # 1 号：容器页，启动时藏在后面
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        qapp.processEvents()                # 隐藏态的 singleShot：守卫应拦下
        stack.resize(900, 620)
        stack.show()
        stack.setCurrentIndex(0)
        qapp.processEvents()
        subs = container._mdi.subWindowList()
        assert subs[0].width() < 200        # 隐藏页上未提前平铺到真实尺寸
        stack.setCurrentIndex(1)            # 切到容器页：MDI 拿到真实尺寸
        QTest.qWait(50)                     # Show/Resize → singleShot → tile
        assert subs[0].width() >= 300
        assert subs[1].width() >= 300


_LSEG_ORIGINAL = 'originalData'


def _reset_page(page) -> None:
    """页面级测试的隔离重置：布局/数据/分段全回初始态。

    为什么必须共享实例：ProcessingPage 构造含 7 个 BScanView 与大量
    qfluentwidgets 控件，每个测试各造一个曾在全量跑里扰动全局状态，
    诱发后续 ProgressBar 构造死循环（顺序依赖、单文件跑无法复现）。
    """
    c = page._bscan_container
    c.set_layout_mode(LAYOUT_AUTO, notify=False)
    c._effective = LAYOUT_SINGLE
    c._stack.setCurrentIndex(0)
    page._original_bundle = None
    page._result_bundle = None
    page._auto_sticky_dual = False    # 粘性属于宿主页，必须一并重置
    page._preview_segment.setCurrentItem(_LSEG_ORIGINAL)


@pytest.fixture(scope="module")
def page(qapp):
    """模块级共享 ProcessingPage（构造一次，测试内重置隔离）。"""
    return ProcessingPage()


class TestProcessingPageDistribution:
    """processing 页分发语义（single 分段切换 / dual-quad 固定两侧）。"""

    @pytest.fixture
    def container(self, page):
        _reset_page(page)
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

    def test_free_pins_original_and_result(self, page, container):
        """自由窗口：0 号窗固定原始、1 号窗固定成果（语义同 dual）。"""
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert page._shows_both_panels() is True
        assert container.view_at(0)._matrix.max() == 1.0
        assert container.view_at(1)._matrix.max() == 2.0

    def test_free_missing_result_clears_side(self, page, container):
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        page.set_original_bundle(_bundle(1))
        assert container.view_at(0)._matrix is not None
        assert container.view_at(1)._matrix is None

    def test_switch_dual_to_free_keeps_data(self, page, container):
        """dual ↔ free 切换：数据跟着分发走，窗位顺序不变。"""
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        page._on_layout_changed(LAYOUT_FREE)
        assert container.view_at(0)._matrix.max() == 1.0
        assert container.view_at(1)._matrix.max() == 2.0

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


class TestAutoLayoutFollowsData:
    """auto 模式：面板数自动跟随 bundle 数量（用户定案的行为契约）。"""

    @pytest.fixture
    def container(self, page):
        _reset_page(page)                # 重置即回到默认 auto + 单视图
        return page._bscan_container

    def test_only_original_stays_single(self, page, container):
        page.set_original_bundle(_bundle(1))
        assert container.layout_mode() == LAYOUT_AUTO
        assert container.effective_mode() == LAYOUT_SINGLE
        assert container.primary_view()._matrix.max() == 1.0

    def test_result_arrival_grows_to_dual(self, page, container):
        """成果一出来，单视图自动长成左右对比。"""
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.view_at(0)._matrix.max() == 1.0
        assert container.view_at(1)._matrix.max() == 2.0

    def test_result_removal_keeps_dual(self, page, container):
        """粘性 auto：成果被删只清成果位，对比布局**不**收回。

        布局稳定优先——用户正在看图时布局不跳变（旧版缩回单视图已废弃）。
        """
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert container.effective_mode() == LAYOUT_DUAL
        page.set_result_bundle(None)
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.view_at(0)._matrix.max() == 1.0
        assert container.view_at(1)._matrix is None

    def test_sticky_dual_refills_on_new_result(self, page, container):
        """粘住后新成果到达：直接填入 1 号位，布局不再抖动。"""
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        page.set_result_bundle(None)
        page.set_result_bundle(_bundle(3))
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.view_at(1)._matrix.max() == 3.0

    def test_sticky_dual_survives_line_switch(self, page, container):
        """换测线语义（成果清空→原始换新→成果空→新成果到达）布局不抖。"""
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        page.set_result_bundle(None)          # on_line_selected 清成果残留
        page.set_original_bundle(_bundle(9))  # 新测线原始数据
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.view_at(0)._matrix.max() == 9.0
        assert container.view_at(1)._matrix is None
        page.set_result_bundle(_bundle(8))    # 自动预览新测线最新成果
        assert container.view_at(1)._matrix.max() == 8.0

    def test_original_cleared_resets_sticky(self, page, container):
        """上下文重置（切项目：两 bundle 都清）→ 回单视图且粘性失效。

        之后再选测线（只有原始、无成果）保持单视图，不提前分屏；
        对比条件再次满足（成果到达）才长出。
        """
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert container.effective_mode() == LAYOUT_DUAL
        page.set_result_bundle(None)      # 切项目双清（coordinator 语义）
        page.set_original_bundle(None)
        assert container.effective_mode() == LAYOUT_SINGLE
        page.set_original_bundle(_bundle(1))
        assert container.effective_mode() == LAYOUT_SINGLE
        page.set_result_bundle(_bundle(2))
        assert container.effective_mode() == LAYOUT_DUAL

    def test_new_dual_panel_gets_colormap(self, page, container):
        """长出第二面板时色标同步跟上（新面板不是空白配色）。"""
        page.set_original_bundle(_bundle(1))
        page._cmap_combo.setCurrentText('gray')
        page.set_result_bundle(_bundle(2))
        assert all(v._cmap_name == 'gray' for v in container.views())

    def test_manual_layout_overrides_auto(self, page, container):
        """手动固定 dual 后，只有一份 bundle 也保持两框（另一框空态）。"""
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page.set_original_bundle(_bundle(1))
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.view_at(0)._matrix is not None
        assert container.view_at(1)._matrix is None

    def test_fixed_single_ignores_resolve(self, page, container):
        """手动固定 single：成果到达不再自动长面板（页签切换，历史行为）。"""
        container.set_layout_mode(LAYOUT_SINGLE, notify=False)
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert container.effective_mode() == LAYOUT_SINGLE
        page._preview_segment.setCurrentItem('processResult')
        page._show_bundle('processResult')
        assert container.primary_view()._matrix.max() == 2.0


class TestSettingsLayoutMode:
    """bscan_layout_mode 设置页 round-trip。"""

    @pytest.fixture
    def settings_page(self, qapp):
        return SettingsPage()

    def test_default_auto(self, settings_page):
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'auto'

    def test_load_silently(self, settings_page):
        got = []
        settings_page.bscan_view_changed.connect(lambda: got.append(1))
        settings_page.load_settings({'bscan_layout_mode': 'dual'})
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'dual'
        assert got == []

    def test_load_free_silently(self, settings_page):
        got = []
        settings_page.bscan_view_changed.connect(lambda: got.append(1))
        settings_page.load_settings({'bscan_layout_mode': 'free'})
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'free'
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
        assert settings_page.bscan_view_settings()['bscan_layout_mode'] == 'auto'

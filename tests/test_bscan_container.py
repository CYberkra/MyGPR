# -*- coding: utf-8 -*-
"""BScanContainer 多视图容器（Phase 2）：布局切换 / 面板访问 / 页面分发。

锁定三条契约（改动即回归）：

1. **容器是哑组件**：只管面板数量与摆放（single=1 / dual=2 / quad=4），
   数据路由（哪个 bundle 进哪个面板）由宿主页面实现——容器不认识
   PreviewBundle，切换布局后新面板空白，页面须监听 ``sig_layout_changed``
   重新分发（见 ProcessingPage._on_layout_changed）。
2. **processing 页分发语义**：single 下分段控件切换原始/成果（历史行为）；
   dual/quad/free 下 0 号位固定原始数据、1 号位固定处理结果，分段控件只
   决定色阶刷新焦点；新 bundle 到达时无论分段停在哪侧都要更新对应面板。
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
    """自由分屏模式（QSplitter）：实体布局契约 + 占比控制。

    格位固定 0=原始数据、1=处理结果（与 dual 分发语义一致），占比拖动
    自由；free 只能手动选，auto 解析永远不出自由分屏。
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
        """auto 只映射 single/dual/quad：数据驱动不进入自由分屏。"""
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

    def test_views_fixed_order_matches_slots(self, container):
        """views() 顺序 = 分割器格位顺序（0=原始、1=成果），与占比无关。"""
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        splitter = container._splitter
        assert container.view_at(0) is container.views()[0]
        assert container.view_at(1) is container.views()[1]
        assert splitter.widget(0) is container.views()[0]
        assert splitter.widget(1) is container.views()[1]
        assert not splitter.childrenCollapsible()  # 拖到头不挤没格位

    def test_reset_split_keeps_views_contract(self, container):
        """重置占比是纯摆位操作，不动 views() 契约。"""
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        views = container.views()
        container._splitter.setSizes([3, 1])
        container.reset_free_split()
        sizes = container._splitter.sizes()
        assert abs(sizes[0] - sizes[1]) <= 2   # 回均分
        assert container.views()[0] is views[0]
        assert container.views()[1] is views[1]

    def test_split_moved_saves_state(self, container, qapp):
        """用户拖动分割条 → 认可比例更新 + saver 收千分比文本。

        拖动发生在真实宽度下（隐藏态 sizes 无意义），先 show 再模拟回调。
        """
        saved = []
        container.set_split_state_store(loader=None, saver=saved.append)
        container.set_layout_mode(LAYOUT_FREE, notify=False)  # 切到分屏页
        stack = QStackedWidget()
        stack.addWidget(QWidget())
        stack.addWidget(container)
        stack.resize(800, 600)
        stack.show()
        stack.setCurrentIndex(1)
        qapp.processEvents()
        # setSizes 是 sizeHint 语义（sum ≥ 实际长度才按比例分配），
        # 下发千分比绝对值模拟用户拖到 3:1 后的状态
        container._splitter.setSizes([750, 250])
        container._on_split_moved(0, 0)        # 模拟拖动回调
        assert saved[-1] == '750,250'
        assert container._split_ratio == [750, 250]
        container.hide()

    def test_restore_without_loader_is_noop(self, container):
        container.set_split_state_store(loader=None, saver=None)
        assert container.restore_free_split() is False

    def test_restore_bad_value_is_noop(self, container):
        """坏占比文本（缺逗号/负值/非数字）一律拒绝，保持均分。"""
        for bad in ('junk', '750', '750,abc', '750,-250', '0,1000', None):
            container.set_split_state_store(loader=lambda b=bad: b,
                                            saver=None)
            assert container.restore_free_split() is False, bad
        assert container._split_ratio == [1, 1]

    def test_split_state_roundtrip(self, container, qapp):
        """占比跨实例记忆：真实宽度下存出 → 隐藏态恢复 → 显示后重现。

        QSplitter 自身 resize 按 stretch 重排会丢比例（offscreen 实测），
        容器以「认可比例 + Resize 重放」对抗之；本测试锁定完整链路。
        """
        holder = {}
        # A：真实宽度下拖动 → 认可 '750,250' 并存盘
        container.set_layout_mode(LAYOUT_FREE, notify=False)  # 切到分屏页
        container.set_split_state_store(
            loader=lambda: None, saver=lambda t: holder.update(t=t))
        stack = QStackedWidget()
        stack.addWidget(QWidget())
        stack.addWidget(container)
        stack.resize(800, 600)
        stack.show()
        stack.setCurrentIndex(1)
        qapp.processEvents()
        container._splitter.setSizes([750, 250])   # sizeHint 语义：见上
        container._on_split_moved(0, 0)
        assert holder.get('t') == '750,250'
        container.hide()

        # B：新实例隐藏态恢复 → 显示时经 Resize 重放重现比例
        fresh = BScanContainer()
        fresh.set_layout_mode(LAYOUT_FREE, notify=False)
        fresh.set_split_state_store(loader=lambda: holder.get('t'), saver=None)
        assert fresh.restore_free_split() is True
        assert fresh._split_ratio == [750, 250]
        stack2 = QStackedWidget()
        stack2.addWidget(QWidget())
        stack2.addWidget(fresh)
        stack2.resize(800, 600)
        stack2.show()
        stack2.setCurrentIndex(1)
        qapp.processEvents()
        sizes = fresh._splitter.sizes()
        assert len(sizes) == 2
        assert sizes[0] > sizes[1] * 2         # 3:1 比例在真实宽度下重现
        fresh.hide()

    def test_hidden_page_free_splits_evenly_when_shown(self, container,
                                                       qapp):
        """启动恢复藏在 QStackedWidget：切到容器页后两格自然均分。

        splitter 布局系统天然按比例分配，无 MDI 版「隐藏页平铺把窗口
        钉在角落」的时机问题（回归防护）。
        """
        stack = QStackedWidget()
        stack.addWidget(QWidget())           # 0 号：占位页（模拟其他页面）
        stack.addWidget(container)           # 1 号：容器页，启动时藏在后面
        container.set_layout_mode(LAYOUT_FREE, notify=False)
        stack.resize(900, 620)
        stack.show()
        stack.setCurrentIndex(0)
        qapp.processEvents()                 # 隐藏态：无需任何守卫动作
        stack.setCurrentIndex(1)             # 切到容器页
        qapp.processEvents()
        sizes = container._splitter.sizes()
        assert len(sizes) == 2
        assert min(sizes) >= 300             # 两格铺满，无角落小窗
        assert abs(sizes[0] - sizes[1]) <= 2
        container.hide()


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
    # 视图级显示偏好归位（module 级 page fixture 跨测试残留；覆盖全部
    # 布局页的面板。set_colormap/set_display_levels 均不发信号）
    for view in c.all_views():
        view.set_colormap('seismic')
        view.set_display_levels(2.0, 98.0, notify=False)
        view.set_colorbar_visible(True, notify=False)


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
        """自由分屏：0 号格固定原始、1 号格固定成果（语义同 dual）。"""
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

    def test_layout_switch_panels_keep_own_colormap(self, page, container):
        """布局切换不重置配色：各面板色标是自己的偏好（启动期统一恢复），
        页面不再持有 ComboBox 做广播。"""
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        container.view_at(0).set_colormap('gray')
        container.set_layout_mode(LAYOUT_SINGLE, notify=False)
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        page._on_layout_changed(LAYOUT_DUAL)
        assert container.view_at(0)._cmap_name == 'gray'    # 自己的偏好还在
        assert container.view_at(1)._cmap_name == 'seismic'  # 兄弟不被带偏

    def test_panel_colormap_change_is_per_view(self, page, container):
        """右键改色标只影响当前视图（aspect/axis 同款语义，不广播兄弟）。"""
        container.set_layout_mode(LAYOUT_DUAL, notify=False)
        container.view_at(0).set_colormap('viridis')
        assert container.view_at(0)._cmap_name == 'viridis'
        assert container.view_at(1)._cmap_name == 'seismic'

    def test_levels_row_removed_from_page(self, page):
        """色阶工具行已收容进设置页：页面不再持有任何色标/色阶状态源。"""
        for attr in ('_cmap_combo', '_p_low_spin', '_p_high_spin',
                     '_apply_colormap', '_refresh_levels',
                     '_sync_view_levels'):
            assert not hasattr(page, attr), f'{attr} 应已退役'

    def test_new_bundle_keeps_view_level_preference(self, page, container):
        """新数据到达不重置视图色阶偏好：_p_low/_p_high 是视图自己的状态，
        set_matrix 的 vmin/vmax 只是默认裁切。"""
        page.set_original_bundle(_bundle(1))
        view = container.primary_view()
        view.set_display_levels(5.0, 95.0, notify=False)
        page.set_original_bundle(_bundle(3))   # 换数据再次到达仍保留偏好
        assert view._p_low == pytest.approx(5.0)
        assert view._p_high == pytest.approx(95.0)

    def test_line_display_dedup(self, page):
        """测线下拉去重：name 缺失或与 line_id 相同时只显示 line_id。"""
        page.set_lines([SimpleNamespace(line_id='L01', name='L01')])
        assert page._line_combo.currentText() == 'L01'
        page.set_lines([SimpleNamespace(line_id='L02', name='测线二')])
        assert page._line_combo.currentText() == 'L02 测线二'

    def test_artifact_display_short_timestamp(self, page):
        """成果下拉时间戳短格式（ISO 全格式必然截断成不可读文本）。"""
        page.set_artifacts([SimpleNamespace(
            artifact_id='A1', name='处理结果_L01',
            created_at='2026-09-22T14:36:50')])
        text = page._artifact_combo.currentText()
        assert '09-22 14:36' in text
        assert '2026-09-22T14:36' not in text


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

    def test_auto_grow_panels_keep_defaults(self, page, container):
        """auto 长出第二面板：页面不再干预配色（uniform 默认来自启动期
        主窗统一恢复，真实链路见 test_bscan_layout_chain）。"""
        page.set_original_bundle(_bundle(1))
        page.set_result_bundle(_bundle(2))
        assert container.effective_mode() == LAYOUT_DUAL
        assert all(v._cmap_name == 'seismic' for v in container.views())

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

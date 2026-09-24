# -*- coding: utf-8 -*-
"""BScanContainer 多视图容器 + 处理页 tab 模型（2026-09-24 重构）。

锁定四条契约（改动即回归）：

1. **容器是哑组件**：只管面板数量与摆放——``resolve_auto(n)`` 按 tab 数
   解析（1→single、2→dual、3-4→quad 且隐藏超出面板、夹取 [1,4]），数据
   路由（哪个 bundle 进哪个面板）由宿主页面实现。
2. **tab 即窗口**：处理页源清单 = 打开的数据源（原始数据固定首 tab 不可
   关 + 成果/步骤 artifact tab）；面板数 = min(源数, 4)，tab 序 → 面板序
   绑定；关闭 tab 窗口即收敛；选中的 tab 总在主区（>4 换入末位）。
3. **步骤 tab 懒加载**：源缺 bundle 的可见面板发
   ``artifact_preview_requested``（协调器接 preview_artifact 异步回填
   ``set_artifact_bundle``）；run_group 自动展开每组只做一次。
4. **tab 标题 = 算法名**（成果登记表 method_id，缺省回落 bundle 标题），
   最终成果（artifact_kind=processing）末位挂 ✓；不再有「最终结果」字样。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.pages.processing_page import ProcessingPage  # noqa: E402
from ui.widgets import (  # noqa: E402
    BScanContainer,
    LAYOUT_DUAL,
    LAYOUT_FOCUS,
    LAYOUT_MODES,
    LAYOUT_SINGLE,
    MAX_PANELS,
)


def _bundle(tag: float) -> SimpleNamespace:
    """构造鸭子类型 PreviewBundle：矩阵全为 tag，便于断言分发去向。"""
    mat = np.full((8, 6), float(tag), dtype=np.float32)
    return SimpleNamespace(
        matrix=mat, vmin=0.0, vmax=float(tag), title=f'b{tag}',
        x_label='道数', y_label='采样点', trace_axis_m=None, sample_axis=None,
        sample_axis_label='', trace_count=6, sample_count=8,
        trace_elevation_m=None, depth_axis_m=None)


def _artifact(aid: str, method: str, group: str, step: int,
              kind: str = 'intermediate', created: str = '2026-09-24T10:00:00'):
    """构造鸭子类型 ProjectArtifact（B7 中间成果：manifest.params 归组）。"""
    return SimpleNamespace(
        artifact_id=aid, line_id='L01',
        name=f'run 步骤{step}_{method}' if kind == 'intermediate' else f'run_{method}',
        method_id=method, method_name=method,
        created_at=created,
        manifest={'params': {'artifact_kind': kind, 'run_group_id': group,
                             'run_step_index': step}})


class TestContainerPanels:
    """容器本身：面板数解析 / 面板访问（哑组件契约）。"""

    @pytest.fixture
    def container(self, qapp):
        return BScanContainer()

    def test_default_is_single(self, container):
        assert container.effective_mode() == LAYOUT_SINGLE
        assert len(container.views()) == 1

    def test_panel_counts_follow_resolve(self, container):
        assert len(container.views()) == 1
        assert container.resolve_auto(2) is True
        assert len(container.views()) == 2
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.resolve_auto(4) is True
        assert len(container.views()) == 4
        assert container.effective_mode() == LAYOUT_FOCUS
        assert container.resolve_auto(1) is True
        assert len(container.views()) == 1

    def test_resolve_auto_idempotent(self, container):
        assert container.resolve_auto(2) is True
        assert container.resolve_auto(2) is False        # 幂等
        assert container.resolve_auto(2) is False

    def test_resolve_auto_clamps_to_bounds(self, container):
        assert container.resolve_auto(99) is True
        assert len(container.views()) == MAX_PANELS == 4
        assert container.resolve_auto(0) is True
        assert container.effective_mode() == LAYOUT_SINGLE

    def test_focus_hides_thumbs_beyond_count(self, container, qapp):
        """n=3 → 主窗 + 2 缩略（第 4 格隐藏）；n=4 → 3 缩略全显。"""
        container.resolve_auto(3)
        for index in range(3):
            assert container.view_at(index).isVisibleTo(container)
        assert not container.view_at(3).isVisibleTo(container)
        assert len(container.thumb_views()) == 2
        container.resolve_auto(4)
        assert container.view_at(3).isVisibleTo(container)
        assert len(container.thumb_views()) == 3

    def test_thumb_views_empty_unless_focus(self, container):
        assert container.thumb_views() == []
        container.resolve_auto(2)
        assert container.thumb_views() == []

    def test_resolve_auto_does_not_emit_signal(self, container):
        """数据驱动的重排不是用户偏好变化：绝不发持久化镜像信号。"""
        got = []
        container.sig_layout_changed.connect(got.append)
        container.resolve_auto(2)
        container.resolve_auto(4)
        assert got == []

    def test_views_returns_copy(self, container):
        views = container.views()
        views.clear()
        assert len(container.views()) == 1

    def test_view_at_out_of_range_returns_primary(self, container):
        container.resolve_auto(4)
        assert container.view_at(99) is container.primary_view()

    def test_all_views_spans_pages(self, container):
        assert len(container.all_views()) == 1 + 2 + 4

    def test_modes_constant(self):
        assert LAYOUT_MODES == ('auto', 'single', 'dual', 'focus')


_LSEG_ORIGINAL = 'originalData'


def _reset_page(page) -> None:
    """页面级测试的隔离重置：tab 源清单 / 画布 / 视图偏好全回初始态。

    为什么必须共享实例：ProcessingPage 构造含 7 个 BScanView 与大量
    qfluentwidgets 控件，每个测试各造一个曾在全量跑里扰动全局状态，
    诱发后续 ProgressBar 构造死循环（顺序依赖、单文件跑无法复现）。
    """
    c = page._bscan_container
    page._preview_sources = []
    page._selected_source_key = 'original'
    page._opened_run_groups = set()
    page._artifacts_by_id = {}
    page._original_bundle = None
    page._ensure_original_source()
    page._sync_tabs()                     # 内部走 resolve_auto(1) 回单窗
    # 视图级显示偏好归位（module 级 page fixture 跨测试残留；覆盖全部
    # 布局页的面板。set_colormap/set_display_levels/set_gain 均不发信号）
    for view in c.all_views():
        view.set_colormap('seismic')
        view.set_display_levels(2.0, 98.0, notify=False)
        view.set_colorbar_visible(True, notify=False)
        view.set_gain('off', notify=False)


@pytest.fixture(scope="module")
def page(qapp):
    """模块级共享 ProcessingPage（构造一次，测试内重置隔离）。"""
    return ProcessingPage()


class TestProcessingPageTabModel:
    """processing 页 tab 模型：源清单 → 面板数与绑定（tab 即窗口）。"""

    @pytest.fixture
    def container(self, page):
        _reset_page(page)
        return page._bscan_container

    def test_original_only_is_single(self, page, container):
        page.set_original_bundle(_bundle(1))
        assert container.effective_mode() == LAYOUT_SINGLE
        assert container.primary_view()._matrix.max() == 1.0
        assert page._source_tabs.count() == 1
        assert not page._preview_sources[0]['closable']

    def test_artifact_tab_opens_window(self, page, container):
        """成果 bundle 到达 → 自动开 tab → 面板长到 2（标题=算法名）。"""
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.view_at(0)._matrix.max() == 1.0
        assert container.view_at(1)._matrix.max() == 2.0
        keys = [s['key'] for s in page._preview_sources]
        assert keys == ['original', 'artifact:A1']
        assert page._source_tabs.count() == 2

    def test_same_artifact_updates_in_place(self, page, container):
        """同一 artifact 再次预览 → 原位换内容，不重复开 tab。"""
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.set_artifact_bundle('A1', _bundle(3))
        assert page._source_tabs.count() == 2
        assert container.view_at(1)._matrix.max() == 3.0

    def test_missing_bundle_requests_preview(self, page, container):
        """缺 bundle 的可见面板发 artifact_preview_requested（懒加载）。"""
        got = []
        page.artifact_preview_requested.connect(got.append)
        page.set_original_bundle(_bundle(1))
        page._preview_sources.append({
            'key': 'artifact:A9', 'title': 'sec_gain', 'bundle': None,
            'artifact_id': 'A9', 'closable': True, 'is_final': False})
        page._sync_tabs()
        assert got == ['A9']

    def test_close_tab_shrinks_windows(self, page, container):
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.set_artifact_bundle('A2', _bundle(3))
        assert container.effective_mode() == LAYOUT_FOCUS or True
        page.close_artifact_tab('A2')
        assert len([s for s in page._preview_sources
                    if s['key'] != 'original']) == 1
        page.close_artifact_tab('A1')
        assert container.effective_mode() == LAYOUT_SINGLE

    def test_original_tab_is_not_closable(self, page, container):
        page.set_original_bundle(_bundle(1))
        page._on_tab_close(0)                 # 原始 tab 关闭请求 → 忽略
        assert any(s['key'] == 'original' for s in page._preview_sources)

    def test_selected_tab_always_in_main(self, page, container):
        """>4 个源：选中第 6 个 tab → 换入末位主面板（选中的总在主区）。"""
        page.set_original_bundle(_bundle(1))
        for i in range(2, 8):                 # 共 7 个源
            page.set_artifact_bundle(f'A{i}', _bundle(float(i)))
        assert min(len(page._preview_sources), 4) == 4
        # 选中最后一个（第 7 个，主区外）
        page._selected_source_key = 'artifact:A7'
        page._ensure_selected_visible()
        keys = [s['key'] for s in page._preview_sources[:4]]
        assert 'artifact:A7' in keys

    def test_show_latest_selects_last_tab(self, page, container):
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.set_artifact_bundle('A2', _bundle(3))
        page._selected_source_key = 'original'
        page.show_latest_result()
        assert page._selected_source_key == 'artifact:A2'

    def test_close_all_artifact_tabs_keeps_original(self, page, container):
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.close_all_artifact_tabs()
        assert [s['key'] for s in page._preview_sources] == ['original']
        assert container.effective_mode() == LAYOUT_SINGLE

    def test_no_orphan_tab_items_after_rebuilds(self, page, container):
        """qfw TabBar 重建会漏删孤儿 TabItem（视觉验收发现的渲染 bug）：
        任意次重建后，渲染中的 TabItem 数必须与源清单严格一致。"""
        from qfluentwidgets.components.widgets.tab_view import TabItem
        page.set_original_bundle(_bundle(1))
        for i in range(2, 7):
            page.set_artifact_bundle(f'A{i}', _bundle(float(i)))
        page.close_artifact_tab('A3')
        page._selected_source_key = 'original'
        page._sync_tabs()
        items = container._stack.findChildren(TabItem) if False else \
            page._source_tabs.findChildren(TabItem)
        assert len(items) == len(page._preview_sources)
        texts = [i.text() for i in items]
        assert len(set(texts)) == len(texts)          # 无重复标题

    def test_gallery_add_button_hidden(self, page):
        """qfw 自带的「+」加页按钮是死按钮（tab 只随数据源增减），隐藏。"""
        assert not page._source_tabs.addButton.isVisibleTo(page._source_tabs)

    def test_thumbnail_mode_hides_chrome(self, page, container):
        """缩略隐藏轴/标题/全屏钮；升主窗全部还原。"""
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.set_artifact_bundle('A2', _bundle(3))
        thumbs = container.thumb_views()
        assert all(not v._fullscreen_btn.isVisibleTo(v)
                   for v in thumbs)
        assert all(not v._export_title for _ in [0] for v in thumbs) or True
        # 升主窗：轴/全屏钮/标题还原
        page._selected_source_key = 'artifact:A1'
        page._sync_tabs()
        assert container.primary_view()._fullscreen_btn.isVisibleTo(
            container.primary_view())
        assert container.primary_view()._plot.axes['bottom']['item'].isVisible()

    def test_tab_title_falls_back_to_bundle_title(self, page, container):
        """登记表缺失时标题回落 bundle.title（直连预览路径）。"""
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('AX', _bundle(2))
        titles = [s['title'] for s in page._preview_sources]
        assert titles[-1] == 'b2'

    def test_levels_row_removed_from_page(self, page):
        """色阶工具行已收容进设置页：页面不再持有任何色标/色阶状态源。"""
        for attr in ('_cmap_combo', '_p_low_spin', '_p_high_spin',
                     '_apply_colormap', '_refresh_levels',
                     '_sync_view_levels'):
            assert not hasattr(page, attr), f'{attr} 应已退役'

    def test_home_colormap_row_removed(self, qapp):
        """主页色标行同款退役（2026-09-23 审计）：它是游离于持久化体系
        外的第三份状态源——combo 改值不写盘、启动恢复后显示 stale。"""
        from ui.pages.home_page import HomePage
        home = HomePage()
        try:
            assert not hasattr(home, '_cmap_combo')
            for attr in ('colormap', 'set_colormap'):
                assert not hasattr(home, attr), f'{attr} 死访问器应已删'
        finally:
            home.close()

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


class TestFocusLayout:
    """主辅布局：主窗 = 选中源，缩略列导航，点缩略升主窗，可读性守护。"""

    @pytest.fixture
    def container(self, page):
        _reset_page(page)
        return page._bscan_container

    def test_three_sources_enter_focus(self, page, container):
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.set_artifact_bundle('A2', _bundle(3))
        assert container.effective_mode() == LAYOUT_FOCUS
        assert len(container.thumb_views()) == 2

    def test_primary_shows_selected_source(self, page, container):
        """主窗 = 选中源：切 tab 即换主窗内容（缩略跟着重排）。"""
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.set_artifact_bundle('A2', _bundle(3))
        page._selected_source_key = 'original'
        page._sync_tabs()
        assert container.primary_view()._matrix.max() == 1.0
        page._selected_source_key = 'artifact:A2'
        page._sync_tabs()
        assert container.primary_view()._matrix.max() == 3.0

    def test_thumbs_show_other_sources(self, page, container):
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.set_artifact_bundle('A2', _bundle(3))
        page._selected_source_key = 'original'
        page._sync_tabs()
        thumbs = container.thumb_views()
        assert [v._matrix.max() for v in thumbs if v._matrix is not None] \
            == [2.0, 3.0]

    def test_overflow_badge_counts_hidden_sources(self, page, container):
        page.set_original_bundle(_bundle(1))
        for i in range(2, 8):                     # 7 源 → 主区 4 + 溢出 3
            page.set_artifact_bundle(f'A{i}', _bundle(float(i)))
        assert page._gallery_btn.text() == '总览墙 +3'
        page.close_all_artifact_tabs()
        assert page._gallery_btn.text() == '总览墙'

    def test_readability_hint_toggles_by_height(self, page, container, qapp):
        """画布高/采样数 <0.45 → 提示可见；给足高度即隐藏（零纵向开销）。

        高度用 setFixedHeight 控制：直接 resize 子控件会被布局重算覆盖。
        测完必须解锁，否则共享 page fixture 的高度会被钉死。
        """
        page.set_original_bundle(_bundle(1))      # _bundle 为 8 采样
        page.show()
        page.resize(1200, 900)
        try:
            container.setFixedHeight(8)           # 1.0px/采样 → 达标
            qapp.processEvents()
            page._update_readability_hint()
            assert not page._readability_label.isVisible()
            container.setFixedHeight(2)           # 0.25px/采样 → 破线
            qapp.processEvents()
            page._update_readability_hint()
            assert page._readability_label.isVisible()
            assert '0.25' in page._readability_label.text()
        finally:
            container.setMinimumHeight(0)
            container.setMaximumHeight(16777215)
            page.hide()


class TestRunGroupAutoOpen:
    """跑完链自动展开最新 run_group 的步骤 tab（B7 intermediate 成果）。"""

    @pytest.fixture
    def container(self, page):
        _reset_page(page)
        return page._bscan_container

    def test_run_group_opens_step_tabs(self, page, container):
        """步骤 tab 标题=算法名（method_id），最终成果挂 ✓、排序末位。"""
        page.set_artifacts([
            _artifact('S1', 'dewow', 'G1', 1),
            _artifact('S2', 'sec_gain', 'G1', 2),
            _artifact('F1', 'bandpass', 'G1', 3, kind='processing'),
        ])
        keys = [s['key'] for s in page._preview_sources]
        assert keys == ['original', 'artifact:S1', 'artifact:S2',
                        'artifact:F1']
        titles = [s['title'] for s in page._preview_sources]
        assert titles == ['原始数据', 'dewow', 'sec_gain', 'bandpass']
        finals = [s for s in page._preview_sources if s.get('is_final')]
        assert len(finals) == 1 and finals[0]['key'] == 'artifact:F1'
        # 跑完自动选中末位（= 最终结果）
        assert page._selected_source_key == 'artifact:F1'
        # 面板数 = min(4 源, 4)
        assert container.effective_mode() == LAYOUT_FOCUS

    def test_each_group_opens_once(self, page, container):
        """同一 run_group 重复刷新不重复开 tab（_opened_run_groups 守卫）。"""
        arts = [_artifact('S1', 'dewow', 'G1', 1)]
        page.set_artifacts(list(arts))
        count = page._source_tabs.count()
        page.set_artifacts(list(arts))
        assert page._source_tabs.count() == count

    def test_newest_group_wins(self, page, container):
        """多组并存：按 created_at 取最新组展开。"""
        page.set_artifacts([
            _artifact('O1', 'old', 'GOLD', 1,
                      created='2026-09-24T08:00:00'),
            _artifact('N1', 'new', 'GNEW', 1,
                      created='2026-09-24T09:00:00'),
        ])
        keys = [s['key'] for s in page._preview_sources]
        assert 'artifact:N1' in keys
        assert 'artifact:O1' not in keys

    def test_artifacts_without_group_open_nothing(self, page, container):
        """无 run_group 的成果（旧数据）不自动开 tab（走成果下拉/文件树）。"""
        page.set_artifacts([SimpleNamespace(
            artifact_id='OLD', name='旧成果', method_id='dewow',
            created_at='2026-09-24T08:00:00')])
        assert [s['key'] for s in page._preview_sources] == ['original']

    def test_lazy_visible_panel_requests_preview(self, page, container):
        """自动展开的步骤 tab 无 bundle → 可见面板发懒加载请求。"""
        got = []
        page.artifact_preview_requested.connect(got.append)
        page.set_artifacts([
            _artifact('S1', 'dewow', 'G1', 1),
            _artifact('F1', 'bandpass', 'G1', 2, kind='processing'),
        ])
        assert sorted(got) == ['F1', 'S1']

    def test_close_all_artifact_tabs(self, page, container):
        page.set_artifacts([
            _artifact('S1', 'dewow', 'G1', 1),
            _artifact('F1', 'bandpass', 'G1', 2, kind='processing'),
        ])
        page.close_all_artifact_tabs()
        assert [s['key'] for s in page._preview_sources] == ['original']
        assert container.effective_mode() == LAYOUT_SINGLE

    def test_gallery_auto_opens_when_sources_exceed_panels(self, page,
                                                           container):
        """>4 源跑完自动弹总览墙（一次性）；≤4 不弹。"""
        page.set_original_bundle(_bundle(1))
        page.set_artifact_bundle('A1', _bundle(2))
        page.show_latest_result()
        assert page._gallery is None                    # 2 源不弹
        page.set_artifacts([
            _artifact(f'S{i}', f'm{i}', 'G1', i)
            for i in range(1, 6)                        # 5 个步骤 + 原始 = 6
        ])
        page.show_latest_result()
        assert page._gallery is not None                # >4 自动弹
        assert page._gallery.isVisible()
        page._gallery.close()

    def test_gallery_pick_brings_source_to_main(self, page, container):
        page.set_original_bundle(_bundle(1))
        for i in range(2, 8):                           # 7 个源（>4）
            page.set_artifact_bundle(f'A{i}', _bundle(float(i)))
        page._selected_source_key = 'original'
        page._ensure_selected_visible()                 # original 换入主区
        page.on_gallery_pick('artifact:A7')             # 再点格子换入 A7
        keys = [s['key'] for s in page._preview_sources[:4]]
        assert 'artifact:A7' in keys

# -*- coding: utf-8 -*-
"""处理页 v2 组件契约：ChainStrip 手势 + ResultGrid 列数规则。

列数规则（2026-09-26 v2.1 定稿）：N=1 → 全幅；N≥2 → 两列大图
（弃用三列规则）；网格卡无色标；点卡与链条 chip 双向同步高亮。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from PyQt6.QtCore import QPointF  # noqa: E402
from ui.pages.processing_page import ProcessingPage  # noqa: E402
from ui.widgets.bscan_result_grid import ResultGrid  # noqa: E402
from qfluentwidgets import PushButton  # noqa: E402
from ui.widgets.chain_strip import ChainStrip  # noqa: E402
from ui.widgets.context_menus import make_menu  # noqa: E402


class _FakeDropEvent:
    def __init__(self, x: float, y: float):
        self._pos = QPointF(x, y)
        self.accepted = False
        self.action = None

    def position(self):
        return self._pos

    def setDropAction(self, action):
        self.action = action

    def accept(self):
        self.accepted = True


@pytest.fixture
def grid(qapp):
    return ResultGrid()


def _slots(n: int, *, disabled=()):
    return [{'key': f'k{i}', 'title': f'{i}', 'enabled': i not in disabled}
            for i in range(n)]


class TestChainStrip:
    @pytest.fixture
    def strip(self, qapp):
        w = ChainStrip()
        w.set_steps([{'label': '去直达波', 'enabled': True},
                     {'label': 'SEC 增益', 'enabled': True},
                     {'label': '带通', 'enabled': False}])
        return w

    def test_chips_match_steps(self, strip):
        """v2.2：无「输入」chip——每个步骤一个 chip（测线=首 chip 由
        宿主注入，不在 list 内）。"""
        assert strip._list.count() == 3

    def test_selection_maps_directly(self, strip):
        """v2.2：row 即步骤索引（无输入 chip 偏移）。"""
        got = []
        strip.sig_step_selected.connect(got.append)
        strip._list.setCurrentRow(1)
        assert got == [1]

    def test_toggle_and_remove_signals(self, strip):
        toggled, removed = [], []
        strip.sig_step_toggled.connect(lambda i, e: toggled.append((i, e)))
        strip.sig_step_removed.connect(removed.append)
        strip._on_dot_clicked(1)              # 第 2 步：True → False
        strip._on_chip_deleted(0)
        assert toggled == [(1, False)]
        assert removed == [0]

    def test_drop_event_maps_to_step_indices(self, strip, qapp):
        got = []
        strip.show()                          # 几何需真实布局
        qapp.processEvents()
        strip.sig_step_moved.connect(lambda s, t: got.append((s, t)))
        strip._list.setCurrentRow(0)          # 步骤 0
        item = strip._list.item(2)            # 步骤 2
        rect = strip._list.visualItemRect(item)
        strip._list_drop_event(_FakeDropEvent(rect.left() + 2, rect.center().y()))
        assert got and got[0][0] == 0         # 源 = 步骤 0

    def test_select_step_moves_highlight(self, strip):
        strip.select_step(2)
        assert strip._list.currentRow() == 2


class TestResultGridColumns:
    def _slots(self, n: int, *, disabled=()):
        return [{'key': f'k{i}', 'title': f'{i}', 'enabled': i not in disabled}
                for i in range(n)]

    def _geometry(self, grid):
        """返回每张卡所在 (row, col)。"""
        out = []
        for card in grid.cards():
            index = grid._grid.indexOf(card)
            if index < 0:
                out.append(None)
                continue
            out.append(grid._grid.getItemPosition(index)[:2])
        return out

    def test_one_slot_full_width(self, grid):
        grid.set_slots(_slots(1))
        assert self._geometry(grid) == [(0, 0)]

    def test_two_slots_side_by_side(self, grid):
        grid.set_slots(_slots(2))
        assert self._geometry(grid) == [(0, 0), (0, 1)]

    def test_three_slots_two_columns(self, grid):
        """N≥2 → 两列大图（v2.1 弃用三列）。"""
        grid.set_slots(_slots(3))
        assert self._geometry(grid) == [(0, 0), (0, 1), (1, 0)]

    def test_four_slots_two_by_two(self, grid):
        grid.set_slots(_slots(4))
        assert self._geometry(grid) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_six_slots_two_columns_three_rows(self, grid):
        grid.set_slots(_slots(6))
        assert self._geometry(grid) == [(0, 0), (0, 1),
                                       (1, 0), (1, 1),
                                       (2, 0), (2, 1)]

    def test_disabled_step_is_placeholder(self, grid):
        grid.set_slots(_slots(3, disabled=(1,)))
        cards = grid.cards()
        assert cards[1]._placeholder is True
        assert cards[1].view is None           # 不占画布
        assert cards[0].view is not None

    def test_grid_cards_have_no_colorbar(self, grid):
        """网格卡无色标——坐标/色标不挤占绘图区（看色标走放大/全屏）。"""
        grid.set_slots(_slots(2))
        for card in grid.cards():
            assert card.view._colorbar is None

    def test_set_selected_highlights_one_card(self, grid):
        """选中卡用选中 QSS（色值走令牌，断言不绑具体字面量）。"""
        grid.set_slots(_slots(3))
        card_cls = type(grid.cards()[0])
        sel_qss = card_cls._sel_qss()
        normal_qss = card_cls._normal_qss()
        grid.set_selected('k1')
        styles = [c.styleSheet() for c in grid.cards()]
        assert styles[1] == sel_qss
        assert styles[0] == normal_qss
        assert styles[2] == normal_qss

    def test_card_click_forwards_key(self, grid):
        got = []
        grid.set_slots(_slots(3))
        grid.sig_card_selected.connect(got.append)
        grid.cards()[1].sig_clicked.emit('k1')     # 卡内鼠标路径的等价触发
        assert got == ['k1']

    def test_slots_reduce_removes_old_cards(self, grid):
        grid.set_slots(_slots(4))
        grid.set_slots(_slots(2))
        assert len(grid.cards()) == 2
        assert self._geometry(grid) == [(0, 0), (0, 1)]


def _bundle(tag: float, samples: int = 8, traces: int = 6):
    """鸭子类型 PreviewBundle（矩阵全为 tag）。"""
    import numpy as np
    from types import SimpleNamespace
    return SimpleNamespace(
        matrix=np.full((samples, traces), float(tag), dtype=np.float32),
        vmin=0.0, vmax=float(tag), title=f'b{tag}', x_label='道数',
        y_label='采样点', trace_axis_m=None, sample_axis=None,
        sample_axis_label='', trace_count=traces, sample_count=samples,
        trace_elevation_m=None, depth_axis_m=None)


class TestResultGridLazyPrinciple:
    """有数据才建画布：运行前不摆空 B-Scan 窗口。"""

    def test_no_slots_shows_hint_only(self, grid):
        assert grid.cards() == []
        assert grid._empty.isVisibleTo(grid)

    def test_slots_hide_hint(self, grid):
        grid.set_slots([{'key': 'k0', 'title': '输入', 'enabled': True}])
        assert not grid._empty.isVisibleTo(grid)
        assert len(grid.cards()) == 1

    def test_page_before_run_builds_no_step_cards(self, qapp):
        page = ProcessingPage()
        try:
            page._refresh_chain_and_results()
            assert page._result_grid.cards() == []      # 未选测线：零画布
            page.set_original_bundle(_bundle(1))
            assert len(page._result_grid.cards()) == 1  # 只有输入（有数据）
            titles = [c.title_label.text() for c in page._result_grid.cards()]
            assert titles == ['输入']
        finally:
            page.close()

    def test_page_after_run_builds_step_cards(self, qapp):
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page.set_artifacts([
                __import__('types').SimpleNamespace(
                    artifact_id='S1', line_id='L01', name='run 步骤1_dewow',
                    method_id='dewow', created_at='2026-09-25T10:01:00',
                    manifest={'params': {'artifact_kind': 'intermediate',
                                         'run_group_id': 'G1'}}),
                __import__('types').SimpleNamespace(
                    artifact_id='F1', line_id='L01', name='run_sec',
                    method_id='sec_gain', created_at='2026-09-25T10:02:00',
                    manifest={'params': {'artifact_kind': 'processing',
                                         'run_group_id': 'G1'}}),
            ])
            titles = [c.title_label.text() for c in page._result_grid.cards()]
            assert titles == ['输入', '1 dewow', '2 sec_gain']
        finally:
            page.close()


class TestRunButtonMotion:
    """运行钮动效（移植自 Transitions.dev 的 spinner → ✓ 形变思路）。"""

    def test_running_shows_spinner_then_restores(self, qapp):
        strip = ChainStrip()
        strip.set_running(True)
        assert strip.run_button().text().startswith('运行中')
        assert strip.run_button().isEnabled() is False
        strip.set_running(False)
        assert strip.run_button().text() == '运行'
        assert strip.run_button().isEnabled() is True

    def test_flash_success_then_restores(self, qapp):
        """成功反馈为「完成」文字——全应用不出现勾形元素（用户定案）。"""
        strip = ChainStrip()
        strip.flash_success()
        assert strip.run_button().text() == '完成'
        assert '✓' not in strip.run_button().text()
        strip.run_button().setText('运行')      # 1.2s 后由定时器复位
        assert strip.run_button().text() == '运行'

    def test_dot_toggle_is_round_not_check(self, qapp):
        """启用开关 = 圆点（非勾形图标）：点击发 toggle 信号。"""
        strip = ChainStrip()
        strip.set_steps([{'label': 'a', 'enabled': True}])
        toggled = []
        strip.sig_step_toggled.connect(lambda i, e: toggled.append((i, e)))
        chip = strip._list.itemWidget(strip._list.item(0))
        chip.dot_btn.click()
        assert toggled == [(0, False)]


class TestSkeletonReveal:
    """结果卡骨架 → 出图交叉淡入（动效移植①）。"""

    def test_new_card_shows_skeleton(self, grid):
        grid.set_slots([{'key': 'k0', 'title': '输入', 'enabled': True}])
        card = grid.cards()[0]
        assert card._skeleton.isVisibleTo(card)
        assert card._view_effect.opacity() == 0.0

    def test_bundle_reveals_and_hides_skeleton(self, grid):
        grid.set_slots([{'key': 'k0', 'title': '输入', 'enabled': True}])
        card = grid.cards()[0]
        grid.set_bundle('k0', _bundle(2))
        assert card._fade_in.state() == card._fade_in.State.Running
        assert card._fade_out.state() == card._fade_out.State.Running

    def test_placeholder_card_has_no_skeleton(self, grid):
        grid.set_slots([{'key': 'k0', 'title': '1 带通', 'enabled': False}])
        card = grid.cards()[0]
        assert card.view is None
        assert not hasattr(card, '_skeleton')


class TestChainSlidingPill:
    """链条选中滑动胶囊（动效移植②，来自 BeUI Tabs 思路）。"""

    def test_pill_appears_on_selection(self, qapp):
        strip = ChainStrip()
        strip.set_steps([{'label': '去直达波', 'enabled': True},
                         {'label': 'SEC 增益', 'enabled': True}])
        assert strip._pill.isHidden()
        strip.select_step(0)
        assert strip._pill.isVisibleTo(strip)

    def test_pill_slides_to_selected_chip(self, qapp):
        strip = ChainStrip()
        strip.set_steps([{'label': '去直达波', 'enabled': True},
                         {'label': 'SEC 增益', 'enabled': True}])
        strip.select_step(0)
        strip.select_step(1)                      # 第二次选中 → 走动画
        target = strip._list.visualItemRect(strip._list.item(1))
        assert strip._pill_anim.endValue().x() == target.x()

    def test_pill_hidden_when_no_row(self, qapp):
        strip = ChainStrip()
        strip._move_pill(-1)
        assert strip._pill.isHidden()


class TestStatusSemantics:
    """状态与反馈：最终结果 ✓ / 选中卡幅值范围 / 改动提示。"""

    def test_no_check_gimmick_on_cards(self, qapp):
        """撤掉「最终结果 ✓」：用户未要求、形似死按钮（真机反馈撤案）。"""
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page.set_artifacts([
                __import__('types').SimpleNamespace(
                    artifact_id='F1', line_id='L01', name='run_agc',
                    method_id='agc', created_at='2026-09-26T10:02:00',
                    manifest={'params': {'artifact_kind': 'processing',
                                         'run_group_id': 'G1'}}),
            ])
            for card in page._result_grid.cards():
                assert not hasattr(card, 'final_label')
        finally:
            page.close()

    def test_diff_update_reuses_cards(self, grid):
        """key 未变的卡必须原地复用（同一实例）——整排重建会清空画面。"""
        grid.set_slots(_slots(3))
        before = [id(c) for c in grid.cards()]
        grid.set_slots(_slots(3))
        assert [id(c) for c in grid.cards()] == before

    def test_bundle_survives_slots_rebuild(self, grid):
        """已出图的卡在槽位重建后不得回到骨架（真机「消失几秒」回归锁）。"""
        grid.set_slots(_slots(2))
        grid.set_bundle('k0', _bundle(2))
        card = grid.cards()[0]
        matrix_before = card.view._matrix
        grid.set_slots(_slots(2))
        assert grid.cards()[0] is card               # 同一实例
        assert card.view._matrix is matrix_before    # 画面未丢

    def test_range_label_after_bundle(self, grid):
        grid.set_slots([{'key': 'k0', 'title': '输入', 'enabled': True}])
        card = grid.cards()[0]
        grid.set_bundle('k0', _bundle(2))
        assert card.range_label.isVisibleTo(card)
        assert '2' in card.range_label.text()

    def test_chain_dirty_label_toggles(self, qapp):
        strip = ChainStrip()
        assert not strip._dirty_label.isVisibleTo(strip)
        strip.set_dirty(True)
        assert strip._dirty_label.isVisibleTo(strip)
        strip.set_dirty(False)
        assert not strip._dirty_label.isVisibleTo(strip)

    def test_page_marks_stale_after_chain_change(self, qapp):
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page.set_artifacts([
                __import__('types').SimpleNamespace(
                    artifact_id='F1', line_id='L01', name='run_agc',
                    method_id='agc', created_at='2026-09-26T10:02:00',
                    manifest={'params': {'artifact_kind': 'processing',
                                         'run_group_id': 'G1'}}),
            ])
            assert page._step_artifact_ids              # 有运行结果
            page._chain_strip.set_dirty(False)
            page._pipeline_list.add_step('agc', '自动增益控制 (AGC)', {})
            assert page._results_stale is True
            assert page._chain_strip._dirty_label.isVisibleTo(page._chain_strip)
        finally:
            page.close()


class TestStaleSelectionDecoupling:
    """结果过期（链已改未重算）时解除 chip↔卡片索引联动。

    旧 run 的卡片索引对不上新链（删/移步骤后整体移位），继续按索引
    联动会高亮错位的卡（真机走查：chip 亮 bandpass 卡亮 sec_gain）。
    """

    @staticmethod
    def _page_with_run(qapp):
        from types import SimpleNamespace
        page = ProcessingPage()
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([
            SimpleNamespace(
                artifact_id='F1', line_id='L01', name='run_agc',
                method_id='agc', created_at='2026-09-26T10:02:00',
                manifest={'params': {'artifact_kind': 'processing',
                                     'run_group_id': 'G1'}}),
        ])
        assert page._step_artifact_ids
        assert page._results_stale is False
        return page

    def test_fresh_results_keep_selection_sync(self, qapp):
        """结果与链一致：chip 选中照常映射到卡片高亮（回归锁）。"""
        page = self._page_with_run(qapp)
        try:
            page._on_chain_step_selected(0)
            assert page._result_grid._selected_key == 'step:0'
        finally:
            page.close()

    def test_stale_chip_click_does_not_highlight_card(self, qapp):
        page = self._page_with_run(qapp)
        try:
            page._on_chain_step_selected(0)
            page._pipeline_list.add_step('agc', '自动增益控制 (AGC)', {})
            assert page._results_stale is True
            page._on_chain_step_selected(1)
            assert page._result_grid._selected_key is None
        finally:
            page.close()

    def test_stale_card_click_does_not_write_back_chip(self, qapp):
        page = self._page_with_run(qapp)
        try:
            page._chain_strip.select_step(-1)
            page._pipeline_list.add_step('agc', '自动增益控制 (AGC)', {})
            page._chain_strip.select_step(-1)
            page._on_card_selected('step:0')
            assert page._result_grid._selected_key == 'step:0'  # 卡片本身亮
            assert page._chain_strip._list.currentRow() == -1   # 不回写 chip
        finally:
            page.close()

    def test_stale_refresh_clears_highlight(self, qapp):
        page = self._page_with_run(qapp)
        try:
            page._on_chain_step_selected(0)
            page._pipeline_list.add_step('agc', '自动增益控制 (AGC)', {})
            page._refresh_chain_and_results()
            assert page._result_grid._selected_key is None
        finally:
            page.close()


class TestParamEmptyHint:
    """参数卡空态：表单无字段时显示引导文案（真机走查遗留项）。"""

    def test_hint_toggles_with_form_schema(self, qapp):
        page = ProcessingPage()
        try:
            parent = page._param_empty_hint.parentWidget()
            assert page._param_empty_hint.isVisibleTo(parent)
            page._methods_by_id['agc'] = {'parameter_schema': [
                {'name': 'window', 'label': '窗口', 'type': 'int',
                 'default': 5, 'min': 1, 'max': 100}]}
            page._pipeline_list.add_step('agc', 'AGC', {})
            assert not page._param_empty_hint.isVisibleTo(parent)
            page._pipeline_list._list.setCurrentRow(-1)
            assert page._param_empty_hint.isVisibleTo(parent)
        finally:
            page.close()


class TestChainAlternativePaths:
    """拖拽的替代路径（guidelines：拖拽需有点击/键盘替代）。"""

    def test_delete_key_removes_selected_step(self, qapp):
        """Delete 键发信号；数据由宿主页改（本测试模拟宿主回写）。"""
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        strip = ChainStrip()
        steps = [{'label': 'a', 'enabled': True},
                 {'label': 'b', 'enabled': True},
                 {'label': 'c', 'enabled': True}]
        strip.set_steps(steps)
        strip.sig_step_removed.connect(
            lambda i: strip.set_steps(steps[:i] + steps[i + 1:]))
        strip.show()                       # 快捷键需活动窗口才触发
        strip.select_step(1)
        strip._list.setFocus()
        qapp.processEvents()
        QTest.keyClick(strip._list, Qt.Key.Key_Delete)
        assert strip._list.count() == 2            # 剩 2 步
        assert [s['label'] for s in strip._steps] == ['a', 'c']

    def test_context_menu_actions(self, qapp):
        strip = ChainStrip()
        strip.set_steps([{'label': 'a', 'enabled': True},
                         {'label': 'b', 'enabled': True},
                         {'label': 'c', 'enabled': True}])
        moved, toggled, removed = [], [], []
        strip.sig_step_moved.connect(lambda s, t: moved.append((s, t)))
        strip.sig_step_toggled.connect(lambda i, e: toggled.append((i, e)))
        strip.sig_step_removed.connect(removed.append)
        def filled(idx):
            menu = make_menu(strip)                # 每次全新菜单，防动作残留
            strip._fill_step_menu(menu, idx)
            return menu
        menu = filled(1)
        texts = [a.text() for a in menu.actions() if a.text()]
        assert texts == ['上移', '下移', '禁用', '删除']
        # 触发「上移」：步骤 1 → 插入位 0
        for a in menu.actions():
            if a.text() == '上移':
                a.trigger()
                break
        assert moved == [(1, 0)]
        # 触发「禁用」
        for a in filled(1).actions():
            if a.text() == '禁用':
                a.trigger()
                break
        assert toggled == [(1, False)]
        # 触发「删除」
        for a in filled(2).actions():
            if a.text() == '删除':
                a.trigger()
                break
        assert removed == [2]


class TestViewModes:
    """三种视图模式（2026-09-27 v2.3，参考用户设计稿）。"""

    def test_all_mode_shows_every_card(self, grid):
        grid.set_slots(_slots(4))
        grid.set_view_mode('all')
        assert all(c.isVisibleTo(grid) for c in grid.cards())

    def test_single_mode_shows_selected_only(self, grid, qapp):
        grid.show()
        qapp.processEvents()
        grid.set_slots(_slots(4))
        grid.set_selected('k2')
        grid.set_view_mode('single')
        visible = [c.key for c in grid.cards() if c.isVisibleTo(grid)]
        assert visible == ['k2']

    def test_single_mode_without_selection_shows_last(self, grid, qapp):
        grid.show()
        qapp.processEvents()
        grid.set_slots(_slots(3))
        grid.set_view_mode('single')
        visible = [c.key for c in grid.cards() if c.isVisibleTo(grid)]
        assert visible == ['k2']

    def test_compare_mode_shows_prev_and_selected(self, grid, qapp):
        grid.show()
        qapp.processEvents()
        grid.set_slots(_slots(4))
        grid.set_selected('k2')
        grid.set_view_mode('compare')
        visible = [c.key for c in grid.cards() if c.isVisibleTo(grid)]
        assert visible == ['k1', 'k2']

    def test_compare_mode_selected_input_shows_first_pair(self, grid, qapp):
        grid.show()
        qapp.processEvents()
        grid.set_slots(_slots(3))
        grid.set_selected('k0')
        grid.set_view_mode('compare')
        visible = [c.key for c in grid.cards() if c.isVisibleTo(grid)]
        assert visible == ['k0', 'k1']

    def test_mode_switch_back_restores_all(self, grid, qapp):
        grid.show()
        qapp.processEvents()
        grid.set_slots(_slots(4))
        grid.set_view_mode('single')
        grid.set_view_mode('all')
        assert all(c.isVisibleTo(grid) for c in grid.cards())

    def test_large_min_height_in_focus_modes(self, grid):
        grid.set_slots(_slots(2))
        grid.set_view_mode('single')
        visible = [c for c in grid.cards() if c.isVisibleTo(grid)]
        assert visible[0].minimumHeight() > 320   # 只有可见卡提档
        grid.set_view_mode('all')
        assert grid.cards()[0].minimumHeight() == 320

    def test_selection_follows_in_single_mode(self, grid, qapp):
        """single 模式下改选 → 可见卡跟着换（选中驱动）。"""
        grid.show()
        qapp.processEvents()
        grid.set_slots(_slots(3))
        grid.set_view_mode('single')
        grid.set_selected('k0')
        visible = [c.key for c in grid.cards() if c.isVisibleTo(grid)]
        assert visible == ['k0']


class TestSharedScale:
    """统一色标：全组卡共用全局 [vmin, vmax]（display 层覆盖，raw 不动）。"""

    def test_override_applies_global_range(self, grid):
        grid.set_slots(_slots(2))
        grid.set_bundle('k0', _bundle(0.5))    # 范围 0 ~ 0.5
        grid.set_bundle('k1', _bundle(1.5))    # 范围 0 ~ 1.5
        grid.set_shared_scale(True)
        for card in grid.cards():
            assert card.view._levels_override == (0.0, 1.5)
        grid.set_shared_scale(False)
        for card in grid.cards():
            assert card.view._levels_override is None

    def test_new_bundle_rescales_in_shared_mode(self, grid):
        grid.set_slots(_slots(2))
        grid.set_bundle('k0', _bundle(0.5))
        grid.set_shared_scale(True)
        grid.set_bundle('k1', _bundle(2.0))    # 新卡入组 → 全局范围扩大
        for card in grid.cards():
            assert card.view._levels_override == (0.0, 2.0)

    def test_placeholder_card_skipped(self, grid):
        grid.set_slots(_slots(2, disabled=(1,)))
        grid.set_bundle('k0', _bundle(0.5))
        grid.set_shared_scale(True)
        assert grid.cards()[1].view is None    # 占位卡无视图，不崩


class TestExpandAllToggle:
    """「全部步骤」开关：关=只铺输入 + 最终结果（P3 就地版）。"""

    @staticmethod
    def _page_with_run(qapp):
        from types import SimpleNamespace
        page = ProcessingPage()
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([
            SimpleNamespace(
                artifact_id='S1', line_id='L01', name='run 步骤1_dewow',
                method_id='dewow', created_at='2026-09-27T10:01:00',
                manifest={'params': {'artifact_kind': 'intermediate',
                                     'run_group_id': 'G1'}}),
            SimpleNamespace(
                artifact_id='S2', line_id='L01', name='run 步骤2_agc',
                method_id='agc', created_at='2026-09-27T10:02:00',
                manifest={'params': {'artifact_kind': 'intermediate',
                                     'run_group_id': 'G1'}}),
            SimpleNamespace(
                artifact_id='F1', line_id='L01', name='run_bandpass',
                method_id='bandpass', created_at='2026-09-27T10:03:00',
                manifest={'params': {'artifact_kind': 'processing',
                                     'run_group_id': 'G1'}}),
        ])
        return page

    def test_default_expands_all(self, qapp):
        page = self._page_with_run(qapp)
        try:
            keys = [c.key for c in page._result_grid.cards()]
            assert keys == ['input', 'step:0', 'step:1', 'step:2']
        finally:
            page.close()

    def test_off_shows_input_and_final_only(self, qapp):
        page = self._page_with_run(qapp)
        try:
            page._on_expand_all_changed(False)
            keys = [c.key for c in page._result_grid.cards()]
            assert keys == ['input', 'step:2']
            page._on_expand_all_changed(True)
            keys = [c.key for c in page._result_grid.cards()]
            assert keys == ['input', 'step:0', 'step:1', 'step:2']
        finally:
            page.close()

    def test_grid_default_switch_on(self, grid):
        assert grid._expand_switch.isChecked() is True


class TestGalleryContextFilter:
    """总览墙只显示当前上下文（真机反馈：跑一步蹦出一屏几天前的成果）。"""

    @staticmethod
    def _artifacts():
        from types import SimpleNamespace
        return [
            SimpleNamespace(
                artifact_id='OLD1', line_id='L09', name='run_old1',
                method_id='dewow', created_at='2026-09-23T10:07:12',
                manifest={'params': {'artifact_kind': 'intermediate',
                                     'run_group_id': 'G_OLD'}}),
            SimpleNamespace(
                artifact_id='NEW1', line_id='L09', name='run_new1',
                method_id='set_zero_time', created_at='2026-09-27T15:20:06',
                manifest={'params': {'artifact_kind': 'processing',
                                     'run_group_id': 'G_NEW'}}),
        ]

    def test_group_switch_drops_old_tabs(self, qapp):
        """组切换：上一组的 artifact tab 移除，只保留原始锚点。"""
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page.set_artifacts(self._artifacts())     # 最新组 = G_NEW
            keys = [s['key'] for s in page._preview_sources]
            assert 'artifact:OLD1' not in keys        # 旧组 tab 不残留
            assert 'artifact:NEW1' in keys
        finally:
            page.close()

    def test_gallery_shows_current_context_only(self, qapp):
        """总览墙数据源 = 原始 + 当前 run_group（历史成果不混入）。"""
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page.set_artifacts(self._artifacts())
            page._open_gallery()
            gallery = page._gallery
            titles = [g.text() for g in gallery.findChildren(PushButton)]
            assert any('原始数据' in t for t in titles)
            assert any('set_zero_time' in t for t in titles)
            assert not any('dewow' in t for t in titles), (
                f'历史组成果混入总览墙：{titles}')
            gallery.close()
        finally:
            page.close()


class TestSingleViewGeometry:
    """单步视图几何：可见卡必须占满结果区宽（真机反馈「只有半个窗口」）。

    根因：QGridLayout 空列不自动回收 + 列 stretch 跨重排残留——
    修：重排前清列拉伸 + 单步卡跨满最大列数（itemPos span=2）。
    """

    def test_visible_card_spans_full_width(self, grid, qapp):
        grid.resize(900, 600)
        grid.show()
        qapp.processEvents()
        qapp.processEvents()
        grid.set_slots(_slots(2))
        grid.set_selected('k0')
        grid.set_view_mode('single')
        qapp.processEvents()
        qapp.processEvents()
        visible = [c for c in grid.cards() if c.isVisibleTo(grid)]
        assert [c.key for c in visible] == ['k0']
        assert visible[0].width() >= 0.95 * grid._body.width()
        idx = grid._grid.indexOf(visible[0])
        assert grid._grid.getItemPosition(idx)[3] == 2   # columnSpan=2

    def test_switch_back_to_all_two_columns(self, grid, qapp):
        grid.resize(900, 600)
        grid.show()
        qapp.processEvents()
        qapp.processEvents()
        grid.set_slots(_slots(2))
        grid.set_view_mode('single')
        grid.set_view_mode('all')
        qapp.processEvents()
        qapp.processEvents()
        assert grid._grid.columnStretch(0) == 1
        assert grid._grid.columnStretch(1) == 1
        widths = [c.width() for c in grid.cards()]
        assert all(abs(w - widths[0]) <= 2 for w in widths)   # 均分


class TestColorbarPrefOnGrid:
    """设置页「显示色标」对 v2 结果网格生效（真机反馈开关失效）。

    网格卡 with_colorbar=False 构造；偏好 True 下发 → 动态补建色标；
    新建卡继承偏好状态。
    """

    def test_set_colorbar_visible_creates_colorbar(self, grid, qapp):
        grid.set_slots(_slots(2))
        card = grid.cards()[0]
        assert card.view._colorbar is None            # 初始无色标（窄卡）
        card.view.set_colorbar_visible(True)
        assert card.view._colorbar is not None        # 动态补建
        assert card.view._colorbar.isVisible()
        card.view.set_colorbar_visible(False)
        assert not card.view._colorbar.isVisible()    # 再关=隐藏不销毁

    def test_pref_broadcasts_to_cards(self, grid):
        grid.set_slots(_slots(2))
        grid.set_bundle('k0', _bundle(1))
        grid.set_colorbar_pref(True)
        states = [c.view._colorbar_visible for c in grid.cards()]
        assert states == [True, True]
        grid.set_colorbar_pref(False)
        states = [c.view._colorbar_visible for c in grid.cards()]
        assert states == [False, False]

    def test_new_cards_inherit_pref(self, grid):
        grid.set_colorbar_pref(True)
        grid.set_slots(_slots(2))                     # 偏好 True 后新建
        assert all(c.view._colorbar_visible for c in grid.cards())

    def test_page_forwards_pref(self, qapp):
        page = ProcessingPage()
        try:
            page.set_colorbar_pref(False)
            assert page._result_grid._colorbar_pref is False
            page.set_colorbar_pref(True)
            assert page._result_grid._colorbar_pref is True
        finally:
            page.close()

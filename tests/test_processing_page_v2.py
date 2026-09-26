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

    def test_chips_are_input_plus_steps(self, strip):
        """输入 chip 固定首位 + 每步一个 chip。"""
        assert strip._list.count() == 4

    def test_selection_offsets_input_chip(self, strip):
        got = []
        strip.sig_step_selected.connect(got.append)
        strip._list.setCurrentRow(2)          # 第 2 个步骤
        assert got == [1]

    def test_toggle_and_remove_signals(self, strip):
        toggled, removed = [], []
        strip.sig_step_toggled.connect(lambda i, e: toggled.append((i, e)))
        strip.sig_step_removed.connect(removed.append)
        strip._on_dot_clicked(1)              # 第 2 步：True → False
        strip._on_chip_deleted(0)
        assert toggled == [(1, False)]
        assert removed == [0]

    def test_drop_event_maps_to_step_indices(self, strip):
        got = []
        strip.sig_step_moved.connect(lambda s, t: got.append((s, t)))
        strip._list.setCurrentRow(1)          # 步骤 0
        item = strip._list.item(3)            # 步骤 2（+1 偏移）
        rect = strip._list.visualItemRect(item)
        strip._list_drop_event(_FakeDropEvent(rect.left() + 2, rect.center().y()))
        assert got and got[0][0] == 0         # 源 = 步骤 0

    def test_select_step_moves_highlight(self, strip):
        strip.select_step(2)
        assert strip._list.currentRow() == 3


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
        grid.set_slots(_slots(3))
        grid.set_selected('k1')
        styles = [c.styleSheet() for c in grid.cards()]
        assert 'rgba(90,156,216' in styles[1]
        assert 'rgba(90,156,216' not in styles[0]
        assert 'rgba(90,156,216' not in styles[2]

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
        strip = ChainStrip()
        strip.flash_success()
        assert strip.run_button().text() == '✓ 完成'
        strip.run_button().setText('运行')      # 1.2s 后由定时器复位
        assert strip.run_button().text() == '运行'


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
        target = strip._list.visualItemRect(strip._list.item(2))
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
        assert strip._list.count() == 3            # 输入 + 2 步
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

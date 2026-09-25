# -*- coding: utf-8 -*-
"""处理页 v2 组件契约：ChainStrip 手势 + ResultGrid 列数规则。

列数规则（用户拍板）：N≤3 → N 列；N≥4 → min(⌈√N⌉, 3)
（2 张横排 / 3 张三列 / 4 张两行两列 / 6 张三列两行）。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from PyQt6.QtCore import QPointF  # noqa: E402
from ui.widgets.bscan_result_grid import ResultGrid  # noqa: E402
from ui.widgets.chain_strip import ChainStrip  # noqa: E402


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
    @pytest.fixture
    def grid(self, qapp):
        return ResultGrid()

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
        grid.set_slots(self._slots(1))
        assert self._geometry(grid) == [(0, 0)]

    def test_two_slots_side_by_side(self, grid):
        grid.set_slots(self._slots(2))
        assert self._geometry(grid) == [(0, 0), (0, 1)]

    def test_three_slots_one_row(self, grid):
        grid.set_slots(self._slots(3))
        assert self._geometry(grid) == [(0, 0), (0, 1), (0, 2)]

    def test_four_slots_two_by_two(self, grid):
        grid.set_slots(self._slots(4))
        assert self._geometry(grid) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_six_slots_three_columns(self, grid):
        grid.set_slots(self._slots(6))
        assert self._geometry(grid) == [(0, 0), (0, 1), (0, 2),
                                       (1, 0), (1, 1), (1, 2)]

    def test_disabled_step_is_placeholder(self, grid):
        grid.set_slots(self._slots(3, disabled=(1,)))
        cards = grid.cards()
        assert cards[1]._placeholder is True
        assert cards[1].view is None           # 不占画布
        assert cards[0].view is not None

    def test_slots_reduce_removes_old_cards(self, grid):
        grid.set_slots(self._slots(4))
        grid.set_slots(self._slots(2))
        assert len(grid.cards()) == 2
        assert self._geometry(grid) == [(0, 0), (0, 1)]

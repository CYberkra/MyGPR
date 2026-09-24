# -*- coding: utf-8 -*-
"""PipelineList 拖拽排序：落点语义 / 行控件保活 / 顺序广播。

拖拽实现契约（2026-09-24）：启用 InternalMove 拿拖拽视觉，但 dropEvent
由 _StepListWidget 委托宿主接管——**不调 Qt 的默认实现**（它挪 item 会
丢/错位 setItemWidget 行控件）；重排走 _steps 数据源整体重建。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtWidgets import QAbstractItemView  # noqa: E402
from ui.widgets.pipeline_list import PipelineList  # noqa: E402


class _FakeDropEvent:
    """伪造 QDropEvent：只暴露 dropEvent 用到的两个接口。"""

    def __init__(self, x: float, y: float):
        self._pos = QPointF(x, y)
        self.action = None
        self.accepted = False

    def position(self) -> QPointF:
        return self._pos

    def setDropAction(self, action) -> None:
        self.action = action

    def accept(self) -> None:
        self.accepted = True


def _steps():
    return [{'method_id': f'm{i}', 'label': f'算法{i}', 'params': {},
             'enabled': True} for i in range(1, 6)]   # 5 步


@pytest.fixture
def widget(qapp):
    w = PipelineList()
    w.set_steps(_steps())
    return w


class TestDragReorder:
    def test_drag_drop_mode_enabled(self, widget):
        """InternalMove 只为拖拽视觉；落点重排由宿主接管。"""
        assert (widget._list.dragDropMode()
                == QAbstractItemView.DragDropMode.InternalMove)

    def test_move_to_middle_reorders(self, widget):
        """源 0 → 目标行 3（上半）：插入到行 2 之后。"""
        got = []
        widget.sig_changed.connect(lambda: got.append(1))
        assert widget._move_step_to(0, 3) is True
        assert [s['method_id'] for s in widget.steps()] == \
            ['m2', 'm3', 'm1', 'm4', 'm5']
        assert got == [1]
        assert widget._list.currentRow() == 2        # 拖动的步骤保持选中

    def test_move_down_uses_insert_semantics(self, widget):
        """源 0 → 目标行 4（下半）：跨 3 行，插到行 3 之后。"""
        widget._move_step_to(0, 4)
        assert [s['method_id'] for s in widget.steps()] == \
            ['m2', 'm3', 'm4', 'm1', 'm5']

    def test_noop_positions(self, widget):
        """落在本位/本位+1 → 无位移、不发信号。"""
        got = []
        widget.sig_changed.connect(lambda: got.append(1))
        assert widget._move_step_to(1, 1) is False
        assert widget._move_step_to(1, 2) is False
        assert widget.steps()[1]['method_id'] == 'm2'
        assert got == []

    def test_row_widgets_survive_reorder(self, widget):
        """重排后每行仍带可用的行控件（enable 框勾选态跟数据走）。"""
        widget._steps[0]['enabled'] = False
        widget._move_step_to(0, 3)
        row = widget._list.itemWidget(widget._list.item(2))
        assert row is not None
        assert row.enabled_box.isChecked() is False

    def test_drop_event_noop_without_selection(self, widget):
        """无选中行（source<0）的拖放直接忽略。"""
        event = _FakeDropEvent(10, 10)
        widget._list_drop_event(event)
        assert event.accepted is False


class TestDropEventDispatch:
    def test_drop_event_reorders_and_accepts(self, widget, qapp):
        """拖行 0 到行 2 中心（上半）→ 插入行 1 之后，事件被接管。"""
        widget._list.setCurrentRow(0)
        item = widget._list.item(2)
        rect = widget._list.visualItemRect(item)
        event = _FakeDropEvent(rect.center().x(), rect.top() + 2)
        widget._list_drop_event(event)
        assert event.accepted is True
        assert [s['method_id'] for s in widget.steps()] == \
            ['m2', 'm1', 'm3', 'm4', 'm5']

    def test_drop_below_center_inserts_after(self, widget, qapp):
        """落点在行下半 → 插到该行之后（行 2 下半 → 顺序不变：1→2+1 无位移）。"""
        widget._list.setCurrentRow(0)
        item = widget._list.item(2)
        rect = widget._list.visualItemRect(item)
        event = _FakeDropEvent(rect.center().x(), rect.bottom() - 2)
        widget._list_drop_event(event)
        assert [s['method_id'] for s in widget.steps()] == \
            ['m2', 'm3', 'm1', 'm4', 'm5']

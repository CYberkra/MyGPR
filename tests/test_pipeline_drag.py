# -*- coding: utf-8 -*-
"""处理链拖拽：行内重排数据回收、方法库拖入插入、insert_step、拖出 MIME。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QMimeData, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QDropEvent  # noqa: E402

from ui.widgets.pipeline_list import METHOD_MIME, PipelineList  # noqa: E402

_STEPS = [
    {'method_id': 'dewow', 'label': '去直流', 'params': {}, 'enabled': True},
    {'method_id': 'agc', 'label': 'AGC 增益', 'params': {}, 'enabled': True},
    {'method_id': 'bp', 'label': '带通滤波', 'params': {}, 'enabled': True},
]


def _steps():
    return [dict(s, params=dict(s['params'])) for s in _STEPS]


# ---------------------------------------------------------------- 行内重排
def test_drag_reorder_syncs_steps_and_keeps_selection(qapp):
    """模拟 InternalMove 落点：takeItem/insertItem 后按 item 序号回收顺序。"""
    widget = PipelineList()
    widget.set_steps(_steps())
    item = widget._list.takeItem(0)
    widget._list.insertItem(2, item)
    widget._sync_order_after_drag(0)  # 被拖步骤的原始序号
    assert [s['method_id'] for s in widget.steps()] == ['agc', 'bp', 'dewow']
    assert widget._list.currentRow() == 2  # 被拖步骤保持选中


def test_drag_sync_aborts_on_count_mismatch(qapp):
    """序号集合对不上（拖拽被打断）：不采纳列表序，按原数据重建。"""
    widget = PipelineList()
    widget.set_steps(_steps())
    item = widget._list.takeItem(0)
    item.setData(Qt.ItemDataRole.UserRole, None)  # 序号丢失
    widget._list.insertItem(2, item)
    widget._sync_order_after_drag(0)
    assert [s['method_id'] for s in widget.steps()] == ['dewow', 'agc', 'bp']


def test_drag_reorder_emits_changed(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    fired = []
    widget.sig_changed.connect(lambda: fired.append(1))
    item = widget._list.takeItem(2)
    widget._list.insertItem(0, item)
    widget._sync_order_after_drag(2)
    assert fired == [1]
    assert [s['method_id'] for s in widget.steps()] == ['bp', 'dewow', 'agc']


# ---------------------------------------------------------------- 方法拖入
def _drop_event(mime, pos=(5, 5)):
    return QDropEvent(QPointF(*pos), Qt.DropAction.CopyAction, mime,
                      Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier)


def test_method_drop_emits_insert_request(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    got = []
    widget.sig_method_drop_requested.connect(
        lambda m, r: got.append((m, r)))
    mime = QMimeData()
    mime.setData(METHOD_MIME, 'gain'.encode('utf-8'))
    widget._list.dropEvent(_drop_event(mime))
    assert len(got) == 1
    assert got[0][0] == 'gain'
    assert 0 <= got[0][1] <= 3


def test_row_at_blank_area_appends(qapp):
    """落点在列表空白处 → 末尾。"""
    widget = PipelineList()
    assert widget._list._row_at(widget._list.rect().bottomRight()) == 0


def test_foreign_mime_falls_back_to_internal_move(qapp):
    """非方法 MIME 走 super()（行内移动路径），不发插入请求。"""
    widget = PipelineList()
    widget.set_steps(_steps())
    got = []
    widget.sig_method_drop_requested.connect(
        lambda m, r: got.append((m, r)))
    mime = QMimeData()
    mime.setData('text/plain', b'hello')
    widget._list.dropEvent(_drop_event(mime))
    assert got == []


# ---------------------------------------------------------------- 插入/追加
def test_insert_step_inserts_at_index_and_selects(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    widget.insert_step(1, 'fk', 'FK 滤波', {})
    assert [s['method_id'] for s in widget.steps()] == \
        ['dewow', 'fk', 'agc', 'bp']
    assert widget._list.currentRow() == 1
    assert widget.steps()[1]['enabled'] is True


def test_insert_step_clamps_out_of_range(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    widget.insert_step(99, 'fk', 'FK 滤波', {})
    assert widget.steps()[-1]['method_id'] == 'fk'


def test_add_step_still_appends(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    widget.add_step('fk', 'FK 滤波', {})
    assert [s['method_id'] for s in widget.steps()] == \
        ['dewow', 'agc', 'bp', 'fk']


# ---------------------------------------------------------------- 拖拽源
def test_method_browser_leaf_drag_mime(qapp):
    from ui.widgets.method_browser import MethodBrowser
    browser = MethodBrowser()
    browser.set_methods([{
        'method_id': 'dewow', 'display_name': '去直流',
        'category_label': '预处理', 'tags': [], 'parameter_schema': [],
    }])
    top = browser._tree.topLevelItem(0)
    leaf = top.child(0)
    mime = browser._tree.mimeData([leaf])
    assert bytes(mime.data(METHOD_MIME)).decode('utf-8') == 'dewow'
    # 分类行拖不出载荷
    assert not browser._tree.mimeData([top]).hasFormat(METHOD_MIME)


def test_method_browser_drag_flags(qapp):
    from ui.widgets.method_browser import MethodBrowser
    browser = MethodBrowser()
    browser.set_methods([{
        'method_id': 'dewow', 'display_name': '去直流',
        'category_label': '预处理', 'tags': [], 'parameter_schema': [],
    }])
    top = browser._tree.topLevelItem(0)
    leaf = top.child(0)
    assert not (top.flags() & Qt.ItemFlag.ItemIsDragEnabled)
    assert leaf.flags() & Qt.ItemFlag.ItemIsDragEnabled
    # 本树不接收落下
    assert not (leaf.flags() & Qt.ItemFlag.ItemIsDropEnabled)

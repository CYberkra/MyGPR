# -*- coding: utf-8 -*-
"""处理链删除撤销：快照回插、保持原索引/启用态/参数、栈清理与上限。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6")

from ui.widgets.pipeline_list import PipelineList  # noqa: E402

_STEPS = [
    {'method_id': 'dewow', 'label': '去直流', 'params': {'w': 5}, 'enabled': True},
    {'method_id': 'agc', 'label': 'AGC 增益', 'params': {'n': 50}, 'enabled': False},
    {'method_id': 'bp', 'label': '带通滤波', 'params': {}, 'enabled': True},
]


def _steps():
    return [dict(s, params=dict(s['params'])) for s in _STEPS]


def test_undo_restores_step_at_original_index(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    widget._remove_step(0)
    assert [s['method_id'] for s in widget.steps()] == ['agc', 'bp']

    assert widget.undo_delete() is True

    assert [s['method_id'] for s in widget.steps()] == ['dewow', 'agc', 'bp']
    assert widget._list.currentRow() == 0          # 恢复行保持选中


def test_undo_preserves_params_and_enabled(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    widget._remove_step(1)

    widget.undo_delete()

    restored = widget.steps()[1]
    assert restored['method_id'] == 'agc'
    assert restored['enabled'] is False            # 原启用态（insert_step 会强制 True）
    assert restored['params'] == {'n': 50}


def test_undo_snapshot_isolated_from_later_edits(qapp):
    """删除后编辑其他步骤参数，不得污染恢复数据。"""
    widget = PipelineList()
    widget.set_steps(_steps())
    widget._remove_step(1)
    widget.update_step_params(1, {'n': 999})       # 此时索引 1 已是 bp

    widget.undo_delete()

    assert widget.steps()[1]['params'] == {'n': 50}
    assert widget.steps()[2]['params'] == {'n': 999}


def test_undo_stack_is_lifo(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    widget._remove_step(0)     # 删 dewow
    widget._remove_step(0)     # 删 agc
    assert [s['method_id'] for s in widget.steps()] == ['bp']

    widget.undo_delete()       # 后删的先恢复
    widget.undo_delete()

    assert [s['method_id'] for s in widget.steps()] == ['dewow', 'agc', 'bp']


def test_undo_empty_stack_returns_false(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    assert widget.can_undo() is False
    assert widget.undo_delete() is False


def test_set_steps_clears_undo_stack(qapp):
    """整体替换 = 新编辑会话，旧撤销快照作废。"""
    widget = PipelineList()
    widget.set_steps(_steps())
    widget._remove_step(0)
    assert widget.can_undo() is True

    widget.set_steps([{'method_id': 'fk', 'label': 'FK', 'params': {}}])
    assert widget.can_undo() is False
    assert widget.undo_delete() is False


def test_undo_clamps_index_when_list_shrunk(qapp):
    """恢复索引越界（期间又删了尾部）→ 夹取到末尾，不崩溃。"""
    widget = PipelineList()
    widget.set_steps(_steps())
    widget._remove_step(0)          # 撤销目标索引 0，快照 = dewow
    widget._remove_step(0)          # 再删 agc
    widget.undo_delete()            # 恢复 agc 到 0
    widget._undo_stack[0] = (99, dict(widget._undo_stack[0][1]))  # 人为越界
    widget.undo_delete()
    assert widget.steps()[-1]['method_id'] == 'dewow'


def test_remove_emits_changed(qapp):
    widget = PipelineList()
    widget.set_steps(_steps())
    fired = []
    widget.sig_changed.connect(lambda: fired.append(1))
    widget._remove_step(1)
    widget.undo_delete()
    assert fired == [1, 1]

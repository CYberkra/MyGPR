# -*- coding: utf-8 -*-
"""StepEditorPopup 契约（P0-3）：启停 pill / 图标动作栏（首末步禁用态、
删除红色警示）/ 参数表单（恢复默认、区内滚动）/ 应用与丢弃；
结构动作即发即关信号；处理页接线（右键 → 弹窗 → 写回 PipelineList）。

offscreen 平台下真实焦点移交不可靠（架构共享知识 5.2）：外部点击关闭
走内部复核函数 ``_close_if_focus_left`` 契约，真机走查补外部点击。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QKeyEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui.widgets.step_editor import StepEditorPopup  # noqa: E402

_SCHEMA = [
    {'name': 'window', 'label': '窗口', 'type': 'int',
     'default': 5, 'min': 1, 'max': 100},
    {'name': 'gain', 'label': '增益', 'type': 'float',
     'default': 2.0, 'min': 0.0, 'max': 10.0},
]


def _snapshot(**overrides) -> dict:
    snapshot = {
        'title': '去直流滤波',
        'method_id': 'dewow',
        'params': {'window': 9, 'gain': 3.5},
        'enabled': True,
        'is_first': False,
        'is_last': False,
        'schema': list(_SCHEMA),
    }
    snapshot.update(overrides)
    return snapshot


@pytest.fixture
def popup(qapp):
    w = StepEditorPopup()
    yield w
    w.close()
    w.deleteLater()


def _press_escape(widget) -> None:
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape,
                      Qt.KeyboardModifier.NoModifier)
    widget.keyPressEvent(event)


class TestPopupWindowContract:
    def test_non_modal_tool_window(self, popup):
        assert popup.isModal() is False
        assert popup.windowFlags() & Qt.WindowType.Tool
        assert popup.width() == 380            # Q4 定案宽度

    def test_all_actions_have_tooltips(self, popup):
        for name, btn in popup._action_buttons.items():
            assert btn.toolTip(), f'{name} 动作缺少 tooltip'
        assert popup._enable_switch.toolTip()

    def test_remove_button_is_danger_red(self, popup):
        """删除红色警示：error 令牌（不跳色铁律），随主题重刷。"""
        from ui.design_tokens import color

        assert color('error') in popup._remove_btn.styleSheet()
        popup.apply_theme(True)
        assert color('error', True) in popup._remove_btn.styleSheet()


class TestSnapshotLoad:
    def test_load_snapshot_fills_form_and_switch(self, popup):
        popup.load_snapshot(1, _snapshot(enabled=False))
        assert popup._index == 1
        assert popup._title_label.text() == '去直流滤波'
        assert popup._enable_switch.isChecked() is False
        assert popup._form.values() == {'window': 9, 'gain': 3.5}
        assert popup._scroll.isVisibleTo(popup) is True
        assert popup._param_hint.isVisibleTo(popup) is False

    def test_first_step_disables_up_last_disables_down(self, popup):
        popup.load_snapshot(0, _snapshot(is_first=True))
        assert popup._up_btn.isEnabled() is False
        assert popup._down_btn.isEnabled() is True
        popup.load_snapshot(2, _snapshot(is_last=True))
        assert popup._up_btn.isEnabled() is True
        assert popup._down_btn.isEnabled() is False

    def test_empty_schema_shows_hint(self, popup):
        popup.load_snapshot(0, _snapshot(schema=[]))
        assert popup._param_hint.isVisibleTo(popup) is True
        assert popup._scroll.isVisibleTo(popup) is False


class TestApplyAndDiscard:
    def test_apply_emits_payload_and_closes(self, popup, qapp):
        got = []
        popup.sig_applied.connect(got.append)
        popup.load_snapshot(1, _snapshot(enabled=False))
        popup.show()
        qapp.processEvents()
        # 弹窗内改参数 + 切 pill：只有「应用」写回
        popup._form.set_values({'window': 21})
        popup._apply_btn.click()
        assert got == [{'index': 1, 'params': {'window': 21, 'gain': 3.5},
                        'enabled': False}]
        assert not popup.isVisible()

    def test_reset_defaults_restores_schema_defaults(self, popup):
        popup.load_snapshot(1, _snapshot(params={'window': 99, 'gain': 8.0}))
        popup._reset_btn.click()
        assert popup._form.values() == {'window': 5, 'gain': 2.0}

    def test_discard_on_close_emits_nothing(self, popup, qapp):
        """Esc 关闭：未应用改动随窗丢弃（数据从未离开弹窗）。"""
        applied, moved = [], []
        popup.sig_applied.connect(applied.append)
        popup.sig_move_requested.connect(lambda *a: moved.append(a))
        popup.load_snapshot(1, _snapshot())
        popup.show()
        qapp.processEvents()
        popup._form.set_values({'window': 42})
        _press_escape(popup)
        assert not popup.isVisible()
        assert applied == []
        assert moved == []

    def test_param_area_scrolls_when_tall(self, popup):
        """参数超区高：区内滚动，窗体高度不爆炸（Q4 定案 320px）。"""
        from ui.widgets.step_editor import _PARAM_AREA_MAX_H

        schema = [dict(item, name=f'p{i}', label=f'参数{i}')
                  for i, item in enumerate(_SCHEMA * 6)]
        popup.load_snapshot(0, _snapshot(schema=schema))
        assert popup._scroll.maximumHeight() <= _PARAM_AREA_MAX_H


class TestStructuralActions:
    @pytest.mark.parametrize('action,signal,args', [
        ('up', 'sig_move_requested', (1, 0)),
        ('down', 'sig_move_requested', (1, 3)),
        ('duplicate', 'sig_duplicate_requested', (1,)),
        ('replace', 'sig_replace_requested', (1,)),
        ('remove', 'sig_remove_requested', (1,)),
    ])
    def test_action_emits_and_closes(self, popup, qapp, action, signal, args):
        got = []
        getattr(popup, signal).connect(lambda *a: got.append(a))
        popup.load_snapshot(1, _snapshot())
        popup.show()
        qapp.processEvents()
        popup._action_buttons[action].click()
        assert got == [args]
        assert not popup.isVisible()          # 结构动作即发即关

    def test_focus_recheck_closes_when_focus_left(self, popup, qapp,
                                                  monkeypatch):
        popup.show()
        qapp.processEvents()
        monkeypatch.setattr(QApplication, 'focusWidget',
                            classmethod(lambda cls: None))
        popup._close_if_focus_left()
        assert not popup.isVisible()


# ============================================================ 页面接线
def _page_with_methods(qapp):
    from ui.pages.processing_page import ProcessingPage

    page = ProcessingPage()
    page.set_methods([
        {'method_id': 'dewow', 'name': 'dewow', 'display_name': '去直流滤波',
         'category': 'filter', 'category_label': '滤波', 'tags': [],
         'parameter_schema': list(_SCHEMA)},
        {'method_id': 'agc', 'name': 'agc', 'display_name': '自动增益控制',
         'category': 'gain', 'category_label': '增益', 'tags': [],
         'parameter_schema': [{'name': 'window', 'label': '窗口',
                               'type': 'int', 'default': 7,
                               'min': 1, 'max': 100}]},
    ])
    return page


class TestPageStepEditorWiring:
    def test_chain_right_click_opens_editor(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._chain_strip.sig_step_editor_requested.emit(0)
            assert page._step_editor is not None
            assert page._step_editor._title_label.text() == '去直流滤波'
            assert page._step_editor.isVisible()
        finally:
            page.close()

    def test_apply_writes_params_and_emits_sig_changed(self, qapp):
        """「应用」→ update_step_params 自动 sig_changed（琥珀 hint 链路原料）。"""
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            emitted: list = []
            page._pipeline_list.sig_changed.connect(lambda: emitted.append(1))
            page._open_step_editor(0)
            page._step_editor._form.set_values({'window': 33})
            page._step_editor._apply_btn.click()
            assert page._pipeline_list.steps()[0]['params']['window'] == 33
            assert emitted, '应用必须经 sig_changed 点亮琥珀 hint'
            assert page._step_editor is None      # 应用后关窗、引用释放
        finally:
            page.close()

    def test_apply_enabled_toggle_uses_existing_path(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._open_step_editor(0)
            page._step_editor._enable_switch.setChecked(False)
            page._step_editor._apply_btn.click()
            steps = page._pipeline_list.steps()
            assert steps[0]['enabled'] is False
            assert steps[0]['params'] == {'window': 5, 'gain': 2.0}
            # chip 变暗视觉（虚线 muted）由 sig_changed → _refresh_chain_and_results 重铺
            assert page._chain_strip._steps[0]['enabled'] is False
        finally:
            page.close()

    def test_unapplied_params_not_written(self, qapp):
        """不点「应用」：参数改动不落 PipelineList（弹窗内快照语义）。"""
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._open_step_editor(0)
            page._step_editor._form.set_values({'window': 77})
            page._step_editor.close()
            assert page._pipeline_list.steps()[0]['params']['window'] == 5
        finally:
            page.close()

    def test_disable_reenable_preserves_params(self, qapp):
        """屏蔽 → 应用 → 再启用 → 应用：参数配置全程保留（PRD A3②
        「屏蔽不丢配置」的往返钉扎）。"""
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._pipeline_list.update_step_params(
                0, {'window': 42, 'gain': 8.0})
            # 屏蔽（应用）
            page._open_step_editor(0)
            page._step_editor._enable_switch.setChecked(False)
            page._step_editor._apply_btn.click()
            steps = page._pipeline_list.steps()
            assert steps[0]['enabled'] is False
            assert steps[0]['params'] == {'window': 42, 'gain': 8.0}
            # 重新启用（应用）：弹窗快照里参数仍是屏蔽前的值
            page._open_step_editor(0)
            assert page._step_editor._enable_switch.isChecked() is False
            assert page._step_editor._form.values() == \
                {'window': 42, 'gain': 8.0}
            page._step_editor._enable_switch.setChecked(True)
            page._step_editor._apply_btn.click()
            steps = page._pipeline_list.steps()
            assert steps[0]['enabled'] is True
            assert steps[0]['params'] == {'window': 42, 'gain': 8.0}
        finally:
            page.close()

    def test_duplicate_step(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._add_method_to_pipeline('agc')
            emitted: list = []
            page._pipeline_list.sig_changed.connect(lambda: emitted.append(1))
            page._duplicate_step(0)
            steps = page._pipeline_list.steps()
            assert [s['method_id'] for s in steps] == ['dewow', 'dewow', 'agc']
            assert steps[1]['params'] == steps[0]['params']
            assert steps[1]['params'] is not steps[0]['params']  # 深拷贝隔离
            # 手动 emit 契约（set_steps 不发信号）：琥珀 hint 经 sig_changed 点亮
            assert emitted == [1]
        finally:
            page.close()

    def test_replace_step_method_keeps_position_resets_params(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._add_method_to_pipeline('agc')
            page._pipeline_list.update_step_params(0, {'window': 55})
            page._replace_step_method(0, 'agc')
            steps = page._pipeline_list.steps()
            assert [s['method_id'] for s in steps] == ['agc', 'agc']
            assert steps[0]['params'] == {'window': 7}   # 新 schema 默认
            assert steps[0]['enabled'] is True
        finally:
            page.close()

    def test_replace_action_opens_method_picker(self, qapp):
        """弹窗「替换算法」→ 选择器 replace 分流（P0-2 接口消费端）。"""
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._open_step_editor(0)
            page._step_editor._action_buttons['replace'].click()
            assert page._method_picker is not None
            assert page._picker_purpose == 'replace'
            assert page._picker_index == 0
            # 双击选择器 → 就地替换
            page._method_picker._browser.sig_add_requested.emit('agc')
            assert [s['method_id']
                    for s in page._pipeline_list.steps()] == ['agc']
        finally:
            page.close()

    def test_remove_action_forwards_to_chain_handler(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._add_method_to_pipeline('agc')
            page._open_step_editor(0)
            page._step_editor._action_buttons['remove'].click()
            assert [s['method_id']
                    for s in page._pipeline_list.steps()] == ['agc']
        finally:
            page.close()

    def test_move_action_forwards_to_chain_handler(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._add_method_to_pipeline('agc')
            page._open_step_editor(0)
            page._step_editor._action_buttons['down'].click()   # 插入位 2
            assert [s['method_id']
                    for s in page._pipeline_list.steps()] == ['agc', 'dewow']
        finally:
            page.close()

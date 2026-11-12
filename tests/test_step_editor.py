"""Inline step editing and shared chain commands."""
import pytest

pytest.importorskip('PyQt6')

from tests.test_processing_workspace_flow import bundle


_SCHEMA = [
    {'name': 'window', 'label': '窗口', 'type': 'int',
     'default': 5, 'min': 1, 'max': 100},
    {'name': 'gain', 'label': '增益', 'type': 'float',
     'default': 2.0, 'min': 0.0, 'max': 10.0},
]


def _trigger_step_command(page, index, command):
    from ui.widgets.context_menus import make_menu

    menu = make_menu(page)
    page._chain_strip._add_step_commands(menu, index)
    next(action for action in menu.actions() if action.data() == command).trigger()
    menu.deleteLater()


def _page_with_methods(qapp):
    from ui.pages.processing_page import ProcessingPage

    page = ProcessingPage()
    page.set_line_label('L01')
    page.set_original_bundle(bundle())
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
            page._pipeline_list.sig_edit_requested.emit(0)
            assert page._editor_uid == page._pipeline_list.step_uid(0)
            assert '去直流滤波' in page._step_tools_title.toolTip()
            assert page._param_form.isVisibleTo(page)
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
            page._param_form._editors['window'].setValue(33)
            assert page.current_pipeline()['steps'][0]['params']['window'] == 5
            page._commit_step_btn.click()
            assert page._pipeline_list.steps()[0]['params']['window'] == 33
            assert emitted, '应用必须经 sig_changed 点亮琥珀 hint'
            assert page._param_form.isVisibleTo(page)
            assert not page._commit_step_btn.isEnabled()
        finally:
            page.close()

    def test_apply_enabled_toggle_uses_existing_path(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            _trigger_step_command(page, 0, 'toggle-step')
            steps = page._pipeline_list.steps()
            assert steps[0]['enabled'] is False
            assert steps[0]['params'] == {'window': 5, 'gain': 2.0}
            # chip 变暗视觉（虚线 muted）由 sig_changed → _refresh_chain_and_results 重铺
            row = page._pipeline_list._list.itemWidget(page._pipeline_list._list.item(0))
            assert row.enabled_box.isChecked() is False
        finally:
            page.close()

    def test_run_waits_until_pending_params_are_applied(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._open_step_editor(0)
            page._param_form._editors['window'].setValue(42)
            runs = []
            page.run_requested.connect(runs.append)
            page._on_run_clicked()
            assert not runs
            assert '应用或放弃' in page._step_feedback.text()
            page._commit_step_btn.click()
            page._on_run_clicked()
            assert runs[0]['steps'][0]['params']['window'] == 42
        finally:
            page.close()

    def test_unapplied_params_not_written(self, qapp):
        """Parameter changes stay pending while interacting with the plot."""
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._open_step_editor(0)
            page._param_form._editors['window'].setValue(77)
            page._workspace.setFocus()
            qapp.processEvents()
            assert page._param_form.values()['window'] == 77
            page._discard_step_btn.click()
            assert page._param_form.values()['window'] == 5
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
            _trigger_step_command(page, 0, 'toggle-step')
            steps = page._pipeline_list.steps()
            assert steps[0]['enabled'] is False
            assert steps[0]['params'] == {'window': 42, 'gain': 8.0}
            # 重新启用（应用）：弹窗快照里参数仍是屏蔽前的值
            page._open_step_editor(0)
            assert page._param_form.values() == \
                {'window': 42, 'gain': 8.0}
            _trigger_step_command(page, 0, 'toggle-step')
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
        """Replacement requires explicit confirmation from the inline library."""
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            _trigger_step_command(page, 0, 'replace-step')
            assert page._method_browser.isVisibleTo(page)
            page._method_browser.sig_add_requested.emit('agc')
            assert page.current_pipeline()['steps'][0]['method_id'] == 'dewow'
            page._commit_step_btn.click()
            assert page._method_browser.isHidden()
            assert [s['method_id']
                    for s in page._pipeline_list.steps()] == ['agc']
        finally:
            page.close()

    def test_remove_action_forwards_to_chain_handler(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._add_method_to_pipeline('agc')
            _trigger_step_command(page, 0, 'remove-step')
            assert [s['method_id']
                    for s in page._pipeline_list.steps()] == ['agc']
        finally:
            page.close()

    def test_move_action_forwards_to_chain_handler(self, qapp):
        page = _page_with_methods(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._add_method_to_pipeline('agc')
            _trigger_step_command(page, 0, 'move-down')   # 插入位 2
            assert [s['method_id']
                    for s in page._pipeline_list.steps()] == ['agc', 'dewow']
        finally:
            page.close()

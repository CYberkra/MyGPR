"""Result tabs are processing inputs; drafts and submitted jobs keep their identity."""
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import Qt
from ui.pages.processing_page import ProcessingPage


def artifact(aid, parent='', step=1, run='R1', method='dewow', params=None):
    metadata = {'lineage_version': 2, 'step_input_artifact_id': parent,
                'run_input_artifact_id': parent, 'run_group_id': run,
                'run_step_index': step, 'artifact_kind': 'processing',
                'lineage': [{'method_id': method, 'params': params or {'window': 23}}]}
    return SimpleNamespace(artifact_id=aid, line_id='L01', name=aid,
                           method_id=method, method_name=method, status='success',
                           created_at=f'2026-10-16T12:00:0{step}',
                           manifest=MappingProxyType({'params': metadata}),
                           shape=(20, 30), dtype='float32', sha256=aid)


def bundle():
    return SimpleNamespace(matrix=np.ones((20, 30), dtype=np.float32), title='原始输入',
                           vmin=-1.0, vmax=1.0, trace_count=30, sample_count=20,
                           x_label='道序号', y_label='采样点', trace_axis_m=None,
                           sample_axis=None, sample_axis_label='', trace_elevation_m=None,
                           depth_axis_m=None)


@pytest.fixture
def page(qapp):
    value = ProcessingPage()
    value.set_lines([SimpleNamespace(line_id='L01', name='L01')])
    value.set_line_label('L01')
    value.set_original_bundle(bundle())
    value.set_artifacts([artifact('A'), artifact('B')])
    yield value
    value.close()
    value.deleteLater()


def test_activated_result_is_the_next_pipeline_input(page):
    assert page.select_artifact('A')
    assert page.current_pipeline()['input_artifact_id'] == 'A'


def test_switching_result_restores_its_own_draft(page):
    page.select_artifact('A')
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    page.select_artifact('B')
    assert page.current_pipeline()['steps'] == []
    page._pipeline_list.add_step('background', '背景抑制', {})
    page.select_artifact('A')
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 41}
    assert len(page.current_pipeline()['steps']) == 1


def test_saved_upstream_pipeline_is_never_replayed_as_new_steps(page):
    page.select_artifact('A')
    assert page.current_pipeline()['steps'] == []


def test_late_preview_does_not_activate_another_result(page):
    page.select_artifact('B')
    page.set_artifact_bundle('A', bundle())
    assert page.current_pipeline()['input_artifact_id'] == 'B'


def test_success_clears_only_submitted_source_and_preserves_other_draft(page):
    page.select_artifact('A')
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    page.set_running(True, 'job')
    assert not page._pipeline_list.isEnabled()
    page.select_artifact('B')
    assert page._pipeline_list.isEnabled()
    page._pipeline_list.add_step('background', '背景抑制', {})
    page.set_running(False, success=True)
    assert page.current_pipeline()['input_artifact_id'] == 'B'
    assert page.current_pipeline()['steps'][0]['method_id'] == 'background'
    page.select_artifact('A')
    assert page.current_pipeline()['steps'] == []


def test_cancel_preserves_submitted_draft(page):
    page.select_artifact('A')
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    page.set_running(True, 'job')
    page.set_running(False, success=False)
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 41}


def test_closing_raw_and_all_views_never_reopens_them(page):
    page.select_artifact('A')
    page._workspace.close_source('original')
    assert page.current_pipeline()['input_artifact_id'] == 'A'
    page._workspace.close_source('artifact:A')
    assert page._workspace.active_key() == ''
    page.set_artifact_bundle('A', bundle())
    assert page._workspace.active_key() == ''


def test_public_close_all_artifact_tabs_updates_the_actual_pipeline_input(page):
    page.select_artifact('A')
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 23})
    page.close_all_artifact_tabs()
    assert page._workspace.active_key() == 'original'
    assert page.current_pipeline()['input_artifact_id'] == ''
    assert page.current_pipeline()['steps'] == []
    page.select_artifact('A')
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 23}


def test_reconfigure_step_uses_parent_and_keeps_existing_results(page):
    page.set_artifacts([artifact('A'), artifact('B', parent='A', params={'window': 23})])
    assert page.reconfigure_artifact('B')
    assert page.current_pipeline()['input_artifact_id'] == 'A'
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 23}
    assert set(page._artifacts_by_id) == {'A', 'B'}


def test_display_changes_do_not_add_processing_steps(page):
    before = page.current_pipeline()
    page._workspace.primary_view().set_colormap('gray')
    assert page.current_pipeline() == before


def test_completion_uses_the_submitted_job_result_not_another_newer_run(page):
    page.select_artifact('A')
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    page.set_running(True, 'job')
    page.set_run_result('job', 'mine')
    page.set_running(False, success=True)
    page.set_artifacts([artifact('A'), artifact('B'), artifact('mine', parent='A'),
                        artifact('unrelated', parent='A', step=2, run='other')])
    assert page.current_pipeline()['input_artifact_id'] == 'mine'


def test_final_only_run_cannot_reconfigure_last_step_without_saved_input(page):
    final = artifact('final', step=2)
    final.manifest = {'params': {**dict(final.manifest['params']),
                      'lineage': [{'method_id': 'zero', 'params': {}},
                                  {'method_id': 'dewow', 'params': {'window': 23}}]}}
    page.set_artifacts([final])
    assert page.reconfigure_artifact('final') is False


def test_new_project_with_reused_line_ids_cannot_inherit_old_draft(page):
    page.select_artifact('A')
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    page.set_project_context('another-project')
    page.set_line_label('L01')
    page.set_original_bundle(bundle())
    page.set_artifacts([artifact('A')])
    page.select_artifact('A')
    assert page.current_pipeline()['steps'] == []


def test_empty_workspace_is_not_a_resumable_data_session(page):
    for key in list(page._workspace.tab_keys()):
        page._workspace.close_source(key)
    assert not page.has_line_session('L01')


def test_preview_failure_stays_on_its_source_and_retry_is_available(page):
    requested = []
    page.artifact_preview_requested.connect(requested.append)
    page.select_artifact('A')
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    page.set_preview_failure('', 'L01', 'A', '文件无法读取')
    assert page._sources['artifact:A']['error'] == '文件无法读取'
    assert 'artifact:A' not in page._workspace._pending
    assert not page._run_btn.isEnabled()
    assert page._preview_retry_btn.isVisibleTo(page)
    page._preview_retry_btn.click()
    assert requested == ['A', 'A']
    assert page.current_pipeline()['input_artifact_id'] == 'A'
    page.set_artifact_bundle('A', bundle())
    assert not page._sources['artifact:A'].get('error')
    assert page._run_btn.isEnabled()


def test_preview_failure_matches_project_line_and_source_without_focus_change(page):
    page.select_artifact('B')
    page.set_preview_failure('other-project', 'L01', 'B', 'wrong project')
    page.set_preview_failure('', 'other-line', 'B', 'wrong line')
    assert not page._sources['artifact:B'].get('error')
    page.set_preview_failure('', 'L01', 'A', 'A missing')
    assert page._sources['artifact:A']['error'] == 'A missing'
    assert page.current_pipeline()['input_artifact_id'] == 'B'
    assert page.retry_preview('artifact:A')
    assert page.current_pipeline()['input_artifact_id'] == 'B'


def test_final_only_history_rebuilds_whole_run_from_frozen_input(page):
    final = artifact('final', parent='A', step=2, run='saved-run')
    final.manifest = {'params': {**dict(final.manifest['params']),
        'pipeline': [{'method_id': 'zero', 'params': {'offset': 3}, 'enabled': True},
                     {'method_id': 'dewow', 'params': {'window': 23}, 'enabled': True}],
        'lineage': [{'method_id': 'zero', 'params': {'offset': 3}},
                    {'method_id': 'dewow', 'params': {'window': 23}}]}}
    page.set_artifacts([artifact('A'), final])
    assert not page.can_reconfigure_artifact('final')
    assert page.reconfigure_run('saved-run')
    assert page.current_pipeline()['input_artifact_id'] == 'A'
    assert [s['method_id'] for s in page.current_pipeline()['steps']] == ['zero', 'dewow']
    assert page.current_pipeline()['steps'][0]['params'] == {'offset': 3}
    assert set(page._artifacts_by_id) == {'A', 'final'}


def test_same_id_new_digest_does_not_reuse_preview_or_draft(page):
    page.select_artifact('A')
    page.set_artifact_bundle('A', bundle())
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    old_key = page._draft_key()
    changed = artifact('A')
    changed.sha256 = 'replacement-content'
    page.set_artifacts([changed, artifact('B')])
    assert page._sources['artifact:A']['bundle'] is None
    assert page.current_pipeline()['steps'] == []
    assert page._draft_key() != old_key
    assert page._drafts[old_key]['steps'][0]['params'] == {'window': 41}
    assert '失效' in page._draft_revision_hint.text()
    page.select_artifact('B')
    page.select_artifact('A')
    assert page.current_pipeline()['steps'] == []


def test_wrong_job_result_cannot_replace_submitted_result(page):
    page.select_artifact('A')
    page.set_running(True, 'mine')
    page.set_run_result('other', 'unrelated')
    assert 'result_id' not in page._run_snapshot
    page.set_run_result('mine', 'actual')
    assert page._run_snapshot['result_id'] == 'actual'


def menu_action(menu, name):
    return next(action for action in menu.actions() if action.data() == name)


def test_source_menu_rechecks_removed_or_revised_target_without_switching_input(page):
    page.select_artifact('B')
    menu = page._source_menu('artifact:A')
    assert page.current_pipeline()['input_artifact_id'] == 'B'
    removed = []
    page.artifact_delete_requested.connect(lambda *args: removed.append(args))
    page.set_artifacts([artifact('B')])
    menu_action(menu, 'delete-source').trigger()
    menu_action(menu, 'continue-source').trigger()
    assert removed == []
    assert page.current_pipeline()['input_artifact_id'] == 'B'
    page.set_artifacts([artifact('A'), artifact('B')])
    menu = page._source_menu('artifact:A')
    revised = artifact('A')
    revised.sha256 = 'new-content'
    page.set_artifacts([revised, artifact('B')])
    menu_action(menu, 'continue-source').trigger()
    assert page.current_pipeline()['input_artifact_id'] == 'B'


def test_history_group_menu_and_keyboard_share_whole_run_action(page):
    final = artifact('final', parent='A', step=2, run='saved-run')
    final.manifest = {'params': {**dict(final.manifest['params']),
        'pipeline': [{'method_id': 'zero', 'params': {}}, {'method_id': 'dewow', 'params': {'window': 23}}]}}
    page.set_artifacts([artifact('A'), final])
    page.select_artifact('B')  # no-op: removed source; current raw is retained
    item = next(page._history_tree.topLevelItem(i) for i in range(page._history_tree.topLevelItemCount())
                if page._history_tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole + 1) == 'saved-run')
    page._history_tree.setCurrentItem(item)
    page._history_menu_binding.show_keyboard()
    menu = page._history_menu_binding.menu
    assert menu_action(menu, 'reconfigure-run').isEnabled()
    menu_action(menu, 'reconfigure-run').trigger()
    assert page.current_pipeline()['input_artifact_id'] == 'A'
    assert len(page.current_pipeline()['steps']) == 2


def test_raw_revision_isolates_old_draft_and_matching_preview_errors_retry(page):
    page.select_original()
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 23})
    old_key = page._draft_key()
    page.set_original_revision('', 'L01', 'new-raw-digest')
    page.set_original_bundle(bundle())
    assert page.current_pipeline()['steps'] == []
    assert page._drafts[old_key]['steps']
    retries = []
    page.original_preview_requested.connect(lambda: retries.append(True))
    page.set_preview_failure('', 'L01', '', 'raw read failure')
    page._preview_retry_btn.click()
    assert retries == [True]
    assert page._workspace.active_key() == 'original'


def test_history_reconfigure_rejects_changed_frozen_input_digest(page):
    final = artifact('final', parent='A', step=2, run='saved-run')
    final.manifest = {'params': {**dict(final.manifest['params']), 'run_input_sha256': 'old-A',
        'pipeline': [{'method_id': 'dewow', 'params': {'window': 23}}]}}
    page.set_artifacts([artifact('A'), final])
    assert not page.reconfigure_run('saved-run')
    assert page._workspace.active_key() == 'original'


def test_repeated_raw_preview_failure_can_retry_again(page):
    requested = []
    page.original_preview_requested.connect(lambda: requested.append(True))
    page.set_preview_failure('', 'L01', '', 'first error')
    assert page.retry_preview('original')
    page.set_preview_failure('', 'L01', '', 'second error')
    assert page.retry_preview('original')
    assert requested == [True, True]


def test_read_only_project_explains_disabled_submission(page):
    page._pipeline_list.add_step('dewow', 'Dewow', {})
    page.set_project_read_only(True)
    assert not page._run_btn.isEnabled()
    assert '只读' in page._run_btn.toolTip()


def test_compact_lineage_tree_handles_deep_saved_chain_without_recursion(page, monkeypatch):
    from ui.file_tree import TreeNode
    node = TreeNode(key='last', kind='artifact', text='last', payload='last')
    for index in range(1050):
        node = TreeNode(key=str(index), kind='artifact', text=str(index), payload=str(index), children=(node,))
    monkeypatch.setattr('ui.pages.processing_page.build_lineage_model', lambda *_args: (node,))
    page._render_trees()
    item, count = page._lineage_tree.topLevelItem(0), 0
    while item is not None:
        count += 1
        item = item.child(0)
    assert count == 1051


def test_replace_run_draft_can_cancel_then_restore_the_exact_old_draft(page, monkeypatch):
    from PyQt6.QtWidgets import QDialog
    page.select_artifact('A')
    page._pipeline_list.add_step('background', '背景', {'axis': 1})
    page._result_name_edit.setText('my pending result')
    page.select_artifact('B')
    before = page.current_pipeline()
    monkeypatch.setattr('ui.pages.processing_page.MessageBox.exec', lambda self: QDialog.DialogCode.Rejected)
    steps = [{'method_id': 'dewow', 'label': 'Dewow', 'params': {'window': 41}, 'enabled': True}]
    assert not page._replace_draft('artifact:A', steps)
    assert page.current_pipeline() == before
    monkeypatch.setattr('ui.pages.processing_page.MessageBox.exec', lambda self: QDialog.DialogCode.Accepted)
    assert page._replace_draft('artifact:A', steps)
    assert page.current_pipeline()['steps'][0]['method_id'] == 'dewow'
    menu_action(page._draft_menu(), 'restore-draft').trigger()
    assert page.current_pipeline()['steps'][0]['method_id'] == 'background'
    assert page.current_pipeline()['steps'][0]['params'] == {'axis': 1}
    assert page.current_pipeline()['result_name'] == 'my pending result'


def test_workspace_split_updates_layout_label_without_rearranging_views(page):
    page._workspace.split_source('original', 'right', copy=True)
    assert page._layout_combo.currentText() == '左右对照'
    page._workspace.split_source('original', 'bottom', copy=True)
    assert page._layout_combo.currentText() == '自定义布局'
    assert len(page._workspace.group_ids()) == 3


def test_open_method_picker_cannot_edit_the_submitted_locked_draft(page):
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 23})
    page.set_running(True, 'mine')
    before = page.current_pipeline()
    page._add_method_to_pipeline('background')
    page._duplicate_step(0)
    page._replace_step_method(0, 'background')
    assert page.current_pipeline() == before


@pytest.mark.parametrize('failed', [False, True])
def test_late_first_raw_preview_never_replaces_a_selected_artifact(qapp, failed):
    value = ProcessingPage()
    value.set_line_label('L01')
    value.set_artifacts([artifact('A')])
    value.select_artifact('A')
    value._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    before = value.current_pipeline()
    if failed:
        value.set_preview_failure('', 'L01', '', 'raw file missing')
    else:
        value.set_original_bundle(bundle())
    assert value.current_pipeline() == before
    assert value._workspace.active_key() == 'artifact:A'
    value.close()
    value.deleteLater()


@pytest.mark.parametrize('failed', [False, True])
def test_late_first_raw_preview_keeps_a_user_closed_workspace_empty(qapp, failed):
    value = ProcessingPage()
    value.set_line_label('L01')
    value.set_artifacts([artifact('A')])
    value.select_artifact('A')
    value._workspace.close_source('artifact:A')
    if failed:
        value.set_preview_failure('', 'L01', '', 'raw file missing')
    else:
        value.set_original_bundle(bundle())
    assert value._workspace.tab_keys() == []
    assert value._workspace.active_key() == ''
    value.close()
    value.deleteLater()

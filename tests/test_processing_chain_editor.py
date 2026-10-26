"""The visible editable chain controls only the active input's pending draft."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest

from ui.pages.processing_page import ProcessingPage
from tests.test_processing_workspace_flow import artifact, bundle


@pytest.fixture
def page(qapp):
    value = ProcessingPage()
    value.set_project_context('P')
    value.set_lines([SimpleNamespace(line_id='L01')])
    value.set_line_label('L01')
    value.set_methods([
        {'method_id': 'dewow', 'display_name': 'Dewow', 'parameter_schema': [
            {'name': 'window', 'label': '窗口', 'type': 'int', 'default': 23, 'min': 1, 'max': 101}]},
        {'method_id': 'agc', 'display_name': 'AGC', 'parameter_schema': [
            {'name': 'window', 'label': '窗口', 'type': 'int', 'default': 9, 'min': 1, 'max': 101}]},
    ])
    value.set_original_bundle(bundle())
    value.set_artifacts([artifact('A'), artifact('B', parent='A')])
    value.resize(1100, 720)
    value.show()
    qapp.processEvents()
    yield value
    value.close()
    value.deleteLater()
    qapp.processEvents()


def click_step(page, index):
    strip = page._chain_strip
    QTest.mouseClick(strip._list.viewport(), Qt.MouseButton.LeftButton,
                     pos=strip._list.visualItemRect(strip._list.item(index)).center())


def test_chain_is_above_canvas_and_left_has_no_duplicate_step_list(page):
    assert page._chain_strip.isVisibleTo(page)
    assert page._pipeline_list.isHidden()
    assert page._chain_strip.mapTo(page, QPoint()).y() < page._workspace.mapTo(page, QPoint()).y()
    assert not page._chain_strip.run_button().isVisibleTo(page)


def test_click_edits_parameters_and_refreshes_summary_without_changing_input(page, qapp):
    page.select_artifact('A')
    page._add_method_to_pipeline('dewow')
    qapp.processEvents()
    click_step(page, 0)
    popup = page._step_editor
    assert popup is not None and popup.isVisible()
    popup._form.set_values({'window': 41})
    popup._apply_btn.click()
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 41}
    assert '41' in page._chain_strip._steps[0]['param_summary']
    assert page.current_pipeline()['input_artifact_id'] == 'A'
    click_step(page, 0)
    assert page._step_editor is not None


def test_chain_reorder_changes_pipeline_and_tab_reorder_does_not(page):
    page._add_method_to_pipeline('dewow')
    page._add_method_to_pipeline('agc')
    page._chain_strip.sig_step_moved.emit(0, 2)
    assert [step['method_id'] for step in page.current_pipeline()['steps']] == ['agc', 'dewow']
    page.select_artifact('A')
    page.select_original()
    before = page.current_pipeline()
    page._workspace.tab_bar().moveTab(0, 1)
    assert page.current_pipeline() == before


def test_switching_inputs_restores_visible_chain_and_invalidates_old_editor(page, qapp):
    page.select_artifact('A')
    page._add_method_to_pipeline('dewow')
    qapp.processEvents()
    click_step(page, 0)
    old = page._step_editor
    page.select_artifact('B')
    assert page._chain_strip._steps == []
    page._add_method_to_pipeline('agc')
    old.sig_applied.emit({'index': 0, 'params': {'window': 77}, 'enabled': True})
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 9}
    page.select_artifact('A')
    assert page._chain_strip._steps[0]['label'] == 'Dewow'
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 23}


def test_running_locks_submitted_chain_only(page):
    page.select_artifact('A')
    page._add_method_to_pipeline('dewow')
    page.set_running(True, 'job')
    assert not page._chain_strip.isEnabled()
    page._chain_strip.sig_step_removed.emit(0)
    assert len(page.current_pipeline()['steps']) == 1
    page.select_artifact('B')
    assert page._chain_strip.isEnabled()
    page._add_method_to_pipeline('agc')
    page.set_running(False, success=False)
    page.select_artifact('A')
    assert page._chain_strip.isEnabled()
    assert len(page.current_pipeline()['steps']) == 1


def test_saved_source_path_opens_real_result_without_replaying_it(page):
    page.select_artifact('B')
    nodes = {button.property('source_key'): button for button in page._saved_path.buttons()}
    assert {'original', 'artifact:A', 'artifact:B'} <= set(nodes)
    nodes['artifact:A'].click()
    assert page.current_pipeline()['input_artifact_id'] == 'A'
    assert page.current_pipeline()['steps'] == []


def test_saved_source_path_shows_only_algorithm_names_and_keeps_parameter_tooltip(page):
    page.select_artifact('B')
    buttons = page._saved_path.buttons()
    assert all('w=' not in button.text() for button in buttons)
    assert any('去漂移' in button.text() for button in buttons)
    assert any('"window": 23' in button.toolTip() for button in buttons)
    assert page._saved_path.parentWidget() is page._chain_panel
    assert page._saved_path.isVisibleTo(page)
    assert page._chain_strip.isVisibleTo(page)


def test_stale_saved_path_button_cannot_activate_reused_id_in_another_project(page):
    page.select_artifact('B')
    old = next(button for button in page._saved_path.buttons()
               if button.property('source_key') == 'artifact:A')
    page.set_project_context('other')
    page.set_line_label('L01')
    page.set_original_bundle(bundle())
    page.set_artifacts([artifact('A'), artifact('B', parent='A')])
    page.select_artifact('B')
    old.click()
    assert page.current_pipeline()['input_artifact_id'] == 'B'


def test_line_switch_closes_editor_before_new_preview_arrives(page, qapp):
    page._add_method_to_pipeline('dewow')
    qapp.processEvents()
    click_step(page, 0)
    popup = page._step_editor
    page.set_line_label('L02')
    assert not popup.isVisible()
    assert page._step_editor is None


def test_old_method_picker_cannot_append_to_a_different_source(page):
    page.select_artifact('A')
    page._open_method_picker()
    old = page._method_picker
    page.select_artifact('B')
    old.sig_method_chosen.emit('dewow')
    assert page.current_pipeline()['steps'] == []
    assert not old.isVisible()


def test_identical_drafts_still_invalidate_the_old_chain_menu(page):
    for source in ('A', 'B'):
        page.select_artifact(source)
        page._add_method_to_pipeline('dewow')
    page.select_artifact('A')
    menu = page._chain_strip._build_overflow_menu()
    old_edit = menu._step_submenus[0].menuActions()[1]
    page.select_artifact('B')
    old_edit.trigger()
    assert page._step_editor is None


def test_reordering_same_summary_steps_invalidates_old_overflow_menu(page):
    page._pipeline_list.set_steps([
        {'method_id': 'dewow', 'label': 'Dewow', 'params': {'window': 23, 'power': 1, 'extra': extra}}
        for extra in (1, 2)])
    page._pipeline_list.sig_changed.emit()
    menu = page._chain_strip._build_overflow_menu()
    old_edit = menu._step_submenus[0].menuActions()[1]
    page._pipeline_list._move_step_to(0, 2)
    old_edit.trigger()
    assert page._step_editor is None


def test_replace_picker_rejects_a_target_moved_after_opening(page):
    page._add_method_to_pipeline('dewow')
    page._add_method_to_pipeline('agc')
    page._open_method_picker(purpose='replace', index=0)
    picker = page._method_picker
    page._pipeline_list._move_step_to(0, 2)
    before = page.current_pipeline()
    picker.sig_method_chosen.emit('dewow')
    assert page.current_pipeline() == before

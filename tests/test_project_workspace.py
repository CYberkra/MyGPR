"""The project route owns startup, empty-project and line-ledger states."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from ui.pages.project_page import ProjectPage


@pytest.fixture
def page(qapp):
    widget = ProjectPage()
    widget.resize(1000, 650)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


def test_startup_empty_and_ledger_are_mutually_exclusive(page):
    assert page.workspace_state() == 'startup'
    assert page.startup.isVisible()
    assert not page._work_widget.isVisible()
    page.set_project_info(SimpleNamespace(project_id='P1', name='North slope'))
    assert page.workspace_state() == 'empty'
    assert not page.startup.isVisible()
    assert page._empty_project.isVisible()
    assert not page._ledger_widget.isVisible()
    assert not page.header_import_btn.isVisible()
    assert page.empty_import_btn.isVisible()
    page.set_lines([SimpleNamespace(line_id='L01', name='Line 1')])
    assert page.workspace_state() == 'ledger'
    assert page._ledger_widget.isVisible()
    assert page.header_import_btn.isVisible()
    assert not page.empty_import_btn.isVisible()
    page.set_project_info(None)
    assert page.workspace_state() == 'startup'
    assert page._lines_table.rowCount() == 0


def test_processing_entry_tracks_single_selection_only(page):
    page.set_project_info(SimpleNamespace(project_id='P1', name='North slope'))
    page.set_lines([SimpleNamespace(line_id='L01'), SimpleNamespace(line_id='L02')])
    emitted = []
    page.line_process_requested.connect(emitted.append)
    assert not page.process_line_btn.isEnabled()
    page.select_line('L02')
    assert page.process_line_btn.isEnabled()
    page.process_line_btn.click()
    assert emitted == ['L02']
    page.set_line_resume_available('L02', True)
    assert page.process_line_btn.text() == '继续处理'
    from PyQt6.QtCore import QItemSelectionModel
    model = page._lines_table.selectionModel()
    model.select(page._lines_table.model().index(0, 0),
                 QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    assert not page.process_line_btn.isEnabled()
    assert page.current_line_id() == ''
    assert not page.sync_btn.isEnabled()
    # A late callback for the prior single line must not restore a hidden target.
    page.set_artifacts([SimpleNamespace(artifact_id='late', line_id='L02')])
    assert page._artifacts == []
    assert not page.preview_artifact_btn.isEnabled()


def test_project_switch_resets_old_lines_and_tool_context(page):
    page.set_project_info(SimpleNamespace(project_id='P1', name='First'))
    page.set_lines([SimpleNamespace(line_id='L01')])
    page.select_line('L01')
    page.file_edit.setText('old.csv')
    page._sensor_edits['rtk'].setText('old-rtk.csv')
    page.set_project_info(SimpleNamespace(project_id='P2', name='Second'))
    assert page.workspace_state() == 'empty'
    assert page.current_line_id() == ''
    assert page.file_edit.text() == ''
    assert page._sensor_edits['rtk'].text() == ''


def test_programmatic_line_selection_replaces_multiselection(page):
    from PyQt6.QtCore import QItemSelectionModel
    page.set_project_info(SimpleNamespace(project_id='P1', name='Project'))
    page.set_lines([SimpleNamespace(line_id='L01'), SimpleNamespace(line_id='L02')])
    model = page._lines_table.selectionModel()
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    for row in range(2):
        model.select(page._lines_table.model().index(row, 0), flags)
    assert not page.process_line_btn.isEnabled()
    page.select_line('L02')
    assert [index.row() for index in model.selectedRows()] == [1]
    assert page.current_line_id() == 'L02'
    assert page.process_line_btn.isEnabled()


def test_import_and_sync_forms_open_as_tools(page, qapp):
    page.set_project_info(SimpleNamespace(project_id='P1', name='Project'))
    assert not page._import_dialog.isVisible()
    page.empty_import_btn.click()
    qapp.processEvents()
    assert page._import_dialog.isVisible()
    page._import_dialog.close()
    page.set_lines([SimpleNamespace(line_id='L01')])
    page.select_line('L01')
    page.show_sensor_tool()
    qapp.processEvents()
    assert page._sync_dialog.isVisible()


def test_startup_commands_are_shared_with_project_menu(page):
    requested = []
    page.startup.new_project_requested.connect(lambda: requested.append('new'))
    page.startup.open_project_requested.connect(lambda: requested.append('open'))
    page.startup.new_btn.click()
    page.startup.open_btn.click()
    assert requested == ['new', 'open']


def test_recent_projects_keep_paths_and_availability(page, tmp_path):
    existing = tmp_path / 'survey'
    existing.mkdir()
    missing = tmp_path / 'missing'
    page.startup.set_recent_projects([str(existing), str(missing)])
    assert page.startup._recent_list.count() == 2
    assert str(existing) in page.startup._recent_list.item(0).text()
    assert '不可用' in page.startup._recent_list.item(1).text()
    opened = []
    page.startup.open_recent_project_requested.connect(opened.append)
    page.startup._open_recent_item(page.startup._recent_list.item(0))
    assert opened == [str(existing)]


def test_changing_line_closes_sensor_tool_and_clears_previous_target_files(page):
    page.set_project_info(SimpleNamespace(project_id='P1', name='Project'))
    page.set_lines([SimpleNamespace(line_id='L01'), SimpleNamespace(line_id='L02')])
    page.select_line('L01')
    page.show_sensor_tool()
    page._sensor_edits['rtk'].setText('line-1-rtk.csv')
    page.select_line('L02')
    assert not page._sync_dialog.isVisible()
    assert page._sensor_edits['rtk'].text() == ''


def test_removing_recent_record_preserves_project_directory_and_shared_menu_model(page, tmp_path):
    from ui.settings_manager import SettingsManager
    settings = SettingsManager(str(tmp_path / 'settings.json'))
    project_root = tmp_path / 'survey'
    project_root.mkdir()
    settings.add_recent_project(str(project_root))
    page.set_settings_manager(settings)
    page.startup._toggle_pin(str(project_root))
    assert settings.get('pinned_recent_projects') == [str(project_root)]
    # Recent-history trimming cannot evict a pinned project from the UI model.
    settings.set('recent_projects', [])
    page.startup.refresh_recents()
    assert page.startup._recent_paths == [str(project_root)]
    page.startup._remove_recent(str(project_root))
    assert project_root.is_dir()
    assert settings.get('recent_projects') == []
    assert settings.get('pinned_recent_projects') == []
    menu = page.startup.project_menu(page.empty_import_btn, can_close=True,
                                     close_requested=page.close_project_requested.emit)
    assert [action.text() for action in menu.actions()] == ['新建项目', '打开项目…', '关闭当前项目']

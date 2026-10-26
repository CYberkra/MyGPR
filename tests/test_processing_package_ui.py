"""Minimal menu/save flow preserves selection and project ownership."""
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QObject, QPoint, pyqtSignal

from ui.coordinator_project import ProjectChain
from ui.controllers.delivery_controller import DeliveryController


@pytest.mark.parametrize('cancel,switch', [(True, ''), (False, ''), (False, 'different'), (False, 'reopen')])
def test_save_dialog_cancel_or_project_change_never_submits_wrong_project(qapp, monkeypatch, cancel, switch):
    from ui import dialogs
    co = MagicMock()
    state = {'project': 'A'}
    co.current_project_id.side_effect = lambda: state['project']
    co.require_project.return_value = True
    co.settings.get.return_value = ''
    chain = ProjectChain(co)
    def choose(*args, **kwargs):
        if switch == 'different':
            state['project'] = 'B'
        elif switch == 'reopen':
            chain.on_project_closed()
            co.goto_page.reset_mock()  # Closing has its own navigation, independent of export.
        return '' if cancel else 'C:/export/result.zip'
    monkeypatch.setattr(dialogs, 'choose_processing_package_path', choose)
    chain.on_processing_package_requested('L02', 'saved-step')
    assert not co.goto_page.called
    assert chain.current_line_id == ''
    if cancel or switch:
        co.delivery_controller.export_processing_package.assert_not_called()
    else:
        co.delivery_controller.export_processing_package.assert_called_once_with(
            'A', 'L02', 'C:/export/result.zip', artifact_id='saved-step')


class Bridge(QObject):
    job_completed = pyqtSignal(str, bool, str, object)
    def watch(self, job_id, *, title):
        # Fast jobs may finish synchronously as soon as the bridge starts watching.
        self.job_completed.emit(job_id, True, '', NS(archive_path='result.zip', warnings=()))


@pytest.mark.parametrize('stale', [False, True])
def test_package_completion_is_connected_before_watch_and_context_guarded(qapp, stale):
    controller = DeliveryController()
    bridge = Bridge()
    backend = NS(submit_processing_result_package=lambda *a, **k: 'package-job')
    controller.set_backend(NS(backend=backend, job_bridge=bridge))
    controller.set_project_context('A')
    if stale:
        def later(job_id, *, title):
            controller.set_project_context('B')
            bridge.job_completed.emit(job_id, True, '', NS(archive_path='A.zip', warnings=()))
        bridge.watch = later
    got = []
    controller.processing_package_generated.connect(got.append)
    assert controller.export_processing_package('A', 'L01', 'result.zip') == 'package-job'
    assert len(got) == (0 if stale else 1)
    assert bridge.receivers(bridge.job_completed) == 0


def test_tree_menu_has_one_export_action_and_does_not_activate_input(qapp, monkeypatch):
    from ui.widgets.file_tree_panel import FileTreePanel, _ROLE_KIND
    panel = FileTreePanel()
    panel.set_project_info(NS(name='Example', project_id='A', root_path=''))
    panel.set_lines([NS(line_id='L01', name='L01', updated_at='', length_m=0,
                        processing_status='', sample_count=72, trace_count=48)])
    panel.set_artifacts([NS(artifact_id='step', line_id='L01', name='Result',
                            created_at='', branch_id='', parent_artifact_id='',
                            method_id='dewow', method_name='dewow', status='committed',
                            shape=(72, 48), data_reference='', manifest={})])
    got, activated, menus = [], [], []
    panel.processing_package_requested.connect(lambda *args: got.append(args))
    panel.line_process_requested.connect(activated.append)
    panel.artifact_focus_requested.connect(lambda *args: activated.append(args))
    monkeypatch.setattr(panel, '_show_context_menu', lambda menu, pos: menus.append(menu))
    leaf = panel._line_id_by_item['L01']
    monkeypatch.setattr(panel._tree, 'itemAt', lambda pos: leaf)
    panel._on_context_menu(QPoint(1, 1))
    actions = [a for a in menus[-1].actions() if a.text() == '打包处理结果…']
    assert len(actions) == 1 and actions[0].isEnabled()
    actions[0].trigger()
    assert got == [('L01', '')] and activated == []
    from PyQt6.QtWidgets import QTreeWidgetItemIterator
    iterator = QTreeWidgetItemIterator(panel._tree)
    result_leaf = None
    while iterator.value():
        if iterator.value().data(0, _ROLE_KIND) == 'artifact':
            result_leaf = iterator.value()
            break
        iterator += 1
    assert result_leaf is not None
    panel._on_artifact_context_menu(result_leaf, QPoint(1, 1))
    result_actions = [a for a in menus[-1].actions() if a.text() == '打包到此步的结果…']
    assert len(result_actions) == 1
    result_actions[0].trigger()
    assert got == [('L01', ''), ('L01', 'step')] and activated == []
    # Old menu callbacks cannot export a line after a project switch.
    panel.set_project_info(NS(name='Other', project_id='B', root_path=''))
    actions[0].trigger()
    assert got == [('L01', ''), ('L01', 'step')]
    panel.close()


def test_native_save_flow_uses_simple_default_name_and_preserves_collision_policy(qapp, monkeypatch, tmp_path):
    from ui import dialogs, file_dialogs
    seen = []
    monkeypatch.setattr(file_dialogs, 'getSaveFileName',
        lambda *args, **kwargs: (seen.append((args, kwargs)) or ('', '')))
    assert dialogs.choose_processing_package_path(None, 'L01', directory=str(tmp_path)) == ''
    args, kwargs = seen[0]
    assert args[2].endswith('L01_处理结果.zip')
    assert args[3] == 'ZIP 压缩包 (*.zip)'
    assert kwargs['confirm_overwrite'] is False

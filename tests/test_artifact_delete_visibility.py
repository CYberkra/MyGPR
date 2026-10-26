"""Deletion confirmation stays visible when requested from the processing page."""
from unittest.mock import MagicMock
from types import SimpleNamespace as NS

from PyQt6.QtCore import QPoint, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget
import pytest
import numpy as np

from ui import dialogs
from ui.coordinator_project import ProjectChain


@pytest.mark.parametrize('kind', ['delete', 'discard'])
def test_delete_confirmation_is_visible_with_hidden_project_page(qapp, kind):
    host = QWidget()
    host.resize(900, 650)
    pages = QStackedWidget(host)
    layout = QVBoxLayout(host)
    layout.addWidget(pages)
    project = QWidget()
    processing = QWidget()
    pages.addWidget(project)
    pages.addWidget(processing)
    pages.setCurrentWidget(processing)
    host.show()
    qapp.processEvents()
    assert not project.isVisible()
    observations = []
    def inspect_and_cancel():
        boxes = host.findChildren(dialogs.MessageBox)
        observations.append(any(box.isVisible() and not box.visibleRegion().isEmpty() for box in boxes))
        # Test never confirms deletion; finish even when the modal is hidden.
        for box in boxes:
            QDialog.done(box, QDialog.DialogCode.Rejected)
    QTimer.singleShot(250, inspect_and_cancel)
    try:
        result = (dialogs.ask_artifact_delete(project, ['Example saved result']) if kind == 'delete'
                  else dialogs.ask_discard_interpretation(project, 'L01'))
        assert result is False
        assert observations == [True]
    finally:
        host.close()
        host.deleteLater()


def test_artifact_updates_refresh_resource_tree_even_for_a_different_line():
    co = MagicMock()
    chain = ProjectChain(co)
    chain.current_line_id = 'L01'
    chain.on_artifacts_updated('L5204', [])
    co.project_controller.refresh_all_artifacts.assert_called_once_with()
    # Export/delete elsewhere must keep the processing input and current page data.
    assert chain.current_line_id == 'L01'
    co.page.assert_not_called()


@pytest.mark.parametrize('current_line', ['L01', 'L02'])
def test_confirmed_result_delete_updates_real_tree_and_preserves_raw_lines(qapp, tmp_path, monkeypatch, current_line):
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from ui.controllers.project_controller import ProjectController
    from ui.controllers import project_controller as module
    from ui.widgets.file_tree_panel import FileTreePanel, _ROLE_PAYLOAD

    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(tmp_path / 'delete-ui', name='Delete UI')
    pid = summary.project_id
    raw = np.random.default_rng(3).normal(size=(32, 24)).astype(np.float32)
    for line in ('L01', 'L02'):
        backend.projects.save_line_dataset(pid, line, raw)
    saved = backend.project_processing.execute_pipeline(pid, 'L02', PipelineDefinition(
        steps=(PipelineStep('dewow', {'window': 7}),)))
    host = QWidget()
    host.resize(950, 680)
    panel = FileTreePanel(host)
    pages = QStackedWidget(host)
    project = QWidget()
    project.set_artifacts = lambda artifacts: None
    pages.addWidget(project)
    pages.addWidget(QWidget())
    pages.setCurrentIndex(1)
    layout = QHBoxLayout(host)
    layout.addWidget(panel)
    layout.addWidget(pages)
    pc = ProjectController()
    pc.set_backend(NS(backend=backend, job_bridge=None))
    pc._current = summary  # Same opened session used by the real command objects.
    monkeypatch.setattr(module, 'run_command', lambda command, **kwargs: command.execute())
    co = MagicMock()
    co.project_controller = pc
    co.require_project.return_value = True
    co.processing.preview_newest_artifact = False
    other_pages = {}
    co.page.side_effect = lambda name: project if name == 'projectInterface' else other_pages.setdefault(name, MagicMock())
    chain = ProjectChain(co)
    chain.current_line_id = current_line
    panel.set_project_info(summary)
    panel.set_lines(list(backend.projects.list_lines(pid)))
    panel.set_artifacts(list(backend.projects.list_artifacts(pid)))
    panel.artifact_delete_requested.connect(lambda line, aid: chain.on_artifact_delete_requested(line, [aid]))
    pc.artifact_descendants_ready.connect(chain.on_artifact_descendants_ready)
    pc.artifacts_updated.connect(chain.on_artifacts_updated)
    pc.all_artifacts_updated.connect(panel.set_artifacts)
    host.show()
    qapp.processEvents()
    observations = []
    def confirm_temp_result():
        boxes = host.findChildren(dialogs.MessageBox)
        observations.append(any(box.isVisible() for box in boxes))
        for box in boxes:
            box.yesButton.click()  # Only the disposable test project is affected.
    try:
        item = next(item for item in panel._items() if item.data(0, _ROLE_PAYLOAD) == saved.artifact_id)
        panel._on_artifact_context_menu(item, QPoint(10, 10))
        menu = panel._context_menu
        action = next(a for a in menu.actions() if a.text() == '删除此结果及依赖成果…')
        QTimer.singleShot(250, confirm_temp_result)
        QTest.mouseClick(menu.view.viewport(), Qt.MouseButton.LeftButton,
                         pos=menu.view.visualItemRect(action.property('item')).center())
        qapp.processEvents()
        assert observations == [True]
        assert backend.projects.list_artifacts(pid, 'L02') == ()
        assert panel._artifact('L02', saved.artifact_id) is None
        assert set(panel._line_id_by_item) == {'L01', 'L02'}
        assert chain.current_line_id == current_line
        for line in ('L01', 'L02'):
            np.testing.assert_array_equal(backend.projects.read_dataset(pid, line).data, raw)
    finally:
        host.close()
        host.deleteLater()
        backend.shutdown()

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QDialog

import ui.field_panels.project_dialogs as project_dialogs_module
from ui.field_panels.project_dialogs import ProjectCreateDialog, ProjectSettingsDialog
from ui.field_workbench_window import FieldWorkbenchWindow


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class AutoAcceptProjectCreateDialog(ProjectCreateDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parent_edit.setText(tempfile.mkdtemp(prefix="mygpr-v090-project-"))
        QTimer.singleShot(0, self.accept)


class AutoAcceptProjectSettingsDialog(ProjectSettingsDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.location_edit.setText("v0.8.90 回归测试测区")
        QTimer.singleShot(0, self.accept)


def test_project_create_direct_ok_does_not_raise(monkeypatch) -> None:
    app = _app()
    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.90")
    monkeypatch.setattr(project_dialogs_module, "ProjectCreateDialog", AutoAcceptProjectCreateDialog)
    try:
        win.show()
        app.processEvents()
        win._action_new_project_dialog()
        app.processEvents()
        assert win.project_root is not None
        assert (Path(win.project_root) / "project.json").exists()
        assert win.project_manifest is not None
        assert win.project_manifest.name == "新建 MyGPR 项目"
    finally:
        win.close()


def test_project_settings_accepted_path_uses_imported_qdialog(monkeypatch) -> None:
    app = _app()
    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.90")
    monkeypatch.setattr(project_dialogs_module, "ProjectCreateDialog", AutoAcceptProjectCreateDialog)
    monkeypatch.setattr(project_dialogs_module, "ProjectSettingsDialog", AutoAcceptProjectSettingsDialog)
    try:
        win.show()
        app.processEvents()
        win._action_new_project_dialog()
        app.processEvents()
        win._action_project_settings_dialog()
        app.processEvents()
        assert win.project_manifest is not None
        assert win.project_manifest.location == "v0.8.90 回归测试测区"
    finally:
        win.close()


def test_qdialog_symbol_is_available_for_operation_handlers() -> None:
    assert project_dialogs_module.QDialog.DialogCode.Accepted == QDialog.DialogCode.Accepted

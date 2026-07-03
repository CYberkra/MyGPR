from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from core.ingest_service import IngestService
from core.project_models import QcItem, QcReportV1
from core.project_service import ProjectService
from ui.gui_base import _configure_qt_cjk_font
from ui.workbench_window import MyGPRWorkbenchWindow, WORKSPACES

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_qt_cjk_font_fallback_registers_windows_font() -> None:
    windows_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if not windows_font.exists():
        pytest.skip("Windows CJK font fallback is not available on this host")
    assert _configure_qt_cjk_font(_app()) in {"Microsoft YaHei", "Microsoft YaHei UI"}


def test_workbench_uses_lifecycle_workspaces_and_data_document_tabs() -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        assert list(win.workspace_buttons) == list(WORKSPACES)
        assert win.active_workspace == "data_management"
        assert win.document_tabs.count() == 1
        assert win.document_tabs.tabText(0) == "项目概览"
        assert "项目管理" not in [win.document_tabs.tabText(i) for i in range(win.document_tabs.count())]
        assert win.project_tree.headerItem().text(0) == "项目资源"
    finally:
        win.close()
        app.processEvents()


def test_workbench_can_create_empty_formal_project(tmp_path: Path) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.create_project(tmp_path / "empty_project", name="Empty")
        assert win.project is not None
        assert win.project.manifest.name == "Empty"
        assert win.project.manifest.temporary is False
        assert win.add_line_action.isEnabled()
    finally:
        win.close()
        app.processEvents()


def test_workbench_layout_keeps_primary_panels_visible_at_minimum_size() -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.resize(1120, 720)
        win.show()
        app.processEvents()
        assert win.project_tree.width() >= 180
        assert win.inspector.width() >= 220
        assert win.document_tabs.width() >= 500
        for button in list(win.workspace_buttons.values()) + [
            win.create_action,
            win.open_action,
            win.import_action,
            win.import_folder_action,
            win.add_line_action,
            win.formalize_action,
            win.sidecar_action,
            win.qc_action,
            win.ack_warning_action,
            win.legacy_action,
            win.save_legacy_action,
        ]:
            assert button.isVisible()
    finally:
        win.close()
        app.processEvents()


def test_workbench_requires_warning_acknowledgement_before_processing(tmp_path: Path) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    project = ProjectService.create(tmp_path / "warning_gate", name="Warning Gate")
    try:
        win.project = project
        report = QcReportV1(
            line_id="L001",
            items=[QcItem("rtk_missing", "warning", "未发现 RTK 辅助文件。")],
            created_at="2026-06-08T00:00:00Z",
        )
        assert win._command_state(report) == "qc_review_required"
        report.items[0].acknowledged = True
        report.items[0].acknowledgement_note = "现场确认无 RTK"
        assert win._command_state(report) == "formal_ready"
    finally:
        win.project = None
        project.close()
        win.close()
        app.processEvents()


def test_global_splitter_layout_restores_without_project(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    app = _app()
    first = MyGPRWorkbenchWindow()
    first.resize(1400, 900)
    first.show()
    app.processEvents()
    first.main_splitter.setSizes([310, 760, 330])
    first.vertical_splitter.setSizes([650, 210])
    first.close()
    app.processEvents()

    second = MyGPRWorkbenchWindow()
    try:
        second.resize(1400, 900)
        second.show()
        app.processEvents()
        horizontal = second.main_splitter.sizes()
        vertical = second.vertical_splitter.sizes()
        assert abs(horizontal[0] - 310) < 20
        assert abs(horizontal[2] - 330) < 20
        assert abs(vertical[1] - 210) < 20
    finally:
        second.close()
        app.processEvents()


def test_open_loose_file_creates_temporary_project_and_qc_document(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_loose_path(source)
        assert win.project is not None
        assert win.project.manifest.temporary is True
        line = win.project.list_lines()[0]
        assert win.selected_line_id == line.line_id
        assert win.run_selected_line_qc().can_process is True
        labels = [win.document_tabs.tabText(i) for i in range(win.document_tabs.count())]
        assert any(label.startswith("质控") for label in labels)
    finally:
        win.close()
        app.processEvents()


def test_formal_project_restores_selected_line_and_documents(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    temporary = IngestService.open_temporary(source)
    formal = IngestService.formalize(temporary, tmp_path / "formal", name="Formal")
    line_id = formal.list_lines()[0].line_id
    formal.close()
    temporary.close()

    app = _app()
    first = MyGPRWorkbenchWindow()
    first.open_project(tmp_path / "formal")
    first.open_line_document(line_id)
    first.close()
    app.processEvents()

    second = MyGPRWorkbenchWindow()
    try:
        second.open_project(tmp_path / "formal")
        assert second.selected_line_id == line_id
        labels = [second.document_tabs.tabText(i) for i in range(second.document_tabs.count())]
        assert any(label.startswith("测线") for label in labels)
    finally:
        second.close()
        app.processEvents()


def test_formalize_and_qc_ui_actions_run_in_background_threads(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_loose_path(source)
        win.formalize_project_async(tmp_path / "formal_async", name="Formal Async")
        assert win._task_threads
        deadline = time.time() + 15
        while time.time() < deadline and (win.project is None or win.project.manifest.temporary):
            app.processEvents()
            time.sleep(0.01)
        assert win.project is not None
        assert win.project.manifest.temporary is False

        win.run_selected_line_qc_async()
        assert win._task_threads
        deadline = time.time() + 15
        while time.time() < deadline and win._task_threads:
            app.processEvents()
            time.sleep(0.01)
        labels = [win.document_tabs.tabText(i) for i in range(win.document_tabs.count())]
        assert any(label.startswith("质控") for label in labels)
    finally:
        win.close()
        app.processEvents()


def test_formal_project_can_add_multiple_lines_in_background(tmp_path: Path) -> None:
    first_source = tmp_path / "first.csv"
    second_source = tmp_path / "second.csv"
    np.savetxt(first_source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    np.savetxt(second_source, np.arange(30, dtype=np.float32).reshape(6, 5), delimiter=",")
    temporary = IngestService.open_temporary(first_source)
    formal = IngestService.formalize(temporary, tmp_path / "formal_multi", name="Multi")
    formal.close()
    temporary.close()

    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(tmp_path / "formal_multi")
        first_line_id = win.project.list_lines()[0].line_id
        win.open_line_document(first_line_id)
        win.import_line_async(second_source)
        deadline = time.time() + 15
        while time.time() < deadline and win._task_threads:
            app.processEvents()
            time.sleep(0.01)
        assert win.project is not None
        assert len(win.project.list_lines()) == 2
        assert win.project_tree.topLevelItem(0).childCount() == 2
        labels = [win.document_tabs.tabText(i) for i in range(win.document_tabs.count())]
        assert "测线 · first" in labels
        assert "测线 · second" in labels
    finally:
        win.close()
        app.processEvents()


def test_workbench_processing_result_tree_opens_saved_version_document(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    temporary = IngestService.open_temporary(source)
    formal = IngestService.formalize(temporary, tmp_path / "formal_result_tree", name="Result Tree")
    line_id = formal.list_lines()[0].line_id
    result = formal.save_processing_result(
        line_id,
        np.arange(20, dtype=np.float32).reshape(5, 4),
        name="Dewow Preview",
        processing_chain=[{"method": "dewow", "params": {"window": 8}}],
    )
    formal.close()
    temporary.close()

    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(tmp_path / "formal_result_tree")
        result_group = win.project_tree.topLevelItem(1)
        assert result_group.text(0) == "处理结果"
        assert result_group.childCount() == 1
        result_item = result_group.child(0)
        assert result_item.text(0) == "Dewow Preview"

        # Use Qt's role enum directly; the child payload should identify a saved
        # processing result rather than an inert JSON path.
        from PyQt6.QtCore import Qt

        assert result_item.data(0, Qt.ItemDataRole.UserRole) == (
            "result",
            line_id,
            result.result_id,
        )
        win.switch_workspace("processing_lab")
        assert win.active_workspace == "processing_lab"
        win.project_tree.setCurrentItem(result_item)
        app.processEvents()

        assert win.active_workspace == "data_management"
        assert win.workspace_stack.currentWidget() is win.document_tabs
        labels = [win.document_tabs.tabText(i) for i in range(win.document_tabs.count())]
        assert "处理结果 · Dewow Preview" in labels
        assert win.inspector_title.text() == "Dewow Preview"
        assert result.result_id in win.inspector_body.text()
    finally:
        win.close()
        app.processEvents()

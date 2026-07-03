from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

import app_qt
from core.ingest_service import IngestService
from ui.legacy_processing_bridge import LegacyProcessingBridge
from ui.field_workbench_window import FieldWorkbenchWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_default_main_window_factory_returns_new_workbench() -> None:
    app = _app()
    win = app_qt.create_main_window("MyGPR Test")
    try:
        assert isinstance(win, FieldWorkbenchWindow)
    finally:
        win.close()
        app.processEvents()


def test_legacy_bridge_only_writes_result_after_explicit_save(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(30, dtype=np.float32).reshape(6, 5), delimiter=",")
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(temporary, tmp_path / "formal", name="Formal")
    temporary.close()
    app = _app()
    bridge = LegacyProcessingBridge(project)
    try:
        line_id = project.list_lines()[0].line_id
        win = bridge.open_line(line_id, state="formal_ready")
        app.processEvents()
        assert list((project.root / "results").rglob("result.json")) == []

        result = bridge.save_current_result(name="Explicit")
        assert result.line_id == line_id
        assert (project.root / "results" / line_id / result.result_id / "result.json").exists()
    finally:
        bridge.close()
        project.close()
        app.processEvents()


def test_legacy_bridge_ignores_unalignable_optional_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(30, dtype=np.float32).reshape(6, 5), delimiter=",")
    (tmp_path / "rtk.csv").write_text(
        "timestamp_s,longitude,latitude\n0,104,30\n1,104.1,30.1\n",
        encoding="utf-8",
    )
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(temporary, tmp_path / "formal_sidecar", name="Formal")
    temporary.close()
    app = _app()
    bridge = LegacyProcessingBridge(project)
    try:
        line_id = project.list_lines()[0].line_id
        win = bridge.open_line(line_id, state="formal_ready")
        app.processEvents()
        assert win.data is not None
        assert win.data.shape == (6, 5)
    finally:
        bridge.close()
        project.close()
        app.processEvents()


def test_workbench_blocks_legacy_processing_when_raw_integrity_fails(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(30, dtype=np.float32).reshape(6, 5), delimiter=",")
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(temporary, tmp_path / "formal_blocked", name="Formal")
    line = project.list_lines()[0]
    copied = project.resolve_relative_path(line.raw_files[0].path)
    copied.chmod(0o666)
    copied.write_text("tampered", encoding="utf-8")
    project.close()
    temporary.close()

    app = _app()
    win = FieldWorkbenchWindow()
    try:
        win.open_project(tmp_path / "formal_blocked")
        win.selected_line_id = line.line_id
        with pytest.raises(PermissionError):
            win.open_selected_in_legacy()
        assert win.legacy_bridge is None
    finally:
        win.close()
        app.processEvents()

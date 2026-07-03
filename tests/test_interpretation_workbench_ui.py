from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.ingest_service import IngestService
from ui.interpretation_workbench_page import InterpretationWorkbenchPage
from ui.workbench_window import MyGPRWorkbenchWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _formal_project(tmp_path: Path) -> Path:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(120, dtype=np.float32).reshape(20, 6), delimiter=",")
    temporary = IngestService.open_temporary(source)
    formal = IngestService.formalize(temporary, tmp_path / "formal", name="Formal")
    formal.close()
    temporary.close()
    return tmp_path / "formal"


def test_workbench_interpretation_workspace_creates_point_line_and_interval(tmp_path: Path) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(_formal_project(tmp_path))
        line_id = win.project.list_lines()[0].line_id
        win.selected_line_id = line_id
        win.switch_workspace("interpretation")
        page = win.interpretation_workbench
        assert isinstance(page, InterpretationWorkbenchPage)
        assert page.line_id == line_id

        page.add_point(trace=2, sample=4, confidence=0.8, label="点")
        page.add_interface_line(points=[(0, 2), (3, 4), (5, 6)], confidence=0.9, label="界面")
        page.add_interval(
            trace_start=1,
            trace_end=4,
            sample_start=5,
            sample_end=10,
            confidence=0.7,
            label="区间",
        )
        assert page.feature_table.rowCount() == 3
        assert (win.project.root / "interpretations" / f"{line_id}.geojson").exists()
    finally:
        win.close()
        app.processEvents()


def test_interpretation_raw_source_switch_ignores_processing_qc_gate(
    tmp_path: Path,
) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(_formal_project(tmp_path))
        line_id = win.project.list_lines()[0].line_id
        win.project.save_processing_result(
            line_id,
            np.ones((20, 6), dtype=np.float32),
            name="Test Result",
            processing_chain=[],
        )
        win.selected_line_id = line_id
        win.switch_workspace("interpretation")
        page = win.interpretation_workbench
        result_index = page.source_combo.findText("处理结果 · Test Result")
        assert result_index >= 0

        page.source_combo.setCurrentIndex(result_index)
        app.processEvents()
        assert page.source_result_id is not None

        page.source_combo.setCurrentIndex(0)
        app.processEvents()
        assert page.source_result_id is None
        assert page.data is not None
        assert page.data.shape == (20, 6)
    finally:
        win.close()
        app.processEvents()

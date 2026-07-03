from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.ingest_service import IngestService
from ui.spatial_synthesis_page import SpatialSynthesisPage
from ui.workbench_window import MyGPRWorkbenchWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_workbench_spatial_workspace_is_real_page(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    temporary = IngestService.open_temporary(source)
    formal = IngestService.formalize(temporary, tmp_path / "formal", name="Spatial")
    formal.close()
    temporary.close()

    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(tmp_path / "formal")
        win.switch_workspace("spatial")
        assert isinstance(win.spatial_synthesis, SpatialSynthesisPage)
        assert win.spatial_synthesis.summary_table.rowCount() >= 1
        assert "无空间定位" in win.spatial_synthesis.status_label.text()
    finally:
        win.close()
        app.processEvents()

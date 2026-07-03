from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.ingest_service import IngestService
from ui.delivery_page import DeliveryPage
from ui.workbench_window import MyGPRWorkbenchWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_workbench_delivery_workspace_runs_checks_and_builds_package(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    temporary = IngestService.open_temporary(source)
    formal = IngestService.formalize(temporary, tmp_path / "formal", name="Delivery")
    formal.close()
    temporary.close()

    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(tmp_path / "formal")
        win.switch_workspace("delivery")
        page = win.delivery_page
        assert isinstance(page, DeliveryPage)
        checks = page.run_checks()
        assert page.check_table.rowCount() >= 1
        assert checks["summary"]["error_count"] == 0
        package = page.build_package("ui_delivery")
        assert package.exists()
        assert win.evidence_table.rowCount() >= 4
        evidence_roles = [
            win.evidence_table.item(row, 0).text()
            for row in range(win.evidence_table.rowCount())
        ]
        assert "成果清单" in evidence_roles
        assert "文件校验清单" in evidence_roles
    finally:
        win.close()
        app.processEvents()

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.ingest_service import IngestService
from core.project_models import QcItem, QcReportV1
from core.project_service import ProjectService
from ui.workbench_window import MyGPRWorkbenchWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_windows_environment_scripts_cover_pywavelets_and_local_installer() -> None:
    root = Path(__file__).resolve().parents[1]
    checker = (root / "check_mygpr_environment.bat").read_text(encoding="utf-8")
    launcher = (root / "start_mygpr.bat").read_text(encoding="utf-8")
    installer = (root / "install_mygpr_environment.bat").read_text(encoding="utf-8")

    assert "pywt" in checker
    for version in ("3.13", "3.12", "3.11", "3.10"):
        assert version in checker
    assert "install_mygpr_environment.bat" in launcher
    assert "requirements-dev.txt" in installer
    assert ".venv" in installer
    for alias in ("启动MyGPR.bat", "启动MyGPR_调试日志.bat", "检查MyGPR环境.bat", "安装MyGPR环境.bat"):
        assert (root / alias).exists()


def test_workbench_status_label_tracks_project_gate_states(tmp_path: Path) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    try:
        assert "未打开工程" in win.workflow_status_label.text()

        win.open_loose_path(source)
        assert "临时检查" in win.workflow_status_label.text()

        temporary = IngestService.open_temporary(source)
        formal = IngestService.formalize(temporary, tmp_path / "formal", name="Formal")
        line_id = formal.list_lines()[0].line_id
        formal.close()
        temporary.close()

        win.open_project(tmp_path / "formal")
        win.selected_line_id = line_id
        report = QcReportV1(
            line_id=line_id,
            items=[QcItem("rtk_missing", "warning", "未发现 RTK 辅助文件。")],
            created_at="2026-06-08T00:00:00Z",
        )
        win._sync_actions(report)
        assert "待确认" in win.workflow_status_label.text()

        report.items[0].acknowledged = True
        report.items[0].acknowledgement_note = "现场确认无 RTK"
        win._sync_actions(report)
        assert "正式就绪" in win.workflow_status_label.text()
    finally:
        win.close()
        app.processEvents()

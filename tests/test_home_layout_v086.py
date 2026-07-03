from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ui.field_workbench_window import FieldWorkbenchWindow


def _app():
    return QApplication.instance() or QApplication([])


def test_home_page_removes_attention_card_and_keeps_overview_sections() -> None:
    app = _app()
    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.86")
    try:
        win.show()
        app.processEvents()
        texts = [label.text() for label in win.findChildren(__import__("PyQt6.QtWidgets").QtWidgets.QLabel)]
        assert "今日关注" not in texts
        assert "项目流程概览" in texts
        assert "模块快速概览" in texts
        assert "最近项目活动" in texts
        assert "交付成果概览" in texts
    finally:
        win.close()

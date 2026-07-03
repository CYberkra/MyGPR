from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QFrame, QLabel

from ui.field_workbench_window import FieldWorkbenchWindow


def _app():
    return QApplication.instance() or QApplication([])


def test_home_activity_card_uses_compact_tiles() -> None:
    app = _app()
    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.87")
    try:
        win.show()
        app.processEvents()
        texts = [label.text() for label in win.findChildren(QLabel)]
        assert "今日关注" not in texts
        assert "最近项目活动" in texts
        tiles = [w for w in win.findChildren(QFrame) if w.objectName() == "activityTile"]
        assert len(tiles) >= 2
    finally:
        win.close()

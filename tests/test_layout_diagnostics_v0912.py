from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget

from ui.field_panels.capture_service import collect_layout_diagnostics


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_layout_diagnostics_collects_explicit_layout_keys_without_breaking_qss_object_names() -> None:
    app = _app()
    window = QWidget()
    window.resize(1536, 816)
    child = QWidget(window)
    child.setObjectName("card")
    child.setProperty("layoutKey", "processingRawBscanCard")
    child.setGeometry(20, 30, 640, 360)
    window.show()
    app.processEvents()

    payload = collect_layout_diagnostics(window, "processing_lab")

    assert payload["workspace_key"] == "processing_lab"
    assert payload["window"] == {"width": 1536, "height": 816}
    assert payload["widgets"]["processingRawBscanCard"]["width"] == 640
    assert payload["widgets"]["processingRawBscanCard"]["height"] == 360
    assert child.objectName() == "card"

    window.close()

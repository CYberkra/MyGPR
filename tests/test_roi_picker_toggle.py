from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app_qt import GPRGuiQt
from ui.autotune_tuning_page import AutoTuneTuningPage


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_manual_roi_picker_defaults_off_and_is_explicitly_gated():
    app = _get_app()
    win = GPRGuiQt("MyGPR V-test")
    try:
        assert win._is_manual_roi_pick_enabled() is False
        assert win.page_auto_tune.btn_pick_roi.isChecked() is False
        assert "全图" in win.page_auto_tune.roi_picker_status_label.text()

        win._set_manual_roi_pick_enabled(True)
        assert win._is_manual_roi_pick_enabled() is True
        assert win.page_auto_tune.btn_pick_roi.isChecked() is True
        assert "框选" in win.page_auto_tune.roi_picker_status_label.text()

        win._set_manual_roi_pick_enabled(False)
        assert win._is_manual_roi_pick_enabled() is False
        assert win.page_auto_tune.btn_pick_roi.isChecked() is False
    finally:
        win.close()
        app.processEvents()


def test_autotune_page_exposes_roi_picker_switch_as_manual_mode_gate():
    app = _get_app()
    page = AutoTuneTuningPage()
    try:
        assert page.btn_pick_roi.text() == "框选"
        assert page.btn_pick_roi.isChecked() is False
        assert page.btn_pick_roi.isEnabled() is False

        page.region_mode_combo.setCurrentIndex(2)  # manual mode
        assert page.state.roi_mode == "manual"
        assert page.btn_pick_roi.isEnabled() is True

        page.btn_pick_roi.setChecked(True)
        assert page.btn_pick_roi.isChecked() is True
        assert "框选" in page.roi_picker_status_label.text()
    finally:
        page.close()
        app.processEvents()

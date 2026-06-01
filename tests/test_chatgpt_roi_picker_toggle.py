from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manual_roi_picker_defaults_off_and_is_gated():
    app_qt = (ROOT / "app_qt.py").read_text(encoding="utf-8")
    assert "self._manual_roi_pick_enabled = False" in app_qt
    assert "def _is_manual_roi_pick_enabled" in app_qt
    assert "self._is_manual_roi_pick_enabled() and \"shift\" in key" in app_qt
    assert "if self._is_manual_roi_pick_enabled() and self._manual_roi_values is not None" in app_qt
    assert "图上 ROI 框选关闭" in app_qt


def test_autotune_page_exposes_roi_picker_switch():
    page = (ROOT / "ui" / "autotune_tuning_page.py").read_text(encoding="utf-8")
    assert "QCheckBox(\"启用图上框选 ROI\")" in page
    assert "self.btn_pick_roi.setChecked(False)" in page
    assert "set_plot_roi_picker_status" in page
    assert "Shift+拖拽" in page

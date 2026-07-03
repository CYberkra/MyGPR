from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_QT = ROOT / "app_qt.py"
BRANDING = ROOT / "ui" / "app_branding.py"
REPORT_CONTROLLER = ROOT / "ui" / "report_export_controller.py"
DISPLAY_MIXIN = ROOT / "ui" / "main_window_display_mixin.py"
SOURCE = APP_QT.read_text(encoding="utf-8")
DISPLAY_SOURCE = DISPLAY_MIXIN.read_text(encoding="utf-8")
BRANDING_SOURCE = BRANDING.read_text(encoding="utf-8")
REPORT_SOURCE = REPORT_CONTROLLER.read_text(encoding="utf-8")


def test_main_bscan_grid_is_disabled_by_default():
    assert "ax.grid(False)" in SOURCE
    assert "ax.grid(True, linestyle=\":\", alpha=0.5)" not in SOURCE
    assert "ax.grid(True, linestyle=\":\", alpha=0.3)" not in SOURCE


def test_main_bscan_toolbar_does_not_duplicate_display_chips():
    assert "self._plot_display_mode_chip = None" in SOURCE
    assert "self._plot_colormap_chip = None" in SOURCE
    assert "self._plot_range_chip = None" in SOURCE
    assert 'plot_toolbar_layout.addWidget(self._plot_display_mode_chip)' not in SOURCE
    assert 'plot_toolbar_layout.addWidget(self._plot_colormap_chip)' not in SOURCE
    assert 'plot_toolbar_layout.addWidget(self._plot_range_chip)' not in SOURCE


def test_high_resolution_toolbar_export_is_present():
    assert "class HiResNavigationToolbar" in BRANDING_SOURCE
    assert "EXPORT_DPI = 600" in BRANDING_SOURCE
    assert "保存高清 B-scan 图像" in BRANDING_SOURCE
    # Toolbar high-resolution export lives in ui/app_branding.py; Evidence report
    # package export was split to ui/report_export_controller.py in V0.8.1.
    assert "self.canvas.figure.savefig(" in BRANDING_SOURCE and "dpi=self.EXPORT_DPI" in BRANDING_SOURCE
    assert "HiResNavigationToolbar" in SOURCE
    assert "self.fig.savefig(image_path, dpi=600" in REPORT_SOURCE


def test_bscan_imshow_keeps_grid_disabled_and_uses_imshow():
    # V0.8.14 intentionally removed forced nearest/resample flags to avoid
    # confusing display-only changes with processing-array changes.
    assert "ax.imshow(" in DISPLAY_SOURCE
    assert "ax.grid(False)" in DISPLAY_SOURCE

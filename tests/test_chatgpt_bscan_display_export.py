from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_QT = ROOT / "app_qt.py"
SOURCE = APP_QT.read_text(encoding="utf-8")


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
    assert "class HiResNavigationToolbar" in SOURCE
    assert "EXPORT_DPI = 600" in SOURCE
    assert "保存高清 B-scan 图像" in SOURCE
    assert "self.fig.savefig(image_path, dpi=600" in SOURCE


def test_bscan_imshow_uses_nearest_interpolation_for_crisp_export():
    assert 'interpolation="nearest"' in SOURCE
    assert "resample=False" in SOURCE

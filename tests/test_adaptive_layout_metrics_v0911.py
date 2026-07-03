from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget

from ui.field_panels.layout_metrics import layout_metrics_for


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_v0912_layout_metrics_fit_windows_125_percent_work_area() -> None:
    app = _app()
    widget = QWidget()
    widget.resize(1536, 816)
    widget.show()
    app.processEvents()

    metrics = layout_metrics_for(widget)

    assert metrics.compact is True
    assert 340 <= metrics.processing_bscan_h <= 460
    # v0.9.24 compact width stays bounded; the action area is compacted instead of widening the side panel.
    assert 264 <= metrics.processing_params_w <= 300
    assert metrics.processing_bottom_max_h <= int(metrics.window_height * 0.22)
    assert metrics.spatial_map_h >= 310
    assert metrics.spatial_map_h > metrics.spatial_dem_h * 3
    assert metrics.delivery_side_min_w <= 300
    assert metrics.delivery_files_max_h <= 145

    widget.close()


def test_v0912_layout_metrics_scale_up_without_exceeding_bounds() -> None:
    app = _app()
    widget = QWidget()
    widget.resize(1920, 1080)
    widget.show()
    app.processEvents()

    metrics = layout_metrics_for(widget)

    assert metrics.compact is False
    assert metrics.content_width > 1600
    assert metrics.processing_bscan_h >= 480
    # v0.9.22 manual-processing chain needs wider side panel (min 268 non-compact)
    # v0.9.24 non-compact min raised to 272
    assert 272 <= metrics.processing_params_w <= 320
    assert metrics.spatial_map_h <= 560
    assert metrics.delivery_cover_w >= 245

    widget.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptive layout metrics for the MyGPR field workbench.

The field pages share a constrained 15.6-inch/1080P target.  This module keeps
plot heights, side-panel widths and bottom-panel heights in one place instead
of scattering fixed ``_compact_value`` calls across individual page builders.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QWidget


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


@dataclass(frozen=True)
class FieldLayoutMetrics:
    """Computed layout values for the field workbench pages."""

    window_width: int
    window_height: int
    content_width: int
    content_height: int
    compact: bool
    spacing: int

    project_table_min_h: int
    project_bottom_max_h: int
    project_ops_min_w: int
    project_ops_max_w: int
    project_summary_map_h: int
    project_summary_map_w: int
    project_preview_bscan_h: int
    project_preview_map_h: int

    processing_bscan_h: int
    processing_bottom_max_h: int
    processing_params_w: int
    processing_line_overview_max_h: int
    processing_line_map_h: int
    processing_param_label_w: int

    target_bscan_h: int
    target_table_max_h: int
    target_info_w: int
    target_preview_h: int

    spatial_map_h: int
    spatial_profile_h: int
    spatial_dem_h: int
    spatial_correlation_h: int
    spatial_info_max_h: int
    spatial_table_max_h: int
    spatial_side_w: int

    delivery_cover_w: int
    delivery_cover_min_h: int
    delivery_report_thumb_h: int
    delivery_toc_min_h: int
    delivery_side_min_w: int
    delivery_side_max_h: int
    delivery_files_max_h: int

    @property
    def is_low_height(self) -> bool:
        return self.content_height < 700

    @property
    def is_narrow(self) -> bool:
        return self.content_width < 1250


def _screen_available_size() -> tuple[int, int]:
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return 1450, 790
    rect = screen.availableGeometry()
    return max(1, rect.width()), max(1, rect.height())


def layout_metrics_for(widget: QWidget) -> FieldLayoutMetrics:
    """Return adaptive field-page layout metrics for the current window.

    The function first uses the actual top-level window geometry.  Before a
    window is shown, Qt may report tiny default sizes; in that case the primary
    screen available geometry is used as a stable fallback.
    """

    top = widget.window() if widget is not None else None
    screen_w, screen_h = _screen_available_size()
    width = int(top.width()) if top is not None and top.width() > 600 else screen_w
    height = int(top.height()) if top is not None and top.height() > 400 else screen_h
    compact = bool(getattr(widget, "compact_mode", height <= 850 or width <= 1550))

    sidebar_w = 185 if compact else 210
    header_h = 60 if compact else 70
    outer_margin = 14 if compact else 18
    spacing = 5 if compact else 6
    content_w = max(920, width - sidebar_w - outer_margin)
    # Leave a small safety band for Qt title/navigation rows and font rounding
    # differences.  Without it, compact 1080P captures can be only a few pixels
    # taller than the visible scroll area after adding title-bar action buttons.
    height_safety = 18 if compact else 0
    content_h = max(560, height - header_h - outer_margin - height_safety)

    # Page-specific primary visual heights.  These values intentionally use
    # tighter ratios plus clamps; they scale across 1366/1536/1920 logical
    # widths while preserving usable plot sizes on 1080P laptops.
    processing_bscan_h = _clamp(content_h * 0.62, 380 if compact else 430, 470 if compact else 550)
    # Tighter bottom panel: keep history visible but minimize dead space.
    processing_bottom_h = _clamp(content_h * 0.14, 105 if compact else 120, 130 if compact else 145)
    processing_params_w = _clamp(content_w * 0.17, 250 if compact else 265, 275 if compact else 310)

    target_bscan_h = _clamp(content_h * 0.52, 370 if compact else 410, 430 if compact else 520)
    target_table_h = _clamp(content_h * 0.14, 90, 105 if compact else 135)

    spatial_map_h = _clamp(content_h * 0.58, 380 if compact else 420, 490 if compact else 560)
    spatial_table_h = _clamp(content_h * 0.14, 90, 120 if compact else 150)

    delivery_side_h = _clamp(content_h * 0.28, 180 if compact else 200, 220 if compact else 270)

    return FieldLayoutMetrics(
        window_width=width,
        window_height=height,
        content_width=content_w,
        content_height=content_h,
        compact=compact,
        spacing=spacing,
        project_table_min_h=_clamp(content_h * 0.22, 135, 185 if compact else 215),
        project_bottom_max_h=_clamp(content_h * 0.12, 95, 115 if compact else 130),
        project_ops_min_w=_clamp(content_w * 0.19, 235, 290 if compact else 330),
        project_ops_max_w=_clamp(content_w * 0.22, 280, 325 if compact else 370),
        project_summary_map_h=_clamp(content_h * 0.12, 80, 110 if compact else 130),
        project_summary_map_w=_clamp(content_w * 0.075, 88, 116 if compact else 132),
        project_preview_bscan_h=_clamp(content_h * 0.10, 65, 82 if compact else 105),
        project_preview_map_h=_clamp(content_h * 0.055, 38, 48 if compact else 62),
        processing_bscan_h=processing_bscan_h,
        processing_bottom_max_h=processing_bottom_h,
        processing_params_w=processing_params_w,
        processing_line_overview_max_h=processing_bottom_h,
        processing_line_map_h=_clamp(processing_bottom_h * 0.42, 52, 64 if compact else 78),
        processing_param_label_w=_clamp(content_w * 0.052, 68, 78 if compact else 92),
        target_bscan_h=target_bscan_h,
        target_table_max_h=target_table_h,
        target_info_w=_clamp(content_w * 0.15, 200 if compact else 220, 225 if compact else 255),
        target_preview_h=_clamp(content_h * 0.065, 44, 55 if compact else 72),
        spatial_map_h=spatial_map_h,
        spatial_profile_h=_clamp(content_h * 0.20, 130 if compact else 150, 175 if compact else 220),
        spatial_dem_h=_clamp(content_h * 0.07, 55 if compact else 65, 75 if compact else 85),
        spatial_correlation_h=_clamp(content_h * 0.10, 70, 95 if compact else 115),
        spatial_info_max_h=_clamp(content_h * 0.10, 70, 95 if compact else 115),
        spatial_table_max_h=spatial_table_h,
        spatial_side_w=_clamp(content_w * 0.25, 310 if compact else 355, 380 if compact else 440),
        delivery_cover_w=_clamp(content_w * 0.22, 280 if compact else 310, 330 if compact else 400),
        delivery_cover_min_h=_clamp(content_h * 0.33, 240 if compact else 280, 300 if compact else 390),
        delivery_report_thumb_h=_clamp(content_h * 0.08, 55, 70 if compact else 90),
        delivery_toc_min_h=_clamp(content_h * 0.33, 240 if compact else 280, 300 if compact else 390),
        delivery_side_min_w=_clamp(content_w * 0.20, 250, 300 if compact else 340),
        delivery_side_max_h=delivery_side_h,
        delivery_files_max_h=_clamp(content_h * 0.12, 80, 100 if compact else 130),
    )


__all__ = ["FieldLayoutMetrics", "layout_metrics_for"]

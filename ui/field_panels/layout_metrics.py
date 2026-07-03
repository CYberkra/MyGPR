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
    # ratios plus clamps; they scale across 1366/1536/1920 logical widths while
    # preserving minimum usable plot sizes on 1080P laptops.
    processing_bscan_h = _clamp(content_h * 0.56, 360 if compact else 390, 430 if compact else 500)
    # The processing history table must show at least the original row plus
    # two recent steps on 1080P/125% screens; otherwise the new project-tree
    # workflow feels hidden even though the data is present.  Keep it below
    # the 22% layout diagnostic cap while giving the log/history rows enough
    # breathing room.
    processing_bottom_h = _clamp(content_h * 0.20, 154 if compact else 168, 170 if compact else 190)
    # The manual-processing chain introduced in v0.9.22 needs a slightly wider
    # side panel than the old single-method parameter strip.  Keep it bounded so
    # the two B-scan panels remain dominant, but do not squeeze spin boxes and
    # action buttons below their usable width on 1080P/125% Windows captures.
    # v0.9.24: 264 min in compact mode with full-screen window (1530x810).
    # Button spacing handled in processing_page.py setSpacing(10).
    processing_params_w = _clamp(content_w * 0.185, 264 if compact else 278, 285 if compact else 320)

    target_bscan_h = _clamp(content_h * 0.46, 320 if compact else 360, 370 if compact else 450)
    target_table_h = _clamp(content_h * 0.18, 112, 140 if compact else 180)

    spatial_map_h = _clamp(content_h * 0.52, 330 if compact else 370, 430 if compact else 500)
    spatial_table_h = _clamp(content_h * 0.22, 135, 185 if compact else 220)

    delivery_side_h = _clamp(content_h * 0.32, 200 if compact else 220, 250 if compact else 300)

    return FieldLayoutMetrics(
        window_width=width,
        window_height=height,
        content_width=content_w,
        content_height=content_h,
        compact=compact,
        spacing=spacing,
        project_table_min_h=_clamp(content_h * 0.26, 155, 210 if compact else 245),
        project_bottom_max_h=_clamp(content_h * 0.19, 140, 160 if compact else 185),
        project_ops_min_w=_clamp(content_w * 0.20, 245, 300 if compact else 340),
        project_ops_max_w=_clamp(content_w * 0.24, 300, 340 if compact else 390),
        project_summary_map_h=_clamp(content_h * 0.16, 96, 126 if compact else 145),
        project_summary_map_w=_clamp(content_w * 0.075, 88, 116 if compact else 132),
        project_preview_bscan_h=_clamp(content_h * 0.13, 84, 110 if compact else 135),
        project_preview_map_h=_clamp(content_h * 0.075, 50, 62 if compact else 82),
        processing_bscan_h=processing_bscan_h,
        processing_bottom_max_h=processing_bottom_h,
        processing_params_w=processing_params_w,
        processing_line_overview_max_h=processing_bottom_h,
        processing_line_map_h=_clamp(processing_bottom_h * 0.42, 52, 64 if compact else 78),
        processing_param_label_w=_clamp(content_w * 0.052, 68, 78 if compact else 92),
        target_bscan_h=target_bscan_h,
        target_table_max_h=target_table_h,
        target_info_w=_clamp(content_w * 0.16, 210 if compact else 235, 235 if compact else 270),
        target_preview_h=_clamp(content_h * 0.075, 50, 64 if compact else 82),
        spatial_map_h=spatial_map_h,
        spatial_profile_h=_clamp(content_h * 0.13, 100 if compact else 115, 130 if compact else 160),
        spatial_dem_h=_clamp(content_h * 0.10, 70 if compact else 80, 90 if compact else 100),
        spatial_correlation_h=_clamp(content_h * 0.135, 88, 118 if compact else 145),
        spatial_info_max_h=_clamp(content_h * 0.17, 105, 140 if compact else 170),
        spatial_table_max_h=spatial_table_h,
        spatial_side_w=_clamp(content_w * 0.28, 345 if compact else 390, 410 if compact else 470),
        delivery_cover_w=_clamp(content_w * 0.24, 300 if compact else 340, 360 if compact else 430),
        delivery_cover_min_h=_clamp(content_h * 0.43, 300 if compact else 350, 380 if compact else 480),
        delivery_report_thumb_h=_clamp(content_h * 0.115, 72, 96 if compact else 124),
        delivery_toc_min_h=_clamp(content_h * 0.43, 300 if compact else 350, 380 if compact else 480),
        delivery_side_min_w=_clamp(content_w * 0.20, 250, 300 if compact else 340),
        delivery_side_max_h=delivery_side_h,
        delivery_files_max_h=_clamp(content_h * 0.17, 105, 135 if compact else 170),
    )


__all__ = ["FieldLayoutMetrics", "layout_metrics_for"]

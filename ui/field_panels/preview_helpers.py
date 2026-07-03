#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common preview drawing helpers for the MyGPR field workbench."""

from __future__ import annotations

import numpy as np

from ui.field_panels.plots import draw_bscan


def _set_comfort_limits(ax, xs, ys, *, min_span: float = 24.0, pad_ratio: float = 0.12) -> None:
    """Set readable limits for short/single-point plan previews.

    Matplotlib's autoscale can either zoom to a point or make a short segment
    float inside a visually empty chart.  This helper gives sparse field data a
    stable, intentionally padded viewport.
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_span = max(x_max - x_min, min_span)
    y_span = max(y_max - y_min, min_span)
    span = max(x_span, y_span)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    pad = span * pad_ratio
    ax.set_xlim(x_mid - span / 2.0 - pad, x_mid + span / 2.0 + pad)
    ax.set_ylim(y_mid - span / 2.0 - pad, y_mid + span / 2.0 + pad)


class FieldPreviewMixin:
    """Shared B-scan and trajectory preview drawing methods."""

    def _draw_current_line_bscan(self, canvas, *, title: str = "") -> None:
        dataset = self.processed_gpr_dataset or self.active_gpr_dataset
        if dataset is not None:
            draw_bscan(
                canvas,
                title=title,
                diff=False,
                gain=1.0,
                data_matrix=dataset.matrix,
                distance_axis_m=dataset.distance_axis_m,
                depth_axis_m=dataset.depth_axis_m,
            )
        else:
            self._draw_empty_plot(canvas, "暂无可预览 B-scan\n请先导入当前项目测线数据")

    def _draw_empty_plot(self, canvas, message: str = "暂无数据") -> None:
        fig = canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#FAFCFE")
        ax.text(
            0.5,
            0.54,
            message,
            ha="center",
            va="center",
            fontsize=10,
            color="#6B7D90",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#E2EAF1")
        fig.tight_layout(pad=0.55)
        canvas.draw_idle()

    def _draw_current_line_strip(self, canvas, *, marker: float | None = None) -> None:
        if self.trajectory_model is None or not getattr(self.trajectory_model, "points", None):
            self._draw_empty_plot(canvas, "暂无测线轨迹\n请先导入含定位信息的测线")
            return
        fig = canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        xs = np.asarray(self.trajectory_model.x, dtype=float)
        ys = np.asarray(self.trajectory_model.y, dtype=float)
        if xs.size == 0 or ys.size == 0:
            self._draw_empty_plot(canvas, "暂无测线轨迹")
            return
        ax.plot(xs, ys, color="#0D91B2", linewidth=1.8, marker="o", markersize=2.4, markevery=max(1, len(xs)//18))
        if marker is not None and len(self.trajectory_model.distance) >= 2:
            try:
                point = self.trajectory_model.interpolate(float(marker))
                ax.scatter([point.x], [point.y], marker="v", s=52, color="#F04438", zorder=3)
            except Exception:
                pass
        _set_comfort_limits(ax, xs, ys, min_span=24.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, color="#E8EEF4", linewidth=0.6)
        ax.set_facecolor("#FAFCFE")
        for spine in ax.spines.values():
            spine.set_color("#D8E2EA")
        fig.tight_layout(pad=0.35)
        canvas.draw_idle()


__all__ = ["FieldPreviewMixin", "_set_comfort_limits"]

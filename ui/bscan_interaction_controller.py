# -*- coding: utf-8 -*-
"""B-scan mouse/keyboard interaction controller for MyGPR.

This controller owns the main Matplotlib B-scan interaction surface: hover
crosshair, trace selection, pan/zoom, manual ROI drag, slider compare dragging,
and view-limit persistence.  It deliberately reads and writes the host window's
existing attributes so V0.8.3 can reduce ``app_qt.py`` without changing runtime
behavior or public GUI method names.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)


class BscanInteractionController:
    """Manage main B-scan canvas interactions and transient plot overlays."""

    def __init__(self, host):
        self.host = host

    def on_main_canvas_press(self, event):
        """Record main-plot mouse press. Default inspect mode does not pan on left drag."""
        host = self.host
        if host._toolbar_mode_active():
            return
        if (
            event.inaxes not in host._main_plot_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return

        button = host._event_button_number(event)
        if event.dblclick:
            host._reset_main_plot_view_to_default()
            return

        ax = event.inaxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        action = "inspect"
        if button in {2, 3}:
            action = "pan"
        elif button == 1 and host._is_manual_roi_pick_enabled():
            action = "roi"
        elif button == 1 and host._is_slider_compare_split_hit(event):
            action = "slider"
        elif button != 1:
            return

        host._main_press_state = {
            "x": float(event.x),
            "y": float(event.y),
            "xdata": float(event.xdata),
            "ydata": float(event.ydata),
            "key": str(event.key or ""),
            "button": button,
            "action": action,
            "axes": ax,
            "dragging": False,
            "xlim_start": (float(xlim[0]), float(xlim[1])),
            "ylim_start": (float(ylim[0]), float(ylim[1])),
            "bbox_width": max(1.0, float(ax.bbox.width)),
            "bbox_height": max(1.0, float(ax.bbox.height)),
        }

    def on_main_canvas_motion(self, event):
        """Update hover readout and handle explicit drag modes."""
        host = self.host
        host._update_plot_coord_label(event)
        host._update_hover_crosshair(event)
        if host._main_press_state is None or host._toolbar_mode_active():
            return
        if event.x is None or event.y is None:
            return

        start = host._main_press_state
        dx = float(event.x) - float(start["x"])
        dy = float(event.y) - float(start["y"])
        if np.hypot(dx, dy) < host._main_drag_threshold_px:
            return

        start["dragging"] = True
        action = str(start.get("action") or "inspect")

        if action == "roi":
            if event.xdata is not None and event.ydata is not None:
                host._draw_drag_roi_preview(start, event)
            return

        if action == "slider":
            if event.xdata is not None:
                host._update_main_slider_compare_ratio_from_event(event)
            return

        if action == "pan":
            host._pan_main_plot_by_pixels(start, dx, dy)
            return

    def on_main_canvas_release(self, event):
        """Handle click selection, ROI commit, slider commit, and pan completion."""
        host = self.host
        if host._main_press_state is None:
            host._capture_main_view_limits_from_axes()
            return

        start = host._main_press_state
        host._main_press_state = None
        host._remove_drag_roi_preview()

        if event.x is None or event.y is None:
            host._capture_main_view_limits_from_axes()
            return

        dx = float(event.x) - float(start["x"])
        dy = float(event.y) - float(start["y"])
        is_drag = bool(start.get("dragging")) and np.hypot(dx, dy) >= host._main_drag_threshold_px
        action = str(start.get("action") or "inspect")

        if is_drag and action == "roi" and event.xdata is not None and event.ydata is not None:
            x0, x1 = sorted([float(start["xdata"]), float(event.xdata)])
            y0, y1 = sorted([float(start["ydata"]), float(event.ydata)])
            if abs(x1 - x0) > 1.0e-9 and abs(y1 - y0) > 1.0e-9:
                host._manual_roi_values = {
                    "dist_start": x0,
                    "dist_end": x1,
                    "time_start": y0,
                    "time_end": y1,
                }
                host._update_manual_roi_status()
                if host.data is not None:
                    host.plot_data(host.data)
            return

        if is_drag and action == "slider":
            if event.xdata is not None:
                host._update_main_slider_compare_ratio_from_event(event, force=True)
            return

        if is_drag and action == "pan":
            host._capture_main_view_limits_from_axes()
            return

        if action == "inspect" and event.xdata is not None:
            host._select_trace_from_x(float(event.xdata))
        host._capture_main_view_limits_from_axes()

    def event_button_number(self, event) -> int | None:
        """Normalize Matplotlib button values to 1/2/3 where possible."""
        button = getattr(event, "button", None)
        try:
            return int(button)
        except Exception:
            name = str(button).lower()
            if "left" in name:
                return 1
            if "middle" in name:
                return 2
            if "right" in name:
                return 3
        return None

    def is_main_slider_compare_active(self) -> bool:
        """Return whether the main plot is in slider-compare mode."""
        host = self.host
        return bool(
            hasattr(host.page_advanced, "slider_compare_var")
            and host.page_advanced.slider_compare_var.isChecked()
        )

    def is_slider_compare_split_hit(self, event) -> bool:
        """Only let slider compare take over when press starts near the split line."""
        host = self.host
        if not host._is_main_slider_compare_active():
            return False
        if (
            event is None
            or event.inaxes not in host._main_plot_axes
            or event.xdata is None
            or host._last_display_trace_axis.size == 0
        ):
            return False
        x_min = float(host._last_display_trace_axis[0])
        x_max = float(host._last_display_trace_axis[-1])
        split_x = x_min + (x_max - x_min) * float(host._main_slider_compare_ratio)
        try:
            split_px = float(event.inaxes.transData.transform((split_x, float(event.ydata)))[0])
            return abs(float(event.x) - split_px) <= float(host._slider_hit_tolerance_px)
        except Exception:
            return False

    def update_main_slider_compare_ratio_from_event(self, event, force: bool = False):
        """Update slider compare split position with draw throttling."""
        host = self.host
        if (
            event is None
            or event.inaxes not in host._main_plot_axes
            or event.xdata is None
            or host._last_display_trace_axis.size == 0
        ):
            return

        x_min = float(host._last_display_trace_axis[0])
        x_max = float(host._last_display_trace_axis[-1])
        span = x_max - x_min
        if abs(span) < 1.0e-9:
            return

        new_ratio = (float(event.xdata) - x_min) / span
        new_ratio = max(0.0, min(1.0, new_ratio))
        if abs(new_ratio - host._main_slider_compare_ratio) < 1.0e-4:
            return

        now = time.monotonic()
        if not force and (now - host._last_slider_compare_draw_ts) < host._slider_compare_draw_interval_s:
            host._main_slider_compare_ratio = new_ratio
            return

        host._main_slider_compare_ratio = new_ratio
        host._last_slider_compare_draw_ts = now
        host._refresh_plot()

    def on_main_canvas_scroll(self, event):
        """Zoom main plot. Shift = x-only, Ctrl = y-only."""
        host = self.host
        if host._toolbar_mode_active() or host._is_main_slider_compare_active():
            return
        if (
            event.inaxes not in host._main_plot_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return

        ax = event.inaxes
        scale = 0.85 if event.button == "up" else 1.18
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = float(event.xdata)
        ydata = float(event.ydata)
        key = str(getattr(event, "key", "") or "").lower()
        scale_x = scale if "control" not in key and "ctrl" not in key else 1.0
        scale_y = scale if "shift" not in key else 1.0
        new_xlim = (
            xdata - (xdata - xlim[0]) * scale_x,
            xdata + (xlim[1] - xdata) * scale_x,
        )
        new_ylim = (
            ydata - (ydata - ylim[0]) * scale_y,
            ydata + (ylim[1] - ydata) * scale_y,
        )
        host._set_clamped_axis_limits(ax, new_xlim, new_ylim)
        host._capture_main_view_limits_from_axes()
        host.canvas.draw_idle()

    def on_main_canvas_key_press(self, event):
        """Esc clears transient states; R toggles ROI; Home/H resets view."""
        host = self.host
        key = str(getattr(event, "key", "") or "").lower()
        if key == "escape":
            host._main_press_state = None
            host._remove_drag_roi_preview()
            host.canvas.draw_idle()
            return
        if key == "r":
            host._set_manual_roi_pick_enabled(not host._is_manual_roi_pick_enabled())
            return
        if key in {"home", "h"}:
            host._reset_main_plot_view_to_default()

    def pan_main_plot_by_pixels(self, start: dict, dx_px: float, dy_px: float):
        """Pan by screen pixels for stable drag behavior."""
        host = self.host
        ax = start.get("axes")
        if ax is None:
            return
        xlim0 = start.get("xlim_start") or ax.get_xlim()
        ylim0 = start.get("ylim_start") or ax.get_ylim()
        bbox_w = max(1.0, float(start.get("bbox_width") or ax.bbox.width or 1.0))
        bbox_h = max(1.0, float(start.get("bbox_height") or ax.bbox.height or 1.0))
        x_span = float(xlim0[1] - xlim0[0])
        y_span = float(ylim0[1] - ylim0[0])
        dx_data = -float(dx_px) / bbox_w * x_span
        dy_data = -float(dy_px) / bbox_h * y_span
        new_xlim = (float(xlim0[0]) + dx_data, float(xlim0[1]) + dx_data)
        new_ylim = (float(ylim0[0]) + dy_data, float(ylim0[1]) + dy_data)
        host._set_clamped_axis_limits(ax, new_xlim, new_ylim)
        now_ts = time.perf_counter()
        if now_ts - float(host._last_main_motion_draw_ts) >= float(host._main_motion_draw_interval_s):
            host.canvas.draw_idle()
            host._last_main_motion_draw_ts = now_ts

    def set_clamped_axis_limits(self, ax, xlim, ylim):
        """Set axis limits while clamping to the current data extent."""
        host = self.host
        clamped_xlim, clamped_ylim = host._clamp_main_view_limits(xlim, ylim)
        ax.set_xlim(*clamped_xlim)
        ax.set_ylim(*clamped_ylim)

    def clamp_main_view_limits(self, xlim, ylim):
        """Clamp pan/zoom limits to avoid empty regions and pathological zoom."""
        host = self.host
        if host._last_plot_extent is None:
            return tuple(xlim), tuple(ylim)
        x0, x1, y0, y1 = [float(v) for v in host._last_plot_extent]
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        data_x_span = max(1.0e-9, xmax - xmin)
        data_y_span = max(1.0e-9, ymax - ymin)

        def _clamp_pair(pair, low, high, min_span, max_span):
            a, b = float(pair[0]), float(pair[1])
            direction = 1.0 if b >= a else -1.0
            lo, hi = sorted([a, b])
            span = hi - lo
            if span < min_span:
                center = (lo + hi) * 0.5
                lo = center - min_span * 0.5
                hi = center + min_span * 0.5
            elif span > max_span:
                span = max_span
                center = (lo + hi) * 0.5
                lo = center - span * 0.5
                hi = center + span * 0.5
            span = hi - lo
            if lo < low:
                lo = low
                hi = low + span
            if hi > high:
                hi = high
                lo = high - span
            lo = max(low, lo)
            hi = min(high, hi)
            return (lo, hi) if direction >= 0 else (hi, lo)

        min_x_span = data_x_span / max(20.0, min(200.0, data_x_span))
        min_y_span = data_y_span / max(20.0, min(200.0, data_y_span))
        return (
            _clamp_pair(xlim, xmin, xmax, min_x_span, data_x_span),
            _clamp_pair(ylim, ymin, ymax, min_y_span, data_y_span),
        )

    def update_plot_coord_label(self, event):
        """Update compact trace/sample/amplitude readout."""
        host = self.host
        if host._plot_coord_label is None:
            return
        now_ts = time.perf_counter()
        if now_ts - float(host._last_main_coord_update_ts) < float(host._main_coord_update_interval_s):
            return
        host._last_main_coord_update_ts = now_ts
        if (
            event.inaxes not in host._main_plot_axes
            or event.xdata is None
            or event.ydata is None
            or host._last_display_trace_axis.size == 0
            or host._last_display_time_axis.size == 0
            or host._last_display_data is None
        ):
            if host._last_coord_label_key != ("empty",):
                host._plot_coord_label.setText("道 --   采样 --   距离 --   时间 --   振幅 --")
                host._last_coord_label_key = ("empty",)
            return

        trace_pos = float(event.xdata)
        time_pos = float(event.ydata)
        trace_idx = int(np.argmin(np.abs(host._last_display_trace_axis - trace_pos)))
        time_idx = int(np.argmin(np.abs(host._last_display_time_axis - time_pos)))
        amplitude = float(host._last_display_data[time_idx, trace_idx])
        trace_value = float(host._last_display_trace_axis[trace_idx])
        time_value = float(host._last_display_time_axis[time_idx])
        raw_trace_idx = (
            int(host._last_display_trace_indices[trace_idx])
            if trace_idx < host._last_display_trace_indices.size
            else trace_idx
        )
        if host._is_main_slider_compare_active():
            extra_hint = "分隔线附近可拖动对比"
        elif host._is_manual_roi_pick_enabled():
            extra_hint = "ROI 框选"
        else:
            extra_hint = "查看"
        label_key = (trace_idx, time_idx, extra_hint)
        if label_key == host._last_coord_label_key:
            return
        host._last_coord_label_key = label_key
        host._plot_coord_label.setText(
            f"道 {raw_trace_idx}   采样 {time_idx}   距离 {trace_value:.2f}   时间 {time_value:.2f}   振幅 {amplitude:.4g}   {extra_hint}"
        )

    def on_main_canvas_leave(self, event):
        """Clear hover readout and transient crosshair when leaving main plot."""
        host = self.host
        if host._plot_coord_label is not None:
            host._plot_coord_label.setText("道 --   采样 --   距离 --   时间 --   振幅 --")
        host._last_coord_label_key = ("empty",)
        host._clear_hover_crosshair_artists()

    def toolbar_mode_active(self) -> bool:
        host = self.host
        return bool(host._main_toolbar is not None and getattr(host._main_toolbar, "mode", ""))

    def select_trace_from_x(self, x_value: float):
        """Select nearest displayed trace by x-coordinate."""
        host = self.host
        if (
            host._last_display_trace_axis.size == 0
            or host._last_display_trace_indices.size != host._last_display_trace_axis.size
        ):
            return
        nearest = int(np.argmin(np.abs(host._last_display_trace_axis - float(x_value))))
        host._set_selected_trace_index(int(host._last_display_trace_indices[nearest]))

    def draw_drag_roi_preview(self, start: dict, event):
        """Draw throttled transient ROI rectangle while dragging."""
        host = self.host
        ax = start.get("axes")
        if ax is None or event.xdata is None or event.ydata is None:
            return
        now = time.monotonic()
        if (now - host._last_roi_preview_draw_ts) < host._roi_preview_draw_interval_s:
            return
        host._last_roi_preview_draw_ts = now

        x0, x1 = sorted([float(start["xdata"]), float(event.xdata)])
        y0, y1 = sorted([float(start["ydata"]), float(event.ydata)])
        if host._drag_roi_preview_patch is None:
            host._drag_roi_preview_patch = Rectangle(
                (x0, y0),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                edgecolor="#38bdf8",
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
                zorder=7,
            )
            ax.add_patch(host._drag_roi_preview_patch)
        else:
            host._drag_roi_preview_patch.set_bounds(x0, y0, abs(x1 - x0), abs(y1 - y0))
        host.canvas.draw_idle()

    def remove_drag_roi_preview(self):
        """Remove transient ROI preview rectangle."""
        host = self.host
        if host._drag_roi_preview_patch is not None:
            try:
                host._drag_roi_preview_patch.remove()
            except Exception:
                pass
            host._drag_roi_preview_patch = None
            try:
                host.canvas.draw_idle()
            except Exception:
                pass

    def capture_main_view_limits_from_axes(self):
        """Capture current single-axes view limits so redraw can preserve zoom/pan."""
        host = self.host
        if len(host._main_plot_axes) != 1:
            host._main_view_limits = None
            return
        ax = host._main_plot_axes[0]
        host._main_view_limits = {
            "xlim": tuple(float(v) for v in ax.get_xlim()),
            "ylim": tuple(float(v) for v in ax.get_ylim()),
        }

    def reset_main_plot_view_to_default(self):
        """Reset main view to the active plot payload's full extent."""
        host = self.host
        host._main_view_limits = None
        active_data, _, _ = host._get_active_plot_payload(host.data)
        fallback_data = host.data if host.data is not None else active_data
        if fallback_data is None:
            return
        host.plot_data(fallback_data)
        host._capture_main_view_limits_from_axes()

    def clear_hover_crosshair_artists(self, draw: bool = True):
        """Remove transient hover crosshair lines without touching selected trace marker."""
        host = self.host
        for artist in list(getattr(host, "_hover_crosshair_artists", []) or []):
            try:
                artist.remove()
            except Exception:
                pass
        host._hover_crosshair_artists = []
        host._hover_crosshair_axes = None
        host._hover_crosshair_last_key = None
        if draw:
            try:
                host.canvas.draw_idle()
            except Exception:
                pass

    def update_hover_crosshair(self, event):
        """Reuse hover artists and throttle updates for smoother B-scan inspection."""
        host = self.host
        if host._toolbar_mode_active():
            host._clear_hover_crosshair_artists()
            return
        if (
            event is None
            or event.inaxes not in host._main_plot_axes
            or event.xdata is None
            or event.ydata is None
            or host._last_display_trace_axis.size == 0
            or host._last_display_time_axis.size == 0
        ):
            host._clear_hover_crosshair_artists()
            return

        now = time.perf_counter()
        if now - float(getattr(host, "_hover_crosshair_last_draw_ts", 0.0)) < float(getattr(host, "_hover_crosshair_draw_interval_s", 1.0 / 45.0)):
            return

        trace_idx = int(np.argmin(np.abs(host._last_display_trace_axis - float(event.xdata))))
        time_idx = int(np.argmin(np.abs(host._last_display_time_axis - float(event.ydata))))
        key = (id(event.inaxes), trace_idx, time_idx)
        if key == host._hover_crosshair_last_key:
            return
        host._hover_crosshair_last_key = key
        host._hover_crosshair_last_draw_ts = now

        x = float(host._last_display_trace_axis[trace_idx])
        y = float(host._last_display_time_axis[time_idx])
        ax = event.inaxes
        try:
            from core.theme_manager import get_theme_manager
            from ui.theme import get_effective_theme_key
            theme_key = get_effective_theme_key(get_theme_manager().get_current_theme(), widget=host)
        except Exception:
            theme_key = "light"
        color = "#5EEAD4" if theme_key == "dark" else "#0F766E"
        alpha_span = 0.075 if theme_key == "light" else 0.12
        alpha_v = 0.34 if theme_key == "light" else 0.52
        alpha_h = 0.20 if theme_key == "light" else 0.34

        if host._last_display_trace_axis.size > 1:
            if trace_idx == 0:
                half_width = abs(float(host._last_display_trace_axis[1] - host._last_display_trace_axis[0])) * 0.5
            else:
                half_width = abs(float(host._last_display_trace_axis[trace_idx] - host._last_display_trace_axis[trace_idx - 1])) * 0.5
            half_width = max(half_width, 0.5)
        else:
            half_width = 0.5

        ymin, ymax = ax.get_ylim()
        polygon_xy = np.array(
            [
                [x - half_width, ymin],
                [x - half_width, ymax],
                [x + half_width, ymax],
                [x + half_width, ymin],
                [x - half_width, ymin],
            ],
            dtype=float,
        )

        artists = list(getattr(host, "_hover_crosshair_artists", []) or [])
        needs_create = len(artists) != 3 or getattr(host, "_hover_crosshair_axes", None) is not ax
        if not needs_create:
            try:
                span, vline, hline = artists
                span.set_xy(polygon_xy)
                span.set_facecolor(color)
                span.set_alpha(alpha_span)
                span.set_visible(True)
                vline.set_xdata([x, x])
                vline.set_color(color)
                vline.set_alpha(alpha_v)
                vline.set_visible(True)
                hline.set_ydata([y, y])
                hline.set_color(color)
                hline.set_alpha(alpha_h)
                hline.set_visible(True)
                host.canvas.draw_idle()
                return
            except Exception:
                host._clear_hover_crosshair_artists(draw=False)
                needs_create = True

        if needs_create:
            span = ax.axvspan(x - half_width, x + half_width, color=color, alpha=alpha_span, zorder=8.5)
            vline = ax.axvline(x, color=color, linewidth=0.85, linestyle="-", alpha=alpha_v, zorder=9)
            hline = ax.axhline(y, color=color, linewidth=0.75, linestyle="-", alpha=alpha_h, zorder=9)
            host._hover_crosshair_artists = [span, vline, hline]
            host._hover_crosshair_axes = ax
            host.canvas.draw_idle()

    def clear_selected_trace_marker_artists(self):
        """Remove selected-trace marker artists without redrawing the whole B-scan."""
        host = self.host
        for artist in list(getattr(host, "_selected_trace_marker_artists", []) or []):
            try:
                artist.remove()
            except Exception:
                pass
        host._selected_trace_marker_artists = []

    def selected_trace_x_position(self) -> float | None:
        """Return current selected trace x-position in display coordinates."""
        host = self.host
        if host._selected_trace_index is None:
            return None
        trace_axis = np.asarray(host._last_display_trace_axis, dtype=np.float32)
        trace_indices = np.asarray(host._last_display_trace_indices, dtype=np.int32)
        if trace_axis.size == 0 or trace_indices.size != trace_axis.size:
            return None
        matches = np.flatnonzero(trace_indices == int(host._selected_trace_index))
        if matches.size == 0:
            return None
        return float(trace_axis[int(matches[0])])

    def refresh_selected_trace_marker_lightweight(self) -> bool:
        """Update only selected-trace vertical marker after trace selection."""
        host = self.host
        if not host._main_plot_axes:
            return False
        host._clear_selected_trace_marker_artists()
        selected_x = host._selected_trace_x_position()
        if selected_x is None:
            host.canvas.draw_idle()
            return True
        for ax in host._main_plot_axes:
            line = ax.axvline(selected_x, color="#0F766E", linewidth=1.1, linestyle=":", alpha=0.72, zorder=8)
            host._selected_trace_marker_artists.append(line)
        host.canvas.draw_idle()
        return True

    def draw_selected_trace_marker(self, axes, axis_info: dict):
        """Draw selected-trace marker during full B-scan rendering."""
        host = self.host
        host._selected_trace_marker_artists = []
        if host._selected_trace_index is None:
            return

        trace_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        trace_indices = np.asarray(axis_info.get("trace_indices", []), dtype=np.int32)
        if trace_axis.size == 0 or trace_indices.size != trace_axis.size:
            return

        matches = np.flatnonzero(trace_indices == int(host._selected_trace_index))
        if matches.size == 0:
            return

        selected_x = float(trace_axis[int(matches[0])])
        for ax in axes:
            line = ax.axvline(selected_x, color="#0F766E", linewidth=1.1, linestyle=":", alpha=0.72, zorder=8)
            host._selected_trace_marker_artists.append(line)

    def draw_manual_roi_marker(self, axes, axis_info: dict):
        """Draw current manual ROI on the main B-scan."""
        host = self.host
        if host._manual_roi_values is None or not axes:
            return

        trace_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        time_axis = np.asarray(axis_info.get("time_axis", []), dtype=np.float32)
        if trace_axis.size == 0 or time_axis.size == 0:
            return

        vals = host._manual_roi_values
        x0 = float(vals["dist_start"])
        x1 = float(vals["dist_end"])
        y0 = float(vals["time_start"])
        y1 = float(vals["time_end"])
        rect_x = min(x0, x1)
        rect_y = min(y0, y1)
        rect_w = abs(x1 - x0)
        rect_h = abs(y1 - y0)
        if rect_w <= 0 or rect_h <= 0:
            return

        for ax in axes[:1]:
            patch = Rectangle(
                (rect_x, rect_y),
                rect_w,
                rect_h,
                fill=False,
                edgecolor="#0ea5e9",
                linewidth=1.4,
                linestyle="--",
                alpha=0.95,
                zorder=6,
            )
            ax.add_patch(patch)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-scan interaction controller GUI smoke tests."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
from PyQt6.QtWidgets import QApplication

from app_qt import GPRGuiQt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def _close_window(app: QApplication, win: GPRGuiQt) -> None:
    win.close()
    app.processEvents()


def test_bscan_interaction_controller_wrappers_and_hover_inspect_smoke():
    app = _get_app()
    win = GPRGuiQt("MyGPR V-test")
    try:
        data = np.linspace(-1.0, 1.0, 40, dtype=np.float32).reshape(8, 5)
        win.shared_data.load_data(data, path="demo.csv", source="test")
        win.data = win.shared_data.current_data
        win.header_info = {"a_scan_length": 8, "num_traces": 5}
        win.original_data = win.shared_data.original_data
        win.plot_data(win.data)
        app.processEvents()

        assert win.bscan_interaction_controller.host is win
        assert win._main_plot_axes
        assert not win._toolbar_mode_active()

        xlim, ylim = win._clamp_main_view_limits((-999, 999), (-999, 999))
        assert len(xlim) == 2
        assert len(ylim) == 2

        win._set_selected_trace_index(2)
        assert win._selected_trace_index == 2
        # Main B-scan no longer draws persistent selected-trace markers; hover
        # readout/crosshair is the visual inspection path.
        assert win._selected_trace_marker_artists == []

        x_pos = win._selected_trace_x_position()
        assert x_pos is not None
        win._select_trace_from_x(float(x_pos))
        assert win._selected_trace_index == 2
        assert win._selected_trace_marker_artists == []

        event = SimpleNamespace(inaxes=win._main_plot_axes[0], xdata=float(x_pos), ydata=0.0, button=1, key="")
        win._update_plot_coord_label(event)
        assert "道" in win._plot_coord_label.text()

        win._clear_hover_crosshair_artists(draw=False)
        assert win._hover_crosshair_artists == []
    finally:
        _close_window(app, win)


def test_bscan_nearest_axis_index_handles_monotonic_axes():
    from ui.bscan_interaction_controller import BscanInteractionController

    inc = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    dec = np.array([4.0, 2.0, 1.0, 0.0], dtype=np.float32)
    assert BscanInteractionController._nearest_axis_index(inc, 1.7) == 2
    assert BscanInteractionController._nearest_axis_index(inc, -10.0) == 0
    assert BscanInteractionController._nearest_axis_index(inc, 99.0) == 3
    assert BscanInteractionController._nearest_axis_index(dec, 1.7) == 1
    assert BscanInteractionController._nearest_axis_index(dec, -10.0) == 3
    assert BscanInteractionController._nearest_axis_index(dec, 99.0) == 0


def test_main_canvas_draw_requests_are_coalesced():
    app = _get_app()
    win = GPRGuiQt("MyGPR V-test")
    try:
        calls = []
        win.canvas.draw_idle = lambda: calls.append("draw")  # type: ignore[method-assign]
        win._last_canvas_draw_flush_ts = 0.0
        win._request_main_canvas_draw("unit", min_interval_s=10.0)
        assert len(calls) == 1
        win._request_main_canvas_draw("unit", min_interval_s=10.0)
        assert len(calls) == 1
        assert win._pending_canvas_draw is True
        win._flush_pending_canvas_draw()
        assert len(calls) == 2
        assert win._pending_canvas_draw is False
    finally:
        _close_window(app, win)

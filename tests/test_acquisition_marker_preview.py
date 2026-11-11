"""Acquisition marks use source row numbers, never original acquisition IDs."""

from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from ui.widgets.bscan_view import BScanView
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtTest import QTest, QSignalSpy


def test_raw_marker_positions_tooltips_clear_and_fullscreen(qapp):
    host = QWidget()
    layout = QVBoxLayout(host)
    view = BScanView(host)
    layout.addWidget(view)
    raw = np.arange(72, dtype=np.float32).reshape(12, 6)
    marks = ({"marker_id": 2, "trace_no": 5, "marker_type": "general", "label": "地物 A"},)
    bundle = SimpleNamespace(
        matrix=raw,
        vmin=0,
        vmax=72,
        trace_count=11,
        trace_indices=np.arange(0, 11, 2),
        source_trace_ids=np.arange(100, 106),
        acquisition_markers=marks,
    )
    host.resize(850, 450)
    host.show()
    view.set_bundle(bundle)
    qapp.processEvents()
    assert view.acquisition_marker_count() == 1
    assert view.acquisition_marker_positions() == [2.5]
    tooltip = view.acquisition_marker_tooltip(0)
    assert "第5道" in tooltip and "地物 A" in tooltip
    np.testing.assert_array_equal(view._matrix, raw)
    view.enter_fullscreen()
    QTest.qWait(400)
    qapp.processEvents()
    assert view.acquisition_marker_count() == 1
    view.exit_fullscreen()
    view.settle_fullscreen()
    view.set_bundle(SimpleNamespace(matrix=raw, vmin=0, vmax=72))
    assert view.acquisition_marker_count() == 0
    view.set_bundle(bundle)
    view.clear()
    assert view.acquisition_marker_count() == 0
    view.close()
    host.close()


def test_marker_visibility_toggle_preserves_points_and_matrix(qapp):
    host = QWidget()
    layout = QVBoxLayout(host)
    view = BScanView(host)
    layout.addWidget(view)
    raw = np.arange(72, dtype=np.float32).reshape(12, 6)
    view.set_bundle(SimpleNamespace(
        matrix=raw,
        vmin=0,
        vmax=72,
        acquisition_markers=({"trace_no": 5, "marker_type": "general", "label": "地物 A"},),
    ))
    host.show()
    qapp.processEvents()
    positions = view.acquisition_marker_positions()
    assert view.acquisition_markers_shown()
    view.set_acquisition_markers_visible(False)
    assert not view.acquisition_markers_shown()
    assert view.acquisition_marker_positions() == positions
    np.testing.assert_array_equal(view._matrix, raw)
    view.set_acquisition_markers_visible(True)
    assert view.acquisition_markers_shown()
    assert view.acquisition_marker_positions() == positions
    np.testing.assert_array_equal(view._matrix, raw)
    host.close()


def test_marker_availability_signal_tracks_bundle_and_clear(qapp):
    view = BScanView()
    raw = np.arange(72, dtype=np.float32).reshape(12, 6)
    marked = SimpleNamespace(
        matrix=raw,
        vmin=0,
        vmax=72,
        acquisition_markers=({"trace_no": 5, "marker_type": "general"},),
    )
    availability = QSignalSpy(view.sig_acquisition_markers_availability_changed)
    view.set_bundle(marked)
    assert [event[0] for event in availability] == [True]
    view.set_bundle(SimpleNamespace(matrix=raw, vmin=0, vmax=72, acquisition_markers=()))
    assert [event[0] for event in availability] == [True, False]
    view.set_bundle(marked)
    view.clear()
    assert [event[0] for event in availability] == [True, False, True, False]
    view.close()

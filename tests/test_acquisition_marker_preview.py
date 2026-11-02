"""Acquisition marks use source row numbers, never original acquisition IDs."""

from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from ui.widgets.bscan_view import BScanView
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtTest import QTest


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
    points = view._acquisition_marker_item.points()
    assert len(points) == 1 and points[0].pos().x() == 2.5
    assert "第5道" in points[0].data() and "地物 A" in points[0].data()
    np.testing.assert_array_equal(view._matrix, raw)
    view.enter_fullscreen()
    QTest.qWait(400)
    qapp.processEvents()
    assert len(view._acquisition_marker_item.points()) == 1
    view.exit_fullscreen()
    view.settle_fullscreen()
    view.set_bundle(SimpleNamespace(matrix=raw, vmin=0, vmax=72))
    assert len(view._acquisition_marker_item.points()) == 0
    view.set_bundle(bundle)
    view.clear()
    assert len(view._acquisition_marker_item.points()) == 0
    view.close()
    host.close()

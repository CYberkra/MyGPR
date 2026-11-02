"""The registered gain parameter is an editable curve, not a JSON text box."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("qfluentwidgets")

from ui.desktop_backend_facade import method_catalog
from ui.widgets.param_form import ParamForm


def test_registered_curve_editor_roundtrips_nodes_and_emits_once(qapp):
    entry = next(item for item in method_catalog() if item.method_id == "manual_gain")
    form = ParamForm()
    form.set_schema(entry.parameter_schema)
    editor = form._editors["nodes"]
    assert type(editor).__name__ == "GainCurveEditor"
    nodes = [{"tau": 0, "db": -6}, {"tau": 0.4, "db": 3}, {"tau": 1, "db": 20}]
    changed = []
    form.sig_changed.connect(lambda: changed.append(True))
    form.set_values({"nodes": json.dumps(nodes)})
    assert changed == []
    assert json.loads(form.values()["nodes"]) == nodes
    editor.setValue([{"tau": 0, "db": 0}, {"tau": 1, "db": 6}])
    assert changed == [True]


def test_gain_dialog_edit_cancel_and_reset_do_not_overwrite_applied_nodes(qapp):
    from ui.widgets.manual_gain_editor import GainCurveDialog, GainCurveEditor

    editor = GainCurveEditor()
    original = editor.value()
    dialog = GainCurveDialog(original)
    dialog.move_node(1, 0.3, 12)
    dialog.add_node(0.6, -4)
    dialog.reject()
    assert editor.value() == original
    assert json.loads(dialog.value())[1] == {"tau": 0.3, "db": 12.0}
    dialog.reset_curve()
    assert json.loads(dialog.value()) == json.loads(original)


def test_gain_dialog_endpoint_order_and_limits_are_visible(qapp):
    from ui.widgets.manual_gain_editor import GainCurveDialog

    dialog = GainCurveDialog()
    dialog.move_node(0, 0.8, 20)
    dialog.move_node(1, 0.9, 100)
    nodes = json.loads(dialog.value())
    assert nodes[0]["tau"] == 0
    assert 0 < nodes[1]["tau"] < nodes[2]["tau"]
    assert nodes[1]["db"] == 60
    assert not dialog.remove_node(0)
    assert dialog.remove_node(1)
    assert not dialog.add_node(0.5, 0)
    assert dialog.status_label.text()


def test_double_click_on_plot_adds_a_real_node(qapp):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtTest import QTest
    from ui.widgets.manual_gain_editor import GainCurveDialog

    dialog = GainCurveDialog()
    dialog.show()
    qapp.processEvents()
    view = dialog.plot.getViewBox()
    scene_position = view.mapViewToScene(QPointF(12, 60))
    point = dialog.plot.mapFromScene(scene_position)
    QTest.mouseDClick(dialog.plot.viewport(), Qt.MouseButton.LeftButton, pos=point)
    qapp.processEvents()
    nodes = json.loads(dialog.value())
    assert len(nodes) == 6
    assert nodes[3]["tau"] == pytest.approx(0.6, abs=0.01)
    assert nodes[3]["db"] == pytest.approx(12, abs=0.5)
    dialog.close()


def test_accepting_curve_editor_updates_the_processing_parameter(qapp):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    from ui.widgets.manual_gain_editor import GainCurveEditor

    editor = GainCurveEditor()
    changed = []
    editor.valueChanged.connect(lambda: changed.append(True))

    def accept_draft():
        dialog = QApplication.activeModalWidget()
        dialog.move_node(2, 0.5, 14)
        dialog.ok_button.click()

    QTimer.singleShot(0, accept_draft)
    editor.button.click()
    assert changed == [True]
    assert json.loads(editor.value())[2]["db"] == 14


def test_dense_valid_curve_stays_ordered_when_numeric_gain_changes(qapp):
    from ui.widgets.manual_gain_editor import GainCurveDialog

    positions = [0, 0.5, 0.5000001, 0.5000002, 1]
    dialog = GainCurveDialog([{"tau": tau, "db": 0} for tau in positions])
    dialog.select_node(2)
    dialog.gain_spin.setValue(10)
    assert [node["tau"] for node in json.loads(dialog.value())] == positions
    dialog.move_node(2, 0.6, 12)
    edited = json.loads(dialog.value())
    assert positions[1] < edited[2]["tau"] < positions[3]


def test_real_mouse_drag_and_right_click_change_only_the_selected_node(qapp):
    from PyQt6.QtCore import QPointF, QEvent, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtTest import QTest
    from ui.widgets.manual_gain_editor import GainCurveDialog

    dialog = GainCurveDialog()
    dialog.show()
    qapp.processEvents()
    view = dialog.plot.getViewBox()

    def point(db, percent):
        return dialog.plot.mapFromScene(view.mapViewToScene(QPointF(db, percent)))

    start, stop = point(0, 50), point(18, 60)
    viewport = dialog.plot.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(stop), QPointF(viewport.mapToGlobal(stop)),
                        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    qapp.sendEvent(viewport, event)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=stop)
    qapp.processEvents()
    nodes = json.loads(dialog.value())
    assert nodes[2]["db"] == pytest.approx(18, abs=0.5)
    assert nodes[2]["tau"] == pytest.approx(0.6, abs=0.01)
    QTest.mouseClick(viewport, Qt.MouseButton.RightButton, pos=point(nodes[2]["db"], nodes[2]["tau"] * 100))
    qapp.processEvents()
    assert len(json.loads(dialog.value())) == 4
    dialog.close()

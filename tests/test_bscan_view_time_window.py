from types import SimpleNamespace
import numpy as np
import pytest


def make_view(qapp, axis=None, label="时间 (ns)"):
    from ui.widgets.bscan_view import BScanView

    axis = np.linspace(0, 900, 501) if axis is None else np.asarray(axis)
    data = np.random.default_rng(3).normal(size=(len(axis), 80)).astype("float32")
    v = BScanView()
    v.resize(800, 500)
    v.set_bundle(
        SimpleNamespace(
            matrix=data,
            vmin=-3,
            vmax=3,
            sample_axis=axis,
            sample_axis_label=label,
            trace_count=80,
            sample_count=len(axis),
        )
    )
    return v, data


def test_time_window_maps_ns_to_preview_rows_without_changing_data(qapp):
    v, a = make_view(qapp)
    before = a.copy()
    try:
        v._plot.vb.setXRange(10, 30, padding=0)
        old_x = v._plot.vb.viewRange()[0]
        assert v.set_view_time_window(36, 180)
        np.testing.assert_allclose(v._plot.vb.viewRange()[1], [20.5, 100.5])
        np.testing.assert_allclose(v._plot.vb.viewRange()[0], old_x)
        np.testing.assert_array_equal(v._matrix, before)
        assert v.view_time_window() == (36, 180)
        v.set_display_mode("envelope")
        np.testing.assert_allclose(v._plot.vb.viewRange()[1], [20.5, 100.5])
        v.set_display_mode("wiggle")
        np.testing.assert_allclose(v._plot.vb.viewRange()[1], [20.5, 100.5])
        np.testing.assert_allclose(v._plot.vb.viewRange()[0], old_x)
        v.restore_full_time_window()
        np.testing.assert_allclose(v._plot.vb.viewRange()[1], [0, 501])
        assert v.view_time_window() is None
    finally:
        v.close()


@pytest.mark.parametrize("start,end", [(100, 20), (-1, 100), (0, 1000), (0, float("nan")), (0, 0)])
def test_invalid_window_keeps_current_view(qapp, start, end):
    v, _ = make_view(qapp)
    try:
        assert v.set_view_time_window(18, 90)
        old = v._plot.vb.viewRange()
        assert not v.set_view_time_window(start, end)
        np.testing.assert_allclose(v._plot.vb.viewRange(), old)
    finally:
        v.close()


def test_window_uses_actual_offset_nonuniform_axis(qapp):
    v, _ = make_view(qapp, [18, 22, 30, 45, 70])
    try:
        assert v.set_view_time_window(22, 45)
        np.testing.assert_allclose(v._plot.vb.viewRange()[1], [1.5, 3.5])
        assert not v.set_view_time_window(0, 45)
    finally:
        v.close()


def test_non_time_axis_disables_ns_window(qapp):
    v, _ = make_view(qapp, [0, 1, 2, 3], label="深度 (m)")
    try:
        assert not v.time_window_available()
        assert not v.set_view_time_window(0, 2)
    finally:
        v.close()


def test_new_data_resets_window_and_fits(qapp):
    v, a = make_view(qapp)
    try:
        assert v.set_view_time_window(0, 100)
        v.set_matrix(a.copy(), -3, 3)
        assert v.view_time_window() is None
        assert not v.time_window_available()
        assert v._plot.vb.viewRange()[1][1] - v._plot.vb.viewRange()[1][0] > 400
    finally:
        v.close()


def test_view_time_dialog_validates_and_resets(qapp):
    from ui.widgets.bscan_time_window import TimeWindowDialog
    from PyQt6.QtWidgets import QDialogButtonBox

    d = TimeWindowDialog(None, (0, 900), (20, 100))
    try:
        d.start_spin.setValue(120)
        assert not d.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
        d.reset_button.click()
        assert d.start_spin.value() == 0 and d.end_spin.value() == 900
        assert d.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    finally:
        d.close()


def test_time_ticks_are_physical_ns_not_sample_indices(qapp):
    v, _ = make_view(qapp)
    try:
        texts = v._convert_left_ticks([20.5, 100.5], 1, 20)
        assert texts is not None
        np.testing.assert_allclose([float(s) for s in texts], [36, 180])
    finally:
        v.close()


def test_custom_dialog_and_internal_refit_keep_selected_time(qapp):
    from PyQt6.QtCore import QTimer
    from ui.widgets.bscan_time_window import TimeWindowDialog

    v, _ = make_view(qapp)
    try:

        def confirm():
            dialog = v.findChild(TimeWindowDialog)
            dialog.start_spin.setValue(36)
            dialog.end_spin.setValue(180)
            dialog.accept()

        QTimer.singleShot(0, confirm)
        v._edit_view_time_window()
        assert v.view_time_window() == (36, 180)
        v._fit_current_mode()
        np.testing.assert_allclose(v._plot.vb.viewRange()[1], [20.5, 100.5])
        v.set_aspect_mode("square", notify=False)
        np.testing.assert_allclose(v._plot.vb.viewRange()[1], [20.5, 100.5])
        v.fit_to_data()
        assert v.view_time_window() is None
    finally:
        v.close()


@pytest.mark.parametrize("axis", [[0, 2, 1], [0, np.nan, 3], [0, 0, 1]])
def test_invalid_physical_axis_cannot_be_used_as_time(qapp, axis):
    v, _ = make_view(qapp, axis)
    try:
        assert not v.time_window_available()
        assert not v.set_view_time_window(0, 1)
    finally:
        v.close()

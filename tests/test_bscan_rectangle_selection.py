"""Display-only rectangle zoom, local color levels, and mouse-mode lifecycle."""
from types import SimpleNamespace
from time import monotonic

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest

from ui.desktop_backend_facade import compute_display_levels
from ui.widgets.bscan_view import BScanView


@pytest.fixture
def scene(qapp):
    matrix = np.random.default_rng(42).normal(size=(100, 60)).astype('float32')
    matrix[:10] *= 1000
    matrix[40:] *= 500
    view = BScanView()
    view.resize(850, 540)
    view.set_bundle(SimpleNamespace(matrix=matrix, vmin=-1000, vmax=1000,
        sample_axis=np.arange(100) * 2., sample_axis_label='时间 (ns)'))
    view.show()
    qapp.processEvents()
    yield view, matrix
    view.close()
    view.deleteLater()
    qapp.processEvents()


def point(view, x, y):
    return view._glw.mapFromScene(view._plot.vb.mapViewToScene(QPointF(x, y)))


def drag(view, start, end, *, release=True):
    viewport = view._glw.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(end), QPointF(viewport.mapToGlobal(end)),
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    from PyQt6.QtWidgets import QApplication
    QApplication.sendEvent(viewport, event)
    if release:
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)


def test_numeric_window_also_rescales_image_and_colorbar(scene):
    view, matrix = scene
    before = matrix.copy()
    assert view.set_view_time_window(20, 78)
    expected = compute_display_levels(matrix[10:40], p_low=2, p_high=98)
    np.testing.assert_allclose(view._image_item.levels, expected)
    np.testing.assert_allclose(view._colorbar.levels(), expected)
    np.testing.assert_array_equal(matrix, before)


def test_rectangle_changes_view_and_uses_only_selected_samples_and_traces(scene):
    view, matrix = scene
    assert callable(getattr(view, 'set_view_selection', None)), 'Rectangle zoom API missing'
    assert view.set_view_selection(QRectF(10, 10, 20, 30))
    np.testing.assert_allclose(view._plot.vb.viewRange(), [[10, 30], [10, 40]])
    expected = compute_display_levels(matrix[10:40, 10:30], p_low=2, p_high=98)
    np.testing.assert_allclose(view._image_item.levels, expected)
    np.testing.assert_allclose(view._colorbar.levels(), expected)
    assert view._matrix is matrix
    for mode in ('envelope', 'wiggle', 'grayscale'):
        view.set_display_mode(mode)
        np.testing.assert_allclose(view._plot.vb.viewRange(), [[10, 30], [10, 40]])
    view.fit_to_data()
    np.testing.assert_allclose(view._image_item.levels,
        compute_display_levels(matrix, p_low=2, p_high=98))


def test_real_mouse_drag_is_one_shot_and_does_not_pick(scene, qapp):
    view, _ = scene
    assert callable(getattr(view, 'start_rectangle_zoom', None)), 'Rectangle mode missing'
    picks = []
    view.set_pick_enabled(True)
    view.sig_point_picked.connect(lambda *args: picks.append(args))
    assert view.start_rectangle_zoom()
    assert view._selection_tool.active
    assert view._glw.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    drag(view, point(view, 10, 10), point(view, 30, 40))
    qapp.processEvents()
    ranges = view._plot.vb.viewRange()
    np.testing.assert_allclose(ranges, [[10, 30], [10, 40]], atol=.3)
    assert not view._selection_tool.active
    assert not picks
    assert view._pick_enabled


@pytest.mark.parametrize('cancel', ['escape', 'right'])
def test_cancel_removes_rubber_band_and_keeps_previous_view(scene, qapp, cancel):
    view, _ = scene
    assert callable(getattr(view, 'start_rectangle_zoom', None)), 'Rectangle mode missing'
    before = view._plot.vb.viewRange()
    view.start_rectangle_zoom()
    start, end = point(view, 10, 10), point(view, 30, 40)
    drag(view, start, end, release=False)
    assert view._selection_tool.rubber.isVisible()
    if cancel == 'escape':
        QTest.keyClick(view._glw.viewport(), Qt.Key.Key_Escape)
    else:
        QTest.mouseClick(view._glw.viewport(), Qt.MouseButton.RightButton, pos=end)
    QTest.mouseRelease(view._glw.viewport(), Qt.MouseButton.LeftButton, pos=end)
    qapp.processEvents()
    assert not view._selection_tool.active
    assert not view._selection_tool.rubber.isVisible()
    np.testing.assert_allclose(view._plot.vb.viewRange(), before)


def test_tiny_click_does_not_zoom_or_pick(scene, qapp):
    view, _ = scene
    assert callable(getattr(view, 'start_rectangle_zoom', None)), 'Rectangle mode missing'
    before = view._plot.vb.viewRange()
    view.start_rectangle_zoom()
    p = point(view, 20, 20)
    drag(view, p, p + QPoint(1, 1))
    qapp.processEvents()
    np.testing.assert_allclose(view._plot.vb.viewRange(), before)
    assert not view._selection_tool.active


def test_local_levels_follow_pan_and_respect_shared_fixed_scale(scene, qapp):
    view, matrix = scene
    assert callable(getattr(view, 'set_view_selection', None)), 'Rectangle zoom API missing'
    view.set_view_selection(QRectF(10, 10, 20, 30))
    view._plot.vb.setYRange(50, 80, padding=0)
    expected = compute_display_levels(matrix[50:80, 10:30], p_low=2, p_high=98)
    deadline = monotonic() + 1.5
    while not np.allclose(view._image_item.levels, expected) and monotonic() < deadline:
        QTest.qWait(20)
    np.testing.assert_allclose(view._image_item.levels, expected)
    view.set_levels_override((-5, 5))
    view.set_view_selection(QRectF(10, 10, 20, 30))
    np.testing.assert_allclose(view._image_item.levels, [-5, 5])
    np.testing.assert_allclose(view._colorbar.levels(), [-5, 5])


def test_reload_clear_and_depth_axes_do_not_leave_a_stale_selection(scene, qapp):
    view, matrix = scene
    assert callable(getattr(view, 'start_rectangle_zoom', None)), 'Rectangle mode missing'
    view.set_view_selection(QRectF(10, 10, 20, 30))
    view.start_rectangle_zoom()
    view.set_matrix(matrix.copy(), -1000, 1000)
    assert not view._selection_tool.active
    assert view._view_selection is None
    assert view.start_rectangle_zoom()  # generic rectangle zoom needs no ns axis
    view.clear()
    assert not view._selection_tool.active
    assert not view.start_rectangle_zoom()


def test_actual_context_menu_can_arm_mode_and_focused_escape_cancels(scene, qapp):
    from ui.widgets.context_menus import SafeRoundMenu
    view, _ = scene
    view.activateWindow()
    qapp.processEvents()
    view._show_context_menu(SimpleNamespace(screenPos=lambda:
        QPointF(view._glw.viewport().mapToGlobal(point(view, 20, 20)))))
    qapp.processEvents()
    menu = view.findChild(SafeRoundMenu)
    assert menu is not None and menu.isVisible()
    assert menu.view.item(0).text() == '框选放大'
    QTest.mouseClick(menu.view.viewport(), Qt.MouseButton.LeftButton,
                     pos=menu.view.visualItemRect(menu.view.item(0)).center())
    qapp.processEvents()
    assert view._selection_tool.active
    focused = qapp.focusWidget()
    assert focused is not None
    QTest.keyClick(focused, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not view._selection_tool.active


def test_reverse_drag_and_clamped_selection_preserve_source(scene, qapp):
    view, matrix = scene
    before = matrix.copy()
    view.start_rectangle_zoom()
    drag(view, point(view, 30, 40), point(view, 10, 10))
    qapp.processEvents()
    np.testing.assert_allclose(view._plot.vb.viewRange(), [[10, 30], [10, 40]], atol=.3)
    assert view.set_view_selection(QRectF(-20, -10, 40, 40))
    np.testing.assert_allclose(view._plot.vb.viewRange(), [[0, 20], [0, 30]])
    old = view._plot.vb.viewRange()
    assert not view.set_view_selection(QRectF(0, 0, 0, 5))
    assert not view.set_view_selection(QRectF(float('nan'), 0, 4, 5))
    np.testing.assert_allclose(view._plot.vb.viewRange(), old)
    np.testing.assert_array_equal(matrix, before)


def test_toggle_scope_and_nonfinite_or_empty_region_are_safe(scene, qapp):
    view, matrix = scene
    matrix[20:30, 20:30] = np.nan
    assert view.set_view_selection(QRectF(20, 20, 10, 10))
    assert np.isfinite(view._image_item.levels).all()
    view._plot.vb.setYRange(200, 250, padding=0)
    QTest.qWait(80)
    assert np.isfinite(view._image_item.levels).all()
    view._toggle_selection_levels(False)
    np.testing.assert_allclose(view._image_item.levels,
        compute_display_levels(matrix, p_low=2, p_high=98))


def test_envelope_and_gain_levels_use_displayed_region(scene, qapp):
    view, matrix = scene
    before = matrix.copy()
    view.set_view_selection(QRectF(10, 10, 20, 30))
    view.set_display_mode('envelope')
    view.set_gain('tvg', db=12)
    target = view._image_item.image
    expected = compute_display_levels(target[10:40, 10:30], p_low=2, p_high=98)
    np.testing.assert_allclose(view._image_item.levels, [0, expected[1]])
    np.testing.assert_array_equal(matrix, before)


def test_regular_pick_still_works_after_selection_and_views_are_independent(scene, qapp):
    view, _ = scene
    other = BScanView()
    other.set_matrix(np.ones((100, 60)), -1, 1)
    before = other._plot.vb.viewRange()
    try:
        view.set_view_selection(QRectF(10, 10, 20, 30))
        picks = []
        view.sig_point_picked.connect(lambda *args: picks.append(args))
        view.set_pick_enabled(True)
        QTest.mouseClick(view._glw.viewport(), Qt.MouseButton.LeftButton, pos=point(view, 20.5, 25.5))
        qapp.processEvents()
        assert picks == [(20, 25)]
        np.testing.assert_allclose(other._plot.vb.viewRange(), before)
    finally:
        other.close()


def test_escape_cancels_selection_before_leaving_fullscreen(scene, qapp):
    from PyQt6.QtWidgets import QWidget, QVBoxLayout
    view, _ = scene
    root = QWidget()
    QVBoxLayout(root).addWidget(view)
    root.show()
    view.enter_fullscreen()
    qapp.processEvents()
    try:
        view.start_rectangle_zoom()
        QTest.keyClick(qapp.focusWidget(), Qt.Key.Key_Escape)
        qapp.processEvents()
        assert not view._selection_tool.active
        assert view.is_fullscreen()
        QTest.keyClick(view._glw, Qt.Key.Key_Escape)
        qapp.processEvents()
        assert not view.is_fullscreen()
    finally:
        view.exit_fullscreen()
        view.setParent(None)
        root.close()

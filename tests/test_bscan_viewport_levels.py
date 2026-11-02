"""Viewport color levels must work without first arming rectangle selection.

Prepared for GitHub CI; user policy prohibits running these tests locally.
"""
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtTest import QTest

from ui.desktop_backend_facade import compute_display_levels
from ui.widgets.bscan_view import BScanView


@pytest.fixture
def viewport(qapp):
    raw = np.random.default_rng(52).normal(size=(100, 60)).astype(np.float32)
    raw[:10] *= 1000
    raw[50:] *= 500
    view = BScanView()
    view.resize(850, 540)
    view.set_matrix(raw, -1000, 1000)
    view.show()
    qapp.processEvents()
    yield view, raw
    view.close()
    view.deleteLater()
    qapp.processEvents()


def wait_levels(view, expected):
    deadline = monotonic() + 1.5
    while monotonic() < deadline and not np.allclose(view._image_item.levels, expected):
        QTest.qWait(20)
    np.testing.assert_allclose(view._image_item.levels, expected)
    np.testing.assert_allclose(view._colorbar.levels(), expected)


def zoom(view, left=10, right=30, top=10, bottom=40):
    view._plot.vb.setRange(xRange=(left, right), yRange=(top, bottom), padding=0)


def test_plain_zoom_recalculates_levels_without_a_rectangle_or_time_axis(viewport):
    view, raw = viewport
    before = raw.copy()
    assert view._view_selection is None and view._view_time_window is None
    view._image_item.setLevels((-9000, 9000))
    zoom(view)
    wait_levels(view, compute_display_levels(raw[10:40, 10:30], p_low=2, p_high=98))
    assert view._view_selection is None and view._view_time_window is None
    assert view._matrix is raw
    np.testing.assert_array_equal(raw, before)


def test_plain_pan_updates_to_the_new_visible_samples(viewport):
    view, raw = viewport
    zoom(view)
    wait_levels(view, compute_display_levels(raw[10:40, 10:30], p_low=2, p_high=98))
    zoom(view, left=20, right=40, top=60, bottom=90)
    wait_levels(view, compute_display_levels(raw[60:90, 20:40], p_low=2, p_high=98))


def test_enabling_follow_in_an_already_zoomed_view_applies_immediately(viewport):
    view, raw = viewport
    view._toggle_selection_levels(False)
    zoom(view)
    global_levels = compute_display_levels(raw, p_low=2, p_high=98)
    QTest.qWait(80)
    np.testing.assert_allclose(view._image_item.levels, global_levels)
    view._toggle_selection_levels(True)
    wait_levels(view, compute_display_levels(raw[10:40, 10:30], p_low=2, p_high=98))
    view._toggle_selection_levels(False)
    wait_levels(view, global_levels)


def test_fixed_shared_levels_stay_locked_until_cleared(viewport):
    view, raw = viewport
    view.set_levels_override((-5, 5))
    zoom(view)
    QTest.qWait(80)
    np.testing.assert_array_equal(view._image_item.levels, [-5, 5])
    view.set_levels_override(None)
    wait_levels(view, compute_display_levels(raw[10:40, 10:30], p_low=2, p_high=98))


def test_envelope_gain_and_contrast_use_current_display_pixels(viewport):
    view, raw = viewport
    before = raw.copy()
    view.set_display_mode("envelope")
    view.set_gain("tvg", db=12)
    view.set_contrast(1.5)
    zoom(view)
    image = view._image_item.image
    _, maximum = compute_display_levels(image[10:40, 10:30], p_low=2, p_high=98)
    wait_levels(view, (0, maximum / 1.5))
    np.testing.assert_array_equal(raw, before)


def test_full_view_after_plain_zoom_restores_whole_matrix_levels(viewport):
    view, raw = viewport
    zoom(view)
    wait_levels(view, compute_display_levels(raw[10:40, 10:30], p_low=2, p_high=98))
    view.fit_to_data()
    wait_levels(view, compute_display_levels(raw, p_low=2, p_high=98))


def test_pending_plain_zoom_then_clear_has_no_stale_render(viewport):
    view, _ = viewport
    zoom(view)
    view.clear()
    QTest.qWait(80)
    assert view._matrix is None and view._image_item.image is None
    assert not view._selection_timer.isActive()


def test_restored_plain_viewport_also_refreshes_levels(viewport):
    view, raw = viewport
    state = view.export_display_state()
    state.update(local_view=False, range=[[10, 30], [10, 40]])
    view.restore_display_state(state, viewport=True)
    wait_levels(view, compute_display_levels(raw[10:40, 10:30], p_low=2, p_high=98))

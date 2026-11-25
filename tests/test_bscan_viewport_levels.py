"""Viewport color levels must work without first arming rectangle selection.

Prepared for GitHub CI; user policy prohibits running these tests locally.
"""
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtTest import QTest

from core.gui_rendering import make_preview_bundle
from ui import desktop_backend_facade as facade
from ui.desktop_backend_facade import compute_display_levels
from ui.widgets.bscan_view import BScanView
from ui.widgets import bscan_view_data


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


def test_full_resolution_bundle_reuses_exact_levels_only_for_matching_pixels(qapp, monkeypatch):
    raw = np.random.default_rng(96).normal(size=(100, 60)).astype(np.float32)
    raw[5, 7] = np.nan
    matrix, _, _ = facade.read_full_resolution_bscan(raw.shape, lambda **kw: (
        raw, np.arange(100), np.arange(60)))
    bundle = facade.build_preview_bundle("L1", matrix, p_low=5, p_high=93)
    expected = compute_display_levels(raw, p_low=5, p_high=93)
    assert bundle.level_percentiles == (5, 93)
    np.testing.assert_array_equal((bundle.vmin, bundle.vmax), expected)
    calls = []

    def calculate(source, **kwargs):
        calls.append(source.shape)
        return compute_display_levels(source, **kwargs)

    monkeypatch.setattr(bscan_view_data, "compute_display_levels", calculate)
    view = BScanView(default_levels=(5, 93))
    try:
        view.resize(850, 540)
        view.show()
        qapp.processEvents()
        calls.clear()
        view.set_bundle(bundle)
        assert not calls
        np.testing.assert_allclose(view._image_item.levels, expected)
        zoom(view)
        view._refresh_selection_levels()
        assert calls[-1] == (30, 20)
        local = compute_display_levels(raw[10:40, 10:30], p_low=5, p_high=93)
        np.testing.assert_allclose(view._image_item.levels, local)
        view.set_levels_follow_view(False)
        calls.clear()
        view.set_display_levels(1, 99)
        assert calls == [raw.shape]
        np.testing.assert_allclose(view._image_item.levels, compute_display_levels(raw, p_low=1, p_high=99))
        view.set_display_levels(5, 93)
        calls.clear()
        view.set_gain("tvg", db=12)
        assert calls and calls[-1] == raw.shape
        gained = compute_display_levels(view._image_item.image, p_low=5, p_high=93)
        np.testing.assert_allclose(view._image_item.levels, gained)
        view.clear()
        view.set_gain("off")
        calls.clear()
        view.set_matrix(matrix, -999, 999)
        assert calls == [raw.shape]
        np.testing.assert_allclose(view._image_item.levels, expected)
    finally:
        view.close()
        view.deleteLater()


@pytest.mark.parametrize('foreign_buffer', [False, True])
def test_mutable_parent_can_change_levels_after_bundle_creation(foreign_buffer):
    raw = np.arange(20, dtype=np.float32).reshape(5, 4)
    if foreign_buffer:
        storage = bytearray(raw.tobytes())
        raw = np.frombuffer(storage, dtype=np.float32).reshape(5, 4)
        read_only_view = np.frombuffer(memoryview(storage).toreadonly(), dtype=np.float32).reshape(5, 4)
    else:
        read_only_view = raw.view()
        read_only_view.setflags(write=False)
    bundle = make_preview_bundle(read_only_view)
    assert bundle.level_percentiles is None
    raw.fill(100)
    assert compute_display_levels(bundle.matrix, p_low=2, p_high=98) != (bundle.vmin, bundle.vmax)


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

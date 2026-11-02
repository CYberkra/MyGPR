"""Multi-view contracts for live, interruptible B-scan fullscreen transitions."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import QCoreApplication, QEvent

from ui.widgets.bscan_workspace import BScanWorkspace


def bundle(value):
    return SimpleNamespace(
        matrix=np.full((12, 16), value, dtype=np.float32),
        vmin=0.0, vmax=float(value), title=str(value),
        x_label='道数', y_label='采样点',
    )


@pytest.fixture
def workspace(qapp, monkeypatch):
    monkeypatch.setattr('ui.widgets.bscan_fullscreen.animations_enabled', lambda: False)
    widget = BScanWorkspace()
    widget.resize(1000, 560)
    widget.set_sources([
        {'key': 'A', 'title': 'A', 'bundle': bundle(1), 'project_id': 'P', 'line_id': 'L'},
        {'key': 'B', 'title': 'B', 'bundle': bundle(2), 'project_id': 'P', 'line_id': 'L'},
    ])
    widget.open_source('A')
    widget.open_source('B')
    widget.split_source('B', 'right')
    widget.show()
    qapp.processEvents()
    yield widget
    for view in widget.views():
        view.settle_fullscreen()
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_fullscreen_preserves_ranges_splitter_and_other_view(workspace):
    first, second = workspace.views()
    first._plot.vb.setRange(xRange=(2, 9), yRange=(3, 10), padding=0)
    second._plot.vb.setRange(xRange=(1, 7), yRange=(2, 8), padding=0)
    first_range = first._plot.vb.viewRange()
    second_range = second._plot.vb.viewRange()
    sizes = workspace._root.sizes()
    first.enter_fullscreen()
    assert first.is_fullscreen()
    assert first._plot.vb.viewRange() == first_range
    assert second._plot.vb.viewRange() == second_range
    assert workspace._root.sizes() == sizes
    first.exit_fullscreen()
    assert not first.is_fullscreen()
    assert first._plot.vb.viewRange() == first_range
    assert second._plot.vb.viewRange() == second_range
    assert workspace._root.sizes() == sizes


def test_views_have_independent_hosts_and_can_settle_separately(workspace):
    first, second = workspace.views()
    first.enter_fullscreen()
    second.enter_fullscreen()
    assert first._fullscreen is not second._fullscreen
    assert first.is_fullscreen() and second.is_fullscreen()
    first.exit_fullscreen()
    assert not first.is_fullscreen()
    assert second.is_fullscreen()
    second.exit_fullscreen()
    assert not second.is_fullscreen()


def test_closing_fullscreen_source_settles_before_workspace_mutation(workspace):
    first, _second = workspace.views()
    first.enter_fullscreen()
    workspace.close_source('A')
    assert not first.is_fullscreen()
    assert workspace.active_key() == 'B'
    assert workspace.tab_keys() == ['B']


def test_f11_shortcut_is_scoped_to_its_view(workspace):
    first, second = workspace.views()
    first._fullscreen_shortcut.activated.emit()
    assert first.is_fullscreen()
    assert not second.is_fullscreen()
    first._fullscreen_shortcut.activated.emit()
    assert not first.is_fullscreen()


def test_transition_does_not_reload_image(workspace, monkeypatch):
    first, _second = workspace.views()
    calls = []
    original = first._image_item.setImage
    monkeypatch.setattr(first._image_item, 'setImage',
                        lambda *args, **kwargs: (calls.append(args), original(*args, **kwargs)))
    first.enter_fullscreen()
    first.exit_fullscreen()
    assert calls == []


def test_fast_retarget_uses_current_geometry_and_finishes_in_origin(workspace, monkeypatch):
    monkeypatch.setattr('ui.widgets.bscan_fullscreen.animations_enabled', lambda: True)
    view = workspace.views()[0]
    origin = view.parentWidget()
    view.enter_fullscreen()
    host = view._fullscreen
    host._animation.setCurrentTime(70)
    middle = host.geometry()
    view.exit_fullscreen()
    assert host.state == 'collapsing'
    assert host._animation.startValue() == middle
    view.enter_fullscreen()
    assert host.state == 'expanding'
    host._animation.setCurrentTime(220)
    assert host.state == 'fullscreen'
    view.exit_fullscreen()
    host._animation.setCurrentTime(220)
    assert view.parentWidget() is origin
    assert not view.is_fullscreen()


def test_link_partner_range_is_not_changed_by_fullscreen(workspace):
    first, second = workspace.views()
    workspace._set_link_enabled(True)
    second._plot.vb.setRange(xRange=(1, 6), yRange=(2, 9), padding=0)
    expected = second._plot.vb.viewRange()
    first.enter_fullscreen()
    assert second._plot.vb.viewRange() == expected
    first.exit_fullscreen()
    assert second._plot.vb.viewRange() == expected


def test_async_refill_updates_fullscreen_view_without_changing_active_source(workspace):
    first, _second = workspace.views()
    activated = []
    workspace.active_changed.connect(activated.append)
    first.enter_fullscreen()
    replacement = bundle(7)
    workspace.set_sources([
        {'key': 'A', 'title': 'A', 'bundle': replacement, 'project_id': 'P', 'line_id': 'L'},
        {'key': 'B', 'title': 'B', 'bundle': bundle(2), 'project_id': 'P', 'line_id': 'L'},
    ])
    assert first._matrix is replacement.matrix
    assert first.is_fullscreen()
    assert workspace.active_key() == 'B'
    assert activated == []


def test_ascan_popup_follows_view_to_fullscreen_and_back(workspace):
    first, _second = workspace.views()
    first.set_ascan_follow(True)
    popup = first._ascan_popup
    original_parent = popup.parentWidget()
    first.enter_fullscreen()
    assert popup.parentWidget() is first._fullscreen
    assert popup.isVisible()
    first.exit_fullscreen()
    assert first._ascan_popup is popup
    assert popup.parentWidget() is original_parent
    popup.hide()


def test_ascan_first_opened_in_fullscreen_survives_host_deletion(workspace, qapp):
    view = workspace.views()[0]
    view.enter_fullscreen()
    view.set_ascan_follow(True)
    popup = view._ascan_popup
    view.exit_fullscreen()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert view._ascan_popup is popup
    assert popup.parentWidget() is view
    assert view._ascan_follow
    popup.show_trace(np.arange(12), trace_index=0)
    assert popup.isVisible()
    view.set_ascan_follow(False)
    view.set_ascan_follow(True)
    assert popup.isVisible()
    popup.hide()

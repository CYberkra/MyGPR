"""Global viewing preferences stay global across tabs, new views and restarts."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QCoreApplication, QEvent, QPointF, QRectF
from PyQt6.QtWidgets import QWidget

from ui.desktop_backend_facade import compute_display_levels
from ui.main_window_bscan_prefs import BScanPreferenceMixin
from ui.pages.settings_page import SettingsPage
from ui.settings_manager import SettingsManager
from ui.widgets.bscan_view import BScanView
from ui.widgets.bscan_workspace import BScanWorkspace


class Preferences(BScanPreferenceMixin):
    def __init__(self, settings, page, settings_page):
        self.settings = settings
        self.pages = {'processingInterface': page, 'settingsInterface': settings_page}
        self.opened = []

    def _page(self, name):
        return self.pages.get(name)

    def _goto_page(self, name):
        self.opened.append(name)


@pytest.fixture
def preferences(qapp, tmp_path):
    host, page = QWidget(), SettingsPage()
    settings = SettingsManager(str(tmp_path/'settings.json'))
    page.load_settings(settings.get_all())
    prefs = Preferences(settings, host, page)
    page.bscan_view_changed.connect(prefs._on_settings_bscan_changed)
    yield prefs, host, page
    host.close()
    page.close()
    host.deleteLater()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def add_view(preferences):
    prefs, host, _ = preferences
    view = BScanView(host)
    raw = np.random.default_rng(52).normal(size=(100, 60)).astype(np.float32)
    raw[:10] *= 1000
    raw[50:] *= 500
    view.set_matrix(raw, -1000, 1000)
    prefs._wire_bscan_preferences(SimpleNamespace(findChildren=lambda _kind: [view]))
    return view, raw


def test_follow_switch_is_global_persistent_and_preserves_current_viewport(preferences):
    prefs, _, page = preferences
    first, raw = add_view(preferences)
    second, _ = add_view(preferences)
    before = raw.copy()
    first._plot.vb.setRange(xRange=(10, 30), yRange=(10, 40), padding=0)
    ranges = first._plot.vb.viewRange()
    page._bscan_levels_follow_check.setChecked(False)
    for view in (first, second):
        assert not view.levels_follow_view()
    assert first._plot.vb.viewRange() == ranges
    np.testing.assert_allclose(first._image_item.levels, compute_display_levels(raw, p_low=2, p_high=98))
    restored = SettingsManager(prefs.settings.settings_file)
    assert restored.get('bscan_levels_follow_view') is False
    fresh_host = QWidget(preferences[1])
    fresh_page = SettingsPage(fresh_host)
    fresh_page.load_settings(restored.get_all())
    fresh_prefs = Preferences(restored, fresh_host, fresh_page)
    reopened, _ = add_view((fresh_prefs, fresh_host, fresh_page))
    assert not reopened.levels_follow_view()
    third, _ = add_view(preferences)
    assert not third.levels_follow_view()
    page._bscan_levels_follow_check.setChecked(True)
    assert all(view.levels_follow_view() for view in (first, second, third))
    np.testing.assert_allclose(first._image_item.levels,
                               compute_display_levels(raw[10:40, 10:30], p_low=2, p_high=98))
    np.testing.assert_array_equal(raw, before)


def test_old_tab_state_cannot_override_global_follow_or_moved_preferences(preferences):
    prefs, _, page = preferences
    view, _ = add_view(preferences)
    page._bscan_levels_follow_check.setChecked(False)
    page._bscan_colorbar_check.setChecked(False)
    page._bscan_cmap_combo.setCurrentText('gray')
    view.restore_display_state({'levels_follow_view': True, 'colorbar': True,
                                'colormap': 'seismic', 'levels': [7, 91]})
    assert not view.levels_follow_view()
    assert not view.colorbar_visible()
    assert view._cmap_name == 'gray'
    assert view.display_levels() == (2., 98.)
    assert 'levels_follow_view' not in view.export_display_state()
    page._bscan_levels_follow_check.setChecked(True)
    view.restore_display_state({'levels_follow_view': False})
    assert view.levels_follow_view()


def test_global_follow_switch_does_not_reset_a_sources_display_gain(preferences):
    _, _, page = preferences
    view, _ = add_view(preferences)
    view.set_gain('tvg', db=18, power=1.6)
    page._bscan_levels_follow_check.setChecked(False)
    assert view.gain_mode() == 'tvg' and view._gain_db == 18 and view._gain_power == 1.6


def test_follow_change_retains_zoom_on_another_views_temporary_aspect(preferences):
    _, _, page = preferences
    first, _ = add_view(preferences)
    second, _ = add_view(preferences)
    second.set_aspect_mode('square')
    first.fit_to_data()
    second._plot.vb.setRange(xRange=(10, 30), yRange=(10, 40), padding=0)
    ranges = second._plot.vb.viewRange()
    page._bscan_levels_follow_check.setChecked(False)
    assert second._plot.vb.viewRange() == ranges and second.aspect_mode() == 'square'


def test_empty_tab_state_applies_global_preferences_instead_of_previous_source_style(preferences):
    _, _, page = preferences
    view, _ = add_view(preferences)
    page._bscan_cmap_combo.setCurrentText('gray')
    view.set_colormap('seismic')
    view.restore_display_state({})
    assert view._cmap_name == 'gray'


def test_global_follow_survives_real_source_switching_and_legacy_workspace_state(preferences):
    from ui.main_window import MyGPRMainWindow

    prefs, host, page = preferences
    workspace = BScanWorkspace(host)
    workspace.view_created.connect(lambda view: MyGPRMainWindow._wire_new_bscan_view(prefs, view))
    prefs._wire_bscan_preferences(host)
    raw = np.arange(24, dtype=np.float32).reshape(4, 6)
    data = SimpleNamespace(matrix=raw, vmin=0., vmax=24.)
    workspace.set_sources([{'key': key, 'title': key, 'bundle': data} for key in ('A', 'B')])
    workspace.open_source('A')
    workspace.open_source('B')
    page._bscan_levels_follow_check.setChecked(False)
    workspace.open_source('A')
    assert not workspace.primary_view().levels_follow_view()
    workspace.open_source('B')
    assert not workspace.primary_view().levels_follow_view()
    assert prefs.settings.get('bscan_levels_follow_view') is False
    workspace.split_source('B', 'right', copy=True)
    assert len(workspace.views()) == 2
    assert all(not view.levels_follow_view() for view in workspace.views())
    state = workspace.save_layout()
    for group in state['root']['children']:
        for tab in group['tabs']:
            tab['state']['levels_follow_view'] = True
            tab['state']['colormap'] = 'gray'
    assert workspace.restore_layout(state)
    assert all(not view.levels_follow_view() for view in workspace.views())
    assert all(view._cmap_name == 'seismic' for view in workspace.views())


def test_global_update_reaches_fullscreen_and_keeps_shared_levels_locked(preferences, qapp):
    prefs, _, page = preferences
    view, _ = add_view(preferences)
    view.set_levels_override((-5, 5))
    view.enter_fullscreen()
    qapp.processEvents()
    assert view in list(prefs._iter_bscan_views())
    page._bscan_levels_follow_check.setChecked(False)
    assert not view.levels_follow_view()
    np.testing.assert_array_equal(view._image_item.levels, [-5, 5])
    view.exit_fullscreen()


def test_settings_roundtrip_and_external_sync_do_not_emit_or_revert_new_preferences(qapp):
    page = SettingsPage()
    calls = []
    page.bscan_view_changed.connect(lambda: calls.append(1))
    try:
        page.load_settings({'bscan_levels_follow_view': False, 'bscan_qc_marks_visible': False})
        assert calls == []
        assert not page._bscan_levels_follow_check.isChecked()
        assert not page._bscan_qc_marks_check.isChecked()
        page.sync_bscan_view_settings({'bscan_levels_follow_view': True, 'bscan_qc_marks_visible': True})
        assert calls == []
        assert page.settings()['bscan_levels_follow_view'] is True
        assert page.settings()['bscan_qc_marks_visible'] is True
    finally:
        page.deleteLater()


def test_display_settings_shortcut_restores_fullscreen_before_navigation(preferences, qapp):
    prefs, host, _ = preferences
    view, _ = add_view(preferences)
    view.enter_fullscreen()
    qapp.processEvents()
    view.sig_display_settings_requested.emit()
    qapp.processEvents()
    assert not view.is_fullscreen() and view._fullscreen is None
    assert view.parentWidget() is host
    assert prefs.opened == ['settingsInterface']


def test_canvas_menu_keeps_viewing_actions_and_opens_the_settings_section(preferences, monkeypatch, qapp):
    from ui.widgets import bscan_view_interaction
    from ui.widgets.context_menus import make_menu

    prefs, _, page = preferences
    view, _ = add_view(preferences)
    menus, focused = [], []

    def capture(parent):
        menu = make_menu(parent)
        monkeypatch.setattr(menu, 'exec', lambda *_args: None)
        menus.append(menu)
        return menu

    monkeypatch.setattr(bscan_view_interaction, 'make_menu', capture)
    monkeypatch.setattr(page, 'focus_bscan_settings', lambda: focused.append(True))
    view._show_context_menu(SimpleNamespace(screenPos=lambda: QPointF(20, 20)))
    actions = {action.text(): action for action in menus[0].actions()}
    assert {'框选放大', '适应全图', '复制图像', '导出图件…', '当前图显示…'} <= actions.keys()
    assert not {'色阶跟随查看范围', '色阶设置…', '显示色标', '显示异常道标记'} & actions.keys()
    actions['当前图显示…'].trigger()
    settings = next(action for action in menus[1].actions() if action.text() == '全局 B-scan 设置…')
    settings.trigger()
    qapp.processEvents()
    assert prefs.opened == ['settingsInterface'] and focused == [True]


def test_fit_all_preserves_aspect_and_does_not_write_preferences(preferences):
    prefs, _, _ = preferences
    view, _ = add_view(preferences)
    view.set_aspect_mode('square', notify=True)
    view.set_view_selection(QRectF(5, 5, 20, 30))
    signals = []
    view.sig_aspect_changed.connect(signals.append)
    view.fit_all()
    assert view.aspect_mode() == prefs.settings.get('bscan_aspect_mode') == 'square'
    assert view._view_selection is None and signals == []


def test_global_tvg_parameters_reach_views_and_win_over_saved_tabs(preferences):
    prefs, _, page = preferences
    first, _ = add_view(preferences)
    second, _ = add_view(preferences)
    old = first.export_display_state()
    page._bscan_gain_combo.setCurrentIndex(page._bscan_gain_combo.findData('tvg'))
    page._bscan_tvg_db_spin.setValue(36)
    page._bscan_tvg_power_spin.setValue(1.8)
    first.restore_display_state(old)
    for view in (first, second):
        assert view.gain_mode() == 'tvg' and view._gain_db == 36 and view._gain_power == 1.8
    assert prefs.settings.get('bscan_tvg_db') == 36

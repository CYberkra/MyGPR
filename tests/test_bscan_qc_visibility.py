"""QC marks are a remembered display preference, never a processing action."""
from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip('PyQt6')
from PyQt6.QtWidgets import QWidget
from ui.widgets.bscan_view import BScanView
from ui.widgets.bscan_workspace import BScanWorkspace


def bundle(mask):
    return SimpleNamespace(matrix=np.arange(24,dtype=np.float32).reshape(4,6),vmin=0.,vmax=24.,
                           trace_qc_bad_mask=np.asarray(mask,dtype=bool))


def test_checkable_display_menu_hides_only_marks_without_a_permanent_canvas_control(qapp):
    from ui.widgets.context_menus import make_menu
    view = BScanView()
    view.show()
    original = bundle([0,1,0,1,0,0])
    view.set_bundle(original)
    qapp.processEvents()
    assert view.findChild(QWidget,'showAnomalousTraceMarks') is None
    menu = make_menu(view)
    menu.addAction(view.qc_marks_menu_action(menu))
    action = next(a for a in menu.actions() if a.text()=='显示异常道标记')
    assert action is not view._qc_marks_action and action.isCheckable() and action.isChecked()
    assert not action.icon().isNull()
    matrix = original.matrix.copy()
    mask = original.trace_qc_bad_mask.copy()
    image = view._image_item.image.copy()
    marks = list(view._qc_marks)
    action.trigger()
    assert not view.qc_marks_visible()
    assert action.icon().isNull()
    assert not any(mark.isVisible() for mark in marks)
    assert view._qc_marks == marks
    np.testing.assert_array_equal(original.matrix,matrix)
    np.testing.assert_array_equal(original.trace_qc_bad_mask,mask)
    np.testing.assert_array_equal(view._image_item.image,image)
    action.trigger()
    assert all(mark.isVisible() for mark in marks)
    menu.close()
    view.close()


def test_hidden_preference_survives_new_data_and_switching_sources(qapp):
    workspace = BScanWorkspace()
    workspace.set_sources([{'key':'A','title':'A','bundle':bundle([1,0,0,0,0,0])},
                           {'key':'B','title':'B','bundle':bundle([0,0,1,1,0,0])}])
    workspace.open_source('A')
    view = workspace.primary_view()
    view.set_qc_marks_visible(False)
    workspace.open_source('B')
    assert len(view._qc_marks) == 2 and not any(m.isVisible() for m in view._qc_marks)
    workspace.open_source('A')
    assert len(view._qc_marks) == 1 and not view._qc_marks[0].isVisible()
    view.set_qc_marks_visible(True)
    assert view._qc_marks[0].isVisible()
    workspace.close()


def test_previously_visited_source_cannot_restore_old_global_visibility(qapp):
    workspace = BScanWorkspace()
    workspace.set_sources([
        {'key': 'A', 'title': 'A', 'bundle': bundle([1, 0, 0, 0, 0, 0])},
        {'key': 'B', 'title': 'B', 'bundle': bundle([0, 0, 1, 1, 0, 0])},
    ])
    workspace.open_source('A')
    workspace.open_source('B')
    workspace.open_source('A')
    view = workspace.primary_view()
    view.set_qc_marks_visible(False)
    workspace.open_source('B')
    assert not view.qc_marks_visible()
    assert not any(mark.isVisible() for mark in view._qc_marks)
    # Old persisted per-tab state must also yield to the global preference.
    view.restore_display_state({'qc_marks': True})
    assert not view.qc_marks_visible()
    assert 'qc_marks' not in view.export_display_state()
    workspace.close()


def test_visibility_persists_across_settings_reload_and_synchronizes_new_views(qapp,tmp_path):
    from ui.main_window_bscan_prefs import BScanPreferenceMixin
    from ui.settings_manager import SettingsManager
    class Preferences(BScanPreferenceMixin):
        def __init__(self,settings,page):
            self.settings,self.pages = settings,{'processingInterface':page}
        def _page(self,_):
            return None
    page = QWidget()
    first,second = BScanView(page),BScanView(page)
    settings = SettingsManager(str(tmp_path/'settings.json'))
    prefs = Preferences(settings,page)
    prefs._wire_bscan_preferences(page)
    first.set_qc_marks_visible(False,notify=True)
    assert not second.qc_marks_visible()
    restored = SettingsManager(str(tmp_path/'settings.json'))
    assert restored.get('bscan_qc_marks_visible') is False
    other_page = QWidget()
    new_view = BScanView(other_page)
    Preferences(restored,other_page)._wire_bscan_preferences(other_page)
    assert not new_view.qc_marks_visible()
    assert not new_view._qc_marks_action.isChecked()
    page.close()
    other_page.close()


def test_global_switch_reaches_a_view_temporarily_reparented_to_fullscreen(qapp,tmp_path):
    from PyQt6.QtWidgets import QVBoxLayout
    from ui.main_window_bscan_prefs import BScanPreferenceMixin
    from ui.settings_manager import SettingsManager
    class Preferences(BScanPreferenceMixin):
        def __init__(self,settings,page):
            self.settings,self.pages = settings,{'processingInterface':page}
        def _page(self,_):
            return None
    host = QWidget()
    page = QWidget(host)
    QVBoxLayout(host).addWidget(page)
    layout = QVBoxLayout(page)
    first,second = BScanView(page),BScanView(page)
    layout.addWidget(first)
    layout.addWidget(second)
    prefs = Preferences(SettingsManager(str(tmp_path/'settings.json')),page)
    prefs._wire_bscan_preferences(page)
    second.set_bundle(bundle([1,0,0,0,0,0]))
    second.enter_fullscreen()
    qapp.processEvents()
    assert second in list(prefs._iter_bscan_views())
    first.set_qc_marks_visible(False,notify=True)
    assert not second.qc_marks_visible()
    assert not second._qc_marks[0].isVisible()
    second._qc_marks_action.trigger()
    assert first.qc_marks_visible() and second.qc_marks_visible()
    assert prefs.settings.get('bscan_qc_marks_visible') is True
    second.exit_fullscreen()
    assert second.qc_marks_visible()
    host.close()


def test_menu_reopen_and_programmatic_preference_updates_keep_check_state(qapp):
    from ui.widgets.context_menus import make_menu
    view = BScanView()
    first = make_menu(view)
    first.addAction(view.qc_marks_menu_action(first))
    action = view._qc_marks_action
    assert action.parent() is view
    changed = []
    view.sig_qc_marks_visible_changed.connect(changed.append)
    view.set_qc_marks_visible(False)
    assert not action.isChecked() and not changed
    first.close()
    first.deleteLater()
    qapp.processEvents()
    reopened = make_menu(view)
    reopened.addAction(view.qc_marks_menu_action(reopened))
    proxy = next(a for a in reopened.actions() if a.text()=='显示异常道标记')
    assert proxy is not action and not proxy.isChecked()
    proxy.trigger()
    assert proxy.isChecked() and action.isChecked() and changed == [True]
    reopened.close()
    view.close()


def test_existing_workspace_more_menu_exposes_the_same_display_only_action(qapp):
    workspace = BScanWorkspace()
    original = bundle([0,1,0,1,0,0])
    workspace.set_sources([{'key':'A','title':'A','bundle':original}])
    workspace.open_source('A')
    view = workspace.primary_view()
    settings = []
    view.sig_display_settings_requested.connect(lambda: settings.append(True))
    menu = workspace.create_context_menu(key='A')
    action = next(a for a in menu.actions() if a.text()=='显示设置…')
    assert not any(a.text() == '显示异常道标记' for a in menu.actions())
    inputs = []
    workspace.source_input_requested.connect(inputs.append)
    action.trigger()
    assert settings == [True] and view.qc_marks_visible() and not inputs
    assert workspace.primary_view() is view
    assert all(m.isVisible() for m in view._qc_marks)
    menu.close()
    workspace.close()


def test_two_live_menus_have_independent_proxies_and_survive_reverse_destruction(qapp):
    from PyQt6.QtCore import QCoreApplication, QEvent
    from ui.widgets.context_menus import make_menu
    view = BScanView()
    menus=[make_menu(view),make_menu(view)]
    for menu in menus:
        menu.addAction(view.qc_marks_menu_action(menu))
    actions=[next(a for a in menu.actions() if a.text()=='显示异常道标记') for menu in menus]
    assert actions[0] is not actions[1]
    menus[1].deleteLater()
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    view.set_qc_marks_visible(False)
    assert not actions[0].isChecked()
    actions[0].trigger()
    assert view.qc_marks_visible() and actions[0].isChecked()
    menus[0].deleteLater()
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    view.set_qc_marks_visible(False)
    assert not view._qc_marks_action.isChecked()
    view.close()

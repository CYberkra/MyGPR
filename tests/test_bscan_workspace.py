"""Qt behavior contracts for the explicitly arranged B-scan workspace."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QSplitter, QTabBar, QVBoxLayout, QWidget

from ui.widgets.bscan_workspace import BScanWorkspace


def bundle(value=1):
    return SimpleNamespace(matrix=np.full((8, 6), value, np.float32),
                           vmin=0.0, vmax=float(value), title=str(value),
                           x_label='道数', y_label='采样点')


def sources(*keys):
    return [dict(key=key, title=key, bundle=bundle(index + 1),
                 artifact_id='' if key == 'raw' else key)
            for index, key in enumerate(keys)]


@pytest.fixture
def workspace(qapp):
    widget = BScanWorkspace()
    widget.resize(900, 520)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def action(menu, command):
    return next(a for a in menu.actions() if a.data() == command)


def test_close_button_preserves_native_close_signal_and_source_after_reorder(workspace):
    workspace.set_sources(sources('raw', 'A', 'B'))
    for key in ('raw', 'A', 'B'):
        workspace.open_source(key)
    tabs = workspace.tab_bar()
    button = tabs.tabButton(0, QTabBar.ButtonPosition.RightSide)
    tabs.moveTab(0, 2)
    requested = []
    tabs.tabCloseRequested.connect(requested.append)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert requested == [2]
    assert workspace.tab_keys() == ['A', 'B']


def test_catalog_does_not_open_tabs_and_many_tabs_stay_in_one_group(workspace):
    workspace.set_sources(sources('raw', 'A', 'B', 'C', 'D', 'E'))
    assert workspace.tab_keys() == []
    assert workspace.active_key() == ''
    assert len(workspace.views()) == 1
    for key in ('raw', 'A', 'B', 'C', 'D', 'E'):
        workspace.open_source(key)
    assert workspace.tab_keys() == ['raw', 'A', 'B', 'C', 'D', 'E']
    assert len(workspace.views()) == 1
    assert len(workspace.all_views()) == 1
    assert not workspace.is_pinned('raw')


def test_moving_into_another_group_preserves_its_current_display_state(workspace):
    workspace.set_sources(sources('A', 'B'))
    workspace.open_source('A')
    origin = workspace.group_ids()[0]
    workspace.open_source('B')
    target = workspace.split_source('B', 'right')
    workspace.group_widget(target).view.set_colormap('gray')
    workspace.move_source('A', target, origin)
    workspace.open_source('B', target)
    assert workspace.primary_view()._cmap_name == 'gray'


def test_raw_and_all_other_tabs_close_without_reopening_on_catalog_refill(workspace):
    catalog = sources('raw', 'A')
    workspace.set_sources(catalog)
    closed = []
    workspace.source_closed.connect(closed.append)
    workspace.open_source('raw')
    workspace.open_source('A')
    workspace.close_source('raw')
    workspace.close_source('A')
    workspace.set_sources(catalog)
    assert closed == ['raw', 'A']
    assert workspace.active_key() == ''
    assert workspace.tab_keys() == []
    assert len(workspace.views()) == 1
    assert workspace.primary_view()._matrix is None


def test_async_refill_renders_only_visible_groups_without_stealing_focus(workspace):
    catalog = sources('raw', 'A', 'B')
    catalog[1]['bundle'] = None
    catalog[2]['bundle'] = None
    workspace.set_sources(catalog)
    requested, activated = [], []
    workspace.preview_requested.connect(requested.append)
    workspace.active_changed.connect(activated.append)
    workspace.open_source('A')
    workspace.open_source('raw')
    assert requested == ['A']
    replacement = bundle(7)
    catalog[1]['bundle'] = replacement
    workspace.set_sources(catalog)
    assert workspace.active_key() == 'raw'
    assert activated == ['A', 'raw']
    assert requested == ['A']
    workspace.open_source('A')
    assert workspace.primary_view()._matrix is replacement.matrix
    assert 'B' not in workspace.tab_keys()


def test_split_move_reorder_and_empty_group_collapse_share_matrix(workspace):
    catalog = sources('raw', 'A', 'B')
    workspace.set_sources(catalog)
    for key in ('raw', 'A', 'B'):
        workspace.open_source(key)
    first = workspace.group_ids()[0]
    second = workspace.split_source('B', 'right')
    assert workspace.group_ids() == [first, second]
    assert len(workspace.views()) == 2
    assert workspace.findChild(QSplitter) is not None
    assert workspace.tab_keys(first) == ['raw', 'A']
    workspace.move_source('raw', second, first, index=0)
    assert workspace.tab_keys(second) == ['raw', 'B']
    workspace.move_source('B', second, second, index=0)
    assert workspace.tab_keys(second) == ['B', 'raw']
    workspace.close_source('A')
    assert workspace.group_ids() == [second]
    assert len(workspace.views()) == 1
    workspace.open_source('B')
    assert workspace.primary_view()._matrix is catalog[2]['bundle'].matrix


def test_duplicate_views_have_independent_preferences_and_final_close_signal(workspace):
    workspace.set_sources(sources('A'))
    workspace.open_source('A')
    original = workspace.group_ids()[0]
    second = workspace.split_source('A', 'bottom', copy=True)
    first_view, second_view = workspace.views()
    first_view.set_display_levels(4, 92)
    second_view.set_display_levels(11, 81)
    assert first_view._matrix is second_view._matrix
    assert first_view.display_levels() != second_view.display_levels()
    closed = []
    workspace.source_closed.connect(closed.append)
    workspace.close_tab('A', second)
    assert closed == []
    assert workspace.group_ids() == [original]
    workspace.close_tab('A', original)
    assert closed == ['A']


def test_pin_close_ranges_and_reopen_use_stable_keys(workspace):
    workspace.set_sources(sources('raw', 'A', 'B', 'C'))
    for key in ('raw', 'A', 'B', 'C'):
        workspace.open_source(key)
    workspace.set_pinned('raw', True)
    workspace.set_pinned('B', True)
    assert not workspace.close_tab('raw')
    workspace.close_tabs('right', 'A')
    assert workspace.tab_keys() == ['raw', 'A', 'B']
    assert workspace.reopen_closed() == 'C'
    assert workspace.active_key() == 'C'
    workspace.close_tabs('others', 'A')
    assert workspace.tab_keys() == ['raw', 'A', 'B']
    workspace.close_tabs('unpinned')
    assert workspace.tab_keys() == ['raw', 'B']
    workspace.set_pinned('raw', False)
    assert workspace.close_tab('raw')
    workspace.close_source('B')  # Authoritative source removal also closes pinned views.
    assert workspace.tab_keys() == []


def test_layout_roundtrip_retains_nested_split_sizes_tabs_pins_and_view_state(workspace, qapp):
    catalog = sources('raw', 'A', 'B', 'C')
    workspace.set_sources(catalog)
    for key in ('raw', 'A', 'B', 'C'):
        workspace.open_source(key)
    first = workspace.group_ids()[0]
    second = workspace.split_source('C', 'right')
    third = workspace.split_source('B', 'bottom', group_id=first)
    workspace.set_pinned('raw', True, first)
    workspace.open_source('A', first)
    workspace.views()[0].set_display_levels(7, 93)
    for splitter in workspace.findChildren(QSplitter):
        splitter.setSizes([200, 450])
    qapp.processEvents()
    state = json.loads(json.dumps(workspace.save_layout()))
    created = []
    workspace.view_created.connect(created.append)
    assert workspace.restore_layout(state)
    assert len(created) == 3
    assert workspace.group_ids() == [first, third, second]
    assert workspace.tab_keys(first) == ['raw', 'A']
    assert workspace.is_pinned('raw', first)
    assert workspace.active_key() == 'A'
    assert workspace.views()[0].display_levels() == (7.0, 93.0)
    assert workspace.save_layout()['root'] == state['root']


def test_restore_skips_removed_sources_and_does_not_fill_empty_layout(workspace):
    workspace.set_sources(sources('raw', 'A'))
    workspace.open_source('A')
    state = workspace.save_layout()
    workspace.set_sources(sources('raw'))
    assert workspace.restore_layout(state)
    assert workspace.tab_keys() == []
    assert workspace.active_key() == ''
    assert len(workspace.views()) == 1
    assert not workspace.restore_layout({'version': 999})


def test_tab_context_menu_never_activates_target_and_delegates_business_actions(workspace, qapp):
    workspace.set_sources(sources('raw', 'A'))
    workspace.open_source('raw')
    workspace.open_source('A')
    requests = []
    workspace.source_context_requested.connect(lambda *args: requests.append(args))
    menu = workspace.create_context_menu(key='raw', global_pos=QPoint(81, 92))
    assert workspace.active_key() == 'A'
    action(menu, 'source-context').trigger()
    assert requests == [('raw', QPoint(81, 92))]
    assert workspace.active_key() == 'A'
    action(menu, 'pin').trigger()
    pinned_menu = workspace.create_context_menu(key='raw')
    assert not action(pinned_menu, 'close').isEnabled()
    menu.close()
    menu.deleteLater()
    pinned_menu.deleteLater()
    qapp.processEvents()


def test_shortcuts_are_scoped_to_workspace_and_reopen_closed(workspace, qapp):
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(workspace)
    editor = QLineEdit()
    layout.addWidget(editor)
    host.show()
    workspace.set_sources(sources('raw', 'A'))
    workspace.open_source('raw')
    workspace.open_source('A')
    editor.setFocus()
    qapp.processEvents()
    QTest.keyClick(editor, Qt.Key.Key_W, Qt.KeyboardModifier.ControlModifier)
    assert workspace.active_key() == 'A'
    bar = workspace.tab_bar()
    bar.setFocus()
    qapp.processEvents()
    QTest.keyClick(bar, Qt.Key.Key_W, Qt.KeyboardModifier.ControlModifier)
    assert workspace.tab_keys() == ['raw']
    QTest.keyClick(bar, Qt.Key.Key_T,
                   Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert workspace.active_key() == 'A'
    workspace.setParent(None)
    host.close()


def deliver_drag(workspace, widget, key, origin, point, *, drop=True):
    mime = workspace.drag_mime_data(key, origin)
    enter = QDragEnterEvent(point, Qt.DropAction.MoveAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, enter)
    move = QDragMoveEvent(point, Qt.DropAction.MoveAction, mime,
                         Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, move)
    assert enter.isAccepted() and move.isAccepted()
    if drop:
        event = QDropEvent(QPointF(point), Qt.DropAction.MoveAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(widget, event)
        assert event.isAccepted()


def test_qt_drop_reorders_tabs_splits_at_edge_and_escape_cancels_hint(workspace, qapp):
    workspace.set_sources(sources('raw', 'A', 'B'))
    for key in ('raw', 'A', 'B'):
        workspace.open_source(key)
    first = workspace.group_ids()[0]
    bar = workspace.tab_bar(first)
    deliver_drag(workspace, bar, 'B', first, QPoint(2, bar.height() // 2))
    assert workspace.tab_keys() == ['B', 'raw', 'A']
    group = workspace.group_widget(first)
    edge = QPoint(group.width() - 3, group.height() // 2)
    deliver_drag(workspace, group, 'A', first, edge, drop=False)
    assert workspace.drop_hint_visible()
    QApplication.sendEvent(bar, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                         Qt.KeyboardModifier.NoModifier))
    assert not workspace.drop_hint_visible()
    assert len(workspace.views()) == 1
    deliver_drag(workspace, group, 'A', first, edge)
    qapp.processEvents()
    assert len(workspace.views()) == 2
    assert workspace.tab_keys(workspace.group_ids()[1]) == ['A']
    assert not workspace.drop_hint_visible()


def test_reset_layout_preserves_open_resources_pins_and_closed_history(workspace):
    workspace.set_sources(sources('raw', 'A', 'B'))
    for key in ('raw', 'A', 'B'):
        workspace.open_source(key)
    workspace.set_pinned('raw', True)
    workspace.split_source('B', 'right')
    workspace.close_source('A')
    workspace.reset_layout()
    assert len(workspace.views()) == 1
    assert workspace.tab_keys() == ['raw', 'B']
    assert workspace.is_pinned('raw')
    assert workspace.active_key() == 'B'
    assert workspace.reopen_closed() == 'A'


def test_clicking_inactive_group_selects_only_clicked_tab_and_more_never_activates(workspace, qapp):
    workspace.set_sources(sources('raw', 'A', 'B'))
    for key in ('raw', 'A', 'B'):
        workspace.open_source(key)
    first = workspace.group_ids()[0]
    second = workspace.split_source('B', 'right')
    assert workspace.active_key() == 'B'
    activated = []
    workspace.active_changed.connect(activated.append)
    first_group = workspace.group_widget(first)
    QTest.mouseClick(first_group.more, Qt.MouseButton.LeftButton)
    assert workspace.active_key() == 'B'
    assert activated == []
    workspace._close_menu()
    qapp.processEvents()
    bar = workspace.tab_bar(first)
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton,
                     pos=bar.tabRect(workspace.tab_keys(first).index('raw')).center())
    assert workspace.active_key() == 'raw'
    assert activated == ['raw']
    assert workspace.tab_keys(second) == ['B']


def test_merge_duplicate_source_preserves_pin_from_either_view(workspace):
    workspace.set_sources(sources('A'))
    workspace.open_source('A')
    second = workspace.split_source('A', 'right', copy=True)
    workspace.set_pinned('A', True, second)
    workspace.reset_layout()
    assert workspace.is_pinned('A')
    assert workspace.tab_keys() == ['A']


def test_error_state_and_catalog_removal_clear_missing_views_without_reopen(workspace):
    workspace.set_sources(sources('raw', 'A'))
    workspace.open_source('raw')
    second = workspace.split_source('A', 'right')
    assert second == ''  # A has never been opened.
    workspace.open_source('A')
    workspace.split_source('A', 'right')
    workspace.set_sources([dict(key='A', title='A', bundle=None, error='文件不可读取')])
    assert workspace.tab_keys() == ['A']
    assert len(workspace.views()) == 1
    assert workspace.primary_view()._matrix is None
    assert workspace.primary_view()._empty_overlay._title_label.text() == '预览加载失败'
    workspace.close_source('A')
    workspace.set_sources(sources('raw', 'A'))
    assert workspace.active_key() == ''
    assert workspace.tab_keys() == []


def test_fluent_menu_keyboard_opens_submenu_and_executes_split(workspace, qapp):
    workspace.set_sources(sources('raw', 'A'))
    workspace.open_source('raw')
    workspace.open_source('A')
    workspace.show_context_menu(key='raw')
    menu = workspace._menu
    qapp.processEvents()
    QTest.keyClick(menu.view, Qt.Key.Key_Home)
    QTest.keyClick(menu.view, Qt.Key.Key_Down)  # pin
    QTest.keyClick(menu.view, Qt.Key.Key_Down)  # arrange submenu
    QTest.keyClick(menu.view, Qt.Key.Key_Right)
    submenu = next(m for m in menu._subMenus if m.title() == '排列与拆分')
    assert submenu.isVisible()
    QTest.keyClick(submenu.view, Qt.Key.Key_Down)  # split right
    QTest.keyClick(submenu.view, Qt.Key.Key_Return)
    assert len(workspace.views()) == 2
    assert workspace.active_key() == 'raw'


def test_real_splitter_handle_drag_resizes_groups_and_cross_group_drop_moves_tab(workspace, qapp):
    workspace.set_sources(sources('raw', 'A', 'B'))
    for key in ('raw', 'A', 'B'):
        workspace.open_source(key)
    first = workspace.group_ids()[0]
    second = workspace.split_source('B', 'right')
    qapp.processEvents()
    splitter = workspace.findChild(QSplitter)
    before = splitter.sizes()
    handle = splitter.handle(1)
    start = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(handle, start + QPoint(80, 0))
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=start + QPoint(80, 0))
    qapp.processEvents()
    assert splitter.sizes()[0] > before[0]
    destination = workspace.group_widget(second)
    deliver_drag(workspace, destination, 'A', first, destination.rect().center())
    assert workspace.tab_keys(first) == ['raw']
    assert workspace.tab_keys(second) == ['B', 'A']
    assert workspace.active_key() == 'A'


def test_showing_workspace_keeps_saved_active_group(qapp):
    widget = BScanWorkspace()
    widget.set_sources(sources('raw', 'A'))
    widget.open_source('raw')
    widget.open_source('A')
    widget.split_source('A', 'right')
    assert widget.active_key() == 'A'
    widget.show()
    qapp.processEvents()
    assert widget.active_key() == 'A'
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

"""Real Qt gestures and stale-action boundaries for the editable draft strip."""
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PyQt6 import sip
from PyQt6.QtGui import QContextMenuEvent, QDrag, QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QListWidget

from ui.widgets.chain_strip import ChainStrip


@pytest.fixture
def strip(qapp):
    widget = ChainStrip()
    widget.resize(800, 38)
    widget.set_steps([{'label': '去直流', 'param_summary': '窗口 11', 'enabled': True},
                      {'label': '增益', 'param_summary': '窗口 21', 'enabled': False},
                      {'label': '带通', 'enabled': True}])
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def chip(strip, index):
    return strip._list.itemWidget(strip._list.item(index))


def test_repeated_label_click_opens_current_step_without_programmatic_click(strip):
    clicked, selected = [], []
    strip.sig_step_clicked.connect(clicked.append)
    strip.sig_step_selected.connect(selected.append)
    strip.select_step(1)
    assert clicked == []
    for _ in range(2):
        QTest.mouseClick(chip(strip, 1).name, Qt.MouseButton.LeftButton)
    assert clicked == [1, 1]
    assert strip._list.currentRow() == 1
    assert selected == []  # already selected; actual clicks remain distinct


def test_summary_and_enabled_state_are_explained_and_same_render_keeps_widgets(strip):
    original = chip(strip, 0)
    assert '窗口 11' in original.toolTip()
    assert '窗口 11' in original.summary.text()
    assert '已禁用' in chip(strip, 1).toolTip()
    strip._list.setFocus()
    strip.select_step(1)
    strip.set_steps([dict(s) for s in strip._steps])
    assert chip(strip, 0) is original
    assert strip._list.currentRow() == 1
    assert strip._list.hasFocus()


def test_context_event_on_label_and_keyboard_edit_are_reachable(strip):
    got = []
    strip.sig_step_editor_requested.connect(got.append)
    label = chip(strip, 1).name
    QApplication.sendEvent(label, QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse, label.rect().center(),
        label.mapToGlobal(label.rect().center())))
    assert got == [1]
    strip._list.setFocus()
    QTest.keyClick(strip._list, Qt.Key.Key_F10, Qt.KeyboardModifier.ShiftModifier)
    QTest.keyClick(strip._list, Qt.Key.Key_Menu)
    assert got == [1, 1, 1]


@pytest.mark.parametrize('disabled', [False, True])
def test_old_overflow_actions_cannot_mutate_a_changed_or_locked_draft(strip, disabled):
    menu = strip._build_overflow_menu()
    selected, edited = [], []
    strip.sig_step_selected.connect(selected.append)
    strip.sig_step_editor_requested.connect(edited.append)
    if disabled:
        strip.setEnabled(False)
    else:
        strip.set_steps([{'label': '新草稿', 'enabled': True}])
    selected.clear()  # draft redraw may restore selection; stale actions may not
    menu._step_submenus[0].actions()[0].trigger()
    menu._step_submenus[0].actions()[1].trigger()
    assert selected == edited == []


class Drop:
    def __init__(self, point):
        self.point = QPointF(point)
        self.accepted = False

    def position(self):
        return self.point

    def setDropAction(self, action):
        pass

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def test_drop_uses_pressed_step_even_when_current_row_changes(strip):
    source = strip._list.visualItemRect(strip._list.item(0)).center()
    QTest.mousePress(strip._list.viewport(), Qt.MouseButton.LeftButton, pos=source)
    strip._list.setCurrentRow(1)
    moved = []
    strip.sig_step_moved.connect(lambda s, t: moved.append((s, t)))
    rect = strip._list.visualItemRect(strip._list.item(2))
    strip._list.dropEvent(Drop(QPoint(rect.right() - 1, rect.center().y())))
    QTest.mouseRelease(strip._list.viewport(), Qt.MouseButton.LeftButton, pos=source)
    assert moved == [(0, 3)]
    assert [s['label'] for s in strip._steps] == ['去直流', '增益', '带通']


def test_button_press_does_not_become_drag_source(strip):
    got = []
    strip.sig_step_toggled.connect(lambda i, enabled: got.append((i, enabled)))
    step = chip(strip, 1)
    step.dot_btn.show()
    QTest.mouseClick(step.dot_btn, Qt.MouseButton.LeftButton)
    assert got == [(1, True)]
    moved = []
    strip.sig_step_moved.connect(lambda s, t: moved.append((s, t)))
    strip._list.dropEvent(Drop(QPoint(1, 1)))
    assert moved == []


def test_hiding_embedded_run_button_reclaims_chain_width(strip, qapp):
    before = strip._list.width()
    strip.set_run_button_visible(False)
    qapp.processEvents()
    assert strip.run_button().isHidden()
    assert strip._list.width() > before


def test_label_drag_reaches_list_drag_start(strip, monkeypatch):
    started = []
    monkeypatch.setattr(type(strip._list), 'startDrag',
                        lambda view, actions: started.append(view._press_source))
    label = chip(strip, 0).name
    center = label.rect().center()
    QTest.mousePress(label, Qt.MouseButton.LeftButton, pos=center)
    moved = center + QPoint(QApplication.startDragDistance() + 15, 0)
    QApplication.sendEvent(label, QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(moved),
        QPointF(label.mapToGlobal(moved)), Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    QTest.mouseRelease(label, Qt.MouseButton.LeftButton, pos=moved)
    assert started == [0]


def test_drop_marker_shows_insertion_boundary_and_clears_on_drop(strip):
    rect = strip._list.visualItemRect(strip._list.item(0))
    QTest.mousePress(strip._list.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    last = strip._list.visualItemRect(strip._list.item(2))

    class Move(Drop):
        def source(self):
            return strip._list

        def acceptProposedAction(self):
            self.accepted = True

    event = Move(QPoint(last.left() + 1, last.center().y()))
    strip._list.dragMoveEvent(event)
    assert event.accepted
    assert strip._list._drop_marker.isVisibleTo(strip)
    assert strip._list._drop_marker.x() < last.left()
    strip._list.dropEvent(event)
    assert strip._list._drop_marker.isHidden()


def test_redraw_and_lock_close_all_existing_overflow_menus(strip, monkeypatch):
    first = strip._build_overflow_menu()
    second = strip._build_overflow_menu()
    closed = []
    for index, menu in enumerate((first, second)):
        monkeypatch.setattr(menu, 'close', lambda ix=index: closed.append(ix))
    strip.set_steps([{'label': '替换草稿', 'enabled': True}])
    assert closed == [0, 1]
    closed.clear()
    strip.setEnabled(False)
    assert closed == [0, 1]


def test_summary_edit_keeps_popup_anchor_identity(strip):
    anchor = strip.step_anchor(0)
    updated = [dict(s) for s in strip._steps]
    updated[0]['param_summary'] = '窗口 101'
    strip.set_steps(updated)
    assert strip.step_anchor(0) is anchor
    assert '窗口 101' in anchor.toolTip()
    assert strip.step_anchor(-1) is None
    assert strip.step_anchor(99) is None


def test_identical_display_in_a_new_context_invalidates_menu_and_drag(strip):
    strip.set_context(('line-a', 'input-a'))
    anchor = strip.step_anchor(0)
    menu = strip._build_overflow_menu()
    edited, moved, selected = [], [], []
    strip.sig_step_editor_requested.connect(edited.append)
    strip.sig_step_moved.connect(lambda source, target: moved.append((source, target)))
    strip.sig_step_selected.connect(selected.append)
    rect = strip._list.visualItemRect(strip._list.item(0))
    QTest.mousePress(strip._list.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    strip.set_context(('line-a', 'input-b'))
    strip.set_steps([dict(step) for step in strip._steps])
    assert strip.step_anchor(0) is anchor
    selected.clear()
    menu._step_submenus[1].actions()[0].trigger()
    menu._step_submenus[0].actions()[1].trigger()
    last = strip._list.visualItemRect(strip._list.item(2))
    strip._list.dropEvent(Drop(QPoint(last.right() - 1, last.center().y())))
    assert edited == moved == selected == []
    assert strip._list._drop_marker.isHidden()


def test_same_context_keeps_live_menu_and_parameter_anchor(strip, monkeypatch):
    strip.set_context(('line-a', 'input-a'))
    anchor = strip.step_anchor(0)
    menu = strip._build_overflow_menu()
    closed, edited = [], []
    monkeypatch.setattr(menu, 'close', lambda: closed.append(True))
    strip.sig_step_editor_requested.connect(edited.append)
    strip.set_context(('line-a', 'input-a'))
    assert strip.step_anchor(0) is anchor
    menu._step_submenus[0].actions()[1].trigger()
    assert edited == [0]
    assert closed == []
    updated = [dict(step) for step in strip._steps]
    updated[0]['param_summary'] = '窗口 101'
    strip.set_steps(updated)
    assert strip.step_anchor(0) is anchor


def test_context_change_cancels_the_strips_active_native_drag(strip, monkeypatch):
    strip.set_context(('line-a', 'input-a'))
    cancelled = []
    monkeypatch.setattr(QDrag, 'cancel', lambda: cancelled.append(True))
    # Simulate a draft switch dispatched inside Qt's nested native drag loop.
    monkeypatch.setattr(QListWidget, 'startDrag',
                        lambda view, actions: strip.set_context(('line-a', 'input-b')))
    rect = strip._list.visualItemRect(strip._list.item(0))
    QTest.mousePress(strip._list.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    strip._list.startDrag(Qt.DropAction.MoveAction)
    assert cancelled == [True]
    assert strip._list._press_source == -1
    assert strip._list._drop_marker.isHidden()


def test_overflow_editor_dispatches_one_explicit_edit_without_selection(strip):
    strip.select_step(2)
    selected, edited = [], []
    strip.sig_step_selected.connect(selected.append)
    strip.sig_step_editor_requested.connect(edited.append)
    menu = strip._build_overflow_menu()
    menu._step_submenus[1].actions()[1].trigger()
    assert edited == [1]
    assert selected == []


def test_overflow_popup_survives_exec_return_and_releases_on_close(strip, qapp):
    # RoundMenu.exec is nonblocking. The popup must outlive this method call.
    strip._on_overflow_menu()
    assert len(strip._overflow_menus) == 1
    menu = strip._overflow_menus[0]
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert not sip.isdeleted(menu)
    assert menu.isVisible()
    menu.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert sip.isdeleted(menu)
    assert strip._overflow_menus == []


def test_context_switch_releases_live_popup_before_the_next_render(strip, qapp):
    strip.set_context(('line-a', 'input-a'))
    strip._on_overflow_menu()
    menu = strip._overflow_menus[0]
    qapp.processEvents()
    assert menu.isVisible()
    strip.set_context(('line-a', 'input-b'))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert sip.isdeleted(menu)
    assert strip._overflow_menus == []
    # Invalidating the next render must never call close() on a deleted wrapper.
    strip.set_steps([{'label': '新草稿', 'enabled': True}])
    strip._on_overflow_menu()
    assert len(strip._overflow_menus) == 1
    strip._overflow_menus[0].close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert strip._overflow_menus == []

"""Real Fluent popup endpoints, dismissal and keyboard behavior with motion reduced."""
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QAbstractAnimation, QCoreApplication, QEvent, QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import Action, InfoBarPosition, MenuAnimationType, MessageBox, RoundMenu

from ui import motion
from ui.widgets.context_menus import add_action, make_menu, show_menu
from ui.widgets.info_bar import InfoBar


@pytest.fixture
def host(qapp):
    widget = QWidget()
    widget.resize(560, 400)
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize('direction', [MenuAnimationType.DROP_DOWN, MenuAnimationType.PULL_UP])
def test_reduced_menu_reaches_its_directional_endpoint(host, monkeypatch, direction):
    monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    menu = make_menu(host)
    add_action(menu, None, 'Open', lambda: None)
    menu.exec(host.mapToGlobal(QPoint(100, 200)), aniType=direction)
    assert menu.aniManager.ani.state() == QAbstractAnimation.State.Stopped
    assert menu.pos() == menu.aniManager.ani.endValue()
    menu.close()


@pytest.mark.parametrize('back_key', [Qt.Key.Key_Escape, Qt.Key.Key_Left])
def test_menu_preference_change_and_keyboard_submenu(host, monkeypatch, back_key):
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    menu = make_menu(host)
    sub = RoundMenu('Group', menu)
    sub.addAction(Action('Open'))
    menu.addMenu(sub)
    closed = []
    menu.closedSignal.connect(lambda: closed.append(True))
    show_menu(menu, host.mapToGlobal(QPoint(20, 20)))
    assert menu.aniManager.ani.state() == QAbstractAnimation.State.Running
    monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    menu.settle_motion()
    QTest.keyClick(menu.view, Qt.Key.Key_Right)
    assert sub.isVisible() and sub.pos() == sub.aniManager.ani.endValue()
    assert sub.aniManager.ani.state() == QAbstractAnimation.State.Stopped
    QTest.keyClick(sub.view, back_key)
    assert not sub.isVisible() and menu.isVisible() and not closed
    QTest.keyClick(menu.view, Qt.Key.Key_Right)
    assert sub.isVisible()
    menu.close()


def test_reduced_infobar_keeps_display_time_and_settles_the_stack(host, monkeypatch, qapp):
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    first = InfoBar.info('First', 'Read me', duration=-1, position=InfoBarPosition.TOP, parent=host)
    assert first.property('slideAni').state() == QAbstractAnimation.State.Running
    monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    first.settle_motion()
    initial_position = first.pos()
    second = InfoBar.info('Next', 'Read me', duration=-1, position=InfoBarPosition.TOP, parent=host)
    assert second.y() >= first.y() + first.height()
    assert second.property('slideAni').state() == QAbstractAnimation.State.Stopped
    first.close()
    assert second.y() == initial_position.y()
    second.close()

    closed = []
    timed = InfoBar.info('Timed', 'Read me', duration=0, parent=host)
    timed.closedSignal.connect(lambda: closed.append(True))
    assert timed.isVisible() and not closed
    qapp.processEvents()
    assert closed == [True]


def test_reduced_modal_dialog_completes_once_with_its_requested_result(host, monkeypatch):
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    box = MessageBox('Replace?', 'Keep the current result?', host)
    completed = []
    box.finished.connect(completed.append)
    box.show()
    assert box.graphicsEffect() is not None
    monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    box.settle_motion()
    assert box.graphicsEffect() is None
    box.cancelButton.click()
    assert not box.isVisible() and completed == [0]
    box.show()
    assert box.graphicsEffect() is None
    box.yesButton.click()
    assert not box.isVisible() and completed == [0, 1]

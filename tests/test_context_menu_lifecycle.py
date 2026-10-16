"""Modal menu callbacks retain the native dispatch receiver until return."""
import pytest

pytest.importorskip('PyQt6')
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import MessageBox, RoundMenu

from ui.widgets.context_menus import _LIVE_MENUS, add_action, make_menu
from ui.widgets.fluent_menu_keyboard import FluentMenuKeyboard


@pytest.mark.parametrize('keyboard', [False, True])
@pytest.mark.parametrize('submenu', [False, True])
def test_modal_cancel_preserves_dispatch_and_eventually_disposes_menu(qapp, keyboard, submenu):
    parent = QWidget()
    parent.resize(500, 400)
    parent.show()
    observations = []
    for _ in range(3):
        root = make_menu(parent)
        menu = RoundMenu('Child', root) if submenu else root

        def modal():
            box = MessageBox('Replace draft?', 'Keep the draft by cancelling.', parent)
            def cancel():
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                observations.append(not sip.isdeleted(root))
                box.cancelButton.click()
            QTimer.singleShot(0, cancel)
            assert box.exec() == 0
            assert not sip.isdeleted(root)

        action = add_action(menu, None, 'Replace', modal)
        if submenu:
            root.addMenu(menu)
        keyboard_handler = FluentMenuKeyboard(root)
        root.exec(parent.mapToGlobal(QPoint(20, 20)), ani=False)
        if submenu:
            menu.exec(parent.mapToGlobal(QPoint(220, 20)), ani=False)
        qapp.processEvents()
        item = action.property('item')
        if keyboard:
            menu.view.setCurrentItem(item)
            QTest.keyClick(menu.view, Qt.Key.Key_Return)
        else:
            QTest.mouseClick(menu.view.viewport(), Qt.MouseButton.LeftButton,
                             pos=menu.view.visualItemRect(item).center())
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        assert sip.isdeleted(root)
        assert id(root) not in _LIVE_MENUS
        assert keyboard_handler is not None
    assert observations == [True, True, True]
    parent.close()
    parent.deleteLater()


def test_action_trigger_remains_synchronous(qapp):
    menu = make_menu()
    called = []
    action = add_action(menu, None, 'Do', lambda: called.append(True))
    action.trigger()
    assert called == [True]
    menu.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

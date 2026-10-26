"""Use the real Fluent popup and Qt pointer events; only the OS drag loop is stubbed."""
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QMimeData, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
import pytest

from ui.widgets import line_drag_combo as module


@pytest.fixture
def combo(qapp):
    value = module.LineDragComboBox()
    value.addItem('L01', userData='L01')
    value.addItem('L02', userData='L02')
    def mime(line):
        result = QMimeData()
        result.setData('application/x-test-line', line.encode())
        return result
    value.mime_factory = mime
    value.resize(220, 36)
    value.show()
    value._showComboMenu()
    qapp.processEvents()
    yield value
    if value.dropMenu and not sip.isdeleted(value.dropMenu):
        value.dropMenu.close()
    value.close()
    value.deleteLater()
    qapp.processEvents()


def row_position(combo):
    menu = combo.dropMenu
    return menu.view.visualItemRect(menu.view.item(1)).center()


def test_single_click_keeps_normal_selection(combo, qapp):
    selected = []
    combo.currentIndexChanged.connect(selected.append)
    QTest.mouseClick(combo.dropMenu.view.viewport(), Qt.MouseButton.LeftButton, pos=row_position(combo))
    qapp.processEvents()
    assert combo.currentData() == 'L02'
    assert selected == [1]
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert combo.dropMenu is None
    combo._toggleComboMenu()
    assert combo.dropMenu.isVisible()


def test_drag_cancel_does_not_select_row_and_closes_popup(combo, qapp, monkeypatch):
    seen = []
    class Drag:
        def __init__(self, parent):
            self.parent = parent
        def setMimeData(self, mime):
            self.mime = mime
        def setPixmap(self, pixmap):
            assert not pixmap.isNull()
        def exec(self, action):
            assert not combo.dropMenu.isVisible()
            seen.append(bytes(self.mime.data('application/x-test-line')))
            return Qt.DropAction.IgnoreAction
    monkeypatch.setattr(module, 'QDrag', Drag)
    menu = combo.dropMenu
    point = row_position(combo)
    QTest.mousePress(menu.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert combo.currentData() == 'L01'  # press alone must not activate
    moved = QPointF(point.x()+30, point.y())
    event = QMouseEvent(QEvent.Type.MouseMove, moved, Qt.MouseButton.NoButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    qapp.sendEvent(menu.view.viewport(), event)
    assert seen == [b'L02']
    assert combo.currentData() == 'L01' and not menu.isVisible()

"""Small Fluent object-menu binding; never attach it to text editors."""
from PyQt6.QtCore import QEvent, QObject, QPoint, Qt
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout,
)

from ui.widgets.context_menus import add_action, bind_menu_keys, make_menu, show_menu


def object_menu(parent):
    return make_menu(parent)


def object_action(menu, command, text, slot, *, enabled=True, reason=''):
    if not enabled and reason:
        text += f'（{reason}）'
    action = add_action(menu, None, text, slot, enabled=enabled)
    action.setData(command)
    if reason and not enabled:
        action.setToolTip(reason)
    return action


def show_details(parent, title, text):
    """Read-only, copyable details; no nested blocking event loop."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.resize(520, 340)
    layout = QVBoxLayout(dialog)
    editor = QPlainTextEdit(dialog)
    editor.setReadOnly(True)
    editor.setPlainText(text)
    layout.addWidget(editor)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.show()
    return dialog


class ObjectMenuBinding(QObject):
    """Pointer hit and keyboard current item share the exact same builder."""

    def __init__(self, page, target, builder):
        super().__init__(page)
        self.target, self.builder = target, builder
        self.viewport = target.viewport() if hasattr(target, 'viewport') else target
        self.menu = None
        if target.focusPolicy() == Qt.FocusPolicy.NoFocus:
            target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        target.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        target.customContextMenuRequested.connect(self.show)
        self.shortcuts = bind_menu_keys(target, self.show_keyboard)
        target.installEventFilter(self)
        if self.viewport is not target:
            self.viewport.installEventFilter(self)
        page.installEventFilter(self)

    def show_keyboard(self):
        item = self.target.currentItem() if hasattr(self.target, 'currentItem') else None
        position = QPoint(-1, -1)
        if item is not None:
            position = self.target.visualItemRect(item).center()
        elif hasattr(self.target, 'currentIndex') and self.target.currentIndex().isValid():
            position = self.target.visualRect(self.target.currentIndex()).center()
        self.show(position)

    def show(self, position):
        self.close()
        menu = self.builder(position)
        if not menu.actions() and not menu._subMenus:
            menu.deleteLater()
            return
        self.menu = menu
        menu.closedSignal.connect(lambda: setattr(self, 'menu', None) if self.menu is menu else None)
        show_menu(menu, self.viewport.mapToGlobal(position if position.x() >= 0 else QPoint(8, 8)))

    def close(self):
        if self.menu is not None:
            self.menu.close()
            self.menu = None

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.Hide:
            self.close()
        if (self.viewport is not self.target and watched is self.viewport
                and kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease)
                and event.button() == Qt.MouseButton.RightButton):
            return True
        return super().eventFilter(watched, event)

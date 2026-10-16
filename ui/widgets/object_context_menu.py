"""Small Fluent object-menu binding; never attach it to text editors."""
from PyQt6.QtCore import QEvent, QObject, QPoint, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout,
)

from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.fluent_menu_keyboard import FluentMenuKeyboard


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
        self.menu = None
        if target.focusPolicy() == Qt.FocusPolicy.NoFocus:
            target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        target.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        target.customContextMenuRequested.connect(self.show)
        self.shortcuts = []
        for sequence in ('Shift+F10', 'Menu'):
            shortcut = QShortcut(QKeySequence(sequence), target)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self.show_keyboard)
            self.shortcuts.append(shortcut)
        target.installEventFilter(self)
        page.installEventFilter(self)

    def show_keyboard(self):
        item = self.target.currentItem() if hasattr(self.target, 'currentItem') else None
        position = self.target.visualItemRect(item).center() if item is not None else QPoint(-1, -1)
        self.show(position)

    def show(self, position):
        self.close()
        menu = self.builder(position)
        self.menu = menu
        menu._object_keyboard = FluentMenuKeyboard(menu)
        menu.closedSignal.connect(lambda: setattr(self, 'menu', None))
        viewport = self.target.viewport() if hasattr(self.target, 'viewport') else self.target
        menu.exec(viewport.mapToGlobal(position if position.x() >= 0 else QPoint(8, 8)))
        menu._object_keyboard.focus_menu(menu)

    def close(self):
        if self.menu is not None:
            self.menu.close()
            self.menu = None

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Hide:
            self.close()
        return super().eventFilter(watched, event)

"""Keyboard navigation for Fluent menus rendered by an embedded QListWidget.

RoundMenu does not register its custom submenu rows with QMenu. Handle those
rows explicitly so the view menu is usable without a pointer, while retaining
the shared Fluent menu styling and lifecycle.
"""
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QAction
from qfluentwidgets import RoundMenu


class FluentMenuKeyboard(QObject):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.targets = {}

        def install(menu):
            for widget in (menu, menu.view, menu.view.viewport()):
                self.targets[id(widget)] = menu
                widget.installEventFilter(self)
            for submenu in menu._subMenus:
                install(submenu)

        install(root)

    @staticmethod
    def _rows(menu):
        return [row for row in range(menu.view.count())
                if menu.view.item(row).flags() & Qt.ItemFlag.ItemIsEnabled
                and menu.view.item(row).flags() & Qt.ItemFlag.ItemIsSelectable]

    def focus_menu(self, menu):
        rows = self._rows(menu)
        if rows and menu.view.currentRow() not in rows:
            menu.view.setCurrentRow(rows[0])
        menu.view.setFocus(Qt.FocusReason.PopupFocusReason)

    def _back(self, menu):
        parent = menu.parentMenu
        if parent is not None:
            menu.hide()
            parent.view.setFocus(Qt.FocusReason.PopupFocusReason)
        else:
            menu.close()

    def _activate(self, menu, item):
        target = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(target, RoundMenu):
            rect = menu.view.visualItemRect(item)
            target.exec(menu.view.viewport().mapToGlobal(rect.topRight()), ani=False)
            self.focus_menu(target)
        elif isinstance(target, QAction) and target.isEnabled():
            # Share pointer dispatch, including its menu lifetime guard while a
            # synchronous action opens a modal dialog.
            menu._onItemClicked(item)

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(watched, event)
        menu = self.targets.get(id(watched))
        if menu is None:
            return False
        key = event.key()
        rows = self._rows(menu)
        current = menu.view.currentRow()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Home, Qt.Key.Key_End):
            if rows:
                if key == Qt.Key.Key_Home:
                    row = rows[0]
                elif key == Qt.Key.Key_End:
                    row = rows[-1]
                else:
                    direction = -1 if key == Qt.Key.Key_Up else 1
                    index = rows.index(current) if current in rows else (-1 if direction > 0 else 0)
                    row = rows[(index + direction) % len(rows)]
                menu.view.setCurrentRow(row)
        elif key in (Qt.Key.Key_Escape, Qt.Key.Key_Left):
            self._back(menu)
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            if current in rows:
                item = menu.view.item(current)
                if key != Qt.Key.Key_Right or isinstance(item.data(Qt.ItemDataRole.UserRole), RoundMenu):
                    self._activate(menu, item)
        else:
            return super().eventFilter(watched, event)
        event.accept()
        return True

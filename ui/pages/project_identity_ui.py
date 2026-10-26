"""Small rename/move entry points; this page helper never writes project data."""

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from qfluentwidgets import FluentIcon as FIF

from ui.dialogs import ask_display_name, ask_project_destination
from ui.widgets.context_menus import add_action


class ProjectIdentityUi(QObject):
    rename_project_requested = pyqtSignal(str, str)
    rename_line_requested = pyqtSignal(str, str, str)
    relocation_requested = pyqtSignal(str, str)

    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.has_active_tasks = lambda: False
        self.shortcut = QShortcut(
            QKeySequence("F2"), page._lines_table, context=Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.shortcut.activated.connect(self.rename_selected)

    def _project_id(self):
        return str(getattr(self.page._summary, "project_id", "") or "")

    def _can_write(self):
        return self.page._can_write() and not self.has_active_tasks()

    def apply_line_name(self, line):
        for row, old in enumerate(self.page._lines):
            if str(old.line_id) == str(line.line_id):
                self.page._lines[row] = line
                self.page._lines_table.item(row, 1).setText(str(line.name))
                self.page._lines_table.item(row, 1).setToolTip(str(line.name))
                break

    def rename_selected(self):
        rows = self.page._lines_table.selectionModel().selectedRows()
        if len(rows) == 1 and rows[0].row() < len(self.page._lines):
            self.rename_line(str(self.page._lines[rows[0].row()].line_id))

    def add_line_rename(self, menu, line_id, *, single=True):
        project_id = self._project_id()
        add_action(
            menu,
            FIF.EDIT,
            "重命名测线…",
            lambda: self.rename_line(line_id) if project_id == self._project_id() else None,
            enabled=single and self._can_write(),
        )

    def rename_line(self, line_id):
        if not self._can_write():
            return
        line = next((line for line in self.page._lines if str(line.line_id) == str(line_id)), None)
        if line is None:
            return
        project_id = self._project_id()
        name = ask_display_name(self.page, "重命名测线", str(line.name), subject=f"测线号：{line_id}（保留）")
        if name and project_id == self._project_id() and self._can_write():
            self.rename_line_requested.emit(project_id, str(line_id), name)

    def add_project_actions(self, menu):
        project_id = self._project_id()
        def current():
            return project_id == self._project_id() and self._can_write()
        add_action(
            menu,
            FIF.EDIT,
            "重命名项目…",
            lambda: self.rename_project() if current() else None,
            enabled=self._can_write(),
        )
        add_action(
            menu, FIF.FOLDER, "迁移项目目录…", lambda: self.relocate() if current() else None, enabled=self._can_write()
        )
        menu.addSeparator()

    def rename_project(self):
        project_id = self._project_id()
        name = ask_display_name(self.page, "重命名项目", str(self.page._summary.name))
        if name and project_id == self._project_id() and self._can_write():
            self.rename_project_requested.emit(project_id, name)

    def relocate(self):
        project_id = self._project_id()
        target = ask_project_destination(self.page, str(self.page._summary.root_path))
        if target and project_id == self._project_id() and self._can_write():
            self.relocation_requested.emit(project_id, target)

"""Small rename/move entry points; this page helper never writes project data."""

import os

from PyQt6.QtCore import QObject, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF

from ui.dialogs import ask_display_name, ask_project_destination
from ui.icons import result_delete_icon
from ui.widgets.context_menus import add_action, make_menu


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

    def on_lines_context_menu(self, pos) -> None:
        page = self.page
        row = page._lines_table.rowAt(pos.y())
        if row < 0 or row >= len(page._lines):
            return
        if not page._lines_table.selectionModel().isRowSelected(row, page._lines_table.rootIndex()):
            page._lines_table.selectRow(row)
        line_id = str(getattr(page._lines[row], 'line_id', '') or '')
        if not line_id:
            return
        source = page._source_path_cache.get(line_id)
        menu = make_menu(page)
        single = len(page._lines_table.selectionModel().selectedRows()) == 1
        add_action(menu, FIF.DEVELOPER_TOOLS, '处理该测线（跳转处理页）',
                   lambda: page.line_process_requested.emit(line_id), enabled=single and not page._busy)
        self.add_line_rename(menu, line_id, single=single)
        add_action(menu, FIF.SYNC, '传感器同步…', page.show_sensor_tool,
                   enabled=single and self._can_write())
        add_action(menu, FIF.DELETE, '删除所选测线',
                   page._on_delete_selected_lines, enabled=self._can_write())
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制数据文件路径',
                   lambda: QApplication.clipboard().setText(source), enabled=bool(source))
        add_action(menu, FIF.FOLDER, '打开数据所在文件夹',
                   lambda: self.open_source_folder(source), enabled=bool(source))
        menu.addSeparator()
        add_action(menu, None, '复制测线号',
                   lambda: QApplication.clipboard().setText(line_id))
        menu.exec(page._lines_table.viewport().mapToGlobal(pos))

    @staticmethod
    def open_source_folder(source: str) -> None:
        folder = os.path.dirname(str(source))
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def on_artifacts_context_menu(self, pos) -> None:
        page = self.page
        row = page._artifacts_table.rowAt(pos.y())
        if row < 0 or row >= len(page._artifacts):
            return
        page._artifacts_table.selectRow(row)
        menu = make_menu(page)
        add_action(menu, FIF.VIEW, '在处理页查看', page._emit_artifact_preview)
        add_action(menu, result_delete_icon(), '删除所选成果',
                   self.on_artifacts_context_delete, enabled=self._can_write())
        menu.exec(page._artifacts_table.viewport().mapToGlobal(pos))

    def on_artifacts_context_delete(self) -> None:
        page = self.page
        row = page._artifacts_table.currentRow()
        if row < 0 or row >= len(page._artifacts):
            return
        artifact_id = str(getattr(page._artifacts[row], 'artifact_id', '') or '')
        if artifact_id:
            page._on_artifact_delete_clicked(artifact_id)

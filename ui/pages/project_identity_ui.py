"""Small rename/move entry points; this page helper never writes project data."""

import os

from PyQt6.QtCore import QObject, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QInputDialog
from qfluentwidgets import FluentIcon as FIF

from ui.dialogs import ask_display_name, ask_project_destination, ask_line_properties
from ui.icons import result_delete_icon
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.object_context_menu import ObjectMenuBinding


class ProjectIdentityUi(QObject):
    rename_project_requested = pyqtSignal(str, str)
    rename_line_requested = pyqtSignal(str, str, str)
    relocation_requested = pyqtSignal(str, str)
    properties_requested = pyqtSignal(str, str, object, bool, object)

    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.has_active_tasks = lambda: False
        self.shortcut = QShortcut(
            QKeySequence("F2"), page._lines_table, context=Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.shortcut.activated.connect(self.rename_selected)
        self.lines_menu = ObjectMenuBinding(page, page._lines_table, self.lines_context_menu)
        self.artifacts_menu = ObjectMenuBinding(page, page._artifacts_table, self.artifacts_context_menu)

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

    def line_properties(self, line_id):
        if not self._can_write():
            return
        line = next((item for item in self.page._lines if str(item.line_id) == str(line_id)), None)
        if line is None:
            return
        project_id = self._project_id()
        result = ask_line_properties(self.page, line)
        if result is not None and project_id == self._project_id() and self._can_write():
            self.properties_requested.emit(project_id, line_id, *result)

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

    def edit_timezone(self):
        project_id = self._project_id()
        if not project_id or not self._can_write():
            return
        current = self.page._info_values['timezone'].text()
        current = current if current and current != '--' else 'UTC'
        values = list(dict.fromkeys((current, 'Asia/Shanghai', 'UTC')))
        value, accepted = QInputDialog.getItem(self.page, '项目时区', 'IANA 时区', values)
        if accepted and value.strip() not in ('', current) and self._can_write() and self._project_id() == project_id:
            self.page.timezone_requested.emit(project_id, value.strip())

    def lines_context_menu(self, pos):
        page = self.page
        menu = make_menu(page)
        row = page._lines_table.rowAt(pos.y())
        if row < 0 or row >= len(page._lines):
            return menu
        line = page._lines[row]
        line_id = str(getattr(line, 'line_id', '') or '')
        if not line_id:
            return menu
        project_id = self._project_id()
        selected = page._lines_table.selectionModel().selectedRows()
        indices = [index.row() for index in selected] if any(index.row() == row for index in selected) else [row]
        targets = tuple(page._lines[index] for index in indices if 0 <= index < len(page._lines))
        source = page._source_path_cache.get(line_id)

        def current(callback):
            if project_id == self._project_id() and any(candidate is line for candidate in page._lines):
                callback()

        def sync():
            if self._can_write():
                index = next(i for i, candidate in enumerate(page._lines) if candidate is line)
                page._lines_table.selectRow(index)
                page.show_sensor_tool()

        def remove():
            if self._can_write() and all(any(candidate is target for candidate in page._lines) for target in targets):
                page.line_delete_requested.emit([str(target.line_id) for target in targets])

        add_action(menu, FIF.DEVELOPER_TOOLS, '在处理页打开',
                   lambda: current(lambda: page.line_process_requested.emit(line_id)), enabled=not page._busy)
        add_action(menu, FIF.EDIT, '测线属性…', lambda: current(lambda: self.line_properties(line_id)),
                   enabled=self._can_write())
        add_action(menu, FIF.EDIT, '重命名测线…', lambda: current(lambda: self.rename_line(line_id)),
                   enabled=self._can_write())
        add_action(menu, FIF.SYNC, '传感器同步…', lambda: current(sync), enabled=self._can_write())
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制数据文件路径',
                   lambda: current(lambda: QApplication.clipboard().setText(source)), enabled=bool(source))
        add_action(menu, FIF.FOLDER, '打开数据所在文件夹',
                   lambda: current(lambda: self.open_source_folder(source)), enabled=bool(source))
        add_action(menu, None, '复制测线号',
                   lambda: current(lambda: QApplication.clipboard().setText(line_id)))
        menu.addSeparator()
        add_action(menu, FIF.DELETE, f'永久删除 {len(targets)} 条测线…',
                   lambda: current(remove), enabled=self._can_write())
        return menu

    @staticmethod
    def open_source_folder(source: str) -> None:
        folder = os.path.dirname(str(source))
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def artifacts_context_menu(self, pos):
        page = self.page
        menu = make_menu(page)
        row = page._artifacts_table.rowAt(pos.y())
        if row < 0 or row >= len(page._artifacts):
            return menu
        artifact = page._artifacts[row]
        artifact_id = str(getattr(artifact, 'artifact_id', '') or '')
        line_id = str(getattr(artifact, 'line_id', '') or page._current_line_id)
        project_id = self._project_id()

        def current():
            return project_id == self._project_id() and any(item is artifact for item in page._artifacts)

        add_action(menu, FIF.VIEW, '在处理页查看',
                   lambda: page.artifact_preview_requested.emit(line_id, artifact_id) if current() else None)
        menu.addSeparator()
        add_action(menu, result_delete_icon(), '永久删除成果…',
                   lambda: page.artifact_delete_requested.emit(line_id, [artifact_id])
                   if current() and self._can_write() else None, enabled=self._can_write())
        return menu

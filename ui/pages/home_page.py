# -*- coding: utf-8 -*-
"""Startup content and compatibility facade for the retired home route.

ProjectPage owns visibility. Previews and tasks use their dedicated workspaces.
"""
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, PrimaryPushButton, PushButton, ScrollArea
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.page_scaffold import make_hint, style_transparent_scroll
from ui.widgets.context_menus import add_action, make_menu


class HomePage(ScrollArea):
    """New/open/recent content embedded in the single project workspace."""

    new_project_requested = pyqtSignal()
    open_project_requested = pyqtSignal()
    open_recent_project_requested = pyqtSignal(str)
    import_line_requested = pyqtSignal()
    goto_page = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary = None
        self._settings = None
        self._recent_paths = []
        self._pinned_paths = []
        style_transparent_scroll(self)
        container = QWidget(self)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(*constants.PAGE_MARGINS)
        outer.addStretch(1)
        self._content = QWidget(container)
        self._content.setMaximumWidth(constants.FORM_COLUMN_MAX_WIDTH)
        content = QVBoxLayout(self._content)
        content.setSpacing(constants.PAGE_SPACING)
        title = BodyLabel('开始一个勘探项目', self._content)
        title.setStyleSheet(f'font-size: {constants.FONT_SIZE_TITLE}pt; font-weight: 600;')
        content.addWidget(title)
        content.addWidget(make_hint('打开已有项目，或新建项目后导入测线。', parent=self._content))
        actions = QHBoxLayout()
        self.open_btn = PrimaryPushButton('打开项目', self._content, FIF.FOLDER)
        self.new_btn = PushButton('新建项目', self._content, FIF.ADD)
        self.new_btn.clicked.connect(self.new_project_requested)
        self.open_btn.clicked.connect(self.open_project_requested)
        actions.addWidget(self.open_btn)
        actions.addWidget(self.new_btn)
        actions.addStretch(1)
        content.addLayout(actions)
        content.addWidget(BodyLabel('最近项目', self._content))
        self._recent_list = QListWidget(self._content)
        self._recent_list.setMinimumHeight(120)
        self._recent_list.setMaximumHeight(280)
        self._recent_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._recent_list.itemActivated.connect(self._open_recent_item)
        self._recent_list.itemClicked.connect(self._open_recent_item)
        self._recent_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._recent_list.customContextMenuRequested.connect(self._recent_context_menu)
        content.addWidget(self._recent_list)
        self._recent_hint = make_hint('暂无最近项目', parent=self._content)
        content.addWidget(self._recent_hint)
        self._startup_error = make_hint('', parent=self._content, key='error')
        self._startup_error.setWordWrap(True)
        self._startup_error.hide()
        content.addWidget(self._startup_error)
        centered = QHBoxLayout()
        centered.addStretch(1)
        centered.addWidget(self._content, 1)
        centered.addStretch(1)
        outer.addLayout(centered)
        outer.addStretch(1)
        self.setWidget(container)
        # Compatibility control, absent from the startup UI.
        self.import_btn = PushButton('导入测线', self)
        self.import_btn.clicked.connect(self.import_line_requested)
        self.import_btn.hide()
        self.set_current_project(None)
        self.set_recent_projects([])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_content'):
            width = max(0, self.viewport().width() - 2 * constants.SPACE_6)
            self._content.setFixedWidth(min(constants.FORM_COLUMN_MAX_WIDTH, width))

    def set_current_project(self, summary) -> None:
        self._summary = summary
        self.import_btn.setEnabled(summary is not None and not bool(getattr(summary, 'read_only', False)))
        self.set_startup_error('')

    def set_preview_bundle(self, bundle) -> None:
        """Legacy coordinator sink; previews belong to processing/interpretation."""

    def mini_jobs(self):
        """The global output panel and Jobs page own task views."""
        return None

    def set_settings_manager(self, settings) -> None:
        self._settings = settings
        self.refresh_recents()

    def refresh_recents(self) -> None:
        if self._settings is not None:
            self._pinned_paths = list(self._settings.get('pinned_recent_projects', []) or [])
            paths = [*self._pinned_paths, *(self._settings.get('recent_projects', []) or [])]
            self.set_recent_projects(list(dict.fromkeys(paths))[:constants.RECENT_PROJECTS_MAX])

    def set_recent_projects(self, paths) -> None:
        self._recent_paths = list(dict.fromkeys(str(p) for p in paths if p))
        ordered = sorted(self._recent_paths, key=lambda p: p not in self._pinned_paths)
        self._recent_list.clear()
        for path in ordered:
            available = Path(path).is_dir()
            prefix = '★ ' if path in self._pinned_paths else ''
            item = QListWidgetItem(f'{prefix}{Path(path).name}\n{path}  ·  {"可打开" if available else "不可用：目录不存在"}')
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self._recent_list.addItem(item)
        self._recent_hint.setVisible(not self._recent_paths)
        self._recent_list.setVisible(bool(self._recent_paths))

    def set_startup_error(self, message: str) -> None:
        self._startup_error.setText(str(message or ''))
        self._startup_error.setVisible(bool(message))

    def _open_recent_item(self, item) -> None:
        path = item.data(Qt.ItemDataRole.UserRole) if item is not None else ''
        if path:
            self.open_recent_project_requested.emit(str(path))

    def _save_recents(self) -> None:
        if self._settings is not None:
            self._settings.set('recent_projects', self._recent_paths)
            self._settings.set('pinned_recent_projects', self._pinned_paths)
            self._settings.save()
        self.set_recent_projects(self._recent_paths)

    def _toggle_pin(self, path: str) -> None:
        if path in self._pinned_paths:
            self._pinned_paths.remove(path)
        else:
            self._pinned_paths.append(path)
        self._save_recents()

    def _remove_recent(self, path: str) -> None:
        self._recent_paths = [p for p in self._recent_paths if p != path]
        self._pinned_paths = [p for p in self._pinned_paths if p != path]
        self._save_recents()

    def _recent_context_menu(self, pos) -> None:
        item = self._recent_list.itemAt(pos)
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole))
        menu = make_menu(self)
        add_action(menu, FIF.FOLDER, '打开项目', lambda: self.open_recent_project_requested.emit(path))
        add_action(menu, FIF.PIN, '取消固定' if path in self._pinned_paths else '固定项目', lambda: self._toggle_pin(path))
        add_action(menu, FIF.COPY, '复制项目路径', lambda: QApplication.clipboard().setText(path))
        add_action(menu, FIF.REMOVE, '从最近列表移除', lambda: self._remove_recent(path))
        menu.exec(self._recent_list.viewport().mapToGlobal(pos))

    def project_menu(self, trigger, *, can_close: bool, close_requested, enabled: bool = True):
        """Share the recent model and commands from every workbench page."""
        self.refresh_recents()
        menu = make_menu(trigger)
        add_action(menu, FIF.ADD, '新建项目', self.new_project_requested.emit, enabled=enabled)
        add_action(menu, FIF.FOLDER, '打开项目…', self.open_project_requested.emit, enabled=enabled)
        if self._recent_paths:
            menu.addSeparator()
        for path in sorted(self._recent_paths, key=lambda p: p not in self._pinned_paths):
            label = ('★ ' if path in self._pinned_paths else '') + Path(path).name
            if not Path(path).is_dir():
                label += ' · 不可用'
            action = add_action(menu, FIF.FOLDER, label,
                                lambda _checked=False, p=path: self.open_recent_project_requested.emit(p), enabled=enabled)
            action.setToolTip(path)
        if can_close:
            menu.addSeparator()
            add_action(menu, FIF.CLOSE, '关闭当前项目', close_requested, enabled=enabled)
        menu.closedSignal.connect(trigger.setFocus)
        return menu


__all__ = ['HomePage']

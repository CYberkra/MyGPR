"""Project launchpad; ProjectPage owns startup / empty / ledger visibility."""
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QSizePolicy, QVBoxLayout, QWidget,
)
from qfluentwidgets import PrimaryPushButton, PushButton, ScrollArea, SearchLineEdit, isDarkTheme
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.controllers.recent_projects_controller import RecentProjectsController
from ui.design_tokens import color, radius
from ui.page_scaffold import make_hint, style_transparent_scroll
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.object_context_menu import ObjectMenuBinding
from ui.widgets.recent_project_list import RecentProjectRow


class HomePage(ScrollArea):
    new_project_requested = pyqtSignal()
    open_project_requested = pyqtSignal()
    restore_requested = pyqtSignal()
    open_recent_project_requested = pyqtSignal(str)
    project_delete_requested = pyqtSignal(str)
    import_line_requested = pyqtSignal()
    goto_page = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary = None
        self._settings = None
        self._recent_paths = []
        self._pinned_paths = []
        self._availability = {}
        self._rows = {}
        self._selected_path = ''
        self._restore_paths = []
        self._user_interacted = False
        self._opening_path = ''
        self._busy = False
        self._backend_ready = True
        self._pinned_only = False
        self._dark = isDarkTheme()
        self._probe = RecentProjectsController(self)
        self._probe.availability_changed.connect(self._on_availability)
        style_transparent_scroll(self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._content = QWidget(self)
        self._content.setObjectName('startupContent')
        self._content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._build_sidebar()
        self._build_browser()
        self.setWidget(self._content)
        # Compatibility sink: import belongs to the opened-project workspace.
        self.import_btn = PushButton('导入测线', self)
        self.import_btn.clicked.connect(self.import_line_requested)
        self.import_btn.hide()
        self.set_current_project(None)
        self._render_recents()
        self.apply_theme(self._dark)

    def _build_sidebar(self) -> None:
        self._sidebar = QWidget(self._content)
        self._sidebar.setObjectName('startupSidebar')
        self._sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._sidebar_layout = QVBoxLayout(self._sidebar)
        self._sidebar_layout.setContentsMargins(20, 28, 20, 20)
        self._sidebar_layout.setSpacing(12)
        self.open_btn = PrimaryPushButton('打开项目', self._sidebar, FIF.FOLDER)
        self.new_btn = PushButton('新建项目', self._sidebar, FIF.ADD)
        self.restore_btn = PushButton('从备份恢复…', self._sidebar)
        self.restore_btn.clicked.connect(self.restore_requested)
        self.open_btn.clicked.connect(lambda: self._request_dialog(False))
        self.new_btn.clicked.connect(lambda: self._request_dialog(True))
        self._action_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        self._action_layout.addWidget(self.open_btn)
        self._action_layout.addWidget(self.new_btn)
        self._action_layout.setSpacing(10)
        self._sidebar_layout.addLayout(self._action_layout)
        self._sidebar_layout.addWidget(self.restore_btn)
        self._filter_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        self.recent_btn = PushButton('最近打开', self._sidebar)
        self.pinned_btn = PushButton('已固定', self._sidebar, FIF.PIN)
        for button, pinned in ((self.recent_btn, False), (self.pinned_btn, True)):
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, p=pinned: self._set_filter(p))
            self._filter_layout.addWidget(button)
        self._filter_layout.setSpacing(6)
        self._sidebar_layout.addSpacing(14)
        self._sidebar_layout.addLayout(self._filter_layout)
        self._sidebar_layout.addStretch(1)
        self._layout.addWidget(self._sidebar)
        for button in (self.open_btn, self.new_btn, self.recent_btn, self.pinned_btn):
            button.setMinimumHeight(38)

    def _build_browser(self) -> None:
        self._main = QWidget(self._content)
        main = QVBoxLayout(self._main)
        main.setContentsMargins(30, 28, 30, 24)
        main.setSpacing(18)
        self._title = QLabel('最近项目', self._main)
        self._title.setObjectName('startupTitle')
        main.addWidget(self._title)
        self.search = SearchLineEdit(self._main)
        self.search.setPlaceholderText('搜索项目')
        self.search.setAccessibleName('搜索项目名称或路径')
        self.search.setMinimumWidth(0)
        self.search.textChanged.connect(self._search_changed)
        main.addWidget(self.search)
        self._browser_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._browser_layout.setSpacing(24)
        self._list_column = QWidget(self._main)
        column = QVBoxLayout(self._list_column)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        self._recent_list = QListWidget(self._list_column)
        self._recent_list.setObjectName('recentProjects')
        self._recent_list.setAccessibleName('最近项目')
        self._recent_list.setMinimumSize(0, 210)
        self._recent_list.setSpacing(3)
        self._recent_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._recent_list.itemActivated.connect(self._open_recent_item)
        self._recent_list.currentItemChanged.connect(self._selection_changed)
        self._recent_list.itemClicked.connect(lambda _item: self._cancel_restore())
        self._menu_binding = ObjectMenuBinding(self, self._recent_list, self._recent_context_menu)
        column.addWidget(self._recent_list, 1)
        self._empty = QWidget(self._list_column)
        empty = QVBoxLayout(self._empty)
        empty.setContentsMargins(0, 32, 0, 32)
        self._recent_hint = QLabel('还没有项目', self._empty)
        self._recent_hint.setObjectName('startupEmpty')
        empty.addStretch(1)
        empty.addWidget(self._recent_hint, 0, Qt.AlignmentFlag.AlignHCenter)
        self._empty_action = PushButton('新建项目', self._empty)
        self._empty_action.clicked.connect(self._empty_action_clicked)
        empty.addWidget(self._empty_action, 0, Qt.AlignmentFlag.AlignHCenter)
        empty.addStretch(1)
        column.addWidget(self._empty, 1)
        self._browser_layout.addWidget(self._list_column, 1)
        self._build_details()
        main.addLayout(self._browser_layout, 1)
        self._startup_error = make_hint('', parent=self._main, key='error')
        self._startup_error.setWordWrap(True)
        self._startup_error.setTextFormat(Qt.TextFormat.PlainText)
        self._startup_error.hide()
        main.addWidget(self._startup_error)
        self._layout.addWidget(self._main, 1)

    def _build_details(self) -> None:
        self._details = QWidget(self._main)
        self._details.setObjectName('startupDetails')
        self._details.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        detail = QVBoxLayout(self._details)
        detail.setContentsMargins(20, 20, 20, 20)
        detail.setSpacing(14)
        self._detail_name = QLabel(self._details)
        self._detail_name.setObjectName('startupProjectName')
        self._detail_path = QLabel(self._details)
        self._detail_path.setObjectName('startupProjectPath')
        self._detail_state = make_hint('', parent=self._details, key='warning')
        for label in (self._detail_name, self._detail_path, self._detail_state):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            detail.addWidget(label)
        self.detail_open_btn = PrimaryPushButton('打开项目', self._details, FIF.FOLDER)
        self.detail_open_btn.setMinimumHeight(38)
        self.detail_open_btn.clicked.connect(lambda: self._request_open(self._selected_path))
        detail.addWidget(self.detail_open_btn)
        self._remove_btn = PushButton('从最近列表移除', self._details)
        self._remove_btn.clicked.connect(lambda: self._remove_recent(self._selected_path))
        detail.addWidget(self._remove_btn)
        detail.addStretch(1)
        self._browser_layout.addWidget(self._details)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not hasattr(self, '_details'):
            return
        width = self.viewport().width()
        narrow, stacked = width < 650, width < 1050
        direction = QBoxLayout.Direction
        self._layout.setDirection(direction.TopToBottom if narrow else direction.LeftToRight)
        if narrow:
            self._sidebar.setMinimumWidth(0)
            self._sidebar.setMaximumWidth(16777215)
        else:
            self._sidebar.setFixedWidth(208)
        self._action_layout.setDirection(direction.LeftToRight if narrow else direction.TopToBottom)
        self._filter_layout.setDirection(direction.LeftToRight if narrow else direction.TopToBottom)
        self._sidebar_layout.setContentsMargins(*((16, 16, 16, 12) if narrow else (20, 28, 20, 20)))
        margin = 16 if narrow else 30
        self._main.layout().setContentsMargins(margin, 24, margin, 24)
        self._browser_layout.setDirection(direction.TopToBottom if stacked else direction.LeftToRight)
        if stacked:
            self._details.setMinimumWidth(0)
            self._details.setMaximumWidth(16777215)
        else:
            self._details.setFixedWidth(250)

    def set_current_project(self, summary) -> None:
        self._summary = summary
        self.import_btn.setEnabled(summary is not None and not bool(getattr(summary, 'read_only', False)))
        if summary is not None:
            self._cancel_restore()
            self._probe.cancel()
        self.set_startup_error('')

    def set_preview_bundle(self, bundle) -> None:
        """Compatibility sink; previews belong to processing/interpretation."""

    def mini_jobs(self):
        return None

    def set_settings_manager(self, settings) -> None:
        self._settings = settings
        self.refresh_recents()

    def refresh_recents(self) -> None:
        if self._settings is not None:
            self._pinned_paths = list(self._settings.get('pinned_recent_projects', []) or [])
            history = list(self._settings.get('recent_projects', []) or [])
            # Reserve pinned records without turning pin order into open history.
            retained = list(dict.fromkeys([*self._pinned_paths, *history]))[:constants.RECENT_PROJECTS_MAX]
            paths = [p for p in dict.fromkeys([*history, *self._pinned_paths]) if p in retained]
            self.set_recent_projects(paths)

    def set_recent_projects(self, paths) -> None:
        self._recent_paths = list(dict.fromkeys(str(p) for p in paths if p))[:constants.RECENT_PROJECTS_MAX]
        self._pinned_paths = [p for p in self._pinned_paths if p in self._recent_paths]
        self._availability = {p: e for p, e in self._availability.items() if p in self._recent_paths}
        self._render_recents()
        self._probe.check(self._recent_paths)

    def _project_display_name(self, path: str) -> str:
        names = self._settings.get('project_display_names', {}) if self._settings is not None else {}
        fallback = Path(path).name or path
        return str(names.get(path) or fallback) if isinstance(names, dict) else fallback

    def _render_recents(self) -> None:
        selected = self._selected_path
        query = self.search.text().strip().casefold()
        paths = [p for p in sorted(self._recent_paths, key=lambda p: p not in self._pinned_paths)
                 if (not self._pinned_only or p in self._pinned_paths)
                 and (not query or query in f'{self._project_display_name(p)}\n{p}'.casefold())]
        self._recent_list.blockSignals(True)
        self._recent_list.clear()
        self._rows.clear()
        for path in paths:
            item = QListWidgetItem(self._item_text(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setSizeHint(QSize(0, 76))
            item.setToolTip(path)
            self._recent_list.addItem(item)
            row = RecentProjectRow(path, self._project_display_name(path), path in self._pinned_paths)
            row.set_error(self._availability.get(path, ''))
            row.pin_requested.connect(self._toggle_pin)
            row.selected.connect(self._select_path)
            row.activated.connect(self._request_open)
            self._recent_list.setItemWidget(item, row)
            self._rows[path] = (item, row)
        self._selected_path = selected if selected in paths else paths[0] if paths else ''
        if self._selected_path:
            self._recent_list.setCurrentItem(self._rows[self._selected_path][0])
        self._recent_list.blockSignals(False)
        self._recent_list.setVisible(bool(paths))
        self._empty.setVisible(not paths)
        self.search.setVisible(bool(self._recent_paths))
        self.recent_btn.setText(f'最近打开  {len(self._recent_paths)}')
        self.pinned_btn.setText(f'已固定  {len(self._pinned_paths)}')
        self.recent_btn.setChecked(not self._pinned_only)
        self.pinned_btn.setChecked(self._pinned_only)
        self._title.setText('已固定' if self._pinned_only else '最近项目')
        self._recent_hint.setText('没有匹配的项目' if query else '还没有固定项目' if self._pinned_only else '还没有项目')
        self._empty_action.setText('清空搜索' if query else '查看最近项目' if self._pinned_only else '新建项目')
        self.set_busy(self._busy)
        self.apply_theme(self._dark)

    def _item_text(self, path: str) -> str:
        name = ('★ ' if path in self._pinned_paths else '') + self._project_display_name(path)
        error = self._availability.get(path, '')
        return f'{name}\n{path}' + (f' · 不可用：{error}' if error else '')

    def _selection_changed(self, current, previous) -> None:
        self._cancel_restore()
        self._selected_path = str(current.data(Qt.ItemDataRole.UserRole)) if current is not None else ''
        self._update_details()

    def _select_path(self, path: str) -> None:
        self._cancel_restore()
        if path in self._rows and not self._busy:
            self._recent_list.setCurrentItem(self._rows[path][0])
            self._recent_list.setFocus(Qt.FocusReason.MouseFocusReason)

    def _update_details(self) -> None:
        path = self._selected_path
        self._details.setVisible(bool(path))
        for key, (_item, row) in self._rows.items():
            row.set_selected(key == path)
        self._detail_name.setText(self._project_display_name(path) if path else '')
        self._detail_path.setText(path)
        error = self._availability.get(path)
        self._detail_state.setText(error if error is not None else '正在检查目录…')
        self._detail_state.setVisible(bool(path) and error != '')
        opening = self._busy and self._opening_path == path
        self.detail_open_btn.setText('正在打开…' if opening else '重试打开' if error else '打开项目')
        self.detail_open_btn.setEnabled(bool(path) and self._backend_ready and not self._busy)
        self._remove_btn.setVisible(bool(error))
        self._remove_btn.setEnabled(not self._busy)

    def _on_availability(self, path: str, error: str) -> None:
        if path not in self._recent_paths and path not in self._restore_paths:
            return
        self._availability[path] = error
        if path in self._rows:
            item, row = self._rows[path]
            item.setText(self._item_text(path))
            row.set_error(error)
        self._update_details()
        if path in self._restore_paths:
            self._restore_paths.remove(path)
            if not error:
                self._restore_paths.clear()
                self._request_open(path)
            elif not self._restore_paths:
                self.set_startup_error(f'无法恢复上次项目：{error}\n{path}')

    def restore_last_project(self, paths) -> None:
        if self._user_interacted or self._summary is not None:
            return
        self._restore_paths = list(dict.fromkeys(str(p) for p in paths if p))[:constants.RECENT_PROJECTS_MAX]
        self._probe.check([*self._restore_paths, *self._recent_paths])

    def _cancel_restore(self) -> None:
        self._user_interacted = True
        self._restore_paths.clear()

    def cancel_pending_restore(self) -> None:
        """A manual dialog/shortcut takes precedence over automatic restoration."""
        self._cancel_restore()

    def _search_changed(self, _text: str) -> None:
        self._cancel_restore()
        self._render_recents()

    def _set_filter(self, pinned: bool) -> None:
        self._cancel_restore()
        self._pinned_only = pinned
        self._render_recents()

    def _empty_action_clicked(self) -> None:
        if self.search.text().strip():
            self.search.clear()
        elif self._pinned_only:
            self._set_filter(False)
        else:
            self._request_dialog(True)

    def _request_dialog(self, new: bool) -> None:
        if self._busy or not self._backend_ready:
            return
        self._cancel_restore()
        self.set_startup_error('')
        (self.new_project_requested if new else self.open_project_requested).emit()

    def set_startup_error(self, message: str) -> None:
        self._startup_error.setText(str(message or ''))
        self._startup_error.setVisible(bool(message))

    def set_backend_ready(self, ready: bool) -> None:
        self._backend_ready = ready
        self.set_busy(self._busy)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._cancel_restore()
        if not busy:
            self._opening_path = ''
        for button in (self.open_btn, self.new_btn, self.restore_btn):
            button.setEnabled(self._backend_ready and not busy)
        self._empty_action.setEnabled(not busy and (bool(self._recent_paths) or self._backend_ready))
        for widget in (self.search, self._recent_list, self.recent_btn, self.pinned_btn):
            widget.setEnabled(not busy)
        self._update_details()

    def _open_recent_item(self, item) -> None:
        if item is not None:
            self._request_open(str(item.data(Qt.ItemDataRole.UserRole)))

    def _request_open(self, path: str) -> None:
        if not path or self._busy or not self._backend_ready:
            return
        self._cancel_restore()
        self._select_path(path)
        self._opening_path = path
        self.set_startup_error('')
        self.open_recent_project_requested.emit(path)
        if not self._busy:
            self._opening_path = ''

    def _save_recents(self) -> None:
        if self._settings is not None:
            self._settings.set('recent_projects', self._recent_paths)
            self._settings.set('pinned_recent_projects', self._pinned_paths)
            self._settings.save()
        self._render_recents()

    def _toggle_pin(self, path: str) -> None:
        if self._busy or path not in self._recent_paths:
            return
        self._cancel_restore()
        focus_pin = path in self._rows and self._rows[path][1].pin_button.hasFocus()
        if path in self._pinned_paths:
            self._pinned_paths.remove(path)
        else:
            self._pinned_paths.append(path)
        self._save_recents()
        if focus_pin and path in self._rows:
            self._rows[path][1].pin_button.setFocus()

    def _remove_recent(self, path: str) -> None:
        if self._busy:
            return
        self._cancel_restore()
        self._recent_paths = [p for p in self._recent_paths if p != path]
        self._pinned_paths = [p for p in self._pinned_paths if p != path]
        self._availability.pop(path, None)
        self._save_recents()

    def _recent_context_menu(self, pos):
        self._cancel_restore()
        menu = make_menu(self)
        item = self._recent_list.itemAt(pos)
        if item is not None:
            path = str(item.data(Qt.ItemDataRole.UserRole))
            add_action(menu, FIF.FOLDER, '打开项目', lambda: self._request_open(path),
                       enabled=self._backend_ready and not self._busy)
            add_action(menu, FIF.PIN, '取消固定' if path in self._pinned_paths else '固定项目',
                       lambda: self._toggle_pin(path), enabled=not self._busy)
            add_action(menu, FIF.COPY, '复制项目路径', lambda: QApplication.clipboard().setText(path))
            add_action(menu, FIF.REMOVE, '从最近列表移除', lambda: self._remove_recent(path), enabled=not self._busy)
            menu.addSeparator()
            add_action(menu, FIF.DELETE, '永久删除项目…',
                       lambda: self.project_delete_requested.emit(path),
                       enabled=self._backend_ready and not self._busy)
        menu.closedSignal.connect(self._recent_list.setFocus)
        return menu

    def project_menu(self, trigger, *, can_close: bool, close_requested, enabled: bool = True):
        self._cancel_restore()
        self.refresh_recents()
        menu = make_menu(trigger)
        enabled = enabled and self._backend_ready and not self._busy
        add_action(menu, FIF.ADD, '新建项目', lambda: self._request_dialog(True), enabled=enabled)
        add_action(menu, FIF.FOLDER, '打开项目…', lambda: self._request_dialog(False), enabled=enabled)
        if self._recent_paths:
            menu.addSeparator()
        for path in sorted(self._recent_paths, key=lambda p: p not in self._pinned_paths):
            label = ('★ ' if path in self._pinned_paths else '') + self._project_display_name(path)
            if self._availability.get(path):
                label += ' · 不可用'
            action = add_action(menu, FIF.FOLDER, label,
                                lambda _checked=False, p=path: self._request_open(p), enabled=enabled)
            action.setToolTip(path)
        if can_close:
            menu.addSeparator()
            add_action(menu, FIF.CLOSE, '关闭当前项目', close_requested, enabled=enabled)
            root = str(getattr(self._summary, 'root_path', '') or '')
            add_action(menu, FIF.DELETE, '永久删除当前项目…',
                       lambda: self.project_delete_requested.emit(root), enabled=enabled and bool(root))
        menu.closedSignal.connect(trigger.setFocus)
        return menu

    def shutdown(self) -> None:
        self._cancel_restore()
        self._probe.cancel()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    def apply_theme(self, dark: bool) -> None:
        self._dark = dark
        self._content.setStyleSheet(
            f'#startupContent {{ background: {color("bg_surface", dark)}; border-radius: {radius("xl")}px; }}'
            f'#startupSidebar, #startupDetails {{ background: {color("bg_subtle", dark)};'
            f' border-radius: {radius("lg")}px; }}'
            f'#startupTitle {{ color: {color("text_primary", dark)}; font-size: 28px; font-weight: 600; }}'
            f'#startupProjectName {{ color: {color("text_primary", dark)}; font-size: 19px; font-weight: 600; }}'
            f'#startupProjectPath {{ color: {color("text_muted", dark)}; font-size: 12px; }}'
            f'#startupEmpty {{ color: {color("text_secondary", dark)}; font-size: 18px; }}'
            f'#recentProjects {{ background: transparent; border: none; outline: none; }}'
            f'#recentProjects::item {{ padding: 0; border: none; background: transparent; }}'
            f'#recentProjects:focus {{ border: 1px solid {color("border_focus", dark)};'
            f' border-radius: {radius("lg")}px; }}')
        for _item, row in self._rows.values():
            row.apply_theme(dark)
        self._detail_state.apply_theme(dark)
        self._startup_error.apply_theme(dark)


__all__ = ['HomePage']

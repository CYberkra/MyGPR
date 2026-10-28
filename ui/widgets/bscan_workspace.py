"""Explicit B-scan tabs and splitter groups; source catalog never opens views.

Each group owns one reusable BScanView. Tabs and duplicated groups refer to the
same PreviewBundle; layout/closed-tab history contains only IDs and view state.
Business actions are delegated through source_context_requested, after choosing
the explicit ``来源操作…`` command in the standard view menu.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from uuid import uuid4

from PyQt6.QtCore import QEvent, QMimeData, QPoint, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QDrag, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QSplitter, QTabBar, QVBoxLayout, QWidget,
)
from qfluentwidgets import FluentIcon as FIF, RoundMenu, Theme, ToolButton, isDarkTheme

from ui.design_tokens import color, rgba
from ui.widgets.fluent_menu_keyboard import FluentMenuKeyboard
from ui.widgets.bscan_view import BScanView
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.bscan_link import compatible
from ui.widgets.bscan_workspace_link import BScanWorkspaceLink


_MIME = 'application/x-mygpr-bscan-tab'
_EDGES = ('left', 'right', 'top', 'bottom')
_EDGE_NAMES = {'left': '左侧', 'right': '右侧', 'top': '上方', 'bottom': '下方'}


def _view_state(view: BScanView) -> dict:
    """Small display state only: neither arrays nor bundles enter persistence."""
    state = {
        'colormap': view._cmap_name,
        'levels': list(view.display_levels()), 'aspect': view.aspect_mode(),
        'x_axis': view.x_axis_mode(), 'y_axis': view.y_axis_mode(),
        'colorbar': view.colorbar_visible(), 'gain': view.gain_mode(),
        'gain_alpha': view._gain_alpha, 'gain_db': view._gain_db,
        'gain_power': view._gain_power, 'display_mode': view.display_mode.value,
        'crosshair': view._crosshair_on,
        'local_view': view._view_selection is not None or view._view_time_window is not None,
        'levels_follow_view': view._selection_follows_levels,
    }
    if view._matrix is not None:
        state['range'] = view._plot.vb.viewRange()
    return state


def _apply_view_state(view: BScanView, state: dict, *, viewport=False) -> None:
    if not state:
        return
    if viewport:
        ranges = state.get('range')
        if isinstance(ranges, (list, tuple)) and len(ranges) == 2:
            try:
                if state.get('local_view'):
                    (left, right), (top, bottom) = ranges
                    view.set_view_selection(QRectF(left, top, right-left, bottom-top))
                else:
                    view._plot.vb.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
            except (TypeError, ValueError, IndexError):
                pass
        return
    try:
        view._selection_follows_levels = bool(state.get('levels_follow_view', True))
        view.set_colormap(str(state.get('colormap', view._cmap_name)))
        view.set_display_levels(*state.get('levels', view.display_levels()))
        view.set_aspect_mode(str(state.get('aspect', view.aspect_mode())))
        view.set_axis_modes(str(state.get('x_axis', view.x_axis_mode())),
                            str(state.get('y_axis', view.y_axis_mode())))
        view.set_colorbar_visible(bool(state.get('colorbar', view.colorbar_visible())))
        view.set_gain(str(state.get('gain', view.gain_mode())),
                      alpha=float(state.get('gain_alpha', view._gain_alpha)),
                      db=float(state.get('gain_db', view._gain_db)),
                      power=float(state.get('gain_power', view._gain_power)))
        view.set_display_mode(state.get('display_mode', view.display_mode.value))
        view._toggle_crosshair(bool(state.get('crosshair', True)))
    except (TypeError, ValueError, KeyError, AttributeError):
        # A malformed session preference must not prevent the data from opening.
        pass


class _SourceTabBar(QTabBar):
    """Real Qt drag source; the workspace handles tab/edge drop destinations."""

    def __init__(self, group):
        super().__init__(group)
        self.group = group
        self._press = None
        self._drag_key = ''
        self.setObjectName('bscanWorkspaceTabs')
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.setUsesScrollButtons(True)
        self.setTabsClosable(True)
        self.setFixedHeight(32)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName('B-scan 数据标签')

    def sync_close_button(self, index, key, title):
        side = QTabBar.ButtonPosition.RightSide
        button = self.tabButton(index, side)
        if not isinstance(button, ToolButton) or button.property('source_key') != key:
            button = ToolButton(FIF.CLOSE, self)
            button.setProperty('source_key', key)
            button.setFixedSize(20, 20)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _checked=False, key=key: self._close_key(key))
            self.setTabButton(index, side, button)
        button.setToolTip(f'关闭 {title}')
        button.setAccessibleName(f'关闭 {title}')
        button.setVisible(key not in self.group.pinned)

    def _close_key(self, key):
        index = self.group.index_of(key)
        if index >= 0:
            self.tabCloseRequested.emit(index)

    def mousePressEvent(self, event):
        index = self.tabAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()  # Opening a context menu never selects its target.
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            if index >= 0:
                self.group.workspace.close_tab(self.tabData(index), self.group.group_id)
            event.accept()
            return
        super().mousePressEvent(event)
        self._press = event.position().toPoint()
        self._drag_key = str(self.tabData(index) or '') if index >= 0 else ''

    def mouseMoveEvent(self, event):
        if (self._press is not None and self._drag_key
                and event.buttons() & Qt.MouseButton.LeftButton
                and (event.position().toPoint() - self._press).manhattanLength()
                >= QApplication.startDragDistance()):
            key, self._drag_key = self._drag_key, ''
            self._press = None
            drag = QDrag(self)
            drag.setMimeData(self.group.workspace.drag_mime_data(key, self.group.group_id))
            index = self.group.index_of(key)
            if index >= 0:
                drag.setPixmap(self.grab(self.tabRect(index)))
            drag.exec(Qt.DropAction.MoveAction)
            self.group.workspace.cancel_drag()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press = None
        self._drag_key = ''
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        index = (self.currentIndex() if event.reason() == event.Reason.Keyboard
                 else self.tabAt(event.pos()))
        key = str(self.tabData(index) or '') if index >= 0 else ''
        self.group.workspace.show_context_menu(self.group.group_id, key, event.globalPos())
        event.accept()


class _ViewGroup(QWidget):
    def __init__(self, workspace, group_id):
        super().__init__(workspace)
        self.workspace, self.group_id = workspace, group_id
        self.setObjectName('bscanWorkspaceGroup')
        self.setMinimumSize(140, 150)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.pinned: set[str] = set()
        self.states: dict[str, dict] = {}
        self.rendered_key = ''
        self.rendered_bundle = None
        self.tabs = _SourceTabBar(self)
        self.more = ToolButton(FIF.MORE, self)
        self.more.setFixedSize(28, 28)
        self.more.setToolTip('视图组操作')
        self.more.setAccessibleName('视图组操作')
        header = QHBoxLayout()
        header.setContentsMargins(2, 0, 2, 0)
        header.setSpacing(2)
        header.addWidget(self.tabs, 1)
        header.addWidget(self.more)
        self.view = BScanView(self)
        self.view.set_title_visible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self.view, 1)
        self.hint = QLabel(self)
        self.hint.setObjectName('bscanDropHint')
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hint.hide()
        self.setFocusProxy(self.tabs)
        self.tabs.currentChanged.connect(lambda _index: workspace._current_changed(self))
        self.tabs.tabBarClicked.connect(lambda index: workspace._tab_clicked(self, index))
        self.tabs.tabCloseRequested.connect(
            lambda index: workspace.close_tab(self.tabs.tabData(index), self.group_id))
        self.more.clicked.connect(lambda: workspace.show_context_menu(
            self.group_id, self.current_key(), self.more.mapToGlobal(QPoint(0, self.more.height()))))
        for child in [self, *self.findChildren(QWidget)]:
            child.setAcceptDrops(True)
            child.installEventFilter(workspace)

    def keys(self):
        return [str(self.tabs.tabData(i)) for i in range(self.tabs.count())]

    def index_of(self, key):
        keys = self.keys()
        return keys.index(key) if key in keys else -1

    def current_key(self):
        return str(self.tabs.tabData(self.tabs.currentIndex()) or '')

    def capture(self):
        if self.rendered_key:
            self.states[self.rendered_key] = _view_state(self.view)


class BScanWorkspace(BScanWorkspaceLink, QWidget):
    active_changed = pyqtSignal(str)
    source_activated_by_user = pyqtSignal(str)
    source_closed = pyqtSignal(str)
    preview_requested = pyqtSignal(str)
    source_context_requested = pyqtSignal(str, object)
    source_input_requested = pyqtSignal(str)
    view_created = pyqtSignal(object)
    layout_changed = pyqtSignal()
    selection_step_requested = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('bscanWorkspace')
        self._token = uuid4().hex
        self._sources: dict[str, dict] = {}
        self._groups: dict[str, _ViewGroup] = {}
        self._counter = 0
        self._mutating = 0
        self._pending: set[str] = set()
        self._closed: list[dict] = []
        self._active_group = ''
        self._last_active = ('', '')
        self._dark = isDarkTheme()
        self._menu = None
        self._link_enabled = False
        self._link_busy = False
        self.selection_step_allowed = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._root = self._new_group()
        self._layout.addWidget(self._root)
        self._active_group = self._root.group_id
        self.setFocusProxy(self._root.tabs)
        self._shortcuts = []
        for key, slot in (
            ('Ctrl+W', self._close_active), ('Ctrl+Shift+T', self.reopen_closed),
            ('Ctrl+Tab', lambda: self._cycle_tab(1)),
            ('Ctrl+Shift+Tab', lambda: self._cycle_tab(-1)),
            ('Ctrl+Alt+Right', lambda: self.split_source(self.active_key(), 'right')),
            ('Ctrl+Alt+Down', lambda: self.split_source(self.active_key(), 'bottom')),
            ('Shift+F10', self._keyboard_menu), ('Menu', self._keyboard_menu),
            ('Escape', self.cancel_drag),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)
        self._sync()

    @contextmanager
    def _change(self):
        self._mutating += 1
        try:
            yield
        finally:
            self._mutating -= 1
            if not self._mutating:
                self._sync()

    def _new_group(self, group_id=None):
        self._counter += 1
        group_id = group_id or f'group-{self._counter}'
        while group_id in self._groups:
            self._counter += 1
            group_id = f'group-{self._counter}'
        group = _ViewGroup(self, group_id)
        self._groups[group_id] = group
        group.view._plot.vb.sigRangeChanged.connect(lambda *_args: self._link_range(group))
        group.view._glw.scene().sigMouseMoved.connect(lambda pos: self._link_cursor(group, pos))
        self.view_created.emit(group.view)
        return group

    def set_sources(self, sources: list[dict]) -> None:
        """Refresh catalog by stable key; never activate or implicitly open tabs."""
        with self._change():
            self._sources = {str(source['key']): dict(source) for source in sources
                             if source.get('key')}
            if self._menu is not None and self._menu.property('source_key') not in self._sources:
                self._close_menu()
            self._pending.intersection_update(self._sources)
            for key, source in self._sources.items():
                if source.get('bundle') is not None or source.get('error'):
                    self._pending.discard(key)
            for group in list(self._groups.values()):
                for key in list(group.keys()):
                    if key not in self._sources:
                        self.close_tab(key, group.group_id, force=True, remember=False)
            self._closed = [record for record in self._closed if record['key'] in self._sources]

    def group_ids(self) -> list[str]:
        def walk(node):
            if isinstance(node, _ViewGroup):
                return [node.group_id]
            return [gid for i in range(node.count()) for gid in walk(node.widget(i))]
        return walk(self._root)

    def _group(self, group_id=None):
        return self._groups.get(group_id or self._active_group)

    def group_widget(self, group_id=None) -> QWidget:
        return self._group(group_id)

    def tab_bar(self, group_id=None) -> QTabBar:
        return self._group(group_id).tabs

    def tab_keys(self, group_id=None) -> list[str]:
        group = self._group(group_id)
        return group.keys() if group else []

    def active_key(self) -> str:
        group = self._group()
        return group.current_key() if group else ''

    def views(self) -> list[BScanView]:
        return [self._groups[gid].view for gid in self.group_ids() if gid in self._groups]

    def all_views(self) -> list[BScanView]:
        return self.views()

    def primary_view(self) -> BScanView:
        return self._group().view

    def open_source(self, key: str, group_id=None) -> bool:
        if key not in self._sources:
            return False
        group = self._group(group_id)
        if group is None:
            return False
        with self._change():
            group.capture()
            index = group.index_of(key)
            if index < 0:
                index = group.tabs.addTab(str(self._sources[key].get('title') or key))
                group.tabs.setTabData(index, key)
            group.tabs.setCurrentIndex(index)
            self._active_group = group.group_id
        return True

    def _current_changed(self, group):
        if not self._mutating:
            group.capture()
            self._active_group = group.group_id
            self._sync()

    def _activate_group(self, group):
        if group.group_id in self._groups and not self._mutating:
            self._active_group = group.group_id
            self._sync()

    def _tab_clicked(self, group, index):
        if index >= 0:
            group.tabs.setCurrentIndex(index)
            self._activate_group(group)
            self.source_activated_by_user.emit(group.current_key())

    def _sync(self):
        if self._active_group not in self._groups:
            self._active_group = self.group_ids()[0]
        for group in self._groups.values():
            for index, key in enumerate(group.keys()):
                source = self._sources.get(key, {})
                title = str(source.get('title') or key)
                group.tabs.setTabText(index, title)
                group.tabs.setTabToolTip(index, str(source.get('tooltip') or f'{title}\n{key}'))
                group.tabs.setTabIcon(index, FIF.PIN.icon() if key in group.pinned else FIF.PHOTO.icon())
                group.tabs.sync_close_button(index, key, title)
            self._render(group)
        self.apply_theme(self._dark, refresh_views=False)
        active = (self._active_group, self.active_key())
        if active != self._last_active:
            self._last_active = active
            self.active_changed.emit(active[1])
        self.layout_changed.emit()

    def _render(self, group):
        key = group.current_key()
        source = self._sources.get(key, {})
        bundle = source.get('bundle')
        changed = key != group.rendered_key
        if changed or bundle is not group.rendered_bundle or bundle is None:
            if changed:
                group.view.clear()
                _apply_view_state(group.view, group.states.get(key, {}))
            if bundle is not None:
                group.view.set_bundle(bundle)
                if changed:
                    _apply_view_state(group.view, group.states.get(key, {}), viewport=True)
            else:
                group.view.clear()
                empty = group.view._empty_overlay
                error = str(source.get('error') or '')
                empty._title_label.setText('预览加载失败' if error else '预览尚未加载' if key else '空工作区')
                empty._hint_label.setText(error or ('正在请求此数据的 B-scan 预览' if key else
                                                  '从项目树或历史记录打开数据'))
            group.rendered_key, group.rendered_bundle = key, bundle
        if key and bundle is None and not source.get('error') and key not in self._pending:
            self._pending.add(key)
            self.preview_requested.emit(key)

    def is_pinned(self, key, group_id=None) -> bool:
        group = self._group(group_id)
        return bool(group and key in group.pinned)

    def set_pinned(self, key, pinned=True, group_id=None) -> bool:
        group = self._group(group_id)
        if group is None or key not in group.keys():
            return False
        if pinned:
            group.pinned.add(key)
        else:
            group.pinned.discard(key)
        self._sync()
        return True

    def close_tab(self, key, group_id=None, *, force=False, remember=True) -> bool:
        group = self._group(group_id)
        if group is None or key not in group.keys() or (key in group.pinned and not force):
            return False
        with self._change():
            group.capture()
            index = group.index_of(key)
            if remember and key in self._sources:
                self._closed.append({'key': key, 'group_id': group.group_id, 'index': index,
                                     'pinned': key in group.pinned,
                                     'state': deepcopy(group.states.get(key, {}))})
                del self._closed[:-32]
            group.tabs.removeTab(index)
            group.states.pop(key, None)
            group.pinned.discard(key)
            self._remove_empty(group)
            if not any(key in other.keys() for other in self._groups.values()):
                self._pending.discard(key)
                self.source_closed.emit(key)
        return True

    def close_source(self, key: str) -> None:
        """Authoritative removal closes every view, including user-pinned tabs."""
        with self._change():
            for gid in list(self.group_ids()):
                self.close_tab(key, gid, force=True)

    def close_tabs(self, scope='unpinned', key=None, group_id=None) -> None:
        group = self._group(group_id)
        if group is None:
            return
        key = key or group.current_key()
        keys = group.keys()
        if scope == 'right':
            keys = keys[keys.index(key) + 1:] if key in keys else []
        elif scope == 'others':
            keys = [candidate for candidate in keys if candidate != key]
        elif scope not in ('all', 'unpinned'):
            raise ValueError('Unknown tab close scope')
        with self._change():
            for candidate in keys:
                self.close_tab(candidate, group.group_id)

    def reopen_closed(self) -> str:
        while self._closed:
            record = self._closed.pop()
            key = record['key']
            if key not in self._sources:
                continue
            group = self._group(record['group_id']) or self._group()
            with self._change():
                group.states[key] = record['state']
                self.open_source(key, group.group_id)
                self.move_source(key, group.group_id, group.group_id, record['index'])
                if record['pinned']:
                    group.pinned.add(key)
            return key
        return ''

    def _split_group(self, target, new, edge):
        orientation = Qt.Orientation.Horizontal if edge in ('left', 'right') else Qt.Orientation.Vertical
        before = edge in ('left', 'top')
        parent = target.parentWidget()
        if isinstance(parent, QSplitter) and parent.orientation() == orientation:
            index = parent.indexOf(target)
            parent.insertWidget(index if before else index + 1, new)
            parent.setSizes([1] * parent.count())
        else:
            splitter = QSplitter(orientation, self)
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(5)
            if isinstance(parent, QSplitter):
                parent.replaceWidget(parent.indexOf(target), splitter)
            else:
                self._layout.replaceWidget(target, splitter)
                self._root = splitter
            splitter.addWidget(new if before else target)
            splitter.addWidget(target if before else new)
            splitter.setSizes([1, 1])
        new.show()

    def split_source(self, key, edge='right', group_id=None, *, copy=False) -> str:
        """Move a tab to an edge group, or share the bundle with ``copy=True``."""
        if edge not in _EDGES:
            raise ValueError('edge must be left, right, top or bottom')
        origin = self._group(group_id)
        if origin is None or key not in origin.keys():
            return ''
        if len(origin.keys()) == 1 and not copy:
            return origin.group_id  # Moving the only tab cannot leave an empty split.
        with self._change():
            origin.capture()
            new = self._new_group()
            self._split_group(origin, new, edge)
            new.states[key] = deepcopy(origin.states.get(key, {}))
            if key in origin.pinned:
                new.pinned.add(key)
            self.open_source(key, new.group_id)
            if not copy:
                self._detach_tab(origin, key)
            self._active_group = new.group_id
        return new.group_id

    def _detach_tab(self, group, key):
        group.tabs.removeTab(group.index_of(key))
        group.states.pop(key, None)
        group.pinned.discard(key)
        self._remove_empty(group)

    def move_source(self, key, target_group_id, source_group_id=None, index=None) -> bool:
        origin, target = self._group(source_group_id), self._group(target_group_id)
        if origin is None or target is None or key not in origin.keys():
            return False
        with self._change():
            origin.capture()
            old_index = origin.index_of(key)
            if origin is target:
                if index is not None:
                    target.tabs.moveTab(old_index, max(0, min(int(index), target.tabs.count() - 1)))
                target.tabs.setCurrentIndex(target.index_of(key))
            else:
                target.capture()
                if key not in target.keys():
                    target.states[key] = deepcopy(origin.states.get(key, {}))
                    position = target.tabs.count() if index is None else max(0, min(int(index), target.tabs.count()))
                    tab_index = target.tabs.insertTab(position, str(self._sources[key].get('title') or key))
                    target.tabs.setTabData(tab_index, key)
                if key in origin.pinned:
                    target.pinned.add(key)
                target.tabs.setCurrentIndex(target.index_of(key))
                self._detach_tab(origin, key)
            self._active_group = target.group_id
        return True

    def _remove_empty(self, group):
        if group.tabs.count() or len(self._groups) == 1:
            return
        parent = group.parentWidget()
        group.view.clear()
        self._groups.pop(group.group_id)
        group.setParent(None)
        group.deleteLater()
        while isinstance(parent, QSplitter) and parent.count() == 1:
            survivor = parent.widget(0)
            grandparent = parent.parentWidget()
            survivor.setParent(None)
            if isinstance(grandparent, QSplitter):
                grandparent.replaceWidget(grandparent.indexOf(parent), survivor)
            else:
                self._layout.replaceWidget(parent, survivor)
                self._root = survivor
            parent.setParent(None)
            parent.deleteLater()
            survivor.show()
            parent = grandparent

    def reset_layout(self) -> None:
        """Merge open tabs, preserving pins/history; never open a catalog item."""
        active = self.active_key()
        active_state = _view_state(self.primary_view())
        with self._change():
            first = self.group_ids()[0]
            for gid in list(self.group_ids())[1:]:
                for key in list(self.tab_keys(gid)):
                    self.move_source(key, first, gid)
            self._active_group = first
            if active:
                self._groups[first].states[active] = active_state
                self._groups[first].rendered_key = ''
                self.open_source(active, first)

    def save_layout(self) -> dict:
        for group in self._groups.values():
            group.capture()

        def encode(node):
            if isinstance(node, _ViewGroup):
                return {'type': 'group', 'id': node.group_id, 'active': node.current_key(),
                        'tabs': [{'key': key, 'pinned': key in node.pinned,
                                  'state': deepcopy(node.states.get(key, {}))} for key in node.keys()]}
            return {'type': 'split', 'orientation': 'horizontal' if node.orientation() == Qt.Orientation.Horizontal else 'vertical',
                    'sizes': node.sizes(), 'children': [encode(node.widget(i)) for i in range(node.count())]}

        return {'version': 1, 'active_group': self._active_group, 'root': encode(self._root),
                'closed': deepcopy(self._closed)}

    def restore_layout(self, state: dict) -> bool:
        """Restore explicitly saved IDs. Missing resources stay closed."""
        seen = set()

        def validate(node, depth=0):
            if not isinstance(node, dict) or depth > 12:
                return False
            if node.get('type') == 'group':
                gid = node.get('id')
                if not isinstance(gid, str) or gid in seen or not isinstance(node.get('tabs'), list):
                    return False
                seen.add(gid)
                return len(seen) <= 64 and all(isinstance(t, dict) and isinstance(t.get('key'), str)
                                               and isinstance(t.get('state', {}), dict) for t in node['tabs'])
            children, sizes = node.get('children'), node.get('sizes')
            return (node.get('type') == 'split' and node.get('orientation') in ('horizontal', 'vertical')
                    and isinstance(children, list) and 2 <= len(children) <= 64
                    and isinstance(sizes, list) and len(sizes) == len(children)
                    and all(isinstance(v, int) and v >= 0 for v in sizes)
                    and all(validate(child, depth + 1) for child in children))

        if not isinstance(state, dict) or state.get('version') != 1 or not validate(state.get('root')):
            return False
        self._close_menu()
        with self._change():
            old_root = self._root
            self._layout.removeWidget(old_root)
            self._groups = {}

            def build(node):
                if node['type'] == 'group':
                    group = self._new_group(node['id'])
                    for tab in node['tabs']:
                        key = tab['key']
                        if key not in self._sources or key in group.keys():
                            continue
                        index = group.tabs.addTab(str(self._sources[key].get('title') or key))
                        group.tabs.setTabData(index, key)
                        group.states[key] = deepcopy(tab.get('state', {}))
                        if tab.get('pinned'):
                            group.pinned.add(key)
                    group.tabs.setCurrentIndex(max(0, group.index_of(node.get('active'))))
                    return group
                orientation = Qt.Orientation.Horizontal if node['orientation'] == 'horizontal' else Qt.Orientation.Vertical
                splitter = QSplitter(orientation, self)
                splitter.setChildrenCollapsible(False)
                splitter.setHandleWidth(5)
                for child in node['children']:
                    splitter.addWidget(build(child))
                splitter.setSizes(node['sizes'])
                return splitter

            self._root = build(state['root'])
            self._layout.addWidget(self._root)
            old_root.hide()
            old_root.deleteLater()
            self._active_group = str(state.get('active_group', ''))
            closed = state.get('closed', [])
            self._closed = [deepcopy(record) for record in closed if isinstance(record, dict)
                            and record.get('key') in self._sources
                            and isinstance(record.get('group_id'), str)
                            and isinstance(record.get('index'), int)
                            and isinstance(record.get('state'), dict)
                            and isinstance(record.get('pinned'), bool)] if isinstance(closed, list) else []
            for group in list(self._groups.values()):
                self._remove_empty(group)
            self._root.show()
        return True

    def _close_active(self):
        group = self._command_group()
        self.close_tab(group.current_key(), group.group_id)

    def _command_group(self):
        widget = QApplication.focusWidget()
        while isinstance(widget, QWidget) and not isinstance(widget, _ViewGroup):
            widget = widget.parentWidget()
        return widget if isinstance(widget, _ViewGroup) and widget.group_id in self._groups else self._group()

    def _cycle_tab(self, direction):
        group = self._command_group()
        if group.tabs.count():
            group.tabs.setCurrentIndex((group.tabs.currentIndex() + direction) % group.tabs.count())

    def _keyboard_menu(self):
        group = self._command_group()
        point = group.tabs.tabRect(group.tabs.currentIndex()).bottomLeft()
        self.show_context_menu(group.group_id, group.current_key(), group.tabs.mapToGlobal(point))

    def create_context_menu(self, group_id=None, key=None, global_pos=None) -> RoundMenu:
        """Frozen target identity; creating/opening menus has no selection effects."""
        group = self._group(group_id)
        gid = group.group_id
        key = group.current_key() if key is None else key
        valid = key in group.keys()
        position = QPoint(global_pos) if global_pos is not None else group.tabs.mapToGlobal(QPoint(8, 28))
        menu = make_menu(self)
        menu.setProperty('source_key', key)

        def add(parent, command, text, slot, *, enabled=True, shortcut=''):
            entry = add_action(parent, None, text, slot, enabled=enabled, shortcut=shortcut)
            entry.setData(command)
            return entry

        if valid:
            can_convert = bool(self.selection_step_allowed and self.selection_step_allowed(gid, key))
            source_snapshot = self._sources[key]
            selection_snapshot = (QRectF(group.view._view_selection)
                                  if group.view._view_selection is not None else None)

            def emit_selection(method):
                current = self._sources.get(key, {})
                live_group = self._groups.get(gid)
                if (live_group is not group or current.get('bundle') is not source_snapshot.get('bundle')
                        or any(current.get(field) != source_snapshot.get(field)
                               for field in ('project_id', 'line_id', 'revision'))
                        or group.view._view_selection != selection_snapshot):
                    return
                if self.selection_step_allowed and self.selection_step_allowed(gid, key):
                    self.selection_step_requested.emit(gid, key, method)

            add(menu, 'selection-crop', '框选范围加入测道裁剪' if can_convert else '框选转处理不可用：请选择当前输入并清空已有步骤',
                lambda: emit_selection('trace_cut'), enabled=can_convert)
            add(menu, 'selection-qc', '框选范围加入坏道标记',
                lambda: emit_selection('trace_qc'), enabled=can_convert)
            can_time = (can_convert and not group.view._showing_elevation
                        and group.view._sample_axis is not None
                        and 'ns' in group.view._sample_axis_label.lower())
            add(menu, 'selection-crop-time', '框选范围加入道与时间裁剪' if can_time else '道与时间裁剪不可用：需要真实时间轴',
                lambda: emit_selection('trace_cut_time'), enabled=can_time)
            can_link = (group.view._matrix is not None and any(
                compatible(group.view, target.view) for target in self._compatible_groups(group)))
            link = add(menu, 'link-views', '联动同测线视图' if can_link else '联动不可用：需要兼容的来源和物理轴',
                       self._set_link_enabled,
                       enabled=can_link or self._link_enabled)
            link.setCheckable(True)
            link.setChecked(self._link_enabled)
            reference = self._sources[key].get('comparison', False)
            add(menu, 'activate', '设为处理输入' if reference else '从此结果继续处理',
                lambda: self.source_input_requested.emit(key) if reference else self.open_source(key, gid))
            pinned = self.is_pinned(key, gid)
            pin = add(menu, 'pin', '取消固定标签' if pinned else '固定标签',
                      lambda: self.set_pinned(key, not pinned, gid))
            pin.setCheckable(True)
            pin.setChecked(pinned)
            split = RoundMenu('排列与拆分', menu)
            for edge, name in _EDGE_NAMES.items():
                add(split, f'split-{edge}', f'拆分到{name}' if len(group.keys()) > 1 else f'拆分到{name}（当前组只有一个标签）',
                    lambda _checked=False, edge=edge: self.split_source(key, edge, gid), enabled=len(group.keys()) > 1)
            split.addSeparator()
            add(split, 'copy-right', '复制到右侧新组', lambda: self.split_source(key, 'right', gid, copy=True))
            move = RoundMenu('移到其他视图组', split)
            targets = [target for target in self.group_ids() if target != gid]
            if not targets:
                add(move, 'no-target', '没有其他视图组', lambda: None, enabled=False)
            for target in targets:
                add(move, f'move-{target}', f'视图组 {self.group_ids().index(target) + 1}',
                    lambda _checked=False, target=target: self.move_source(key, target, gid))
            split.addMenu(move)
            menu.addMenu(split)
            add(menu, 'close', '关闭标签（先取消固定）' if pinned else '关闭标签',
                lambda: self.close_tab(key, gid), enabled=not pinned, shortcut='Ctrl+W')
            ranges = RoundMenu('关闭范围', menu)
            keys = group.keys()
            for scope, text, candidates in (
                ('others', '关闭其他标签', [k for k in keys if k != key]),
                ('right', '关闭右侧标签', keys[keys.index(key) + 1:]),
                ('unpinned', '关闭组内未固定标签', keys),
            ):
                add(ranges, f'close-{scope}', text,
                    lambda _checked=False, scope=scope: self.close_tabs(scope, key, gid),
                    enabled=any(k not in group.pinned for k in candidates))
            menu.addMenu(ranges)
            menu.addSeparator()
            add(menu, 'source-context', '来源操作…',
                lambda: self.source_context_requested.emit(key, position),
                enabled=bool(self.receivers(self.source_context_requested)))
        add(menu, 'reopen', '重新打开已关闭标签', self.reopen_closed,
            enabled=bool(self._closed), shortcut='Ctrl+Shift+T')
        add(menu, 'reset-layout', '恢复为单组标签', self.reset_layout,
            enabled=len(self._groups) > 1)
        menu._workspace_keyboard = FluentMenuKeyboard(menu)
        return menu

    def show_context_menu(self, group_id=None, key=None, global_pos=None):
        self._close_menu()
        group = self._group(group_id)
        if group is None:
            return
        position = global_pos or group.tabs.mapToGlobal(QPoint(8, 28))
        self._menu = self.create_context_menu(group.group_id, key, position)
        self._menu.closedSignal.connect(lambda: setattr(self, '_menu', None))
        self._menu.exec(position)
        self._menu._workspace_keyboard.focus_menu(self._menu)

    def _close_menu(self):
        if self._menu is not None:
            self._menu.close()
            self._menu = None

    def drag_mime_data(self, key, group_id=None) -> QMimeData:
        mime = QMimeData()
        mime.setData(_MIME, json.dumps({'workspace': self._token,
                                     'group': group_id or self._active_group, 'key': key}).encode())
        return mime

    def _drag_source(self, mime):
        if not mime.hasFormat(_MIME):
            return None
        try:
            value = json.loads(bytes(mime.data(_MIME)))
            group = self._group(value.get('group'))
            if value.get('workspace') == self._token and group and value.get('key') in group.keys():
                return value
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _drop_target(self, group, point):
        bar_point = group.tabs.mapFrom(group, point)
        if group.tabs.rect().contains(bar_point):
            index = group.tabs.tabAt(bar_point)
            if index < 0:
                index = group.tabs.count()
            elif bar_point.x() > group.tabs.tabRect(index).center().x():
                index += 1
            return 'tabs', index
        width, height = group.width(), group.height()
        threshold_x, threshold_y = min(64, width // 5), min(64, height // 5)
        if point.x() < threshold_x:
            return 'left', None
        if point.x() > width - threshold_x:
            return 'right', None
        if point.y() < threshold_y:
            return 'top', None
        if point.y() > height - threshold_y:
            return 'bottom', None
        return 'center', None

    def _show_hint(self, group, edge, index):
        for other in self._groups.values():
            other.hint.hide()
        rect = group.rect().adjusted(3, 3, -3, -3)
        if edge == 'tabs':
            x = (group.tabs.tabRect(index).left() if index < group.tabs.count()
                 else group.tabs.tabRect(group.tabs.count() - 1).right() + 1 if group.tabs.count() else 2)
            point = group.tabs.mapTo(group, QPoint(x, 0))
            rect = QRect(point.x(), point.y(), 4, group.tabs.height())
        elif edge in ('left', 'right'):
            rect.setWidth(rect.width() // 2)
            if edge == 'right':
                rect.moveRight(group.width() - 3)
        elif edge in ('top', 'bottom'):
            rect.setHeight(rect.height() // 2)
            if edge == 'bottom':
                rect.moveBottom(group.height() - 3)
        group.hint.setText('' if edge == 'tabs' else f'拆分到{_EDGE_NAMES[edge]}' if edge in _EDGES else '移入此视图组')
        group.hint.setGeometry(rect)
        group.hint.show()
        group.hint.raise_()

    def drop_hint_visible(self) -> bool:
        return any(not group.hint.isHidden() for group in self._groups.values())

    def cancel_drag(self):
        for group in self._groups.values():
            group.hint.hide()
        QDrag.cancel()

    def eventFilter(self, watched, event):
        event_type = event.type()
        if event_type == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.cancel_drag()
        group = watched
        while isinstance(group, QWidget) and not isinstance(group, _ViewGroup):
            group = group.parentWidget()
        if not isinstance(group, _ViewGroup) or group.group_id not in self._groups:
            return super().eventFilter(watched, event)
        if (event_type == QEvent.Type.FocusIn and watched is group.tabs
                and event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason,
                                       Qt.FocusReason.ShortcutFocusReason)):
            self._activate_group(group)
        if (event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton
                and (watched is group.view or group.view.isAncestorOf(watched))):
            self._activate_group(group)
        if event_type in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            source = self._drag_source(event.mimeData())
            if source is None:
                return super().eventFilter(watched, event)
            point = watched.mapTo(group, event.position().toPoint())
            edge, index = self._drop_target(group, point)
            origin = self._groups[source['group']]
            if edge in _EDGES and origin is group and len(origin.keys()) == 1:
                self.cancel_drag()
                event.ignore()
                return True
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
            if event_type == QEvent.Type.Drop:
                self.cancel_drag()
                if edge in _EDGES:
                    if origin is group:
                        self.split_source(source['key'], edge, origin.group_id)
                    else:
                        with self._change():
                            new = self._new_group()
                            self._split_group(group, new, edge)
                            self.move_source(source['key'], new.group_id, origin.group_id)
                else:
                    if origin is group and index is not None and origin.index_of(source['key']) < index:
                        index -= 1
                    self.move_source(source['key'], group.group_id, origin.group_id, index)
            else:
                self._show_hint(group, edge, index)
            return True
        if event_type == QEvent.Type.DragLeave:
            for candidate in self._groups.values():
                candidate.hint.hide()
        return super().eventFilter(watched, event)

    def apply_theme(self, dark, *, refresh_views=True):
        self._dark = bool(dark)
        tool_style = (
            'ToolButton{background:transparent;border:0;border-radius:4px;}'
            f'ToolButton:hover{{background:{color("bg_muted", dark)};}}'
            f'ToolButton:pressed{{background:{rgba("primary", .12, dark)};}}')
        for group in self._groups.values():
            group.more.setStyleSheet(tool_style)
            for index, key in enumerate(group.keys()):
                icon = FIF.PIN if key in group.pinned else FIF.PHOTO
                group.tabs.setTabIcon(index, icon.icon(Theme.DARK if dark else Theme.LIGHT))
                close = group.tabs.tabButton(index, QTabBar.ButtonPosition.RightSide)
                if isinstance(close, ToolButton):
                    close.setStyleSheet(tool_style)
                    close.setIcon(FIF.CLOSE.icon(Theme.DARK if dark else Theme.LIGHT))
            border = (rgba('primary', .40, dark) if group.group_id == self._active_group
                      else color('border_default', dark))
            group.setStyleSheet(
                f'QWidget#bscanWorkspaceGroup {{background:{color("bg_surface", dark)};border:1px solid {border};}}'
                f'QTabBar#bscanWorkspaceTabs::tab {{background:{color("bg_subtle", dark)};color:{color("text_secondary", dark)};'
                'padding:5px 10px;min-width:64px;max-width:220px;border:0;'
                'border-top-left-radius:6px;border-top-right-radius:6px;}'
                f'QTabBar#bscanWorkspaceTabs::tab:selected {{background:{color("bg_surface", dark)};color:{color("text_primary", dark)};'
                f'font-weight:600;border-bottom:2px solid {color("primary", dark)};}}'
                f'QTabBar#bscanWorkspaceTabs::tab:!selected:hover {{background:{color("bg_muted", dark)};}}'
                f'QTabBar#bscanWorkspaceTabs:focus {{border-bottom:1px solid {color("border_focus", dark)};}}'
                'ToolButton{background:transparent;border:0;border-radius:4px;}'
                f'ToolButton:hover{{background:{color("bg_muted", dark)};}}'
                f'ToolButton:pressed{{background:{rgba("primary", .12, dark)};}}'
                f'QLabel#bscanDropHint {{background:{rgba("primary", .16, dark)};border:2px solid {color("primary", dark)};'
                f'color:{color("text_primary", dark)};}}')
            if refresh_views:
                group.view.apply_theme(dark)

    def hideEvent(self, event):
        self._close_menu()
        self.cancel_drag()
        self._set_link_enabled(False)
        super().hideEvent(event)


__all__ = ['BScanWorkspace']

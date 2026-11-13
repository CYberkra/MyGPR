"""One searchable model/view list for an entire B-scan workspace."""
from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from PyQt6.QtCore import QAbstractTableModel, QItemSelectionModel, QModelIndex, QSignalBlocker, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableView, QToolButton, QVBoxLayout, QWidget,
)
from qfluentwidgets import isDarkTheme

from ui import file_dialogs
from ui.desktop_backend_facade import EventInterval, MarkerRecord, marker_rows_for_view
from ui.widgets.context_menus import RoundMenu, add_action, make_menu, show_menu
from ui.widgets.object_context_menu import ObjectMenuBinding, object_action
from ui.widgets.marker_editor import MarkerEditor
from ui.widgets.marker_style import marker_style


class MarkerTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = ()

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = tuple(rows)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 4

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return ('名称', '用途', '位置或范围', '状态')[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 3 and row['status'] == '可定位':
                return ''
            return row[('name', 'purpose', 'position', 'status')[index.column()]]
        if role == Qt.ItemDataRole.UserRole:
            return row
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{row['name']} · {row['purpose']}\n测线 {row['line_id']} · {row['position']}\n{row['status']}\n{row['record'].note}".rstrip()
        return None


class MarkerFilter(QSortFilterProxyModel):
    query = ''
    kind = ''
    purpose = ''
    status = ''

    def filterAcceptsRow(self, source_row, parent):
        row = self.sourceModel().rows[source_row]
        record = row['record']
        interval = isinstance(record, EventInterval)
        return (not self.query or self.query in (row['name'] + ' ' + record.note).casefold()) and (
            not self.kind or (self.kind == '区间') == interval) and (
            not self.purpose or self.purpose == record.purpose) and (
            not self.status or (self.status == '已排除' and record.excluded)
            or (self.status == '可定位' and bool(row['projection'].positions or row['projection'].spans)))


class MarkerTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setMinimumSectionSize(44)
        self.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(30)
        self.setMinimumHeight(0)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setAlternatingRowColors(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = max(260, self.viewport().width())
        weights = (.32, .22, .26) if width < 400 else (.35, .19, .24)
        widths = tuple(int(width * weight) for weight in weights)
        for column, size in enumerate((*widths, width - sum(widths))):
            self.setColumnWidth(column, size)


class MarkerPanel(QWidget):
    operation = pyqtSignal(str, object)
    selected = pyqtSignal(str)
    locate = pyqtSignal(str)
    adjust = pyqtSignal(str)
    close_requested = pyqtSignal()
    navigate = pyqtSignal(str, object)

    def __init__(self, parent=None, *, allow_processing=False, embedded=True):
        super().__init__(parent)
        self.setObjectName('markerListPanel')
        self.allow_processing = allow_processing
        self.processing_source = None
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.view = None
        self._views = {}
        self._requests = {}
        self.editor = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.scope = QComboBox(self)
        self.scope.setAccessibleName('标记范围')
        self.scope.addItems(['当前测线', '整个项目'])
        header.addWidget(self.scope)
        header.addStretch()
        self.count_label = QLabel(self)
        self.count_label.setObjectName('markerCount')
        header.addWidget(self.count_label)
        if embedded:
            close = QToolButton(self)
            close.setText('×')
            close.setAccessibleName('关闭标记列表')
            close.setToolTip('关闭列表')
            close.setFixedSize(28, 28)
            close.clicked.connect(self.close_requested)
            header.addWidget(close)
        layout.addLayout(header)
        filters = QHBoxLayout()
        self.search = QLineEdit(self)
        self.search.setPlaceholderText('搜索名称或备注')
        self.search.setClearButtonEnabled(True)
        filters.addWidget(self.search, 1)
        self.filter_btn = QPushButton('筛选 ▾', self)
        self.filter_btn.clicked.connect(self._show_filters)
        filters.addWidget(self.filter_btn)
        layout.addLayout(filters)
        self.model = MarkerTableModel(self)
        self.proxy = MarkerFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = MarkerTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._row_menu = ObjectMenuBinding(self, self.table, self._menu)
        self.table.doubleClicked.connect(lambda _: self.edit_selected())
        self.table.selectionModel().currentRowChanged.connect(self._selection)
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.new_point = QPushButton('新增点', self)
        self.new_interval = QPushButton('新区间', self)
        self.new_point.clicked.connect(lambda: self.edit(MarkerRecord()) if self.view else None)
        self.new_interval.clicked.connect(self.make_interval)
        self.table.selectionModel().selectionChanged.connect(self._sync_interval_label)
        if allow_processing:
            selection = QPushButton('所选区间 ▾', self)
            selection.clicked.connect(lambda: self._show_selection_menu(selection))
            buttons.addWidget(selection)
        export = QPushButton('导出 ▾', self)
        export.clicked.connect(lambda: self._show_export_menu(export))
        for button in (self.new_point, self.new_interval, export):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.hide()
        layout.addWidget(self.message)
        self.scope.currentIndexChanged.connect(self._scope)
        self.search.textChanged.connect(self._filter)
        self._shortcuts = []
        for key, callback in [('Return', self.locate_selected), ('F2', self.edit_selected)]:
            shortcut = QShortcut(QKeySequence(key), self.table)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        self.apply_theme(isDarkTheme())

    def apply_theme(self, dark):
        self.setStyleSheet(marker_style(dark))

    def set_view(self, view):
        if view is self.view:
            return
        self._row_menu.close()
        self.view = view
        self._views = {view.document.line_id: view} if view else {}
        self.scope.blockSignals(True)
        self.scope.setCurrentIndex(0)
        self.scope.blockSignals(False)
        self._populate(view.rows if view else ())
        for button in (self.new_point, self.new_interval):
            button.setEnabled(bool(view and not view.read_only))

    def _populate(self, rows):
        selected = {row['uid'] for row in self._selected_rows()}
        current = self.table.currentIndex().data(Qt.ItemDataRole.UserRole)
        selection = self.table.selectionModel()
        blocker = QSignalBlocker(selection)
        self.model.set_rows(rows)
        if self.proxy.purpose not in {row['purpose'] for row in rows}:
            self.proxy.purpose = ''
        self._filter()
        for index, row in enumerate(self.model.rows):
            mapped = self.proxy.mapFromSource(self.model.index(index, 0))
            if mapped.isValid() and row['uid'] in selected:
                selection.select(mapped, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
            if mapped.isValid() and current and row['uid'] == current['uid']:
                selection.setCurrentIndex(mapped, QItemSelectionModel.SelectionFlag.NoUpdate)
        del blocker
        self._sync_interval_label()
        self._set_message('')

    def _filter(self, *_):
        self.proxy.query = self.search.text().strip().casefold()
        active = sum(bool(value) for value in (self.proxy.kind, self.proxy.purpose, self.proxy.status))
        self.filter_btn.setText(f'筛选 {active} ▾' if active else '筛选 ▾')
        self.proxy.invalidateFilter()
        visible, total = self.proxy.rowCount(), self.model.rowCount()
        self.count_label.setText(f'{total} 条记录' if visible == total else f'{visible} / {total} 条记录')

    def _show_filters(self):
        menu = make_menu(self)
        purposes = sorted({row['purpose'] for row in self.model.rows if row['purpose']})
        for field, label, default, values in (
                ('kind', '类型', '点与区间', ('点', '区间')),
                ('purpose', '用途', '全部用途', purposes),
                ('status', '状态', '全部状态', ('可定位', '已排除'))):
            submenu = RoundMenu(label, menu)
            for value in ('', *values):
                action = add_action(submenu, None, value or default,
                                    lambda _checked=False, field=field, value=value: self._set_filter(field, value))
                action.setCheckable(True)
                action.setChecked(getattr(self.proxy, field) == value)
            menu.addMenu(submenu)
        show_menu(menu, self.filter_btn.mapToGlobal(self.filter_btn.rect().bottomLeft()))

    def _set_filter(self, field, value):
        setattr(self.proxy, field, value)
        self._filter()

    def _set_message(self, text):
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def _scope(self):
        if not self.view:
            return
        if self.scope.currentIndex():
            self.request('project', {'project_id': self.view.document.project_id})
        else:
            self._populate(self.view.rows)

    def _selected_rows(self):
        return [index.data(Qt.ItemDataRole.UserRole) for index in self.table.selectionModel().selectedRows()]

    def _sync_interval_label(self, *_):
        rows = self._selected_rows()
        two_points = len(rows) == 2 and all(isinstance(row['record'], MarkerRecord) for row in rows)
        self.new_interval.setText('建立区间' if two_points else '新区间')

    def select_id(self, uid):
        for index, row in enumerate(self.model.rows):
            if row['uid'] == uid:
                mapped = self.proxy.mapFromSource(self.model.index(index, 0))
                if mapped.isValid():
                    self.table.selectRow(mapped.row())
                    self.table.scrollTo(mapped)
                break

    def _selection(self, current, previous):
        row = current.data(Qt.ItemDataRole.UserRole)
        if row:
            self.selected.emit(row['uid'])

    def locate_selected(self):
        rows = self._selected_rows()
        if rows:
            self.locate.emit(rows[0]['uid'])

    def edit_selected(self):
        rows = self._selected_rows()
        if rows:
            self.edit(rows[0]['record'], self._views.get(rows[0]['line_id'], self.view))

    def edit(self, record, view=None):
        view = view or self.view
        if view is None:
            return
        if self.editor and self.editor.isVisible():
            self.editor.raise_()
            return
        self.editor = MarkerEditor(view, record, self)
        self.editor.save_requested.connect(lambda value: self.request('save', {'view': view, 'record': value}))
        self.editor.show()

    def make_interval(self):
        if not self.view:
            return
        rows = self._selected_rows()
        points = [row['record'] for row in rows if isinstance(row['record'], MarkerRecord)]
        if len(rows) != 2:
            points = []
        target = self.view
        if len(points) == 2:
            if len({row['line_id'] for row in rows}) != 1:
                self._set_message('两个边界标记须属于同一测线')
                return
            target = self._views.get(rows[0]['line_id'], self.view)
        record = EventInterval()
        if len(points) == 2 and all(point.anchor for point in points):
            points.sort(key=lambda point: point.anchor.trace)
            a, b = points
            if a.anchor.source_id != b.anchor.source_id:
                self._set_message('两个边界标记须属于同一数据版本')
                return
            record = replace(record, start=a.anchor, end=b.anchor,
                             start_marker_id=a.uid, end_marker_id=b.uid,
                             start_marker_revision=a.revision, end_marker_revision=b.revision)
        self.edit(record, target)

    def _menu(self, position):
        row = self.table.indexAt(position).data(Qt.ItemDataRole.UserRole)
        if not row:
            return make_menu(self)
        line = row['line_id']
        context = self._views.get(line)
        return self.record_menu(row, context, lambda: self._views.get(line) is context)

    def record_menu(self, row, context, current, *, locate=None, adjust=None):
        menu = make_menu(self)
        if context is None:
            return menu
        uid, record, projection = row['uid'], row['record'], row['projection']
        editable = not context.read_only
        positioned = bool(projection.positions or projection.spans) and not record.excluded

        def add(command, title, callback, **kwargs):
            return object_action(menu, command, title, lambda: callback() if current() else None, **kwargs)

        add('locate-marker', '定位', locate or (lambda: self.locate.emit(uid)),
            enabled=positioned, reason='当前结果无法定位')
        add('edit-marker', '编辑…' if editable else '详情…', lambda: self.edit(record, context))
        if isinstance(record, EventInterval) and (adjust or context is self.view and self.allow_processing):
            add('adjust-marker', '调整范围', adjust or (lambda: self.adjust.emit(uid)),
                enabled=editable and positioned, reason='需要可编辑且可定位的区间')
        if isinstance(record, MarkerRecord) and record.origin == 'acquisition':
            add('exclude-marker', '恢复使用' if record.excluded else '排除使用',
                lambda: self.request('save', {'view': context, 'record': replace(record, excluded=not record.excluded)}),
                enabled=editable, reason='项目只读')
        for title, page in [('在处理页查看', 'processing'), ('在解释页查看', 'interpretation'), ('在空间页定位', 'spatial')]:
            add('open-' + page, title, lambda page=page: self.navigate.emit(page, row),
                enabled=bool(row['geometry']) if page == 'spatial' else positioned, reason='缺少定位信息')
        if getattr(record, 'origin', 'project') != 'acquisition':
            menu.addSeparator()
            add('delete-marker', '删除项目记录…', lambda: self.delete_record(row, context, current),
                enabled=editable, reason='项目只读')
        return menu

    def _show_selection_menu(self, button):
        menu = make_menu(self)
        view, rows = self.view, tuple(self._selected_rows())
        reason = self._selection_reason(view, rows)
        for title, mode in [('保留…', 'keep'), ('剔除…', 'exclude')]:
            object_action(menu, mode, title,
                lambda _checked=False, mode=mode: self.preview(mode, rows) if view is self.view else None,
                enabled=not reason, reason=reason)
        show_menu(menu, button.mapToGlobal(button.rect().bottomLeft()))

    def _selection_reason(self, view, rows):
        if not view or view.read_only:
            return '需要可编辑的项目'
        if self.processing_source is not None:
            active = self.processing_source()
            if active is None or (active.document.project_id, active.document.line_id, active.target_id) != (
                    view.document.project_id, view.document.line_id, view.target_id):
                return '请先将该成果设为处理输入'
        if not rows or any(not isinstance(row['record'], EventInterval) for row in rows):
            return '请选择事件区间'
        if any(row['line_id'] != view.document.line_id or row['target_id'] != view.target_id for row in rows):
            return '请选择当前输入的区间'
        for row in rows:
            if not row['record'].excluded and (issue := row.get('boundary_issue')):
                return f'“{row["name"]}”的{issue}，请编辑区间并保存起止范围'
        if any(not row['record'].closed or row['record'].excluded or not row['projection'].spans for row in rows):
            return '选择中有开放、已排除或无法映射到道的区间'
        return ''

    def _show_export_menu(self, button):
        menu = make_menu(self)
        views = tuple(self._views.values()) if self.scope.currentIndex() else (self.view,) if self.view else ()
        scope = '整个项目' if self.scope.currentIndex() else '当前测线'
        self._add_export_formats(menu, views, scope)
        if 0 < self.proxy.rowCount() < self.model.rowCount():
            filtered = RoundMenu('当前筛选结果', menu)
            self._add_export_formats(filtered, self._filtered_views(views), '当前筛选结果')
            menu.addMenu(filtered)
        show_menu(menu, button.mapToGlobal(button.rect().bottomLeft()))

    def _add_export_formats(self, menu, views, scope):
        for extension in ('csv', 'json', 'geojson'):
            object_action(menu, 'export-' + extension, f'{extension.upper()} · {scope}…',
                lambda _checked=False, ext=extension: self.export(ext, views, scope), enabled=bool(views))

    def _filtered_views(self, views):
        by_line = {}
        for index in range(self.proxy.rowCount()):
            row = self.proxy.index(index, 0).data(Qt.ItemDataRole.UserRole)
            by_line.setdefault(row['line_id'], set()).add(row['uid'])
        result = []
        for view in views:
            ids = by_line.get(view.document.line_id)
            if not ids:
                continue
            document = replace(view.document,
                markers=tuple(record for record in view.document.markers if record.uid in ids),
                intervals=tuple(record for record in view.document.intervals if record.uid in ids))
            result.append(replace(view, document=document, rows=tuple(row for row in view.rows if row['uid'] in ids)))
        return tuple(result)

    def preview(self, mode, rows=None):
        rows = self._selected_rows() if rows is None else rows
        reason = self._selection_reason(self.view, rows)
        if reason:
            self._set_message(reason)
            return
        self.request('preview', {'view': self.view, 'ids': [row['uid'] for row in rows], 'mode': mode})

    def delete_record(self, row, context, current):
        answer = QMessageBox.question(self, '删除标记', f"删除项目记录“{row['name']}”？有依赖时将拒绝删除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Yes and current():
            self.request('delete', {'view': context, 'uid': row['uid']})

    def export(self, extension, views, scope):
        path, _ = file_dialogs.getSaveFileName(self, '导出标记 · ' + scope, f'markers.{extension}', f'{extension.upper()} (*.{extension})')
        if path:
            self.request('export', {'views': views, 'path': path, 'format': extension})

    def request(self, action, payload):
        token = uuid4().hex
        self._requests[token] = (action, payload)
        self.operation.emit(action, {'token': token, **payload})

    def deliver(self, token, result, error):
        pending = self._requests.pop(token, None)
        if pending is None:
            return
        action, payload = pending
        if action == 'preview' and payload['view'] is not self.view:
            return
        if error:
            self._set_message(error)
            if action == 'save' and self.editor:
                self.editor.failed(error)
            return
        if action == 'save' and self.editor:
            self.editor.accept()
        if action in ('save', 'delete'):
            if self.scope.currentIndex():
                self.request('project', {'project_id': result.document.project_id})
                return
            if self.view and self.view.document.line_id == result.document.line_id and self.view.target_id == result.target_id:
                self.set_view(result)
        elif action == 'project':
            if result and self.view is None:
                self.set_view(result[0])
            current = next((item for item in result if self.view and
                            item.document.line_id == self.view.document.line_id), None)
            if current:
                # A project-list refresh must also refresh the execution context.
                # Keeping the previous revision here makes the next preview stale
                # immediately after a successful save.
                updated = replace(self.view, document=current.document, timezone=current.timezone,
                                  interpretation_links=current.interpretation_links,
                                  processing_history=current.processing_history)
                self.view = replace(updated, rows=marker_rows_for_view(updated))
            self.scope.blockSignals(True)
            self.scope.setCurrentIndex(1)
            self.scope.blockSignals(False)
            self._views = {view.document.line_id: view for view in result}
            self._populate(tuple(row for view in result for row in view.rows))
        elif action == 'preview':
            title = '保留所选区间' if result['mode'] == 'keep' else '剔除所选区间'
            ranges = '、'.join(str(a + 1) if a + 1 == b else f'{a + 1}–{b}'
                              for a, b in result['ranges'])
            segments = len(result['output_segments'])
            input_name = result['input_artifact_id'] or '原始数据'
            text = (f"{result['line_id']} · {input_name}\n"
                    f"选中 {result['affected_count']} / {result['input_count']} 道\n"
                    f"输出 {result['output_count']} 道")
            if segments > 1:
                text += f' · {segments} 段'
            box = QMessageBox(QMessageBox.Icon.Question, title, text,
                              QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, self)
            box.setTextFormat(Qt.TextFormat.PlainText)
            if len(ranges) <= 100:
                box.setInformativeText(f'道号 {ranges}')
            else:
                box.setDetailedText('\n'.join(f'道号 {a + 1}–{b}' for a, b in result['ranges']))
            box.button(QMessageBox.StandardButton.Ok).setText('生成成果')
            box.button(QMessageBox.StandardButton.Cancel).setText('取消')
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if box.exec() == QMessageBox.StandardButton.Ok and payload['view'] is self.view:
                self.request('execute', {'preview': result})
        elif action == 'execute':
            self._set_message('已提交任务')
        elif action == 'export':
            self._set_message(str(result))

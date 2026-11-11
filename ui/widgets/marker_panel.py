"""One searchable model/view list for an entire B-scan workspace."""
from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMenu, QMessageBox, QPushButton, QTableView, QVBoxLayout, QWidget,
)

from ui.desktop_backend_facade import EventInterval, MarkerRecord
from ui.widgets.marker_editor import MarkerEditor


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
            return row[('name', 'purpose', 'position', 'status')[index.column()]]
        if role == Qt.ItemDataRole.UserRole:
            return row
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{row['name']}\n测线 {row['line_id']}\n{row['record'].note}"
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


class MarkerPanel(QWidget):
    operation = pyqtSignal(str, object)
    selected = pyqtSignal(str)
    locate = pyqtSignal(str)
    adjust = pyqtSignal(str)
    close_requested = pyqtSignal()
    navigate = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('markerListPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.view = None
        self._views = {}
        self._requests = {}
        self.editor = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        header = QHBoxLayout()
        self.scope = QComboBox(self)
        self.scope.addItems(['当前测线', '整个项目'])
        header.addWidget(self.scope, 1)
        close = QPushButton('关闭', self)
        close.clicked.connect(self.close_requested)
        header.addWidget(close)
        layout.addLayout(header)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText('搜索名称或备注')
        layout.addWidget(self.search)
        filters = QHBoxLayout()
        self.kind = QComboBox(self)
        self.kind.addItems(['点与区间', '点', '区间'])
        self.purpose = QComboBox(self)
        self.purpose.addItem('全部用途')
        self.status = QComboBox(self)
        self.status.addItems(['全部状态', '可定位', '已排除'])
        for combo in (self.kind, self.purpose, self.status):
            filters.addWidget(combo)
            combo.currentIndexChanged.connect(self._filter)
        layout.addLayout(filters)
        self.model = MarkerTableModel(self)
        self.proxy = MarkerFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        for column, width in enumerate((90, 70, 90, 80)):
            self.table.setColumnWidth(column, width)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        self.table.doubleClicked.connect(lambda _: self.edit_selected())
        self.table.selectionModel().currentRowChanged.connect(self._selection)
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.new_point = QPushButton('新增点', self)
        self.new_interval = QPushButton('新区间', self)
        more = QPushButton('操作 ▾', self)
        self.new_point.clicked.connect(lambda: self.edit(MarkerRecord(anchor=self.view.anchor(0))) if self.view else None)
        self.new_interval.clicked.connect(self.make_interval)
        more.clicked.connect(lambda: self._menu(more.mapTo(self.table.viewport(), more.rect().bottomLeft())))
        for button in (self.new_point, self.new_interval, more):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.scope.currentIndexChanged.connect(self._scope)
        self.search.textChanged.connect(self._filter)
        self._shortcuts = []
        for key, callback in [('Return', self.locate_selected), ('F2', self.edit_selected)]:
            shortcut = QShortcut(QKeySequence(key), self.table)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def set_view(self, view):
        if view is self.view:
            return
        self.view = view
        self._views = {view.document.line_id: view} if view else {}
        self.scope.blockSignals(True)
        self.scope.setCurrentIndex(0)
        self.scope.blockSignals(False)
        self._populate(view.rows if view else ())
        for button in (self.new_point, self.new_interval):
            button.setEnabled(bool(view and not view.read_only))

    def _populate(self, rows):
        self.model.set_rows(rows)
        value = self.purpose.currentText()
        self.purpose.blockSignals(True)
        self.purpose.clear()
        self.purpose.addItems(['全部用途', *sorted({row['purpose'] for row in rows})])
        self.purpose.setCurrentText(value)
        self.purpose.blockSignals(False)
        self._filter()
        self.message.setText(f'{len(rows)} 条记录')

    def _filter(self, *_):
        self.proxy.query = self.search.text().strip().casefold()
        self.proxy.kind = self.kind.currentText() if self.kind.currentIndex() else ''
        self.proxy.purpose = self.purpose.currentText() if self.purpose.currentIndex() > 0 else ''
        self.proxy.status = self.status.currentText() if self.status.currentIndex() else ''
        self.proxy.invalidateFilter()

    def _scope(self):
        if not self.view:
            return
        if self.scope.currentIndex():
            self.request('project', {'project_id': self.view.document.project_id})
        else:
            self._populate(self.view.rows)

    def _selected_rows(self):
        return [index.data(Qt.ItemDataRole.UserRole) for index in self.table.selectionModel().selectedRows()]

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
        target = self.view
        if len(points) == 2:
            if len({row['line_id'] for row in rows}) != 1:
                self.message.setText('两个边界标记须属于同一测线')
                return
            target = self._views.get(rows[0]['line_id'], self.view)
        record = EventInterval()
        if len(points) == 2 and all(point.anchor for point in points):
            points.sort(key=lambda point: point.anchor.trace)
            a, b = points
            if a.anchor.source_id != b.anchor.source_id:
                self.message.setText('两个边界标记须属于同一数据版本')
                return
            record = replace(record, start=a.anchor, end=b.anchor,
                             start_marker_id=a.uid, end_marker_id=b.uid,
                             start_marker_revision=a.revision, end_marker_revision=b.revision)
        self.edit(record, target)

    def _menu(self, position):
        menu = QMenu(self)
        rows = self._selected_rows()
        editable = bool(self.view and not self.view.read_only)
        menu.addAction('定位', self.locate_selected).setEnabled(bool(rows))
        menu.addAction('详情', self.edit_selected).setEnabled(bool(rows))
        menu.addAction('建立区间', self.make_interval).setEnabled(editable)
        menu.addAction('调整范围', lambda: self.adjust.emit(rows[0]['uid'])).setEnabled(
            editable and len(rows) == 1 and isinstance(rows[0]['record'], EventInterval))
        menu.addSeparator()
        for title, mode in [('保留所选区间', 'keep'), ('剔除所选区间', 'exclude')]:
            menu.addAction(title, lambda checked=False, mode=mode: self.preview(mode)).setEnabled(
                editable and bool(rows) and all(isinstance(row['record'], EventInterval) for row in rows))
        menu.addAction('删除项目记录', self.delete_selected).setEnabled(editable and bool(rows))
        menu.addSeparator()
        for title, page in [('在处理页查看', 'processing'), ('在空间页定位', 'spatial')]:
            menu.addAction(title, lambda checked=False, page=page: self.navigate.emit(page, rows[0])).setEnabled(bool(rows))
        for extension in ('json', 'csv', 'geojson'):
            menu.addAction('导出 ' + extension.upper(), lambda checked=False, extension=extension: self.export(extension)).setEnabled(bool(self.view))
        menu.exec(self.table.viewport().mapToGlobal(position))

    def preview(self, mode):
        rows = self._selected_rows()
        if any(row['line_id'] != self.view.document.line_id for row in rows):
            self.message.setText('处理范围必须属于当前测线和当前输入成果')
            return
        self.request('preview', {'view': self.view, 'ids': [row['uid'] for row in rows], 'mode': mode})

    def delete_selected(self):
        rows = self._selected_rows()
        if len(rows) != 1:
            self.message.setText('每次删除一条项目记录；采集记录请在详情中排除使用')
            return
        row = rows[0]
        if QMessageBox.question(self, '删除标记', f"删除项目记录“{row['name']}”？有依赖时将拒绝删除。") == QMessageBox.StandardButton.Yes:
            self.request('delete', {'view': self._views[row['line_id']], 'uid': row['uid']})

    def export(self, extension):
        path, _ = QFileDialog.getSaveFileName(self, '导出标记', f'markers.{extension}', f'{extension.upper()} (*.{extension})')
        if path:
            self.request('export', {'views': tuple(self._views.values()) if self.scope.currentIndex() else (self.view,),
                                    'path': path, 'format': extension})

    def request(self, action, payload):
        token = uuid4().hex
        self._requests[token] = (action, payload)
        self.operation.emit(action, {'token': token, **payload})

    def deliver(self, token, result, error):
        pending = self._requests.pop(token, None)
        if pending is None:
            return
        action, payload = pending
        if error:
            self.message.setText(error)
            if action == 'save' and self.editor:
                self.editor.failed(error)
            return
        if action == 'save' and self.editor:
            self.editor.accept()
        if action in ('save', 'delete'):
            if self.view and self.view.document.line_id == result.document.line_id and self.view.target_id == result.target_id:
                self.set_view(result)
        elif action == 'project':
            if result and self.view is None:
                self.set_view(result[0])
            self.scope.blockSignals(True)
            self.scope.setCurrentIndex(1)
            self.scope.blockSignals(False)
            self._views = {view.document.line_id: view for view in result}
            self._populate(tuple(row for view in result for row in view.rows))
        elif action == 'preview':
            text = (f"输入：{result['input_artifact_id'] or '原始数据'}\n"
                    f"合并后选中 {result['affected_count']} 道；输入 {result['input_count']} 道；输出 {result['output_count']} 道\n"
                    f"输出分段 {len(result['output_segments'])} 段\n"
                    + '\n'.join(f'道 {a + 1}–{b}' for a, b in result['ranges']))
            box = QMessageBox(QMessageBox.Icon.Question, '确认道范围处理', text,
                              QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, self)
            box.button(QMessageBox.StandardButton.Ok).setText('运行并生成新成果')
            if box.exec() == QMessageBox.StandardButton.Ok:
                self.request('execute', {'preview': result})
        elif action == 'execute':
            self.message.setText('已提交任务：' + str(result))
        elif action == 'export':
            self.message.setText(str(result))

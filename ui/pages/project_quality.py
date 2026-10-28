"""Application-side project data-quality report center."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QPlainTextEdit, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QListWidget, QListWidgetItem,
)
from qfluentwidgets import ComboBox, PrimaryPushButton, PushButton

from ui.page_scaffold import card_title, make_hint


class QualityCenterDialog(QDialog):
    check_project_requested = pyqtSignal()
    check_line_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()
    locate_requested = pyqtSignal(str, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reports = []
        self.setWindowTitle('项目数据质检')
        self.setMinimumSize(860, 560)
        self.setModal(False)
        layout = QVBoxLayout(self)
        title = card_title('测线质量与问题定位')
        title.setParent(self)
        layout.addWidget(title)
        actions = QHBoxLayout()
        self._project_check_btn = PrimaryPushButton('检查整个项目', self)
        self._line_check_btn = PushButton('复检选中测线', self)
        self._refresh_btn = PushButton('读取已有报告', self)
        self._status_filter = ComboBox(self)
        for text, data in (('全部状态', ''), ('通过', 'passed'), ('警告', 'warning'), ('失败', 'failed')):
            self._status_filter.addItem(text, userData=data)
        actions.addWidget(self._project_check_btn)
        actions.addWidget(self._line_check_btn)
        actions.addWidget(self._refresh_btn)
        actions.addStretch(1)
        actions.addWidget(self._status_filter)
        layout.addLayout(actions)
        self._table = QTableWidget(0, 7, self)
        self._table.setHorizontalHeaderLabels(
            ['测线', '状态', '采样×道', '时窗/ns', '长度/m', '有效率', '轨迹点'])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._show_selected_details)
        layout.addWidget(self._table, 2)
        hint = make_hint('问题详情与修复建议', parent=self)
        layout.addWidget(hint)
        self._details = QPlainTextEdit(self)
        self._details.setReadOnly(True)
        layout.addWidget(self._details, 1)
        self._locations = QListWidget(self)
        self._locations.setMaximumHeight(120)
        layout.addWidget(self._locations)
        self._locate_btn = PushButton('定位所选问题道', self)
        self._locate_btn.setEnabled(False)
        layout.addWidget(self._locate_btn)
        self._locations.currentRowChanged.connect(lambda row: self._locate_btn.setEnabled(row >= 0))
        self._locations.itemDoubleClicked.connect(lambda _item: self._locate_selected())
        self._locate_btn.clicked.connect(self._locate_selected)
        self._project_check_btn.clicked.connect(self.check_project_requested)
        self._line_check_btn.clicked.connect(self._check_selected)
        self._refresh_btn.clicked.connect(self.refresh_requested)
        self._status_filter.currentIndexChanged.connect(self._refill)

    def set_reports(self, reports):
        self._reports = list(reports or [])
        self._refill()

    def selected_line_id(self):
        row = self._table.currentRow()
        item = self._table.item(row, 0) if row >= 0 else None
        return str(item.data(Qt.ItemDataRole.UserRole) or '') if item else ''

    def _refill(self, *_args):
        wanted = str(self._status_filter.currentData() or '')
        selected = self.selected_line_id()
        reports = [r for r in self._reports if not wanted or str(getattr(r, 'status', '')) == wanted]
        self._table.setRowCount(0)
        for report in reports:
            row = self._table.rowCount()
            self._table.insertRow(row)
            values = (
                str(getattr(report, 'line_id', '')),
                str(getattr(report, 'status_label', '') or getattr(report, 'status', '')),
                f"{getattr(report, 'sample_count', 0)}×{getattr(report, 'trace_count', 0)}",
                f"{float(getattr(report, 'time_window_ns', 0) or 0):g}",
                f"{float(getattr(report, 'length_m', 0) or 0):g}",
                f"{float(getattr(report, 'finite_ratio', 0) or 0):.2%}",
                str(getattr(report, 'trajectory_points', 0)),
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, str(getattr(report, 'line_id', '')))
                self._table.setItem(row, col, item)
            if str(getattr(report, 'line_id', '')) == selected:
                self._table.selectRow(row)
        self._line_check_btn.setEnabled(bool(reports))
        self._show_selected_details()

    def _selected_report(self):
        line_id = self.selected_line_id()
        return next((r for r in self._reports if str(getattr(r, 'line_id', '')) == line_id), None)

    def _show_selected_details(self):
        self._locations.clear()
        report = self._selected_report()
        if report is None:
            self._details.setPlainText('选择一条测线查看问题详情。')
            return
        lines = [
            f"测线：{getattr(report, 'line_id', '')}",
            f"检查时间：{getattr(report, 'checked_at', '')}",
            f"方向判断：{getattr(report, 'orientation_message', '')}",
            f"建议：{getattr(report, 'suggested_action', '')}",
            f"抽样质检：{'是' if getattr(report, 'sampled', False) else '否'}",
            '',
        ]
        issues = list(getattr(report, 'issues', ()) or ())
        if not issues:
            lines.append('未发现问题。')
        for issue in issues:
            start, end = getattr(issue, "trace_start", None), getattr(issue, "trace_end", None)
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= report.trace_count:
                item = QListWidgetItem(f"道 {start + 1}–{end} · {issue.message}")
                item.setData(Qt.ItemDataRole.UserRole, (start, end))
                self._locations.addItem(item)
            lines.extend([
                f"[{getattr(issue, 'severity', '')}] {getattr(issue, 'code', '')}",
                str(getattr(issue, 'message', '') or ''),
                f"建议：{getattr(issue, 'suggestion', '') or '—'}",
                '',
            ])
        self._details.setPlainText('\n'.join(lines))

    def _locate_selected(self):
        item = self._locations.currentItem()
        line_id = self.selected_line_id()
        if item is not None and line_id:
            start, end = item.data(Qt.ItemDataRole.UserRole)
            self.locate_requested.emit(line_id, start, end)

    def _check_selected(self):
        line_id = self.selected_line_id()
        if line_id:
            self.check_line_requested.emit(line_id)

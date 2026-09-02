"""JobTable / MiniJobList — 任务状态控件（SPEC §5.6）。

状态中文映射：queued 排队 / running 运行中 / completed 已完成 /
failed 失败 / cancelled 已取消。
徽章配色：完成 #22c55e、失败 #ef4444、运行 #3b82f6、排队/取消 #9ca3af。
running 状态显示进度条。
"""

import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel,
                             QStackedLayout, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)
from qfluentwidgets import CaptionLabel, ProgressBar, PushButton, ScrollArea

from ui.motion import animate_badge_color, animate_progress

_STATUS_TEXT = {
    'queued': '排队',
    'running': '运行中',
    'completed': '已完成',
    'failed': '失败',
    'cancelled': '已取消',
}

_STATUS_BADGE = {
    'queued': '#9ca3af',
    'running': '#3b82f6',
    'completed': '#22c55e',
    'failed': '#ef4444',
    'cancelled': '#9ca3af',
}

_ACTIVE_STATUSES = ('queued', 'running')

_BADGE_QSS = ('QLabel { padding: 2px 10px; border-radius: 10px; '
              'font-size: 12px; font-weight: bold; '
              'color: #ffffff; background-color: %s; }')


def _make_status_badge(status: str) -> QLabel:
    badge = QLabel(_STATUS_TEXT.get(status, status))
    badge.setStyleSheet(_BADGE_QSS % _STATUS_BADGE.get(status, '#9ca3af'))
    return badge


class JobTable(QWidget):
    """任务中心表格：列 = 标题 / 状态徽章 / 进度条 / 消息 / 操作(取消)。"""

    cancel_requested = pyqtSignal(str)

    _COL_TITLE, _COL_STATUS, _COL_PROGRESS, _COL_MESSAGE, _COL_ACTION = range(5)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = {}     # job_id -> row index
        self._badges = {}   # job_id -> 状态徽章 QLabel（复用，不重建）

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            ['标题', '状态', '进度', '消息', '操作'])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self._COL_TITLE,
                                    QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._COL_MESSAGE,
                                    QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # P2-6：无任务时空态占位（QStackedLayout 切换，避免纯空白）
        empty_page = QWidget(self)
        empty_layout = QVBoxLayout(empty_page)
        empty_label = CaptionLabel('暂无任务', empty_page)
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_label.setStyleSheet('color: #9ca3af; font-size: 13px;')
        empty_layout.addWidget(empty_label)
        self._stack = QStackedLayout()
        self._stack.addWidget(empty_page)   # index 0 = 空态
        self._stack.addWidget(self._table)  # index 1 = 表格
        layout.addLayout(self._stack)

    def _update_empty_state(self) -> None:
        """无任务时显示空态占位。"""
        self._stack.setCurrentIndex(0 if not self._rows else 1)

    # ------------------------------------------------------------- 接口
    def upsert_job(self, job_id: str, title: str) -> None:
        if job_id in self._rows:
            self._table.item(self._rows[job_id], self._COL_TITLE).setText(title)
            return
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._rows[job_id] = row
        self._table.setItem(row, self._COL_TITLE, QTableWidgetItem(title))
        self._update_empty_state()

        badge = _make_status_badge('queued')
        holder = QWidget(self._table)
        lay = QHBoxLayout(holder)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.addWidget(badge)
        lay.addStretch(1)
        self._table.setCellWidget(row, self._COL_STATUS, holder)
        marker = QTableWidgetItem('')
        marker.setData(Qt.ItemDataRole.UserRole, 'queued')
        self._table.setItem(row, self._COL_STATUS, marker)

        bar = ProgressBar(self._table)
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setVisible(False)
        self._table.setCellWidget(row, self._COL_PROGRESS, bar)
        self._table.setItem(row, self._COL_MESSAGE, QTableWidgetItem(''))
        cancel_btn = PushButton('取消', self._table)
        cancel_btn.clicked.connect(
            lambda _checked=False, jid=job_id: self.cancel_requested.emit(jid))
        self._table.setCellWidget(row, self._COL_ACTION, cancel_btn)
        self._badges[job_id] = badge

    def update_progress(self, job_id: str, completed: int, total: int,
                        message: str) -> None:
        row = self._rows.get(job_id)
        if row is None:
            return
        bar = self._table.cellWidget(row, self._COL_PROGRESS)
        if total and total > 0:
            bar.setRange(0, int(total))
            animate_progress(bar, min(int(completed), int(total)))
        else:
            bar.setRange(0, 100)
            animate_progress(bar, int(completed))
        bar.setVisible(True)
        self._table.item(row, self._COL_MESSAGE).setText(message or '')

    def set_status(self, job_id: str, status: str) -> None:
        row = self._rows.get(job_id)
        if row is None:
            return
        badge = self._badges.get(job_id)
        if badge is not None:
            end_hex = _STATUS_BADGE.get(status, '#9ca3af')
            badge.setText(_STATUS_TEXT.get(status, status))
            # 从徽章当前背景色渐变到新状态色（qss 模板同源，见 ui.motion）
            match = re.search(r'background-color:\s*(#[0-9a-fA-F]{6})',
                              badge.styleSheet())
            start_hex = match.group(1) if match else '#9ca3af'
            animate_badge_color(badge, _BADGE_QSS, start_hex, end_hex)
        item = self._table.item(row, self._COL_STATUS)
        if item is not None:
            item.setData(Qt.ItemDataRole.UserRole, status)
        bar = self._table.cellWidget(row, self._COL_PROGRESS)
        bar.setVisible(status == 'running')
        cancel_btn = self._table.cellWidget(row, self._COL_ACTION)
        cancel_btn.setEnabled(status in _ACTIVE_STATUSES)

    def clear_finished(self) -> None:
        """移除终态行（供"清理已完成"按钮）。"""
        for job_id in [j for j, r in self._rows.items()
                       if self._status_of(r) not in _ACTIVE_STATUSES]:
            self._table.removeRow(self._rows[job_id])
            del self._rows[job_id]
            self._badges.pop(job_id, None)
        self._rows = {j: i for i, j in enumerate(
            self._rows_ordered_ids())}
        self._update_empty_state()

    # ------------------------------------------------------------- 内部
    def _rows_ordered_ids(self):
        pairs = sorted(self._rows.items(), key=lambda kv: kv[1])
        return [j for j, _ in pairs]

    def _status_of(self, row):
        item = self._table.item(row, self._COL_STATUS)
        return item.data(Qt.ItemDataRole.UserRole) if item else None


class MiniJobList(QWidget):
    """右侧折叠面板"任务"tab：仅显示活动任务（标题 + 进度条 + 状态）。"""

    cancel_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs = {}   # job_id -> dict(row_widget, bar, status_label, status)

        self._scroll = ScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            'QScrollArea { background-color: transparent; border: none; }')
        self._container = QWidget(self._scroll)
        self._container.setStyleSheet('background-color: transparent;')
        self._box = QVBoxLayout(self._container)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(6)
        # P2-6：无活动任务时空态占位
        self._empty_label = CaptionLabel('暂无任务', self._container)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet('color: #9ca3af; font-size: 13px;')
        self._box.addWidget(self._empty_label)
        self._box.addStretch(1)
        self._scroll.setWidget(self._container)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)

    # ------------------------------------------------------------- 接口
    def upsert_job(self, job_id: str, title: str) -> None:
        entry = self._jobs.get(job_id)
        if entry is not None:
            entry['title_label'].setText(title)
            return
        row_widget = QWidget(self._container)
        lay = QVBoxLayout(row_widget)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        top = QHBoxLayout()
        title_label = QLabel(title, row_widget)
        status_badge = _make_status_badge('queued')
        top.addWidget(title_label, 1)
        top.addWidget(status_badge)
        lay.addLayout(top)

        bottom = QHBoxLayout()
        bar = ProgressBar(row_widget)
        bar.setRange(0, 100)
        bar.setValue(0)
        cancel_btn = PushButton('取消', row_widget)
        cancel_btn.setFixedWidth(60)
        cancel_btn.clicked.connect(
            lambda _checked=False, jid=job_id: self.cancel_requested.emit(jid))
        bottom.addWidget(bar, 1)
        bottom.addWidget(cancel_btn)
        lay.addLayout(bottom)

        self._box.insertWidget(self._box.count() - 1, row_widget)
        self._jobs[job_id] = {
            'widget': row_widget, 'title_label': title_label,
            'badge': status_badge, 'bar': bar, 'cancel': cancel_btn,
            'status': 'queued',
        }
        self._refresh_visibility()

    def update_progress(self, job_id: str, completed: int, total: int,
                        message: str) -> None:
        entry = self._jobs.get(job_id)
        if entry is None:
            return
        bar = entry['bar']
        if total and total > 0:
            bar.setRange(0, int(total))
            animate_progress(bar, min(int(completed), int(total)))
        else:
            bar.setRange(0, 100)
            animate_progress(bar, int(completed))
        if message:
            entry['title_label'].setToolTip(message)

    def set_status(self, job_id: str, status: str) -> None:
        entry = self._jobs.get(job_id)
        if entry is None:
            return
        old_hex = _STATUS_BADGE.get(entry['status'], '#9ca3af')
        entry['status'] = status
        badge = entry['badge']
        end_hex = _STATUS_BADGE.get(status, '#9ca3af')
        badge.setText(_STATUS_TEXT.get(status, status))
        animate_badge_color(badge, _BADGE_QSS, old_hex, end_hex)
        entry['cancel'].setEnabled(status in _ACTIVE_STATUSES)
        self._refresh_visibility()

    def remove_inactive(self) -> None:
        """移除终态任务行。"""
        for job_id in [j for j, e in self._jobs.items()
                       if e['status'] not in _ACTIVE_STATUSES]:
            self._remove_row(job_id)
        self._refresh_visibility()

    # ------------------------------------------------------------- 内部
    def _remove_row(self, job_id):
        entry = self._jobs.pop(job_id, None)
        if entry is None:
            return
        self._box.removeWidget(entry['widget'])
        entry['widget'].deleteLater()

    def _refresh_visibility(self):
        """仅显示活动任务；无活动任务时显示空态占位。"""
        any_active = False
        for entry in self._jobs.values():
            active = entry['status'] in _ACTIVE_STATUSES
            entry['widget'].setVisible(active)
            any_active = any_active or active
        self._empty_label.setVisible(not any_active)

"""JobTable / MiniJobList — 任务状态控件（SPEC §5.6）。

状态中文映射：queued 排队 / running 运行中 / completed 已完成 /
failed 失败 / cancelled 已取消。
任务徽章使用与主题配对的前景和底色。确认了总量的活动任务显示进度。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel,
                             QStackedLayout, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)
from qfluentwidgets import ProgressBar, PushButton, ScrollArea

from ui.motion import animate_progress, stop_progress
from ui.page_scaffold import HintLabel, style_transparent_scroll
from ui.theme_helpers import badge_colors, badge_qss_pair

_STATUS_TEXT = {
    'queued': '排队',
    'running': '运行中',
    'completed': '已完成',
    'failed': '失败',
    'cancelled': '已取消',
    'cancelling': '正在取消…',
}

_STATUS_COLOR_KEY = {
    'running': 'info',
    'completed': 'success',
    'failed': 'error',
}

_ACTIVE_STATUSES = ('queued', 'running')


def _set_status_badge(badge: QLabel, status: str) -> None:
    badge.setText(_STATUS_TEXT.get(status, status))
    badge.setStyleSheet(badge_qss_pair(*badge_colors(_STATUS_COLOR_KEY.get(status, 'neutral'))))


def _make_status_badge(status: str) -> QLabel:
    badge = QLabel()
    _set_status_badge(badge, status)
    return badge


def set_job_progress(bar, completed: int, total: int, *, active: bool = True) -> None:
    """A missing total has no percentage. Only animate a bar the user can see."""
    if not active or total <= 0:
        stop_progress(bar, 0)
        bar.hide()
        return
    completed = max(0, min(int(completed), int(total)))
    bar.setRange(0, int(total))
    bar.show()
    if bar.isVisible():
        animate_progress(bar, completed)
    else:
        stop_progress(bar, completed)


def progress_caption(completed: int, total: int, message: str) -> str:
    if total <= 0:
        return message or '运行中…'
    percent = max(0, min(completed, total)) * 100 // total
    return f'{percent}% · {message}' if message else f'{percent}%'


class JobTable(QWidget):
    """任务中心表格：列 = 标题 / 状态徽章 / 进度条 / 消息 / 操作(取消)。"""

    cancel_requested = pyqtSignal(str)
    # 行数/状态变化（宿主据此同步「清理已完成」的可用态：空表点了没反应）
    rows_changed = pyqtSignal()
    action_requested = pyqtSignal(str)

    _COL_TITLE, _COL_STATUS, _COL_PROGRESS, _COL_MESSAGE, _COL_ACTION = range(5)

    def finished_count(self) -> int:
        """终态（非活动）任务行数。"""
        return sum(self._is_prunable(job_id, row) for job_id, row in self._rows.items())

    def _is_prunable(self, job_id, row):
        batch = self._parent_batch.get(job_id)
        return (self._status_of(row) not in _ACTIVE_STATUSES and
                (batch is None or self._status_of(self._rows[batch]) not in _ACTIVE_STATUSES))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = {}     # job_id -> row index
        self._badges = {}   # job_id -> 状态徽章 QLabel（复用，不重建）
        self._bars = {}
        self._cancel_pending = set()
        self._statuses = None
        self._actions = {}
        self._batches = {}
        self._parent_batch = {}
        self._expanded_batches = set()

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
        self._empty_label = HintLabel('暂无任务', empty_page)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self._empty_label)
        self._stack = QStackedLayout()
        self._stack.addWidget(empty_page)   # index 0 = 空态
        self._stack.addWidget(self._table)  # index 1 = 表格
        layout.addLayout(self._stack)

    def _update_empty_state(self) -> None:
        status_messages = {
            ('queued', 'running', 'failed'): '没有排队、运行中或失败的任务',
            ('completed', 'cancelled', 'failed'): '无已结束任务',
        }
        filtered = status_messages.get(tuple(self._statuses or ()), '无匹配任务')
        self._empty_label.setText(filtered if self._rows and self._statuses is not None else '暂无任务')
        self._stack.setCurrentIndex(int(any(not self._table.isRowHidden(row)
                                            for row in self._rows.values())))

    def status_count(self, *statuses) -> int:
        return sum(self._status_of(row) in statuses for job_id, row in self._rows.items()
                   if job_id not in self._batches)

    def set_visible_statuses(self, statuses) -> None:
        self._statuses = statuses
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        for job_id, row in self._rows.items():
            batch = self._parent_batch.get(job_id)
            scope = self._rows[batch] if batch is not None else row
            hidden = (batch is not None and batch not in self._expanded_batches
                      or self._statuses is not None and self._status_of(scope) not in self._statuses)
            self._table.setRowHidden(row, hidden)
        self._update_empty_state()

    def apply_theme(self, dark: bool) -> None:
        """主题切换：徽章配色按新主题重刷（空态 HintLabel 自刷，主窗口遍历调用）。"""
        for job_id, badge in self._badges.items():
            _set_status_badge(badge, self._display_status(job_id))

    # ------------------------------------------------------------- 接口
    def upsert_job(self, job_id: str, title: str, *, before=None) -> None:
        if job_id in self._rows:
            self._table.item(self._rows[job_id], self._COL_TITLE).setText(title)
            return
        row = self._table.rowCount() if before is None else self._rows[before]
        self._table.insertRow(row)
        if before is not None:
            self._rows = {key: index + int(index >= row) for key, index in self._rows.items()}
        self._rows[job_id] = row
        title_item = QTableWidgetItem(title)
        title_item.setToolTip(title)
        self._table.setItem(row, self._COL_TITLE, title_item)
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

        # The view controls its index widget's visibility when scrolling and
        # filtering. Keep the actual bar inside it so unknown totals stay blank.
        progress_cell = QWidget(self._table)
        progress_layout = QHBoxLayout(progress_cell)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        bar = ProgressBar(progress_cell)
        progress_layout.addWidget(bar)
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.hide()
        self._bars[job_id] = bar
        self._table.setCellWidget(row, self._COL_PROGRESS, progress_cell)
        self._table.setItem(row, self._COL_MESSAGE, QTableWidgetItem(''))
        cancel_btn = PushButton('取消', self._table)
        cancel_btn.clicked.connect(
            lambda _checked=False, jid=job_id: self.activate_job(jid))
        self._table.setCellWidget(row, self._COL_ACTION, cancel_btn)
        self._badges[job_id] = badge
        self._refresh_rows()
        self.rows_changed.emit()

    def update_progress(self, job_id: str, completed: int, total: int,
                        message: str) -> None:
        row = self._rows.get(job_id)
        if row is None:
            return
        bar = self._bars[job_id]
        active = self._status_of(row) in _ACTIVE_STATUSES
        set_job_progress(bar, completed, total, active=active)
        item = self._table.item(row, self._COL_MESSAGE)
        item.setText(progress_caption(completed, total, message) if active else message or '')
        item.setToolTip(message or '')
        item.setData(Qt.ItemDataRole.UserRole, message or '')

    def set_status(self, job_id: str, status: str) -> None:
        row = self._rows.get(job_id)
        if row is None:
            return
        item = self._table.item(row, self._COL_STATUS)
        item.setData(Qt.ItemDataRole.UserRole, status)
        if status not in _ACTIVE_STATUSES:
            self._cancel_pending.discard(job_id)
            set_job_progress(self._bars[job_id], 0, 0)
            message = self._table.item(row, self._COL_MESSAGE)
            message.setText(message.data(Qt.ItemDataRole.UserRole) or '')
        _set_status_badge(self._badges[job_id], self._display_status(job_id))
        self._sync_action(job_id)
        if job_id in self._parent_batch:
            self._update_batch(self._parent_batch[job_id])
        self._refresh_rows()
        self.rows_changed.emit()

    def _display_status(self, job_id):
        return 'cancelling' if job_id in self._cancel_pending else self._status_of(self._rows[job_id])

    def set_cancel_pending(self, job_id):
        if self.can_cancel(job_id):
            self._cancel_pending.add(job_id)
            _set_status_badge(self._badges[job_id], 'cancelling')
            self._sync_action(job_id)
            self.rows_changed.emit()

    def set_job_action(self, job_id: str, label: str) -> None:
        if job_id in self._rows:
            self._actions[job_id] = label
            self._sync_action(job_id)

    def _sync_action(self, job_id):
        row = self._rows[job_id]
        active = self.can_cancel(job_id)
        if job_id in self._batches:
            label = '收起测线' if job_id in self._expanded_batches else '查看测线'
        else:
            label = '取消' if active or job_id in self._cancel_pending else self._actions.get(job_id, '')
        button = self._table.cellWidget(row, self._COL_ACTION)
        button.setText(label)
        button.setVisible(bool(label))
        button.setEnabled(bool(label) and job_id not in self._cancel_pending)

    def activate_job(self, job_id):
        if job_id in self._batches:
            self._expanded_batches.symmetric_difference_update({job_id})
            self._sync_action(job_id)
            self._refresh_rows()
        elif self.can_cancel(job_id):
            self.cancel_job(job_id)
        elif job_id in self._rows and self._actions.get(job_id):
            self.action_requested.emit(job_id)

    def add_batch(self, batch_id, job_ids):
        members = tuple(job_id for job_id in job_ids if job_id in self._rows)
        if not members:
            return
        self._batches[batch_id] = members
        self._parent_batch.update(dict.fromkeys(members, batch_id))
        before = min(members, key=self._rows.__getitem__)
        self.upsert_job(batch_id, f'批量处理 · {len(members)} 条测线', before=before)
        self._update_batch(batch_id)

    def _update_batch(self, batch_id):
        members = self._batches[batch_id]
        statuses = [self._status_of(self._rows[job_id]) for job_id in members]
        active = sum(status in _ACTIVE_STATUSES for status in statuses)
        failed = statuses.count('failed')
        done = len(members) - active
        counts = [f'{statuses.count("completed")} 成功']
        if failed:
            counts.append(f'{failed} 失败')
        if 'cancelled' in statuses:
            counts.append(f'{statuses.count("cancelled")} 已取消')
        status = next(state for state in ('running', 'queued', 'failed', 'completed', 'cancelled')
                      if state in statuses)
        self.update_progress(batch_id, done, len(members), ' · '.join(counts))
        self.set_status(batch_id, status)

    def clear_finished(self) -> None:
        """移除终态行（供"清理已完成"按钮）。

        先收集要删的行号、按行号倒序 removeRow（倒序保证前排删除
        不影响后排行号），再按可视顺序重建 _rows 映射——循环中边删边用
        旧映射会让非相邻多任务的行号漂移，删错/删不掉。
        """
        finished = sorted(
            ((job_id, row) for job_id, row in self._rows.items()
             if self._is_prunable(job_id, row)),
            key=lambda pair: pair[1], reverse=True)
        if not finished:
            return
        for job_id, row in finished:
            self._table.removeRow(row)
            self._rows.pop(job_id, None)
            self._badges.pop(job_id, None)
            self._bars.pop(job_id, None)
            self._cancel_pending.discard(job_id)
            self._actions.pop(job_id, None)
            self._batches.pop(job_id, None)
            self._parent_batch.pop(job_id, None)
            self._expanded_batches.discard(job_id)
        # 删除后幸存行的相对顺序不变，按旧行号升序重排即为新行号
        survivors = sorted(self._rows.items(), key=lambda kv: kv[1])
        self._rows = {job_id: index for index, (job_id, _old) in
                      enumerate(survivors)}
        self._refresh_rows()
        self.rows_changed.emit()

    def remove_inactive(self) -> None:
        """与 MiniJobList 同构的清理接口：JobHub 对三视图统一分发用。"""
        self.clear_finished()

    def context_table(self):
        return self._table

    def job_id_at(self, row: int) -> str:
        return next((job_id for job_id, index in self._rows.items() if index == row), '')

    def job_snapshot(self, job_id: str) -> dict:
        row = self._rows.get(job_id)
        if row is None:
            return {}
        title = self._table.item(row, self._COL_TITLE)
        message = self._table.item(row, self._COL_MESSAGE)
        status = str(self._status_of(row) or '')
        return {
            'job_id': job_id,
            'title': title.text() if title is not None else '',
            'message': (message.data(Qt.ItemDataRole.UserRole) or '') if message is not None else '',
            'status': status,
            'status_text': self._badges[job_id].text(),
        }

    def can_cancel(self, job_id: str) -> bool:
        row = self._rows.get(job_id)
        return (row is not None and job_id not in self._batches and job_id not in self._cancel_pending
                and self._status_of(row) in _ACTIVE_STATUSES)

    def cancel_job(self, job_id: str) -> None:
        if self.can_cancel(job_id):
            self.cancel_requested.emit(job_id)

    # ------------------------------------------------------------- 内部
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
        style_transparent_scroll(self._scroll)
        self._container = QWidget(self._scroll)
        self._container.setStyleSheet('background-color: transparent;')
        self._box = QVBoxLayout(self._container)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(6)
        # P2-6：无活动任务时空态占位
        self._empty_label = HintLabel('暂无任务', self._container)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        bar.hide()
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
        set_job_progress(bar, completed, total, active=entry['status'] in _ACTIVE_STATUSES)
        if message:
            entry['title_label'].setToolTip(message)

    def set_status(self, job_id: str, status: str) -> None:
        entry = self._jobs.get(job_id)
        if entry is None:
            return
        entry['status'] = status
        _set_status_badge(entry['badge'], status)
        if status not in _ACTIVE_STATUSES:
            set_job_progress(entry['bar'], 0, 0)
        entry['cancel'].setEnabled(status in _ACTIVE_STATUSES)
        self._refresh_visibility()

    def remove_inactive(self) -> None:
        """移除终态任务行。"""
        for job_id in [j for j, e in self._jobs.items()
                       if e['status'] not in _ACTIVE_STATUSES]:
            self._remove_row(job_id)
        self._refresh_visibility()

    def apply_theme(self, dark: bool) -> None:
        """主题切换：徽章配色按新主题重刷（空态 HintLabel 自刷，主窗口遍历调用）。"""
        for entry in self._jobs.values():
            _set_status_badge(entry['badge'], entry['status'])

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

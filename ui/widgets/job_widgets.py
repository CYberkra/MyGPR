"""JobTable / MiniJobList — 任务状态控件（SPEC §5.6）。

状态中文映射：queued 排队 / running 运行中 / completed 已完成 /
failed 失败 / cancelled 已取消。
徽章配色：白字彩底，颜色随主题查表（ui.theme_helpers.status_color，
深色取同色相高亮度变体，保证深底对比度）；running 状态显示进度条。
"""

import re

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QHeaderView, QLabel,
                             QStackedLayout, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)
from qfluentwidgets import CaptionLabel, ProgressBar, PushButton, ScrollArea

from ui.motion import animate_badge_color, animate_progress
from ui.page_scaffold import style_transparent_scroll
from ui.theme_helpers import BADGE_QSS, status_color
from ui.widgets.context_menus import FIF, add_action, make_menu

_STATUS_TEXT = {
    'queued': '排队',
    'running': '运行中',
    'completed': '已完成',
    'failed': '失败',
    'cancelled': '已取消',
}

# 状态 → status_color 语义键（queued/cancelled 灰色同 disabled）
_STATUS_COLOR_KEY = {
    'queued': 'disabled',
    'running': 'info',
    'completed': 'success',
    'failed': 'error',
    'cancelled': 'disabled',
}

_ACTIVE_STATUSES = ('queued', 'running')

_EMPTY_LABEL_QSS = 'color: %s; font-size: 13px;'


def _status_badge_color(status: str) -> str:
    """状态徽章底色（随主题）：未知状态回落 disabled 灰。"""
    return status_color(_STATUS_COLOR_KEY.get(status, 'disabled'))


def _make_status_badge(status: str) -> QLabel:
    badge = QLabel(_STATUS_TEXT.get(status, status))
    badge.setStyleSheet(BADGE_QSS % _status_badge_color(status))
    return badge


def _restyle_status_badge(badge: QLabel, status: str) -> None:
    """不换文字只按状态重刷徽章配色（主题切换路径用，无渐变动画）。"""
    badge.setStyleSheet(BADGE_QSS % _status_badge_color(status))


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
        self._table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(
            self._on_context_menu)
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
        empty_label.setStyleSheet(
            _EMPTY_LABEL_QSS % status_color('disabled'))
        empty_layout.addWidget(empty_label)
        self._empty_label = empty_label
        self._stack = QStackedLayout()
        self._stack.addWidget(empty_page)   # index 0 = 空态
        self._stack.addWidget(self._table)  # index 1 = 表格
        layout.addLayout(self._stack)

    def _update_empty_state(self) -> None:
        """无任务时显示空态占位。"""
        self._stack.setCurrentIndex(0 if not self._rows else 1)

    def apply_theme(self, dark: bool) -> None:
        """主题切换：徽章与空态占位文字色按新主题重刷（主窗口遍历调用）。"""
        self._empty_label.setStyleSheet(
            _EMPTY_LABEL_QSS % status_color('disabled'))
        for job_id, badge in self._badges.items():
            _restyle_status_badge(badge, self._status_of(self._rows[job_id]))

    # ------------------------------------------------------------- 接口
    def upsert_job(self, job_id: str, title: str) -> None:
        if job_id in self._rows:
            self._table.item(self._rows[job_id], self._COL_TITLE).setText(title)
            return
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._rows[job_id] = row
        title_item = QTableWidgetItem(title)
        title_item.setToolTip(title)
        self._table.setItem(row, self._COL_TITLE, title_item)
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
        message_item = self._table.item(row, self._COL_MESSAGE)
        message_item.setText(message or '')
        message_item.setToolTip(message or '')

    def set_status(self, job_id: str, status: str) -> None:
        row = self._rows.get(job_id)
        if row is None:
            return
        badge = self._badges.get(job_id)
        if badge is not None:
            end_hex = _status_badge_color(status)
            badge.setText(_STATUS_TEXT.get(status, status))
            # 从徽章当前背景色渐变到新状态色（qss 模板同源，见 ui.motion）
            match = re.search(r'background-color:\s*(#[0-9a-fA-F]{6})',
                              badge.styleSheet())
            start_hex = match.group(1) if match else _status_badge_color(status)
            animate_badge_color(badge, BADGE_QSS, start_hex, end_hex)
        item = self._table.item(row, self._COL_STATUS)
        if item is not None:
            item.setData(Qt.ItemDataRole.UserRole, status)
        bar = self._table.cellWidget(row, self._COL_PROGRESS)
        bar.setVisible(status == 'running')
        cancel_btn = self._table.cellWidget(row, self._COL_ACTION)
        cancel_btn.setEnabled(status in _ACTIVE_STATUSES)

    def clear_finished(self) -> None:
        """移除终态行（供"清理已完成"按钮）。

        先收集要删的行号、按行号倒序 removeRow（倒序保证前排删除
        不影响后排行号），再按可视顺序重建 _rows 映射——循环中边删边用
        旧映射会让非相邻多任务的行号漂移，删错/删不掉。
        """
        finished = sorted(
            ((job_id, row) for job_id, row in self._rows.items()
             if self._status_of(row) not in _ACTIVE_STATUSES),
            key=lambda pair: pair[1], reverse=True)
        if not finished:
            return
        for job_id, row in finished:
            self._table.removeRow(row)
            self._rows.pop(job_id, None)
            self._badges.pop(job_id, None)
        # 删除后幸存行的相对顺序不变，按旧行号升序重排即为新行号
        survivors = sorted(self._rows.items(), key=lambda kv: kv[1])
        self._rows = {job_id: index for index, (job_id, _old) in
                      enumerate(survivors)}
        self._update_empty_state()

    def remove_inactive(self) -> None:
        """与 MiniJobList 同构的清理接口：JobHub 对三视图统一分发用。"""
        self.clear_finished()

    def active_job_ids(self) -> list[str]:
        """仍在排队/运行的任务 id 列表（主窗口退出前检查用）。"""
        return [job_id for job_id, row in self._rows.items()
                if self._status_of(row) in _ACTIVE_STATUSES]

    def focus_job(self, job_id: str) -> None:
        """选中并滚动到指定任务行（迷你任务列表点击定位用）。"""
        row = self._rows.get(str(job_id))
        if row is None:
            return
        self._table.selectRow(row)
        item = self._table.item(row, self._COL_TITLE)
        if item is not None:
            self._table.scrollToItem(item)

    # ------------------------------------------------------------- 内部
    def _status_of(self, row):
        item = self._table.item(row, self._COL_STATUS)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _job_id_of(self, row: int):
        for job_id, r in self._rows.items():
            if r == row:
                return job_id
        return None

    def _copy_text(self, text: str) -> None:
        if text:
            QApplication.clipboard().setText(text)

    def _on_context_menu(self, pos) -> None:
        """任务行右键：复制标题/消息（失败任务可复制错误信息）、取消、清理。"""
        row = self._table.rowAt(pos.y())
        job_id = self._job_id_of(row) if row >= 0 else None
        menu = self._build_context_menu(job_id, row)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _build_context_menu(self, job_id, row: int):
        """构造右键菜单（与 exec 分离，便于测试检查动作）。"""
        menu = make_menu(parent=self._table)
        if job_id is not None:
            title_item = self._table.item(row, self._COL_TITLE)
            message_item = self._table.item(row, self._COL_MESSAGE)
            title = title_item.text() if title_item is not None else ''
            message = message_item.text() if message_item is not None else ''
            add_action(menu, FIF.COPY, '复制标题',
                       lambda: self._copy_text(title))
            add_action(menu, FIF.COPY, '复制消息',
                       lambda: self._copy_text(message),
                       enabled=bool(message))
            add_action(menu, FIF.CANCEL, '取消任务',
                       lambda: self.cancel_requested.emit(job_id),
                       enabled=self._status_of(row) in _ACTIVE_STATUSES)
            menu.addSeparator()
        add_action(menu, FIF.DELETE, '清理已完成', self.clear_finished)
        return menu


class MiniJobList(QWidget):
    """右侧折叠面板"任务"tab：仅显示活动任务（标题 + 进度条 + 状态）。

    任务行可点击（除取消按钮外的区域）：发 ``job_clicked``，由接线器
    跳任务页并定位该任务（JobTable.focus_job）。
    """

    cancel_requested = pyqtSignal(str)
    job_clicked = pyqtSignal(str)
    # 空白区右键「打开任务中心」→ 仅跳页不定位（JobHub.on_open_job_center）
    open_job_center_requested = pyqtSignal()

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
        self._empty_label = CaptionLabel('暂无任务', self._container)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            _EMPTY_LABEL_QSS % status_color('disabled'))
        self._box.addWidget(self._empty_label)
        self._box.addStretch(1)
        self._scroll.setWidget(self._container)

        # 右键：任务行上 = 打开任务中心/复制标题/取消；空白区 = 打开任务中心
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

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
        # 行可点击：跳任务页定位；取消按钮自行消费点击，不走这里
        row_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        row_widget.setToolTip('点击跳转到任务中心')
        row_widget.installEventFilter(self)
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
        old_hex = _status_badge_color(entry['status'])
        entry['status'] = status
        badge = entry['badge']
        end_hex = _status_badge_color(status)
        badge.setText(_STATUS_TEXT.get(status, status))
        animate_badge_color(badge, BADGE_QSS, old_hex, end_hex)
        entry['cancel'].setEnabled(status in _ACTIVE_STATUSES)
        self._refresh_visibility()

    def remove_inactive(self) -> None:
        """移除终态任务行。"""
        for job_id in [j for j, e in self._jobs.items()
                       if e['status'] not in _ACTIVE_STATUSES]:
            self._remove_row(job_id)
        self._refresh_visibility()

    def apply_theme(self, dark: bool) -> None:
        """主题切换：徽章与空态占位文字色按新主题重刷（主窗口遍历调用）。"""
        self._empty_label.setStyleSheet(
            _EMPTY_LABEL_QSS % status_color('disabled'))
        for entry in self._jobs.values():
            _restyle_status_badge(entry['badge'], entry['status'])

    # ------------------------------------------------------------- 内部
    def eventFilter(self, watched, event):
        """任务行左键点击 → job_clicked（取消按钮自行消费，不到这里）。"""
        if (event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton):
            for job_id, entry in self._jobs.items():
                if entry['widget'] is watched:
                    self.job_clicked.emit(job_id)
                    break
        return super().eventFilter(watched, event)

    def _remove_row(self, job_id):
        entry = self._jobs.pop(job_id, None)
        if entry is None:
            return
        self._box.removeWidget(entry['widget'])
        entry['widget'].deleteLater()

    # ------------------------------------------------------------- 右键菜单
    def _on_context_menu(self, pos) -> None:
        """迷你列表右键：行上走任务动作，空白区只留「打开任务中心」。"""
        menu = self._build_context_menu(self._job_id_at(pos))
        menu.exec(self.mapToGlobal(pos))

    def _job_id_at(self, pos):
        """pos（本控件坐标）命中的任务行 id；空白区返回 None。"""
        container_pos = self._container.mapFrom(self, pos)
        for job_id, entry in self._jobs.items():
            widget = entry['widget']
            if widget.isVisible() and widget.geometry().contains(container_pos):
                return job_id
        return None

    def _build_context_menu(self, job_id):
        """构造右键菜单（与 exec 分离，便于测试检查动作）。

        行上：打开任务中心（定位该任务）/ 复制标题 / 取消；
        空白区：仅「打开任务中心」（仅跳页，走 open_job_center_requested）。
        """
        menu = make_menu(parent=self)
        if job_id is not None:
            entry = self._jobs[job_id]
            add_action(menu, FIF.LINK, '打开任务中心',
                       lambda: self.job_clicked.emit(job_id))
            add_action(
                menu, FIF.COPY, '复制任务标题',
                lambda: QApplication.clipboard().setText(
                    entry['title_label'].text()))
            add_action(menu, FIF.CANCEL, '取消任务',
                       lambda: self.cancel_requested.emit(job_id),
                       enabled=entry['status'] in _ACTIVE_STATUSES)
        else:
            add_action(menu, FIF.LINK, '打开任务中心',
                       self.open_job_center_requested.emit)
        return menu

    def _refresh_visibility(self):
        """仅显示活动任务；无活动任务时显示空态占位。"""
        any_active = False
        for entry in self._jobs.values():
            active = entry['status'] in _ACTIVE_STATUSES
            entry['widget'].setVisible(active)
            any_active = any_active or active
        self._empty_label.setVisible(not any_active)

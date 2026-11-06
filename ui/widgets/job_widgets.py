"""MiniJobList — 任务状态控件（SPEC §5.6）。

状态中文映射：queued 排队 / running 运行中 / completed 已完成 /
failed 失败 / cancelled 已取消。
徽章配色：白字彩底，颜色随主题查表（ui.theme_helpers.status_color，
深色取同色相高亮度变体，保证深底对比度）；running 状态显示进度条。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import ProgressBar, PushButton, ScrollArea

from ui.motion import animate_badge_color, animate_progress
from ui.page_scaffold import HintLabel, style_transparent_scroll
from ui.theme_helpers import BADGE_QSS, status_color

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


def _status_badge_color(status: str) -> str:
    """状态徽章底色（随主题）：未知状态回落 disabled 灰。"""
    return status_color(_STATUS_COLOR_KEY.get(status, 'disabled'))


def _make_status_badge(status: str) -> QLabel:
    badge = QLabel(_STATUS_TEXT.get(status, status))
    badge.setStyleSheet(BADGE_QSS % _status_badge_color(status))
    return badge


def _restyle_status_badge(badge: QLabel, status: str) -> None:
    """不换文字只按状态重刷徽章配色（主题切换路径用，无渐变动画）。"""
    color = _status_badge_color(status)
    animate_badge_color(badge, BADGE_QSS, color, color)


class MiniJobList(QWidget):
    """Compact task rows with optional, bounded completion history."""

    cancel_requested = pyqtSignal(str)

    def __init__(self, parent=None, *, history_limit: int = 0):
        super().__init__(parent)
        self._jobs = {}
        self._history_limit = history_limit
        self._recent: dict[str, None] = {}

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
        details = PushButton(row_widget)
        top.addWidget(details)
        details.hide()
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

        self._box.insertWidget(self._box.count() - 1 - len(self._recent), row_widget)
        self._jobs[job_id] = {
            'widget': row_widget, 'title_label': title_label,
            'badge': status_badge, 'bar': bar, 'cancel': cancel_btn, 'details': details,
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
        active = status in _ACTIVE_STATUSES
        entry['cancel'].setEnabled(active)
        entry['cancel'].setVisible(active)
        entry['bar'].setVisible(active)
        self._refresh_visibility()

    def finish_job(self, job_id: str, message: str, action=None) -> None:
        entry = self._jobs.get(job_id)
        if entry is None or not self._history_limit or job_id in self._recent:
            return
        if entry['status'] == 'failed':
            from ui.widgets.object_context_menu import show_details

            title = entry['title_label'].text()
            action = ('详情…', lambda: show_details(self, title, message or '任务执行失败'))
        if action is not None:
            label, callback = action
            button = entry['details']
            button.setText(label)
            button.clicked.connect(callback)
            button.show()
        self._recent[job_id] = None
        self._box.removeWidget(entry['widget'])
        self._box.insertWidget(self._box.count() - len(self._recent), entry['widget'])
        if len(self._recent) > self._history_limit:
            oldest = next(iter(self._recent))
            self._recent.pop(oldest)
            self._remove_row(oldest)
        self._refresh_visibility()

    def remove_inactive(self) -> None:
        """移除终态任务行。"""
        for job_id in [j for j, e in self._jobs.items()
                       if e['status'] not in _ACTIVE_STATUSES and j not in self._recent]:
            self._remove_row(job_id)
        self._refresh_visibility()

    def apply_theme(self, dark: bool) -> None:
        """主题切换：徽章配色按新主题重刷（空态 HintLabel 自刷，主窗口遍历调用）。"""
        for entry in self._jobs.values():
            _restyle_status_badge(entry['badge'], entry['status'])

    # ------------------------------------------------------------- 内部
    def _remove_row(self, job_id):
        entry = self._jobs.pop(job_id, None)
        if entry is None:
            return
        self._box.removeWidget(entry['widget'])
        entry['widget'].deleteLater()

    def _refresh_visibility(self):
        """Show active rows and retained results."""
        any_visible = False
        for job_id, entry in self._jobs.items():
            visible = entry['status'] in _ACTIVE_STATUSES or job_id in self._recent
            entry['widget'].setVisible(visible)
            any_visible = any_visible or visible
        self._empty_label.setVisible(not any_visible)

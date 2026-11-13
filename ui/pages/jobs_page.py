# -*- coding: utf-8 -*-
"""Job controls hosted in the output panel."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import ComboBox, PushButton

from ui import constants
from ui.widgets import JobTable
from ui.widgets.object_context_menu import ObjectMenuBinding, object_action, object_menu, show_details


class JobsPage(QWidget):
    """任务中心页面。"""

    cancel_requested = pyqtSignal(str)   # job_id
    prune_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect_internal()

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 2, 8, 4)
        root.setSpacing(4)

        button_row = QHBoxLayout()
        button_row.setSpacing(constants.CARD_SPACING)
        self._filter = ComboBox(self)
        self._filter.addItem('进行中与失败', userData=('queued', 'running', 'failed'))
        self._filter.addItem('全部任务', userData=None)
        self._filter.addItem('已结束', userData=('completed', 'cancelled', 'failed'))
        button_row.addWidget(self._filter)
        button_row.addStretch(1)
        self._prune_btn = PushButton('清理已结束记录', self)
        self._prune_btn.setToolTip('从列表清除已结束任务记录')
        button_row.addWidget(self._prune_btn)
        root.addLayout(button_row)
        self._job_table = JobTable(self)
        root.addWidget(self._job_table, 1)

    # ============================================================ 内部接线
    def _connect_internal(self) -> None:
        self._job_context = ObjectMenuBinding(self, self._job_table.context_table(), self._job_menu)
        self._job_table.rows_changed.connect(self._job_context.close)
        self._job_table.cancel_requested.connect(self.cancel_requested)
        self._prune_btn.clicked.connect(self._on_prune_clicked)
        self._filter.currentIndexChanged.connect(self._update_filter)
        self._update_filter()
        # 没有终态任务时按钮禁用：否则点了没反应（空表/全是运行中）
        self._job_table.rows_changed.connect(self._sync_prune_enabled)
        self._sync_prune_enabled()

    # ============================================================ 公共接口（供主窗口接线）
    def job_table(self) -> JobTable:
        """JobTable 访问器：主窗口经其 upsert/update_progress/set_status。"""
        return self._job_table

    def _update_filter(self):
        self._job_table.set_visible_statuses(self._filter.currentData())

    # ============================================================ 内部逻辑
    def _sync_prune_enabled(self) -> None:
        """按终态任务数同步「清理已完成」可用态（无可清理 → 禁用）。"""
        counter = getattr(self._job_table, 'finished_count', None)
        count = int(counter()) if callable(counter) else 0
        self._prune_btn.setEnabled(count > 0)
        self._prune_btn.setVisible(count > 0)

    def _on_prune_clicked(self) -> None:
        """清理已完成：本地移除终态行并发 prune_requested。"""
        self._job_table.clear_finished()
        self.prune_requested.emit()

    def _job_menu(self, pos):
        menu = object_menu(self)
        row = self._job_table.context_table().indexAt(pos).row()
        job_id = self._job_table.job_id_at(row)

        if job_id:
            snapshot = self._job_table.job_snapshot(job_id)
            title = snapshot['title']
            message = snapshot['message']
            status = snapshot['status_text']
            text = f'{title}\n状态：{status}\n任务编号：{job_id}'
            if message:
                text += f'\n\n{message}'
            object_action(menu, 'job-details', '任务详情…',
                          lambda: show_details(self, '任务详情', text))
            object_action(menu, 'copy-id', '复制任务编号',
                          lambda: QApplication.clipboard().setText(job_id))
            object_action(menu, 'copy-message', '复制当前消息',
                          lambda: QApplication.clipboard().setText(message),
                          enabled=bool(message), reason='此任务尚无消息')

            object_action(menu, 'cancel-job', '取消任务',
                          lambda: self._job_table.cancel_job(job_id),
                          enabled=self._job_table.can_cancel(job_id), reason='任务已结束或不可取消')
        return menu

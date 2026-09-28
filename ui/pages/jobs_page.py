# -*- coding: utf-8 -*-
"""JobsPage — 任务中心（SPEC §6.8）。

卡内：按钮行 PushButton('清理已完成') + JobTable(stretch)。
页面为 JobTable 的薄包装：暴露 job_table() 访问器，
转发 cancel_requested，并在清理时发 prune_requested。
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import PushButton

from ui import constants
from ui.page_scaffold import make_card
from ui.widgets import JobTable


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
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)

        card, card_layout = make_card('任务记录', parent=self)

        button_row = QHBoxLayout()
        button_row.setSpacing(constants.CARD_SPACING)
        self._prune_btn = PushButton('清理已完成', card)
        self._prune_btn.setToolTip('清除已完成/失败的任务行')
        button_row.addWidget(self._prune_btn)
        button_row.addStretch(1)
        card_layout.addLayout(button_row)

        self._job_table = JobTable(card)
        card_layout.addWidget(self._job_table, 1)
        root.addWidget(card, 1)

    # ============================================================ 内部接线
    def _connect_internal(self) -> None:
        self._job_table.cancel_requested.connect(self.cancel_requested)
        self._prune_btn.clicked.connect(self._on_prune_clicked)
        # 没有终态任务时按钮禁用：否则点了没反应（空表/全是运行中）
        self._job_table.rows_changed.connect(self._sync_prune_enabled)
        self._sync_prune_enabled()

    # ============================================================ 公共接口（供主窗口接线）
    def job_table(self) -> JobTable:
        """JobTable 访问器：主窗口经其 upsert/update_progress/set_status。"""
        return self._job_table

    # ============================================================ 内部逻辑
    def _sync_prune_enabled(self) -> None:
        """按终态任务数同步「清理已完成」可用态（无可清理 → 禁用）。"""
        counter = getattr(self._job_table, 'finished_count', None)
        count = int(counter()) if callable(counter) else 0
        self._prune_btn.setEnabled(count > 0)

    def _on_prune_clicked(self) -> None:
        """清理已完成：本地移除终态行并发 prune_requested。"""
        self._job_table.clear_finished()
        self.prune_requested.emit()

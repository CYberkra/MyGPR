"""LogPanel — 右侧全局折叠面板容器（SPEC §5.7）。

CardWidget 容器 max 380px/min 0，margins 6/spacing 6；
顶部 SegmentedWidget("日志","任务") + QStackedWidget；
日志 tab：QTextEdit 只读（QSS 逐字照 SPEC §1）+ 按钮行（"清空"宽60）；
任务 tab：MiniJobList。
append_log 自动加 [HH:MM:SS] 前缀 + 级别关键词着色 + 滚到底。
"""

from datetime import datetime

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QStackedWidget, QTextEdit, QVBoxLayout
from qfluentwidgets import CardWidget, PushButton, SegmentedWidget
from ui import constants, file_dialogs

from .job_widgets import MiniJobList

# SPEC §1 日志 QSS（浅色主题下也用深底），逐字
_LOG_QSS_INITIAL = """QTextEdit {
    background-color: #2b2b2b;
    color: #e0e0e0;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 5px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}"""

_LOG_QSS_DARK = """QTextEdit {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 5px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}"""

_LOG_QSS_LIGHT = """QTextEdit {
    background-color: #f5f5f5;
    color: #333;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 5px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}"""

# 级别关键词 → 颜色（SPEC §1 关键词规则 + constants 语义色单轨，任务 F 候选 4）
_LEVEL_RULES = (
    (('ERROR', '失败', '错误'), constants.LOG_COLOR_ERROR),
    (('WARNING', '警告'), constants.LOG_COLOR_WARNING),
    (('SUCCESS', '成功', '完成'), constants.LOG_COLOR_SUCCESS),
    (('INFO',), constants.LOG_COLOR_INFO),
)


def _level_color(msg: str):
    upper = msg.upper()
    for keywords, color in _LEVEL_RULES:
        for kw in keywords:
            if kw in msg or kw in upper:
                return color
    return None


class LogPanel(CardWidget):
    """右侧全局折叠面板：日志 / 任务 两个 tab。"""

    cancel_job_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.setMaximumWidth(380)
        self.setMinimumHeight(400)

        # 顶部：SegmentedWidget("日志","任务") + QStackedWidget
        self._segmented = SegmentedWidget(self)
        self._stacked = QStackedWidget(self)
        self._stacked.setMinimumWidth(364)

        # 日志 tab
        self._log_edit = QTextEdit(self)
        self._log_edit.setReadOnly(True)
        self._log_edit.setStyleSheet(_LOG_QSS_INITIAL)
        clear_btn = PushButton('清空', self)
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._log_edit.clear)
        export_btn = PushButton('导出…', self)
        export_btn.setFixedWidth(70)
        export_btn.setToolTip('日志文本保存到文件')
        export_btn.clicked.connect(self._export_log)

        log_page = CardWidget(self)
        log_layout = QVBoxLayout(log_page)
        log_layout.setContentsMargins(6, 6, 6, 6)
        log_layout.setSpacing(6)
        log_layout.addWidget(self._log_edit, 1)
        btn_row = QHBoxLayout()
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(export_btn)
        btn_row.addStretch(1)
        log_layout.addLayout(btn_row)

        # 任务 tab
        self._mini_jobs = MiniJobList(self)
        self._mini_jobs.cancel_requested.connect(self.cancel_job_requested)

        self._stacked.addWidget(log_page)
        self._stacked.addWidget(self._mini_jobs)

        self._segmented.addItem(routeKey='log', text='日志',
                                onClick=lambda: self._stacked.setCurrentIndex(0))
        self._segmented.addItem(routeKey='jobs', text='任务',
                                onClick=lambda: self._stacked.setCurrentIndex(1))
        self._segmented.setCurrentItem('log')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self._segmented)
        layout.addWidget(self._stacked, 1)

    # ------------------------------------------------------------- 接口
    def append_log(self, msg: str) -> None:
        """自动加 [HH:MM:SS] 前缀 + 级别着色 + 滚到底。"""
        stamp = datetime.now().strftime('[%H:%M:%S]')
        text = '%s %s' % (stamp, msg)
        color = _level_color(msg)
        if color:
            html = ('<span style="color:%s;">%s</span>'
                    % (color, self._escape(text)))
        else:
            html = self._escape(text)
        self._log_edit.append(html)
        cursor = self._log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_edit.setTextCursor(cursor)
        self._log_edit.ensureCursorVisible()

    def mini_jobs(self) -> MiniJobList:
        return self._mini_jobs

    def _export_log(self) -> None:
        """日志全文保存为 .txt（纯文本，不含着色标记）。"""
        path, _selected = file_dialogs.getSaveFileName(
            self, '导出日志',
            'mygpr_log_%s.txt' % datetime.now().strftime('%Y%m%d_%H%M%S'),
            '文本文件 (*.txt)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(self._log_edit.toPlainText())
        except OSError as exc:
            self.append_log('WARNING 日志导出失败: %s' % exc)

    def apply_theme(self, dark: bool) -> None:
        """主题换肤：深色 #1e1e1e/#333；浅色 #f5f5f5/#ddd/#333。"""
        self._log_edit.setStyleSheet(
            _LOG_QSS_DARK if dark else _LOG_QSS_LIGHT)

    @staticmethod
    def _escape(text: str) -> str:
        return (text.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;'))

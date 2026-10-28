"""Trace-aligned sensor availability, shown without implying field accuracy."""
import json

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui import constants
from ui.design_tokens import color, radius, rgba
from ui.sensor_summary import SensorCompletenessReport
from ui.theme_helpers import ui_font


_STATES = {
    'complete': ('完整', 'success'),
    'partial': ('部分有效', 'warning'),
    'invalid': ('无有效值', 'warning'),
    'missing': ('未提供', 'text_muted'),
    'incomplete': ('字段不全', 'warning'),
    'unreadable': ('字段不可读', 'error'),
    'mismatch': ('长度不符', 'warning'),
}
_LABELS = {'RTK经纬度': 'RTK 经纬度', '高度': '高度计', 'IMU姿态': 'IMU 姿态',
           'IMU加速度': 'IMU 加速度', 'IMU角速度': 'IMU 角速度', '采集时间': '采集时间戳'}
_NOTE = '有效率仅表示字段可用，不代表 RTK 固定解、精度或时间同步已验收。'


class SensorDetailsPanel(QWidget):
    """Read-only summary rows. No metadata reads, interpolation or text parsing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dark = False
        self._summary_text = ''
        self._message_error = False
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(constants.SPACE_3)
        self._message = self._label('')
        column.addWidget(self._message)
        self._title = self._label('')
        column.addWidget(self._title)
        self._table = QWidget(self)
        self._grid = QGridLayout(self._table)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(constants.SPACE_4)
        self._grid.setVerticalSpacing(constants.SPACE_1)
        self._headers = []
        for col, text in enumerate(('传感器 / 信息', '状态', '有效道数', '缺失 / 无效')):
            label = self._label(text)
            if col >= 2:
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._headers.append(label)
            self._grid.addWidget(label, 0, col)
            self._grid.setColumnStretch(col, (3, 2, 4, 2)[col])
        self._cells = []
        self._separators = []
        for _ in range(6):
            self._append_row()
        column.addWidget(self._table)
        self._note = self._label(_NOTE)
        column.addWidget(self._note)
        self._diagnostics_label = self._label('')
        self._diagnostics_label.hide()
        column.addWidget(self._diagnostics_label)
        self.set_message('选择测线后可查看传感器完整性。')
        self.apply_theme(False)

    def _label(self, text):
        label = QLabel(text, self)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                      Qt.TextInteractionFlag.TextSelectableByKeyboard)
        return label

    def _append_row(self):
        index = len(self._cells)
        cells = tuple(self._label('') for _ in range(4))
        cells[1].setAlignment(Qt.AlignmentFlag.AlignCenter)
        for col, label in enumerate(cells):
            if col >= 2:
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label.setMinimumHeight(constants.BTN_HEIGHT)
            self._grid.addWidget(label, 2 * index + 1, col)
        separator = QFrame(self)
        separator.setFixedHeight(1)
        self._grid.addWidget(separator, 2 * index + 2, 0, 1, 4)
        self._cells.append(cells)
        self._separators.append(separator)

    def _clear_rows(self):
        for cells in self._cells:
            for label in cells:
                label.clear()
                label.setToolTip('')
                label.setProperty('sensorState', '')
                label.hide()
        for separator in self._separators:
            separator.hide()

    def set_message(self, text: str, *, error: bool = False):
        self._summary_text = str(text)
        self._message_error = bool(error)
        self._message.setText(self._summary_text)
        self._message.show()
        self._title.clear()
        self._clear_rows()
        for widget in (self._title, self._table, self._note):
            widget.hide()
        self.setAccessibleDescription(self._summary_text)
        self.apply_theme(self._dark)

    def set_summary(self, line_id: str, report: SensorCompletenessReport | str | None):
        heading = f'当前测线：{line_id}'
        if report is None:
            self.set_message(f'{heading}\n正在检查传感器完整性…')
            return
        if isinstance(report, str):
            # Keep text-only clients compatible, without reconstructing values from prose.
            self.set_message(f'{heading}\n{report}')
            return
        summary = f'{heading}\n{report.as_text()}'
        if report.error or not report.rows:
            self.set_message(summary, error=bool(report.error))
            return
        self._summary_text = summary
        self._message.clear()
        self._message.hide()
        self._title.setText(f'{heading} · 共 {report.total} 道')
        for widget in (self._title, self._table, self._note):
            widget.show()
        self._clear_rows()
        while len(self._cells) < len(report.rows):
            self._append_row()
        for index, row in enumerate(report.rows):
            cells = self._cells[index]
            status, _tone = _STATES[row.state]
            if row.valid_count is None:
                valid = '—' if row.state in ('missing', 'incomplete') else '无法统计'
            else:
                valid = f'{row.valid_count} / {row.total}（{row.valid_count / row.total:.1%}）'
            missing = f'{row.missing_count} 道' if row.missing_count is not None else '无法统计'
            values = (_LABELS.get(row.label, row.label), status, valid, missing)
            for label, text in zip(cells, values):
                label.setText(text)
                label.setToolTip(text)
                label.show()
            cells[1].setProperty('sensorState', row.state)
            self._separators[index].show()
        self.setAccessibleDescription(summary)
        self.apply_theme(self._dark)

    def set_sync_result(self, result):
        if result is None:
            self._diagnostics_label.clear()
            self._diagnostics_label.hide()
            return
        summary = str(getattr(result, 'summary', '') or '')
        diagnostics = getattr(result, 'diagnostics', {}) or {}
        config = getattr(result, 'config', {}) or {}
        text = '同步诊断与本次配置\n' + (summary + '\n' if summary else '')
        if diagnostics:
            text += json.dumps(dict(diagnostics), ensure_ascii=False, indent=2, default=str)
        if config:
            text += '\n配置\n' + json.dumps(dict(config), ensure_ascii=False, indent=2, default=str)
        self._diagnostics_label.setText(text.strip())
        self._diagnostics_label.show()

    def text(self) -> str:
        """Plain-text equivalent for accessibility and existing summary consumers."""
        return self._summary_text

    def apply_theme(self, dark: bool):
        self._dark = bool(dark)
        for label in (*self._headers, self._note):
            label.setFont(ui_font(constants.FONT_SIZE_SECONDARY))
            label.setStyleSheet(f'color: {color("text_muted", dark)}; background: transparent;')
        self._title.setFont(ui_font(constants.FONT_SIZE_BODY, QFont.Weight.DemiBold))
        self._title.setStyleSheet(f'color: {color("text_primary", dark)}; background: transparent;')
        self._message.setFont(ui_font(constants.FONT_SIZE_BODY))
        self._message.setStyleSheet(
            f'color: {color("error" if self._message_error else "text_secondary", dark)}; '
            'background: transparent;')
        for cells in self._cells:
            for col, label in enumerate(cells):
                label.setFont(ui_font(constants.FONT_SIZE_BODY))
                if col == 1:
                    state = str(label.property('sensorState') or '')
                    tone = _STATES.get(state, ('', 'text_muted'))[1]
                    label.setFont(ui_font(constants.FONT_SIZE_SECONDARY, QFont.Weight.DemiBold))
                    label.setStyleSheet(
                        f'color: {color(tone, dark)}; background: {rgba(tone, 0.10, dark)}; '
                        f'border-radius: {radius("sm")}px; padding: 0 {constants.SPACE_1}px;')
                else:
                    label.setStyleSheet(
                        f'color: {color("text_primary" if col == 0 else "text_secondary", dark)}; '
                        'background: transparent;')
        for separator in self._separators:
            separator.setStyleSheet(f'background: {color("border_default", dark)};')

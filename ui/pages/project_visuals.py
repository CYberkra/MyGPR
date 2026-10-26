"""Presentation helpers for the project ledger, details and semantic theme.

These helpers only configure widgets; project state and signals stay on the page.
"""
import ntpath

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QFrame, QHeaderView, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QTableWidget, QVBoxLayout, QWidget,
)

from ui import constants
from ui.design_tokens import color, radius
from ui.page_scaffold import style_transparent_scroll
from ui.theme_helpers import ui_font


class LineDetailsPanel(QWidget):
    """Compact selected-line metadata; paths remain exact, selectable plain text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('lineDetailsPanel')
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(constants.SPACE_4)
        self._message = self._label('选择一条测线')
        column.addWidget(self._message)
        self._metrics = QWidget(self)
        row = QHBoxLayout(self._metrics)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(constants.SPACE_6)
        self._captions = []
        self._values = []
        for caption in ('测线号', '采样点数', '雷达道数', '数据检查'):
            field = QWidget(self._metrics)
            field.setMinimumWidth(0)
            field_column = QVBoxLayout(field)
            field_column.setContentsMargins(0, 0, 0, 0)
            field_column.setSpacing(constants.SPACE_1)
            title = self._label(caption, field)
            value = self._label('', field, selectable=True)
            field_column.addWidget(title)
            field_column.addWidget(value)
            row.addWidget(field, 1)
            self._captions.append(title)
            self._values.append(value)
        column.addWidget(self._metrics)
        self._separator = QFrame(self)
        self._separator.setFixedHeight(1)
        column.addWidget(self._separator)
        self._source = QWidget(self)
        source_column = QVBoxLayout(self._source)
        source_column.setContentsMargins(0, 0, 0, 0)
        source_column.setSpacing(constants.SPACE_1)
        self._source_caption = self._label('数据来源', self._source)
        self._source_name = self._label('', self._source, selectable=True)
        self._source_path = self._label('', self._source, selectable=True)
        source_column.addWidget(self._source_caption)
        source_column.addWidget(self._source_name)
        source_column.addWidget(self._source_path)
        column.addWidget(self._source)
        self.set_message('选择一条测线')

    def _label(self, text, parent=None, *, selectable=False):
        label = QLabel(text, parent or self)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        if selectable:
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse |
                Qt.TextInteractionFlag.TextSelectableByKeyboard)
        return label

    def set_message(self, text: str) -> None:
        self._message.setText(text)
        self._message.show()
        for value in self._values:
            value.clear()
        self._source_name.clear()
        self._source_path.clear()
        self._source_path.setToolTip('')
        for widget in (self._metrics, self._separator, self._source):
            widget.hide()

    def set_line(self, line_id, samples, traces, quality, source, *, source_ready: bool) -> None:
        self._message.hide()
        for widget in (self._metrics, self._separator, self._source):
            widget.show()
        for label, value in zip(self._values, (line_id, samples, traces, quality)):
            label.setText(str(value))
        path = str(source or '')
        self._source_name.setText(
            ntpath.basename(path) if path else
            ('未记录数据来源' if source_ready else '正在查询数据来源…'))
        self._source_path.setText(path)
        self._source_path.setToolTip(path)
        self._source_path.setVisible(bool(path))

    def apply_theme(self, dark: bool) -> None:
        for label in (*self._captions, self._source_caption, self._message):
            label.setFont(ui_font(constants.FONT_SIZE_SECONDARY))
            label.setStyleSheet(f'color: {color("text_muted", dark)}; background: transparent;')
        for label in self._values:
            label.setFont(ui_font(constants.FONT_SIZE_SECTION, QFont.Weight.DemiBold))
            label.setStyleSheet(f'color: {color("text_primary", dark)}; background: transparent;')
        self._source_name.setFont(ui_font(constants.FONT_SIZE_BODY, QFont.Weight.DemiBold))
        self._source_name.setStyleSheet(
            f'color: {color("text_primary", dark)}; background: transparent;')
        self._source_path.setFont(ui_font(constants.FONT_SIZE_SECONDARY))
        self._source_path.setStyleSheet(
            f'color: {color("text_secondary", dark)}; background: transparent;')
        self._separator.setStyleSheet(f'background: {color("border_default", dark)};')


def configure_ledger_table(table: QTableWidget, *, artifacts: bool = False) -> None:
    table.setObjectName('projectArtifactsTable' if artifacts else 'projectLinesTable')
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(constants.BTN_HEIGHT + constants.SPACE_1)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setMinimumSectionSize(constants.SPACE_6 * 2)
    header.setSectionResizeMode(0 if artifacts else 1, QHeaderView.ResizeMode.Stretch)
    if artifacts:
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(5, 56)
    widths = ((1, 120), (2, 96), (3, 160), (4, 88)) if artifacts else (
        (0, 76), (2, 64), (3, 72), (4, 80), (5, 96), (6, 104))
    for column, width in widths:
        table.setColumnWidth(column, width)


def make_detail_scroll(label: QWidget, parent: QWidget) -> QScrollArea:
    """Keep full source paths and scientific caveats reachable in short panes."""
    scroll = QScrollArea(parent)
    style_transparent_scroll(scroll)
    scroll.viewport().setObjectName('projectDetailViewport')
    content = QWidget(scroll)
    content.setObjectName('projectDetailContent')
    column = QVBoxLayout(content)
    column.setContentsMargins(*(constants.SPACE_2,) * 4)
    column.addWidget(label)
    column.addStretch(1)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    scroll.setWidget(content)
    # setWidget enables autoFillBackground on its child. Disable it afterwards:
    # otherwise a system-dark Window palette paints a dark slab in light mode.
    content.setAutoFillBackground(False)
    scroll.viewport().setAutoFillBackground(False)
    content.setStyleSheet('QWidget#projectDetailContent { background: transparent; }')
    scroll.viewport().setStyleSheet(
        'QWidget#projectDetailViewport { background: transparent; }')
    return scroll


def apply_project_theme(page, dark: bool) -> None:
    """Use the workbench palette for the page's existing presentation widgets."""
    page._dark = bool(dark)
    heading_qss = (f'font-size: {constants.FONT_SIZE_BODY}pt; font-weight: 600; '
                   f'color: {color("text_primary", dark)};')
    page._project_heading.setStyleSheet(heading_qss)
    page._selected_heading.setStyleSheet(heading_qss)
    page._line_detail.apply_theme(dark)
    page._sensor_detail.apply_theme(dark)
    for table in (page._lines_table, page._artifacts_table):
        table.setStyleSheet(
            f'QTableWidget {{ background: {color("bg_surface", dark)}; '
            f'color: {color("text_primary", dark)}; '
            f'border: 1px solid {color("border_default", dark)}; '
            f'border-radius: {radius("sm")}px; '
            f'font-size: {constants.FONT_SIZE_BODY}pt; '
            f'selection-background-color: {color("primary_subtle", dark)}; '
            f'selection-color: {color("text_primary", dark)}; }}'
            f'QTableWidget::item {{ padding: 0 {constants.SPACE_2}px; border: none; }}'
            f'QTableWidget::item:selected {{ background: {color("primary_subtle", dark)}; '
            f'color: {color("text_primary", dark)}; border: none; }}'
            f'QHeaderView::section {{ background: {color("bg_subtle", dark)}; '
            f'color: {color("text_muted", dark)}; border: none; '
            f'border-bottom: 1px solid {color("border_default", dark)}; '
            f'padding: {constants.SPACE_1}px {constants.SPACE_2}px; '
            f'font-size: {constants.FONT_SIZE_SECONDARY}pt; }}')
    page._detail_tabs.setStyleSheet(
        f'QTabWidget#projectDetailTabs::pane {{ border: none; '
        f'border-top: 1px solid {color("border_default", dark)}; }}'
        f'QTabBar::tab {{ background: transparent; color: {color("text_muted", dark)}; '
        f'font-size: {constants.FONT_SIZE_SECONDARY}pt; '
        f'padding: {constants.SPACE_2}px {constants.SPACE_3}px; '
        f'border: none; border-bottom: 2px solid transparent; }}'
        f'QTabBar::tab:selected {{ color: {color("text_primary", dark)}; '
        f'border-bottom: 2px solid {color("primary", dark)}; }}'
        f'QTabBar::tab:hover {{ background: {color("bg_subtle", dark)}; }}')
    page._empty_project.setStyleSheet(
        f'BodyLabel#projectEmptyLabel {{ color: {color("text_muted", dark)}; '
        f'font-size: {constants.FONT_SIZE_BODY}pt; }}')
    style_ledger_items(page._lines_table, page._artifacts_table, dark)


def style_ledger_items(lines_table: QTableWidget, artifacts_table: QTableWidget,
                       dark: bool) -> None:
    for table in (lines_table, artifacts_table):
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                item = table.item(row, col)
                if item is None:
                    continue
                item.setToolTip(item.text())
                font = QFont(table.font())
                font.setWeight(QFont.Weight.DemiBold if col == 0 and table is lines_table
                               else QFont.Weight.Normal)
                numeric = table is lines_table and col in (2, 3, 4)
                if numeric:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    font.setPointSize(constants.FONT_SIZE_SECONDARY)
                item.setFont(font)
                item.setForeground(QBrush(QColor(color(
                    'text_muted' if numeric or not item.flags() else 'text_primary', dark))))

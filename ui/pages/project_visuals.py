"""Presentation helpers for the project ledger, details and semantic theme.

These helpers only configure widgets; project state and signals stay on the page.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import QHeaderView, QScrollArea, QSizePolicy, QTableWidget, QVBoxLayout, QWidget

from ui import constants
from ui.design_tokens import color, radius
from ui.page_scaffold import style_transparent_scroll


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
    content = QWidget(scroll)
    column = QVBoxLayout(content)
    column.setContentsMargins(0, 0, 0, 0)
    column.addWidget(label)
    column.addStretch(1)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    scroll.setWidget(content)
    return scroll


def apply_project_theme(page, dark: bool) -> None:
    """Use the workbench palette for the page's existing presentation widgets."""
    page._dark = bool(dark)
    heading_qss = (f'font-size: {constants.FONT_SIZE_BODY}pt; font-weight: 600; '
                   f'color: {color("text_primary", dark)};')
    page._project_heading.setStyleSheet(heading_qss)
    page._selected_heading.setStyleSheet(heading_qss)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable table helpers for field-workbench pages."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QSizePolicy, QTableWidget, QTableWidgetItem


ROW_BG = QColor("#FFFFFF")
ALT_ROW_BG = QColor("#F8FBFD")
SELECTED_ROW_BG = QColor("#DDF4F7")
TEXT_FG = QColor("#243447")
STATUS_OK = QColor("#16A05D")
STATUS_WARN = QColor("#B7791F")
STATUS_BAD = QColor("#E5484D")
MUTED_FG = QColor("#6B7D90")


class FieldTableMixin:
    """Small table factory/fill helpers shared by multiple field panels.

    The table helper intentionally sets explicit item foreground/background
    colors.  Native Windows inactive-selection palettes can otherwise render
    selected rows as a dark block in screenshots, which makes engineering data
    tables look disabled or erroneous.  State is carried primarily by text/icon
    color instead of heavy full-row fills.
    """

    def _table(self, headers: list[str], rows: int) -> QTableWidget:
        table = QTableWidget(rows, len(headers))
        table.setObjectName("dataTable")
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Keep inactive Windows focus from painting a heavy dark current row.
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setAlternatingRowColors(False)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.setMinimumHeight(70)
        table.setMinimumWidth(0)
        table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        table.horizontalHeader().setMinimumSectionSize(36)
        table.verticalHeader().setDefaultSectionSize(22)
        table.horizontalHeader().setFixedHeight(25)
        return table

    def _fill_table(self, table: QTableWidget, rows: list[tuple], *, highlight_row: int | None = None, sort_column: int | None = None) -> None:
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                text = str(val)
                item = QTableWidgetItem(text)
                if sort_column is None:
                    base_bg = SELECTED_ROW_BG if highlight_row is not None and r == highlight_row else (ALT_ROW_BG if r % 2 else ROW_BG)
                    item.setBackground(base_bg)
                item.setForeground(TEXT_FG)
                if "●" in text and ("通过" in text or "固定" in text or "完成" in text or "已" in text):
                    item.setForeground(STATUS_OK)
                elif "⚠" in text or "浮动" in text or "待" in text:
                    item.setForeground(STATUS_WARN)
                elif "✕" in text or "失败" in text or "错误" in text:
                    item.setForeground(STATUS_BAD)
                table.setItem(r, c, item)
        for row in range(table.rowCount()):
            table.setRowHeight(row, 22)
        if sort_column is not None:
            table.setSortingEnabled(True)
            table.sortItems(sort_column, Qt.SortOrder.AscendingOrder)
            table.setAlternatingRowColors(True)


__all__ = ["FieldTableMixin"]

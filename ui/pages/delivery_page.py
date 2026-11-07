# -*- coding: utf-8 -*-
"""Project spatial results and version management."""

import os
import json
from collections.abc import Mapping

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QHeaderView, QListWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    InfoBar, InfoBarPosition, LineEdit, PrimaryPushButton, ScrollArea,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.page_scaffold import (make_card, make_form_row, make_hint,
                              rebuild_check_list, style_transparent_scroll,
                              wrap_centered)
from ui.widgets import EmptyStateOverlay
from ui.widgets import (clear_invalid, make_separator, mark_invalid,
                        validate_non_empty)
from ui.widgets.object_context_menu import ObjectMenuBinding, object_action, object_menu, show_details

def _get(obj, key, default=''):
    """鸭子类型取值：dict 键优先，其次对象属性。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _line_display(line) -> str:
    """测线显示文本：'名称 (线号)'，名称缺失或与线号相同时只显示线号。"""
    line_id = str(_get(line, 'line_id', '') or _get(line, 'id', ''))
    name = str(_get(line, 'name', '') or line_id)
    return '%s (%s)' % (name, line_id) if name != line_id else line_id


class DeliveryPage(QWidget):
    """成果与交付页面。"""

    spatial_requested = pyqtSignal(dict)   # {'name': str, 'line_ids': list[str]}
    spatial_export_requested = pyqtSignal(dict)
    spatial_delete_requested = pyqtSignal(str)
    spatial_current_requested = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._spatial_results = []
        self._spatial_revision = 0
        self._project_root = ''
        self._build_ui()
        self._connect_internal()

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        scroll = ScrollArea(self)
        style_transparent_scroll(scroll)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(scroll)
        content.setObjectName('deliveryScrollContent')
        content.setStyleSheet(
            'QWidget#deliveryScrollContent { background-color: transparent; }')
        root = QVBoxLayout(content)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)
        scroll.setWidget(wrap_centered(content, constants.FORM_COLUMN_MAX_WIDTH))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._build_spatial_card(root)
        root.addStretch(1)

    def _build_spatial_card(self, root: QVBoxLayout) -> None:
        """卡片1：空间成果（主操作进卡头行，P2-3）。"""
        self._spatial_btn = PrimaryPushButton('生成空间成果', self)
        # 未就绪门控：无测线时禁用，set_lines 导入测线后点亮
        self._spatial_btn.setEnabled(False)
        self._spatial_btn.setToolTip('导入测线后可用')
        spatial_card, spatial_layout = make_card(
            '空间成果', parent=self, header_action=self._spatial_btn)
        self._spatial_name_edit = LineEdit(spatial_card)
        self._spatial_name_edit.setPlaceholderText('例如：全场剖面拼接成果')
        spatial_layout.addLayout(make_form_row(
            '成果名称:', self._spatial_name_edit, parent=spatial_card,
            trailing_stretch=False))

        self._lines_list = QListWidget(spatial_card)
        self._lines_list.setAccessibleName('选择测线，可多选')
        self._lines_list.setStyleSheet(
            'QListWidget::indicator { width: 14px; height: 14px; border-radius: 3px; '
            'border: 1px solid palette(mid); background-color: palette(base); }'
            'QListWidget::indicator:checked { border: 1px solid palette(highlight); '
            'background-color: palette(highlight); }'
        )
        self._lines_list.setMinimumHeight(120)
        spatial_layout.addWidget(self._lines_list)
        self._lines_empty_hint = make_hint('暂无可选测线', parent=spatial_card)
        self._lines_empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spatial_layout.addWidget(self._lines_empty_hint)
        self._lines_list.setVisible(False)

        spatial_layout.addWidget(make_separator())

        self._spatial_table = QTableWidget(0, 4, spatial_card)
        self._spatial_table.setHorizontalHeaderLabels(
            ['名称', '测线数', '创建时间', '状态'])
        self._spatial_table.verticalHeader().setVisible(False)
        self._spatial_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._spatial_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._spatial_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents)
        self._spatial_table.setMinimumHeight(140)
        spatial_layout.addWidget(self._spatial_table)
        self._spatial_empty = EmptyStateOverlay(
            self._spatial_table, icon=FIF.SEND, title='暂无空间成果')
        root.addWidget(spatial_card)

    def _connect_internal(self) -> None:
        self._spatial_context = ObjectMenuBinding(self, self._spatial_table, self._spatial_result_menu)
        self._spatial_btn.clicked.connect(self._on_spatial_clicked)
    # ============================================================ 公共接口（供主窗口接线）
    def set_spatial_results(self, results: list) -> None:
        """空间成果列表 → 结果表格（名称/测线数/创建时间）。"""
        results = list(results or [])
        self._spatial_context.close()
        self._spatial_revision += 1
        self._spatial_results = results
        self._spatial_empty.setVisible(not results)
        self._spatial_table.setRowCount(len(results))
        for row, item in enumerate(results):
            name = str(_get(item, 'name', '') or _get(item, 'title', ''))
            line_ids = _get(item, 'line_ids', None)
            if line_ids is None:
                n_lines = _get(item, 'line_count', 0)
            else:
                n_lines = len(line_ids)
            created = str(_get(item, 'created_at', '')
                          or _get(item, 'created', '') or '--')
            self._spatial_table.setItem(row, 0, QTableWidgetItem(name))
            self._spatial_table.setItem(row, 1, QTableWidgetItem(str(n_lines)))
            self._spatial_table.setItem(row, 2, QTableWidgetItem(created))
            status = str(_get(item, 'status', '') or '')
            options = _get(item, 'options', {}) or {}
            files = _get(item, 'files', {}) or {}
            summary = _get(item, 'summary', {}) or {}
            warnings = list(summary.get('preflight_warnings', []) or [])
            text = '带警告' if status == 'generated_with_warnings' or warnings else '已生成'
            if 'amplitude_geotiff' in files:
                text += ' · 振幅切片'
            elif 'surface_control_points_geojson' in files:
                text += ' · 已生成曲面控制点'
            if 'amplitude_geotiff' not in files and (options.get('surface_generated') is False or (files and 'surface_mesh' not in files)):
                text += ' · 未生成曲面'
            status_item = QTableWidgetItem(text)
            status_item.setToolTip('\n'.join(map(str, warnings)))
            self._spatial_table.setItem(row, 3, status_item)
            self._spatial_table.item(row, 0).setData(
                Qt.ItemDataRole.UserRole, str(_get(item, 'result_id', '') or _get(item, 'artifact_id', '')))

    def _spatial_result_menu(self, pos):
        menu = object_menu(self)
        row = self._spatial_table.indexAt(pos).row()
        if not 0 <= row < len(self._spatial_results):
            object_action(menu, 'copy-count', '复制成果数量',
                          lambda: QApplication.clipboard().setText(str(len(self._spatial_results))))
            return menu
        result, revision = self._spatial_results[row], self._spatial_revision
        result_id = str(_get(result, 'result_id', '') or _get(result, 'artifact_id', ''))

        def valid():
            return (revision == self._spatial_revision and row < len(self._spatial_results)
                    and self._spatial_results[row] is result)

        metadata = {}
        for key in ('result_id', 'name', 'revision', 'status', 'line_ids', 'created_at',
                    'coordinate_system', 'vertical_datum', 'stale', 'files', 'options', 'sources'):
            value = _get(result, key, '')
            metadata[key] = dict(value) if isinstance(value, Mapping) else value
        text = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
        object_action(menu, 'result-details', '参数与来源…',
                      lambda: show_details(self, '空间成果详情', text) if valid() else None)
        object_action(menu, 'copy-id', '复制成果标识',
                      lambda: QApplication.clipboard().setText(result_id) if valid() else None,
                      enabled=bool(result_id), reason='记录没有稳定成果标识')
        object_action(menu, 'copy-name', '复制成果名称',
                      lambda: QApplication.clipboard().setText(str(_get(result, 'name', ''))) if valid() else None)
        menu.addSeparator()
        object_action(menu, 'set-current-result', '设为当前空间成果',
                      lambda: self.spatial_current_requested.emit(result_id) if valid() else None,
                      enabled=bool(result_id))
        object_action(menu, 'open-result', '打开成果目录',
                      lambda: self._open_spatial_result(result_id) if valid() else None,
                      enabled=bool(result_id and self._project_root))
        object_action(menu, 'export-result', '导出成果…',
                      lambda: self._export_spatial_result(result_id) if valid() else None,
                      enabled=bool(result_id))
        object_action(menu, 'delete-result', '永久删除空间成果…',
                      lambda: self._delete_spatial_result(result_id) if valid() else None,
                      enabled=bool(result_id and not self._busy))
        return menu

    def set_project_root(self, root):
        self._project_root = os.path.normpath(str(root or '')) if root else ''

    def _open_spatial_result(self, result_id):
        path = os.path.join(self._project_root, 'spatial', 'results', result_id)
        if os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _export_spatial_result(self, result_id):
        from ui.dialogs import choose_spatial_export_path
        path, format_name = choose_spatial_export_path(self, result_id)
        if path:
            self.spatial_export_requested.emit({
                'result_id': result_id,
                'destination': path,
                'format_name': format_name,
            })

    def _delete_spatial_result(self, result_id):
        self.spatial_delete_requested.emit(result_id)

    def set_busy(self, busy: bool) -> None:
        """忙态：禁用全部操作按钮。"""
        self._busy = bool(busy)
        self._update_spatial_btn_enabled()

    def _update_spatial_btn_enabled(self) -> None:
        """「生成空间成果」双门控：非忙态且有可选测线（P2-3 未就绪禁用）。"""
        self._spatial_btn.setEnabled(
            not self._busy and self._lines_list.count() > 0)

    def selected_line_ids(self) -> list:
        """当前勾选的测线 id 列表。"""
        ids = []
        for i in range(self._lines_list.count()):
            item = self._lines_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def spatial_name(self) -> str:
        return self._spatial_name_edit.text().strip()

    def set_lines(self, lines: list) -> None:
        """可选测线列表（主窗口注入；dict/对象鸭子类型，取 line_id 与 name）。

        重建列表时保持已有勾选状态（rebuild_check_list 同 spatial_page
        模式），避免刷新测线集合后丢掉用户已勾选的测线；新出现的测线维持
        默认不勾选。
        """
        rebuild_check_list(
            self._lines_list, lines or [],
            key_fn=lambda line: str(
                _get(line, 'line_id', '') or _get(line, 'id', '')),
            text_fn=_line_display,
            default_checked=False)
        # 空态显隐：有测线显示勾选列表，无测线显示引导文案
        has_lines = self._lines_list.count() > 0
        self._lines_list.setVisible(has_lines)
        self._lines_empty_hint.setVisible(not has_lines)
        self._update_spatial_btn_enabled()

    # ============================================================ 内部逻辑
    def _on_spatial_clicked(self) -> None:
        if self._busy:
            return
        ok, msg = validate_non_empty(self._spatial_name_edit.text(),
                                     '成果名称')
        if not ok:
            mark_invalid(self._spatial_name_edit, msg)
            InfoBar.warning(title='空间成果', content=msg,
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        clear_invalid(self._spatial_name_edit)
        line_ids = self.selected_line_ids()
        if not line_ids:
            InfoBar.warning(title='空间成果', content='请至少勾选一条测线',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        self.spatial_requested.emit(
            {'name': self.spatial_name(), 'line_ids': line_ids})

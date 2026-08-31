# -*- coding: utf-8 -*-
"""InterpretationPage — 界面解释标注（SPEC §6.6，UAV-GPR 解译工作流重设计）。

布局（保持既有美术风格：卡片 + 灰白底 + 圆角）：
- 顶部工具卡片（两行，防叠字）：
    行1 = 测线 / 数据（原始|成果）/ 打开标注会话 + 右侧会话状态
    行2 = 自动追踪 / 吸附 / 平滑 | 撤销 / 重做 | 保存标注 + 操作提示
- 主区左（stretch）：剖面标注卡片 = BScanView（pick 模式，点击追加点；
  overlay #fbbf24）。pick/overlay 均为原始数据坐标（BScanView 内部完成
  降采样坐标换算），与后端编辑会话坐标系一致。
- 主区右（固定 320px）：
    标注点列表卡片 = 表格(#/道/采样点/时间ns/估计深度m) + 删除选中/清空；
    深度换算卡片 = 介电常数 εr（默认 9.0），深度 = ½·c·t/√εr。
- 底部信息条：点数 / 会话状态。

页面纯展示 + 发信号：内部维护当前点列，pick 点击追加并发 points_changed；
删除/清空同样通过 points_changed → 控制器 replace_points 写回会话。
会话未打开时编辑按钮、点列管理与 pick 前置禁用（P1-6），避免"点了才报错"。
"""

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CaptionLabel, CardWidget, ComboBox, DoubleSpinBox, PrimaryPushButton,
    PushButton, SubtitleLabel,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.widgets import BScanView, make_separator

_OVERLAY_COLOR = constants.CHART_OVERLAY_COLOR   # 标注散点颜色（SPEC §6.6）
_C_M_PER_NS = 0.29979        # 真空光速 c (m/ns)


def _page_title(text: str) -> SubtitleLabel:
    """页面标题：SubtitleLabel 微软雅黑 12pt Bold 居中（SPEC §1）。"""
    label = SubtitleLabel(text)
    label.setFont(QFont(constants.FONT_FAMILY, 12, QFont.Weight.Bold))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def _card_title(text: str) -> SubtitleLabel:
    """卡片标题：SubtitleLabel 微软雅黑 10pt Bold（SPEC §1）。"""
    label = SubtitleLabel(text)
    label.setFont(QFont(constants.FONT_FAMILY, 10, QFont.Weight.Bold))
    return label


class InterpretationPage(QWidget):
    """界面解释标注页面。"""

    open_session_requested = pyqtSignal(str)  # artifact_id（''=原始数据）
    auto_trace_requested = pyqtSignal()
    snap_requested = pyqtSignal()
    smooth_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    save_requested = pyqtSignal()
    points_changed = pyqtSignal(list)      # 当前标注点列 [(trace, sample), ...]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = []           # [(trace_index, sample_index), ...] 原始数据坐标
        self._bundle = None         # 当前预览 bundle（取时间轴做深度换算）
        self._busy = False
        self._session_active = False  # 会话未打开时禁用编辑按钮与 pick
        self._build_ui()
        self._connect_internal()
        # 初始即应用"未开会话"禁用态（P1-6），不依赖外部首次调用
        self._update_edit_enabled()

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)
        root.addWidget(_page_title('界面解释标注'))
        root.addWidget(self._build_tool_card())

        body = QHBoxLayout()
        body.setSpacing(constants.PAGE_SPACING)
        body.addWidget(self._build_bscan_card(), 1)
        body.addWidget(self._build_side_column(), 0)
        root.addLayout(body, 1)

        root.addWidget(self._build_info_bar())

    def _build_tool_card(self) -> CardWidget:
        """顶部工具卡片：两行布局（行1 会话来源 / 行2 编辑操作）。"""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(*constants.CARD_MARGINS)
        layout.setSpacing(constants.CARD_SPACING)
        layout.addWidget(_card_title('标注工具'))

        # ---- 行1：会话来源（测线 / 数据版本 / 打开会话）+ 状态
        row1 = QHBoxLayout()
        row1.setSpacing(constants.CARD_SPACING)
        line_label = CaptionLabel('测线:', card)
        row1.addWidget(line_label)
        self._line_label = CaptionLabel('--', card)
        self._line_label.setStyleSheet('font-weight: bold;')
        row1.addWidget(self._line_label)
        row1.addSpacing(8)
        artifact_label = CaptionLabel('数据:', card)
        row1.addWidget(artifact_label)
        self._artifact_combo = ComboBox(card)
        self._artifact_combo.addItem('原始数据')
        self._artifact_combo.setCurrentIndex(0)
        self._artifact_combo.setMinimumWidth(180)
        self._artifact_combo.setToolTip('选择标注对象：原始数据或某个处理成果')
        row1.addWidget(self._artifact_combo)
        self._open_session_btn = PushButton('打开标注会话', card, FIF.EDIT)
        self._open_session_btn.setToolTip('在所选数据上打开标注会话')
        row1.addWidget(self._open_session_btn)
        row1.addStretch(1)
        status_label = CaptionLabel('状态:', card)
        row1.addWidget(status_label)
        self._session_status_label = CaptionLabel('未打开会话', card)
        self._session_status_label.setStyleSheet(
            'color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        row1.addWidget(self._session_status_label)
        layout.addLayout(row1)

        # ---- 行2：编辑操作（追踪/吸附/平滑 | 撤销/重做 | 保存）
        row2 = QHBoxLayout()
        row2.setSpacing(constants.CARD_SPACING)
        self._auto_trace_btn = PushButton('自动追踪', card, FIF.SEARCH)
        self._auto_trace_btn.setToolTip('基于已有标注点自动追踪同相轴')
        self._snap_btn = PushButton('吸附', card, FIF.PIN)
        self._snap_btn.setToolTip('把标注点吸附到邻近信号极值位置')
        self._smooth_btn = PushButton('平滑', card)
        self._smooth_btn.setToolTip('对标注点列做滑动中值平滑')
        for btn in (self._auto_trace_btn, self._snap_btn, self._smooth_btn):
            row2.addWidget(btn)
        row2.addWidget(make_separator(vertical=True))
        self._undo_btn = PushButton('撤销', card, FIF.CANCEL)
        self._redo_btn = PushButton('重做', card, FIF.SYNC)
        row2.addWidget(self._undo_btn)
        row2.addWidget(self._redo_btn)
        row2.addWidget(make_separator(vertical=True))
        self._save_btn = PrimaryPushButton('保存标注', card, FIF.SAVE)
        self._save_btn.setToolTip('保存当前标注点列到项目')
        row2.addWidget(self._save_btn)
        row2.addStretch(1)
        hint = CaptionLabel('提示：在剖面图上左键点击拾取标注点', card)
        hint.setStyleSheet(
            'color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        row2.addWidget(hint)
        layout.addLayout(row2)
        return card

    def _build_bscan_card(self) -> CardWidget:
        """剖面标注卡片：BScanView（pick 模式 + overlay）。"""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(*constants.CARD_MARGINS)
        layout.setSpacing(constants.CARD_SPACING)
        layout.addWidget(_card_title('剖面标注'))
        self._bscan = BScanView(card)
        self._bscan.setMinimumHeight(420)
        self._bscan.set_pick_enabled(False)  # 会话打开前禁用 pick（P1-6）
        layout.addWidget(self._bscan, 1)
        return card

    def _build_side_column(self) -> QWidget:
        """右列（固定 320px）：标注点列表 + 深度换算。"""
        column = QWidget(self)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(constants.PAGE_SPACING)
        column.setFixedWidth(320)

        # ---------------- 标注点列表
        points_card = CardWidget(column)
        points_layout = QVBoxLayout(points_card)
        points_layout.setContentsMargins(*constants.CARD_MARGINS)
        points_layout.setSpacing(constants.CARD_SPACING)
        header_row = QHBoxLayout()
        header_row.addWidget(_card_title('标注点列表'))
        header_row.addStretch(1)
        self._points_count_label = CaptionLabel('0 个点', points_card)
        self._points_count_label.setStyleSheet(
            'color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        header_row.addWidget(self._points_count_label)
        points_layout.addLayout(header_row)

        self._points_table = QTableWidget(0, 5, points_card)
        self._points_table.setHorizontalHeaderLabels(
            ['#', '道', '采样点', '时间(ns)', '深度(m)'])
        self._points_table.verticalHeader().setVisible(False)
        self._points_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._points_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._points_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        table_header = self._points_table.horizontalHeader()
        table_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._points_table.setMinimumHeight(180)
        points_layout.addWidget(self._points_table, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(constants.CARD_SPACING)
        self._remove_point_btn = PushButton('删除选中', points_card, FIF.DELETE)
        self._remove_point_btn.setToolTip('删除列表中选中的标注点')
        self._clear_points_btn = PushButton('清空', points_card)
        self._clear_points_btn.setToolTip('清空全部标注点')
        btn_row.addWidget(self._remove_point_btn, 1)
        btn_row.addWidget(self._clear_points_btn, 1)
        points_layout.addLayout(btn_row)
        layout.addWidget(points_card, 1)

        # ---------------- 深度换算
        depth_card = CardWidget(column)
        depth_layout = QVBoxLayout(depth_card)
        depth_layout.setContentsMargins(*constants.CARD_MARGINS)
        depth_layout.setSpacing(constants.CARD_SPACING)
        depth_layout.addWidget(_card_title('深度换算'))
        diel_row = QHBoxLayout()
        diel_row.setSpacing(constants.CARD_SPACING)
        diel_label = CaptionLabel('介电常数 εr:', depth_card)
        diel_row.addWidget(diel_label)
        self._dielectric_spin = DoubleSpinBox(depth_card)
        self._dielectric_spin.setRange(1.0, 81.0)
        self._dielectric_spin.setDecimals(1)
        self._dielectric_spin.setSingleStep(0.5)
        self._dielectric_spin.setValue(constants.DEFAULT_DIELECTRIC)
        self._dielectric_spin.setToolTip(
            '介质相对介电常数，用于把双程走时换算成估计深度\n'
            '常见介质：空气≈1，干砂≈3-5，湿砂≈20-30，水≈81')
        diel_row.addWidget(self._dielectric_spin, 1)
        depth_layout.addLayout(diel_row)
        formula_hint = CaptionLabel('深度 = ½ · c · t / √εr（c = 0.30 m/ns）',
                                    depth_card)
        formula_hint.setWordWrap(True)
        formula_hint.setStyleSheet(
            'color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        depth_layout.addWidget(formula_hint)
        layout.addWidget(depth_card)
        return column

    def _build_info_bar(self) -> CardWidget:
        """底部信息条：点数 / 会话状态。"""
        card = CardWidget(self)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(*constants.CARD_MARGINS)
        layout.setSpacing(constants.CARD_SPACING)
        self._info_label = CaptionLabel('标注点数: 0 | 会话状态: 未打开会话',
                                        card)
        layout.addWidget(self._info_label, 1)
        return card

    # ============================================================ 内部接线
    def _connect_internal(self) -> None:
        self._open_session_btn.clicked.connect(self._on_open_session_clicked)
        self._auto_trace_btn.clicked.connect(self.auto_trace_requested)
        self._snap_btn.clicked.connect(self.snap_requested)
        self._smooth_btn.clicked.connect(self.smooth_requested)
        self._undo_btn.clicked.connect(self.undo_requested)
        self._redo_btn.clicked.connect(self.redo_requested)
        self._save_btn.clicked.connect(self.save_requested)
        self._bscan.sig_point_picked.connect(self._on_point_picked)
        self._remove_point_btn.clicked.connect(self._on_remove_selected_point)
        self._clear_points_btn.clicked.connect(self._on_clear_points)
        self._dielectric_spin.valueChanged.connect(
            lambda _v: self._refresh_points_table())

    # ============================================================ 公共接口（供主窗口接线）
    def set_bundle(self, bundle) -> None:
        """剖面预览 bundle → BScanView；同时缓存时间轴用于深度换算。"""
        self._bundle = bundle
        self._bscan.set_bundle(bundle)
        self._refresh_points_table()

    def set_line_label(self, text: str) -> None:
        """当前测线标签（顶部工具卡片）。"""
        self._line_label.setText(text or '--')

    def set_artifacts(self, artifacts) -> None:
        """处理成果列表 → 数据下拉（原始数据 + 各成果）；保持旧选择，否则默认原始数据。"""
        current = self._current_artifact_id()
        self._artifact_combo.blockSignals(True)
        self._artifact_combo.clear()
        self._artifact_combo.addItem('原始数据')  # index 0 = 原始数据
        for artifact in (artifacts or []):
            artifact_id = str(getattr(artifact, 'artifact_id', '') or '')
            if not artifact_id:
                continue
            name = str(getattr(artifact, 'name', '') or artifact_id)
            self._artifact_combo.addItem(f'成果: {name}')
            self._artifact_combo.setItemData(self._artifact_combo.count() - 1, artifact_id)
        if current:
            found = self._index_of_artifact(current)
            self._artifact_combo.setCurrentIndex(found if found >= 0 else 0)
        else:
            self._artifact_combo.setCurrentIndex(0)
        self._artifact_combo.blockSignals(False)

    def set_session_info(self, text: str) -> None:
        """会话状态文案（顶部状态 CaptionLabel + 底部信息条）。"""
        text = text or '未打开会话'
        self._session_status_label.setText(text)
        self._refresh_info()

    def set_points(self, points: list) -> None:
        """整列替换标注点（原始数据坐标）→ overlay（#fbbf24）+ 点列表。"""
        self._points = [(int(t), int(s)) for t, s in (points or [])]
        self._bscan.set_overlay_points(self._points, _OVERLAY_COLOR)
        self._refresh_points_table()
        self._refresh_info()

    def set_busy(self, busy: bool) -> None:
        """忙态：禁用全部操作按钮；会话状态由 set_session_active 控制。"""
        self._busy = bool(busy)
        self._open_session_btn.setEnabled(not self._busy)
        self._update_edit_enabled()

    def set_session_active(self, active: bool) -> None:
        """会话状态：未开会话禁用编辑按钮与 pick（P1-6），避免"点了才报错"。"""
        self._session_active = bool(active)
        self._update_edit_enabled()

    # ============================================================ 内部逻辑
    def _on_open_session_clicked(self) -> None:
        """打开标注会话 → 携带当前数据选择（''=原始数据，否则为成果 artifact_id）。"""
        self.open_session_requested.emit(self._current_artifact_id())

    def _current_artifact_id(self) -> str:
        index = self._artifact_combo.currentIndex()
        if index <= 0:
            return ''
        return str(self._artifact_combo.itemData(index) or '')

    def _index_of_artifact(self, artifact_id: str) -> int:
        for i in range(1, self._artifact_combo.count()):
            if str(self._artifact_combo.itemData(i) or '') == artifact_id:
                return i
        return -1

    def _update_edit_enabled(self) -> None:
        enabled = self._session_active and not self._busy
        for btn in (self._auto_trace_btn, self._snap_btn, self._smooth_btn,
                    self._undo_btn, self._redo_btn, self._save_btn,
                    self._remove_point_btn, self._clear_points_btn):
            btn.setEnabled(enabled)
        self._bscan.set_pick_enabled(enabled)

    def _on_point_picked(self, trace: int, sample: int) -> None:
        """pick 点击追加点（原始数据坐标）→ overlay/列表刷新 + points_changed。"""
        self._points.append((int(trace), int(sample)))
        self._bscan.set_overlay_points(self._points, _OVERLAY_COLOR)
        self._refresh_points_table()
        self._refresh_info()
        self.points_changed.emit(list(self._points))

    def _on_remove_selected_point(self) -> None:
        """删除列表选中行对应的标注点。"""
        row = self._points_table.currentRow()
        if not (0 <= row < len(self._points)):
            return
        del self._points[row]
        self._emit_points_updated()

    def _on_clear_points(self) -> None:
        """清空全部标注点。"""
        if not self._points:
            return
        self._points = []
        self._emit_points_updated()

    def _emit_points_updated(self) -> None:
        """点列变更统一出口：overlay + 表格 + 信息条 + 信号。"""
        self._bscan.set_overlay_points(self._points, _OVERLAY_COLOR)
        self._refresh_points_table()
        self._refresh_info()
        self.points_changed.emit(list(self._points))

    # ---------------- 点列表与深度换算
    def _sample_time_ns(self, sample: int):
        """采样点 → 双程走时(ns)：优先用 bundle 时间轴（线性外推回原始坐标）。

        无时间轴（纯索引数据）时返回 None，表格对应列显示 '--'。
        """
        bundle = self._bundle
        if bundle is None:
            return None
        axis = getattr(bundle, 'sample_axis', None)
        if axis is None or len(axis) < 2:
            return None
        sample_count = int(getattr(bundle, 'sample_count', 0) or 0)
        if sample_count < 2:
            return None
        # 降采样轴近似线性：按原始样本序号在 [axis[0], axis[-1]] 上插值
        frac = min(max(sample / max(sample_count - 1, 1), 0.0), 1.0)
        return float(axis[0]) + frac * float(axis[-1] - axis[0])

    def _estimate_depth_m(self, time_ns: float):
        """双程走时(ns) → 估计深度(m)：d = ½·c·t/√εr。"""
        eps = float(self._dielectric_spin.value())
        if eps <= 0.0:
            return None
        return 0.5 * _C_M_PER_NS * float(time_ns) / math.sqrt(eps)

    def _refresh_points_table(self) -> None:
        """重建标注点表格（道/采样点/时间/深度列）。"""
        table = self._points_table
        table.blockSignals(True)
        table.setRowCount(0)
        for index, (trace, sample) in enumerate(self._points):
            row = table.rowCount()
            table.insertRow(row)
            time_ns = self._sample_time_ns(sample)
            depth_m = (self._estimate_depth_m(time_ns)
                       if time_ns is not None else None)
            values = (
                str(index + 1),
                str(trace + 1),
                str(sample + 1),
                ('%.2f' % time_ns) if time_ns is not None else '--',
                ('%.3f' % depth_m) if depth_m is not None else '--',
            )
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)
        table.blockSignals(False)
        self._points_count_label.setText('%d 个点' % len(self._points))

    def _refresh_info(self) -> None:
        self._info_label.setText(
            '标注点数: %d | 会话状态: %s'
            % (len(self._points), self._session_status_label.text()))

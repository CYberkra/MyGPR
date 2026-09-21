# -*- coding: utf-8 -*-
"""InterpretationPage — 界面解释标注（SPEC §6.6，UAV-GPR 解译工作流重设计）。

布局（保持既有美术风格：卡片 + 灰白底 + 圆角）：
- 顶部工具卡片（两行，防叠字）：
    行1 = 测线 / 数据（原始|成果）/ 打开标注会话 + 右侧会话状态
    行2 = 自动追踪 / 吸附 / 平滑 | 撤销 / 重做 | 保存标注 + 操作提示
- 主区左（stretch）：剖面标注卡片 = BScanView（pick 模式，点击追加点；
  overlay #fbbf24）。pick/overlay 均为原始数据坐标（BScanView 内部完成
  降采样坐标换算），与后端编辑会话坐标系一致。
- 主区右（可折叠侧栏，展开 SIDE_TOOL_WIDTH px）：
    标注点列表卡片 = 表格(#/道/采样点/时间ns/估计深度m) + 点数计数 +
    删除选中/清空（计数与按钮同行：计数居左、按钮居右）；
    深度换算卡片 = 介电常数 εr（默认 9.0），深度 = ½·c·t/√εr；
    速度分析卡片 = 绕射双曲线拟合（Phase 2.1）。
- 底部信息条（横条卡片：card_title「标注信息」+ 点数 / 会话状态）。

页面纯展示 + 发信号：内部维护当前点列，pick 点击追加并发 points_changed；
删除/清空同样通过 points_changed → 控制器 replace_points 写回会话。
会话未打开时编辑按钮、点列管理与 pick 前置禁用（P1-6），避免"点了才报错"。
"""

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QHBoxLayout, QHeaderView, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CaptionLabel, CardWidget, ComboBox, DoubleSpinBox, PrimaryPushButton,
    PushButton,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.page_scaffold import (card_title, make_card, make_scroll_column,
                              refill_combo)
from ui.theme_helpers import status_color
from ui.widgets import BScanView, CollapsiblePanel, make_separator
from ui.widgets.context_menus import add_action, make_menu

_OVERLAY_COLOR = constants.CHART_OVERLAY_COLOR   # 标注散点颜色（SPEC §6.6）
_C_M_PER_NS = 0.29979        # 真空光速 c (m/ns)


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
    velocity_requested = pyqtSignal(list)  # 拾取点 [(trace, sample), ...] → 速度分析

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = []           # [(trace_index, sample_index), ...] 原始数据坐标
        self._bundle = None         # 当前预览 bundle（取时间轴做深度换算）
        self._busy = False
        self._velocity_running = False  # 提交后到回调前的在飞窗口，防重复提交
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
        root.addWidget(self._build_tool_card())

        body = QHBoxLayout()
        body.setSpacing(constants.PAGE_SPACING)
        body.addWidget(self._build_bscan_card(), 1)
        body.addWidget(self._build_side_column(), 0)
        root.addLayout(body, 1)

        root.addWidget(self._build_info_bar())

    def _build_tool_card(self) -> CardWidget:
        """顶部工具卡片：两行布局（行1 会话来源 / 行2 编辑操作）。"""
        card, layout = make_card('标注工具', parent=self)

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
            'color: %s; font-size: 11px;' % status_color('disabled'))
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
            'color: %s; font-size: 11px;' % status_color('disabled'))
        row2.addWidget(hint)
        layout.addLayout(row2)
        return card

    def _build_bscan_card(self) -> CardWidget:
        """剖面标注卡片：BScanView（pick 模式 + overlay）。"""
        card, layout = make_card('剖面标注', parent=self)
        self._bscan = BScanView(card)
        self._bscan.setMinimumHeight(420)
        self._bscan.set_pick_enabled(False)  # 会话打开前禁用 pick（P1-6）
        layout.addWidget(self._bscan, 1)
        return card

    def _build_side_column(self) -> QWidget:
        """右栏（展开 SIDE_TOOL_WIDTH px，可折叠）：标注点列表 + 深度换算 + 速度分析。"""
        scroll, layout = make_scroll_column(constants.SIDE_TOOL_WIDTH)
        panel = CollapsiblePanel(
            'right', expand_width=constants.SIDE_TOOL_WIDTH, collapse_width=40,
            parent=self)
        panel.set_content_widget(scroll)

        # ---------------- 标注点列表
        points_card, points_layout = make_card('标注点列表')

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
        # Delete 键删除选中点（与处理链/项目页测线表同约定）
        self._delete_point_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete), self._points_table,
            context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_point_shortcut.activated.connect(
            self._on_remove_selected_point)
        # 右键 = 复制该点信息 / 删除选中点 / 清空全部（与按钮、Delete 键同约定）
        self._points_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._points_table.customContextMenuRequested.connect(
            self._on_points_context_menu)
        points_layout.addWidget(self._points_table, 1)

        # 点数计数与删除/清空同行：计数居左，按钮居右
        btn_row = QHBoxLayout()
        btn_row.setSpacing(constants.CARD_SPACING)
        self._points_count_label = CaptionLabel('0 个点', points_card)
        self._points_count_label.setStyleSheet(
            'color: %s; font-size: 11px;' % status_color('disabled'))
        btn_row.addWidget(self._points_count_label)
        btn_row.addStretch(1)
        self._remove_point_btn = PushButton('删除选中', points_card, FIF.DELETE)
        self._remove_point_btn.setToolTip('删除列表中选中的标注点 (Delete)')
        self._clear_points_btn = PushButton('清空', points_card)
        self._clear_points_btn.setToolTip('清空全部标注点')
        btn_row.addWidget(self._remove_point_btn)
        btn_row.addWidget(self._clear_points_btn)
        points_layout.addLayout(btn_row)
        layout.addWidget(points_card, 1)

        # ---------------- 深度换算
        depth_card, depth_layout = make_card('深度换算')
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
            'color: %s; font-size: 11px;' % status_color('disabled'))
        depth_layout.addWidget(formula_hint)
        layout.addWidget(depth_card)

        # ---------------- 速度分析（Phase 2.1）
        velocity_card, velocity_layout = make_card('速度分析')
        self._velocity_btn = PushButton('拟合速度模型', velocity_card)
        self._velocity_btn.setToolTip(
            '用当前标注点拟合绕射双曲线：t² = A·x² + B·x + C ⇒ v = 2/√A\n'
            '需要 ≥3 个拾取点；结果写回测线速度模型并重算深度轴')
        self._velocity_btn.setEnabled(False)  # 会话打开且点数足够后启用
        velocity_layout.addWidget(self._velocity_btn)
        self._velocity_result_label = CaptionLabel('尚未拟合', velocity_card)
        self._velocity_result_label.setWordWrap(True)
        velocity_layout.addWidget(self._velocity_result_label)
        layout.addWidget(velocity_card)
        layout.addStretch(1)
        return panel

    def _build_info_bar(self) -> CardWidget:
        """底部信息条（横条卡片）：标题 + 点数 / 会话状态。"""
        card = CardWidget(self)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(*constants.CARD_MARGINS)
        layout.setSpacing(constants.CARD_SPACING)
        layout.addWidget(card_title('标注信息'))
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
        self._velocity_btn.clicked.connect(
            lambda: self.velocity_requested.emit(list(self._points)))

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
        def _artifact_id(artifact) -> str:
            return str(getattr(artifact, 'artifact_id', '') or '')

        refill_combo(
            self._artifact_combo,
            [a for a in (artifacts or []) if _artifact_id(a)],
            lambda a: f'成果: {str(getattr(a, "name", "") or _artifact_id(a))}',
            _artifact_id,
            previous_data=self._current_artifact_id(),
            prepend=(('原始数据', ''),))   # index 0 = 原始数据

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
        self._update_edit_enabled()

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

    def _update_edit_enabled(self) -> None:
        enabled = self._session_active and not self._busy
        for btn in (self._auto_trace_btn, self._snap_btn, self._smooth_btn,
                    self._undo_btn, self._redo_btn, self._save_btn,
                    self._remove_point_btn, self._clear_points_btn):
            btn.setEnabled(enabled)
        # 速度分析还需 ≥3 个拾取点（双曲线 3 参数拟合下限）且不在飞
        self._velocity_btn.setEnabled(
            enabled and not self._velocity_running and len(self._points) >= 3)
        self._bscan.set_pick_enabled(enabled)

    def set_velocity_running(self, running: bool) -> None:
        """速度分析在飞态：提交后置 True，完成/失败回调置 False，防重复提交。"""
        self._velocity_running = bool(running)
        self._velocity_result_label.setText(
            '拟合中…' if running else '尚未拟合')
        self._update_edit_enabled()

    def set_velocity_result(self, line_id: str, result: dict) -> None:
        """速度分析完成 → 卡片显示拟合证据（v/εr/x0/z0/RMSE）。"""
        self._velocity_running = False
        self._update_edit_enabled()
        body = (result or {}).get('evidence', {}).get('body', {}) \
            if isinstance(result, dict) else {}
        if not body:
            self._velocity_result_label.setText('拟合完成（无证据返回）')
            return
        lines = [
            '测线: %s' % (line_id or '--'),
            '速度: %.4f m/ns (εr=%.2f)' % (
                float(body.get('v_m_ns', 0.0)),
                float(body.get('dielectric_constant', 0.0))),
            '绕射点: x=%.2f m, z=%.2f m' % (
                float(body.get('x0_m', 0.0)), float(body.get('z0_m', 0.0))),
            'RMSE: %.3f ns | 拾取点: %d' % (
                float(body.get('rmse_ns', 0.0)),
                int(body.get('pick_count', 0))),
        ]
        self._velocity_result_label.setText('\n'.join(lines))

    def set_velocity_failed(self, message: str) -> None:
        """速度分析失败 → 卡片显示错误。

        空串（或 coordinator 项目关闭重置）→ 恢复「尚未拟合」。
        两种路径都解除在飞态，防止任务在飞时关项目导致按钮永久禁用。
        """
        self._velocity_running = False
        self._update_edit_enabled()
        self._velocity_result_label.setText(
            ('拟合失败: %s' % message) if message else '尚未拟合')

    def _on_point_picked(self, trace: int, sample: int) -> None:
        """pick 点击追加点（原始数据坐标）→ overlay/列表末行增量插入 + points_changed。"""
        self._points.append((int(trace), int(sample)))
        self._bscan.set_overlay_points(self._points, _OVERLAY_COLOR)
        self._append_points_row(len(self._points) - 1)
        self._refresh_info()
        self._update_edit_enabled()
        self.points_changed.emit(list(self._points))

    def _on_remove_selected_point(self) -> None:
        """删除列表选中行对应的标注点（局部删行 + 后续行号重排）。"""
        row = self._points_table.currentRow()
        if not (0 <= row < len(self._points)):
            return
        del self._points[row]
        table = self._points_table
        table.removeRow(row)
        for r in range(row, table.rowCount()):
            item = table.item(r, 0)
            if item is not None:
                item.setText(str(r + 1))
        self._points_count_label.setText('%d 个点' % len(self._points))
        self._emit_points_updated(table_updated=True)

    def _on_points_context_menu(self, pos) -> None:
        """标注点表右键：右击行先选中（与 Delete 键同一目标行语义）再弹菜单。"""
        row = self._points_table.rowAt(pos.y())
        if 0 <= row < len(self._points):
            self._points_table.selectRow(row)
        menu = self._build_points_menu(row)
        menu.exec(self._points_table.viewport().mapToGlobal(pos))

    def _build_points_menu(self, row: int):
        """构造标注点右键菜单（与 exec 分离，便于测试检查动作）。

        编辑动作与「删除选中/清空」按钮同门控：会话已打开且不在忙态。
        """
        menu = make_menu(parent=self._points_table)
        editable = self._session_active and not self._busy
        has_row = 0 <= row < len(self._points)
        if has_row:
            trace, sample = self._points[row]
            summary = '道 %d, 采样点 %d' % (trace + 1, sample + 1)
            time_ns = self._sample_time_ns(sample)
            if time_ns is not None:
                summary += ', %.2f ns' % time_ns
            add_action(menu, FIF.COPY, '复制该点信息',
                       lambda: QApplication.clipboard().setText(summary))
            menu.addSeparator()
        add_action(menu, FIF.DELETE, '删除选中点',
                   self._on_remove_selected_point,
                   enabled=editable and has_row)
        add_action(menu, FIF.DELETE, '清空全部',
                   self._on_clear_points,
                   enabled=editable and bool(self._points))
        return menu

    def _on_clear_points(self) -> None:
        """清空全部标注点。"""
        if not self._points:
            return
        self._points = []
        self._emit_points_updated()

    def _emit_points_updated(self, *, table_updated: bool = False) -> None:
        """点列变更统一出口：overlay + 表格 + 信息条 + 信号。

        table_updated=True 表示调用方已完成表格局部更新（单点删除），
        此处跳过整表重建。
        """
        self._bscan.set_overlay_points(self._points, _OVERLAY_COLOR)
        if not table_updated:
            self._refresh_points_table()
        self._refresh_info()
        self._update_edit_enabled()
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

    def _point_row_values(self, index: int, trace: int, sample: int) -> tuple:
        """单行五列文本（# / 道 / 采样点 / 时间(ns) / 深度(m)）。"""
        time_ns = self._sample_time_ns(sample)
        depth_m = (self._estimate_depth_m(time_ns)
                   if time_ns is not None else None)
        return (
            str(index + 1),
            str(trace + 1),
            str(sample + 1),
            ('%.2f' % time_ns) if time_ns is not None else '--',
            ('%.3f' % depth_m) if depth_m is not None else '--',
        )

    def _append_points_row(self, index: int) -> None:
        """末行增量插入（拾取热路径）：O(1) 追加，不重建整张表。"""
        table = self._points_table
        trace, sample = self._points[index]
        row = table.rowCount()
        table.insertRow(row)
        for col, text in enumerate(self._point_row_values(index, trace, sample)):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, col, item)
        self._points_count_label.setText('%d 个点' % len(self._points))

    def _refresh_points_table(self) -> None:
        """整表重建（外部整列替换 / 介电常数变化 / 清空）。"""
        table = self._points_table
        table.blockSignals(True)
        table.setRowCount(0)
        for index, (trace, sample) in enumerate(self._points):
            row = table.rowCount()
            table.insertRow(row)
            for col, text in enumerate(
                    self._point_row_values(index, trace, sample)):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)
        table.blockSignals(False)
        self._points_count_label.setText('%d 个点' % len(self._points))

    def _refresh_info(self) -> None:
        self._info_label.setText(
            '标注点数: %d | 会话状态: %s'
            % (len(self._points), self._session_status_label.text()))

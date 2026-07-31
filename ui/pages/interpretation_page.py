# -*- coding: utf-8 -*-
"""InterpretationPage — 界面解释标注（SPEC §6.6）。

- 顶部工具卡片：测线 Label + 打开标注会话 | 自动追踪/吸附/平滑/撤销/重做 |
  PrimaryPushButton('保存标注') + 状态 CaptionLabel
- BScanView（pick 模式，点击追加点；overlay 显示当前标注点列，颜色 #fbbf24）
- 底部信息条：点数/会话状态

页面纯展示 + 发信号：内部维护当前点列，pick 点击追加并发 points_changed。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel, CardWidget, PrimaryPushButton, PushButton, SubtitleLabel,
)

from ui import constants
from ui.widgets import BScanView

_OVERLAY_COLOR = '#fbbf24'   # 标注散点颜色（SPEC §6.6）


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


def _create_separator(vertical: bool = False) -> QFrame:
    """分隔线工厂：HLine/VLine + Sunken + color:#e0e0e0（SPEC §1）。"""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine if vertical
                       else QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet('color: #e0e0e0;')
    return line


class InterpretationPage(QWidget):
    """界面解释标注页面。"""

    open_session_requested = pyqtSignal()
    auto_trace_requested = pyqtSignal()
    snap_requested = pyqtSignal()
    smooth_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    save_requested = pyqtSignal()
    points_changed = pyqtSignal(list)      # 当前标注点列 [(trace, sample), ...]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = []           # [(trace_index, sample_index), ...]
        self._busy = False
        self._build_ui()
        self._connect_internal()

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)
        root.addWidget(_page_title('界面解释标注'))

        # ---------------- 顶部工具卡片
        tool_card = CardWidget(self)
        tool_layout = QVBoxLayout(tool_card)
        tool_layout.setContentsMargins(*constants.CARD_MARGINS)
        tool_layout.setSpacing(constants.CARD_SPACING)
        tool_layout.addWidget(_card_title('标注工具'))

        row = QHBoxLayout()
        row.setSpacing(constants.CARD_SPACING)
        line_label = CaptionLabel('测线:', tool_card)
        line_label.setMinimumWidth(40)
        row.addWidget(line_label)
        self._line_label = CaptionLabel('--', tool_card)
        row.addWidget(self._line_label)
        self._open_session_btn = PushButton('打开标注会话', tool_card)
        row.addWidget(self._open_session_btn)
        row.addWidget(_create_separator(vertical=True))
        self._auto_trace_btn = PushButton('自动追踪', tool_card)
        self._snap_btn = PushButton('吸附', tool_card)
        self._smooth_btn = PushButton('平滑', tool_card)
        self._undo_btn = PushButton('撤销', tool_card)
        self._redo_btn = PushButton('重做', tool_card)
        for btn in (self._auto_trace_btn, self._snap_btn, self._smooth_btn,
                    self._undo_btn, self._redo_btn):
            row.addWidget(btn)
        row.addWidget(_create_separator(vertical=True))
        self._save_btn = PrimaryPushButton('保存标注', tool_card)
        row.addWidget(self._save_btn)
        row.addStretch(1)
        status_label = CaptionLabel('状态:', tool_card)
        row.addWidget(status_label)
        self._session_status_label = CaptionLabel('未打开会话', tool_card)
        self._session_status_label.setStyleSheet(
            'color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        row.addWidget(self._session_status_label)
        tool_layout.addLayout(row)
        root.addWidget(tool_card)

        # ---------------- B-Scan（pick 模式 + overlay）
        bscan_card = CardWidget(self)
        bscan_layout = QVBoxLayout(bscan_card)
        bscan_layout.setContentsMargins(*constants.CARD_MARGINS)
        bscan_layout.setSpacing(constants.CARD_SPACING)
        bscan_layout.addWidget(_card_title('剖面标注'))
        self._bscan = BScanView(bscan_card)
        self._bscan.setMinimumHeight(320)
        self._bscan.set_pick_enabled(True)
        bscan_layout.addWidget(self._bscan, 1)
        root.addWidget(bscan_card, 1)

        # ---------------- 底部信息条：点数/会话状态
        info_card = CardWidget(self)
        info_layout = QHBoxLayout(info_card)
        info_layout.setContentsMargins(*constants.CARD_MARGINS)
        info_layout.setSpacing(constants.CARD_SPACING)
        self._info_label = CaptionLabel('标注点数: 0 | 会话状态: 未打开会话',
                                        info_card)
        info_layout.addWidget(self._info_label, 1)
        root.addWidget(info_card)

    # ============================================================ 内部接线
    def _connect_internal(self) -> None:
        self._open_session_btn.clicked.connect(self.open_session_requested)
        self._auto_trace_btn.clicked.connect(self.auto_trace_requested)
        self._snap_btn.clicked.connect(self.snap_requested)
        self._smooth_btn.clicked.connect(self.smooth_requested)
        self._undo_btn.clicked.connect(self.undo_requested)
        self._redo_btn.clicked.connect(self.redo_requested)
        self._save_btn.clicked.connect(self.save_requested)
        self._bscan.sig_point_picked.connect(self._on_point_picked)

    # ============================================================ 公共接口（供主窗口接线）
    def set_bundle(self, bundle) -> None:
        """剖面预览 bundle → BScanView。"""
        self._bscan.set_bundle(bundle)

    def set_line_label(self, text: str) -> None:
        """当前测线标签（顶部工具卡片）。"""
        self._line_label.setText(text or '--')

    def set_session_info(self, text: str) -> None:
        """会话状态文案（顶部状态 CaptionLabel + 底部信息条）。"""
        text = text or '未打开会话'
        self._session_status_label.setText(text)
        self._refresh_info()

    def set_points(self, points: list) -> None:
        """整列替换标注点 → BScanView overlay（#fbbf24）。"""
        self._points = [(int(t), int(s)) for t, s in (points or [])]
        self._bscan.set_overlay_points(self._points, _OVERLAY_COLOR)
        self._refresh_info()

    def set_busy(self, busy: bool) -> None:
        """忙态：禁用全部操作按钮。"""
        self._busy = bool(busy)
        for btn in (self._open_session_btn, self._auto_trace_btn,
                    self._snap_btn, self._smooth_btn, self._undo_btn,
                    self._redo_btn, self._save_btn):
            btn.setEnabled(not self._busy)

    # ============================================================ 内部逻辑
    def _on_point_picked(self, trace: int, sample: int) -> None:
        """pick 点击追加点 → overlay 刷新 + points_changed。"""
        self._points.append((int(trace), int(sample)))
        self._bscan.set_overlay_points(self._points, _OVERLAY_COLOR)
        self._refresh_info()
        self.points_changed.emit(list(self._points))

    def _refresh_info(self) -> None:
        self._info_label.setText(
            '标注点数: %d | 会话状态: %s'
            % (len(self._points), self._session_status_label.text()))

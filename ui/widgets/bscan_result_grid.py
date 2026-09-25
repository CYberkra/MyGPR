# -*- coding: utf-8 -*-
"""ResultGrid — 结果网格（处理页 v2 主区下半）。

设计（2026-09-25 v2）：
- 一屏看多步：列数 = ``N≤3 → N``，``N≥4 → min(⌈√N⌉, 3)``
  （2 张横排 / 3 张三列 / 4 张两行两列 / 6 张三列两行）；
- 单元间距 16（8pt 栅格），**单元最小高 320**——行数多时纵向滚动，
  绝不把剖面压破 0.45px/采样可读性红线；
- 卡头**极简**：序号 + 算法名，⤢ 图标 hover 才显（常驻噪音更少）；
- 禁用步骤 → 虚线占位卡（保留 1:1 对应，不占画布）。

网格只认「槽位」（key/title/enabled），数据由宿主页按 key 回填。
"""
import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QScrollArea, QSizePolicy, QVBoxLayout, QWidget)
from qfluentwidgets import FluentIcon as FIF, ToolButton

from ui.widgets.bscan_view import BScanView
from ui.widgets.empty_state import EmptyStateOverlay

_CELL_MIN_HEIGHT = 320
_GRID_SPACING = 16
_MAX_COLUMNS = 3


class _ResultCard(QFrame):
    """单张结果：极简卡头（序号 + 算法名）+ B-Scan + hover 操作钮。"""

    sig_expand_requested = pyqtSignal(str)   # slot key
    sig_compare_requested = pyqtSignal(str)  # slot key（P2 接线）

    def __init__(self, key: str, title: str, *, placeholder: bool = False,
                 parent=None):
        super().__init__(parent)
        self.key = key
        self._placeholder = placeholder
        self.setObjectName('resultCard')
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(_CELL_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            '#resultCard{background:transparent}' if not placeholder else
            '#resultCard{border:1px dashed rgba(140,140,140,0.55);'
            'border-radius:6px}')

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self.title_label = QLabel(title, self)
        self.title_label.setStyleSheet(
            'color:#8A8A85' if placeholder else '')
        head.addWidget(self.title_label)
        head.addStretch(1)
        self.expand_btn = ToolButton(FIF.FULL_SCREEN, self)
        self.expand_btn.setFixedSize(20, 20)
        self.expand_btn.setToolTip('放大 / 全屏浏览该结果')
        self.expand_btn.clicked.connect(
            lambda: self.sig_expand_requested.emit(self.key))
        head.addWidget(self.expand_btn)
        self.compare_btn = ToolButton(FIF.VIEW, self)
        self.compare_btn.setFixedSize(20, 20)
        self.compare_btn.setToolTip('与另一张结果并排对比（P2 接线）')
        self.compare_btn.clicked.connect(
            lambda: self.sig_compare_requested.emit(self.key))
        head.addWidget(self.compare_btn)
        self.expand_btn.setVisible(False)
        self.compare_btn.setVisible(False)

        body = QVBoxLayout(self)
        body.setContentsMargins(4, 2, 4, 4)
        body.setSpacing(4)
        body.addLayout(head)
        if placeholder:
            self.view = None
            hint = QLabel('已禁用（未参与本次运行）', self)
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setStyleSheet('color:#8A8A85')
            body.addWidget(hint, 1)
        else:
            self.view = BScanView(self)
            body.addWidget(self.view, 1)

    def set_bundle(self, bundle) -> None:
        if self.view is None:
            return
        if bundle is None:
            self.view.clear()       # 尚无输入数据 → 空态
        else:
            self.view.set_bundle(bundle)

    def clear(self) -> None:
        if self.view is not None:
            self.view.clear()

    def enterEvent(self, event) -> None:
        if not self._placeholder:
            self.expand_btn.setVisible(True)
            self.compare_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.expand_btn.setVisible(False)
        self.compare_btn.setVisible(False)
        super().leaveEvent(event)


class ResultGrid(QWidget):
    """结果网格：按槽位铺卡片，列数自适应，纵向可滚。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards = []                 # 顺序 = 槽位顺序
        self._keys = {}                  # key → card
        self._body = QWidget(self)
        self._grid = QGridLayout(self._body)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(_GRID_SPACING)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)
        self._scroll.setWidget(self._body)

        # 运行前不摆空画布：只给一句引导（有数据才建 B-Scan 视图）
        self._empty = EmptyStateOverlay(
            self, icon=None, title='暂无结果',
            hint='选择测线后显示输入数据；点「运行」后按步骤显示各步结果')
        self._empty.setVisible(True)

    # ---------------------------------------------------------------- 槽位
    def set_slots(self, slots) -> None:
        """slots: [{key, title, enabled}]（宿主页为数据源；这里只铺卡）。"""
        for card in self._cards:
            self._grid.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        self._keys = {}
        self._empty.setVisible(not (slots or []))
        for spec in (slots or []):
            card = _ResultCard(
                spec['key'], spec.get('title', ''),
                placeholder=not bool(spec.get('enabled', True)),
                parent=self._body)
            self._cards.append(card)
            self._keys[spec['key']] = card
        self._reflow()

    def _reflow(self) -> None:
        """列数规则 + 落格（占位卡同样占一格，保持 1:1）。"""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    self._grid.removeWidget(widget)
        n = len(self._cards)
        if n == 0:
            return
        cols = n if n <= _MAX_COLUMNS else min(
            int(math.ceil(math.sqrt(n))), _MAX_COLUMNS)
        for index, card in enumerate(self._cards):
            self._grid.addWidget(card, index // cols, index % cols)

    def set_bundle(self, key: str, bundle) -> None:
        card = self._keys.get(key)
        if card is not None:
            card.set_bundle(bundle)

    def clear_all(self) -> None:
        for card in self._cards:
            card.clear()

    def card(self, key: str):
        return self._keys.get(key)

    def cards(self) -> list:
        return list(self._cards)

# -*- coding: utf-8 -*-
"""ResultGrid — 结果网格（处理页 v2 主区下半）。

设计（2026-09-26 v2.1，用户定稿）：
- **两列大图**：``N=1 → 全幅``，``N≥2 → 2 列``——单格更大，剖面更可读
  （2026-09-26 弃用三列规则）；
- 网格卡**无色标**（``BScanView(with_colorbar=False)``），坐标/色标不再
  挤占绘图区；看色标与精读走放大 / 全屏；
- 单元间距 16（8pt 栅格），**单元最小高 320**——行数多时纵向滚动；
- 卡头**极简**：序号 + 算法名，⤢ 图标 hover 才显；
- **选中同步高亮**：点卡 → 发 :attr:`sig_card_selected`；宿主调
  :meth:`set_selected` 高亮对应卡（与链条 chip 双向同步）；
- 禁用步骤 → 虚线占位卡（保留 1:1 对应，不占画布）。

网格只认「槽位」（key/title/enabled），数据由宿主页按 key 回填。
"""
import math

from PyQt6.QtCore import (QPropertyAnimation, Qt, pyqtSignal)
from PyQt6.QtWidgets import (QFrame, QGraphicsOpacityEffect,
                             QGridLayout, QHBoxLayout, QLabel, QScrollArea,
                             QSizePolicy, QVBoxLayout, QWidget)
from qfluentwidgets import FluentIcon as FIF, ToolButton

from ui.widgets.bscan_view import BScanView
from ui.widgets.empty_state import EmptyStateOverlay

_CELL_MIN_HEIGHT = 320
_GRID_SPACING = 16
_MAX_COLUMNS = 2


class _Skeleton(QWidget):
    """结果卡骨架：预览未回填时的微光占位（动效移植①）。

    移植 Transitions.dev 的「Skeleton loader and reveal」思路——Qt 侧用
    不透明度脉冲 + 横向扫光条（QPropertyAnimation）近似 shimmer，不引入
    新依赖；出图后与画布做交叉淡入（skeleton 淡出 / 画布淡入）。
    """

    def __init__(self, host=None):
        super().__init__(host)
        from PyQt6.QtCore import QPropertyAnimation, QRect
        from PyQt6.QtGui import QColor, QPainter
        self._QPropertyAnimation = QPropertyAnimation
        self._QRect = QRect
        self._QColor = QColor
        self._QPainter = QPainter
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._shimmer_x = -0.35
        self._anim = QPropertyAnimation(self, b'geometry')
        self._anim.setDuration(1400)
        self._anim.setLoopCount(-1)
        self._anim.setStartValue(0)
        self._anim.setEndValue(1000)
        self._anim.valueChanged.connect(lambda _v: self.update())

    def showEvent(self, event) -> None:
        if self._anim.state() != self._anim.State.Running:
            self._anim.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._anim.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:
        painter = self._QPainter(self)
        painter.setRenderHint(self._QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, self._QColor(150, 150, 150, 26))
        # 扫光条：按动画进度在卡宽上平移
        progress = (self._anim.currentValue() or 0) / 1000.0
        width = max(rect.width() * 0.35, 40)
        x = rect.left() + (rect.width() + width) * progress - width
        band = self._QRect(int(x), rect.top(), int(width), rect.height())
        painter.fillRect(band, self._QColor(200, 200, 200, 34))
        painter.end()


class _ResultCard(QFrame):
    """单张结果：极简卡头（序号 + 算法名）+ B-Scan + hover 操作钮。"""

    sig_expand_requested = pyqtSignal(str)   # slot key
    sig_compare_requested = pyqtSignal(str)  # slot key（P2 接线）
    sig_clicked = pyqtSignal(str)            # 点卡 → 宿主选中对应步骤

    _SEL_QSS = ('#resultCard{border:2px solid rgba(90,156,216,0.85);'
                'border-radius:6px}')
    _NORMAL_QSS = '#resultCard{border:2px solid transparent;border-radius:6px}'

    def __init__(self, key: str, title: str, *, placeholder: bool = False,
                 parent=None):
        super().__init__(parent)
        self.key = key
        self._placeholder = placeholder
        self.setObjectName('resultCard')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(self._NORMAL_QSS)
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
            # 网格卡无色标：坐标/色标不挤占绘图区（看色标走放大/全屏）
            self.view = BScanView(self, with_colorbar=False)
            self._skeleton = _Skeleton(self.view)
            self._skeleton.setGeometry(self.view.rect())
            self._skeleton.show()          # 建卡即占位：等预览回填
            # 画布透明度效果（出图时淡入，与骨架交叉）
            self._view_effect = QGraphicsOpacityEffect(self.view)
            self._view_effect.setOpacity(0.0)
            self.view.setGraphicsEffect(self._view_effect)
            body.addWidget(self.view, 1)
            self._fade_out = QPropertyAnimation(self._skeleton, b'windowOpacity')
            self._fade_out.setDuration(240)
            self._fade_out.setStartValue(1.0)
            self._fade_out.setEndValue(0.0)
            self._fade_out.finished.connect(self._skeleton.hide)
            self._fade_in = QPropertyAnimation(self._view_effect, b'opacity')
            self._fade_in.setDuration(240)
            self._fade_in.setStartValue(0.0)
            self._fade_in.setEndValue(1.0)

    def set_bundle(self, bundle) -> None:
        if self.view is None:
            return
        if bundle is None:
            self.view.clear()       # 尚无输入数据 → 空态（骨架继续占位）
            return
        self.view.set_bundle(bundle)
        self._reveal()

    def _reveal(self) -> None:
        """出图：骨架淡出 + 画布淡入（交叉淡入，避免「空白→突然有图」）。"""
        if self._skeleton.isHidden():
            return
        self._fade_out.start()
        self._fade_in.start()

    def clear(self) -> None:
        if self.view is not None:
            self.view.clear()

    def set_selected(self, selected: bool) -> None:
        """选中高亮：与链条 chip 同步（描边 2px 主题蓝）。"""
        self.setStyleSheet(self._SEL_QSS if selected else self._NORMAL_QSS)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.sig_clicked.emit(self.key)
        super().mousePressEvent(event)

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

    sig_card_selected = pyqtSignal(str)  # 点卡 → 宿主选中对应步骤

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
            card.sig_clicked.connect(self.sig_card_selected)
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

    def set_selected(self, key: str) -> None:
        """高亮指定槽位卡，其余恢复常规描边。"""
        for card_key, card in self._keys.items():
            card.set_selected(card_key == key)

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

# -*- coding: utf-8 -*-
"""CollapsibleCard — 标题行可点击收起的卡片（垂直方向）。

与 :class:`ui.widgets.collapsible_panel.CollapsiblePanel`（横向折叠的**侧栏**）
同机制不同轴：那一个收的是整个栏宽、按钮是贴边纵向长条；这一个收的是**卡内
内容区高度**、按钮在标题行右侧。两者刻意不合并——侧栏折叠要参与
``PanelStateMixin`` 的中栏宽度让位计算，卡片折叠不参与。

为什么需要它（空间页 2026-10-06 实测）：左栏「三维显示」卡 6 个控件占 260px
= 左栏高度的 28.8%，而它们全部只连着 ``Trajectory3DView``——切到平面地图 /
高程剖面 / 深度切片时一行都不生效，却仍然占着位置。收成一行标题后释放 232px。

折叠态**不销毁**子控件（``setVisible`` 而非 reparent/重建），所以
DoubleSpinBox 的值、SwitchButton 的开关态天然保留，无需另存。
"""

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QTransform
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (CardWidget, FluentIcon as FIF, ToolButton,
                            isDarkTheme)

from ui import constants
from ui.page_scaffold import card_title

__all__ = ['CollapsibleCard', 'make_collapsible_card']

_DURATION_MS = 220
# QWIDGETSIZE_MAX：QWidget 默认 maximumHeight 上限。展开后解除高度限制，
# 交回布局决定（不设会让卡片在有 stretch 的栏里不肯长高）。
_QWIDGETSIZE_MAX = 16777215


def _down_chevron() -> QIcon:
    """CHEVRON_RIGHT_MED 顺时针旋转 90° 得到向下 chevron（旋转而非翻转：
    chevron 上下不对称，翻转会得到指向错误的形状）。"""
    pm = FIF.CHEVRON_RIGHT_MED.icon().pixmap(16, 16)
    return QIcon(pm.transformed(QTransform().rotate(90)))


class CollapsibleCard(CardWidget):
    """标题行右侧带 chevron 的可折叠卡片。

    用法::

        card, body = make_collapsible_card('三维显示', parent=self)
        body.addWidget(widget)

    ``body`` 是卡内的 QVBoxLayout——收起的只是它的可见性，控件本身一直在，
    所以状态不丢。``set_collapsed`` 走 220ms OutCubic 高度动画（与侧栏一致）。
    """

    sig_collapsed = pyqtSignal(bool)

    def __init__(self, title: str, *, parent=None, collapsed: bool = False):
        super().__init__(parent)
        # 美术打磨 C 同款：CardWidget 深色下实测渲染白底，显式主题底色。
        # **刻意不实现 apply_theme()**：``make_card`` 造的 20+ 张卡都没有，
        # 主题切换靠主窗口 ``findChildren`` 单轮遍历里的 ``widget.update()``
        # 触发 CardWidget 重读 ``isDarkTheme()``。本类保持同一策略——单点
        # 特例反而会让下一个人以为卡片底色需要手动维护。
        self.setBackgroundColor(
            QColor('#2d2e32') if isDarkTheme() else QColor('#ffffff'))

        self._collapsed = bool(collapsed)
        self._animating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(*constants.CARD_MARGINS)
        root.setSpacing(constants.CARD_SPACING)

        # 标题行：标题 + stretch + chevron 按钮
        header = QHBoxLayout()
        header.setSpacing(constants.CARD_SPACING)
        self._title = card_title(title)
        self._title.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        header.addWidget(self._title, 1)
        self._toggle_btn = ToolButton(FIF.CHEVRON_RIGHT_MED, self)
        self._toggle_btn.setFixedSize(*constants.TOOL_BTN_SIZE)
        self._toggle_btn.setToolTip('展开' if self._collapsed else '收起')
        self._toggle_btn.clicked.connect(self.toggle)
        header.addWidget(self._toggle_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(header)

        # 内容区：独立 QWidget 承载，收起时整体隐藏（子控件状态不丢）
        self._body = QWidget(self)
        self._body.setObjectName('collapsibleCardBody')
        self._body.setStyleSheet(
            'QWidget#collapsibleCardBody { background-color: transparent; }')
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(constants.CARD_SPACING)
        root.addWidget(self._body, 0)

        # 高度动画：驱动 body 的 maximumHeight（0 ↔ 自然高度）
        self._animation = QPropertyAnimation(self._body, b'maximumHeight', self)
        self._animation.setDuration(_DURATION_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

        self._toggle_btn.setIcon(FIF.CHEVRON_RIGHT_MED.icon()
                                 if self._collapsed else _down_chevron())
        if self._collapsed:
            self._body.setVisible(False)
            self._body.setMaximumHeight(0)

    # ------------------------------------------------------------ 访问
    def body_layout(self) -> QVBoxLayout:
        """卡内内容布局（挂控件用）。"""
        return self._body_layout

    def is_collapsed(self) -> bool:
        return self._collapsed

    def body_height(self) -> int:
        """内容区自然高度（动画目标值）。

        必须在布局已激活时读——未show 过 / 未做过布局的控件 sizeHint 可能
        仍是默认值（离屏探针里尤其明显：孤儿控件不给布局就量到 100×30）。

        **与可见性无关**：``QWidget.sizeHint()`` 对隐藏控件同样有效（Qt 的
        sizeHint 不看 isVisible），所以收起态也能问出正确高度——展开动画
        才能从 0 续跑到真实值。``_on_animation_finished`` 里的收敛判定也
        依赖这一点，否则展开后会永远等不到「maximumHeight == 目标」。
        """
        self._body_layout.activate()
        return max(self._body.sizeHint().height(),
                   self._body.minimumSizeHint().height())

    # ------------------------------------------------------------ 状态
    def set_collapsed(self, collapsed: bool, animate: bool = True) -> None:
        """折叠 / 展开内容区。

        动画中途再次调用**不丢弃**：与 CollapsiblePanel 同策略——停表后从当前
        maximumHeight 续跑到新目标，视觉上平滑换向而非跳变。
        """
        collapsed = bool(collapsed)
        if self._collapsed == collapsed and not self._animating:
            return

        self._animating = True
        self._collapsed = collapsed
        target = 0 if collapsed else self.body_height()
        self._animation.stop()
        self._animation.setStartValue(self._body.maximumHeight())
        self._animation.setEndValue(target)

        if collapsed:
            self._body.setVisible(False)
        # chevron 指示的是「点击后的动作」：收起态给 ›（点开），展开态给 ⌄
        # （点收起）——与 Windows 侧栏 / Win11 设置页的展开器同一约定。
        self._toggle_btn.setIcon(FIF.CHEVRON_RIGHT_MED.icon()
                                 if collapsed else _down_chevron())
        self._toggle_btn.setToolTip('展开' if collapsed else '收起')

        if animate:
            self._animation.start()
        else:
            self._body.setMaximumHeight(target)
            self._on_animation_finished()

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    # ------------------------------------------------------------ 内部
    def _on_animation_finished(self) -> None:
        # stop() 也会发 finished（Qt 语义），续跑场景下会误清 _animating
        if self._body.maximumHeight() != (0 if self._collapsed
                                          else self.body_height()):
            return
        self._animating = False
        if not self._collapsed:
            self._body.setVisible(True)
            self._body.setMaximumHeight(_QWIDGETSIZE_MAX)
        self.sig_collapsed.emit(self._collapsed)


def make_collapsible_card(title: str, *, parent=None,
                          collapsed: bool = False) -> tuple:
    """可折叠卡片工厂：返回 ``(card, body_layout)``。

    与 :func:`ui.page_scaffold.make_card` 同构（都返回「卡 + 内容布局」），
    差别只在标题行多了 chevron、内容区可收起——调用方换用它是**一行改动**。
    """
    card = CollapsibleCard(title, parent=parent, collapsed=collapsed)
    return card, card.body_layout()

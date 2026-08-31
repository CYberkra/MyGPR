# -*- coding: utf-8 -*-
"""CollapsiblePanel — 可折叠侧栏容器。

把任意内容 QWidget 包裹成可横向折叠/展开的侧栏。
折叠按钮为沿面板边缘的纵向长条（chevron 图标 + 主题色淡底），
展开/折叠时都容易发现；折叠后按钮铺满整个窄条。

与全局 LogPanel 的折叠动画风格保持一致：
- QPropertyAnimation(maximumWidth) 220ms OutCubic
- 主题色淡底 + hover 加深
"""

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, pyqtSignal
from PyQt6.QtGui import QIcon, QTransform
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon as FIF, PushButton, themeColor

_EXPAND_BUTTON_WIDTH = 16


def chevron_left_icon() -> QIcon:
    """CHEVRON_RIGHT_MED 水平翻转得到向左 chevron。"""
    pm = FIF.CHEVRON_RIGHT_MED.icon().pixmap(16, 16)
    return QIcon(pm.transformed(QTransform().scale(-1, 1)))


def collapse_button_qss() -> str:
    c = themeColor()
    r, g, b = c.red(), c.green(), c.blue()
    return (
        'PushButton {'
        f' background-color: rgba({r}, {g}, {b}, 0.10);'
        f' border: 1px solid rgba({r}, {g}, {b}, 0.40);'
        ' border-radius: 6px; padding: 0; }'
        'PushButton:hover {'
        f' background-color: rgba({r}, {g}, {b}, 0.22);'
        f' border: 1px solid rgba({r}, {g}, {b}, 0.60); }}'
        'PushButton:pressed {'
        f' background-color: rgba({r}, {g}, {b}, 0.32); }}')


class CollapsiblePanel(QWidget):
    """可折叠侧栏面板。

    参数:
        side: 'left' 或 'right'，决定折叠按钮放置侧。
        expand_width: 展开时的最大宽度。
        collapse_width: 折叠后的最小宽度。
    """

    sig_collapsed = pyqtSignal(bool)

    def __init__(self, side: str = 'right', expand_width: int = 400,
                 collapse_width: int = 40, parent=None):
        super().__init__(parent)
        self._side = side if side in ('left', 'right') else 'right'
        self._expand_width = int(expand_width)
        self._collapse_width = int(collapse_width)
        self._collapsed = False
        self._animating = False

        # 内容容器
        self._content = QWidget(self)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)

        # 折叠按钮：贴边的纵向长条，主题色淡底 + chevron 图标，
        # 常态可见、hover 加深，一眼可发现。
        self._collapse_btn = PushButton(self)
        self._collapse_btn.setFixedWidth(_EXPAND_BUTTON_WIDTH)
        # PushButton 默认垂直 sizePolicy 为 Fixed，需显式放开才能纵向铺满
        self._collapse_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._collapse_btn.setStyleSheet(collapse_button_qss())
        self._collapse_btn.clicked.connect(self.toggle)
        self._update_button_icon()

        # 布局：按钮 + 内容（或 内容 + 按钮），按钮纵向铺满
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        if self._side == 'left':
            layout.addWidget(self._collapse_btn, 0)
            layout.addWidget(self._content, 1)
        else:
            layout.addWidget(self._content, 1)
            layout.addWidget(self._collapse_btn, 0)

        # 动画
        self._animation = QPropertyAnimation(self, b'maximumWidth', self)
        self._animation.setDuration(220)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

        self.setMaximumWidth(self._expand_width)

    # ------------------------------------------------------------ 内容
    def set_content_widget(self, widget: QWidget) -> None:
        """设置面板内容部件（面板接管其生命周期）。"""
        self._content_layout.addWidget(widget, 1)

    def content_widget(self) -> QWidget:
        """返回内容容器，便于外部直接 addLayout。"""
        return self._content

    # ------------------------------------------------------------ 状态
    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool, animate: bool = True) -> None:
        """折叠或展开面板。"""
        collapsed = bool(collapsed)
        if self._collapsed == collapsed and not self._animating:
            return
        if self._animating:
            return

        self._animating = True
        self._collapsed = collapsed

        target = self._collapse_width if collapsed else self._expand_width
        current = self.maximumWidth()
        self._animation.stop()
        self._animation.setStartValue(current)
        self._animation.setEndValue(target)

        # 折叠后按钮铺满窄条，作为唯一的展开入口
        self._collapse_btn.setFixedWidth(
            self._collapse_width - 4 if collapsed else _EXPAND_BUTTON_WIDTH)

        if collapsed:
            self._content.setVisible(False)
            self._update_button_icon()

        if animate:
            self._animation.start()
        else:
            self.setMaximumWidth(target)
            self._on_animation_finished()

    def toggle(self) -> None:
        """切换折叠/展开。"""
        self.set_collapsed(not self._collapsed)

    # ------------------------------------------------------------ 内部
    def _on_animation_finished(self) -> None:
        self._animating = False
        if not self._collapsed:
            self._content.setVisible(True)
            self._update_button_icon()
        self.sig_collapsed.emit(self._collapsed)

    def _update_button_icon(self) -> None:
        if self._side == 'left':
            # 左栏：展开时点击向左收起（chevron ←），折叠后向右展开（chevron →）
            icon = FIF.CHEVRON_RIGHT_MED.icon() if self._collapsed \
                else chevron_left_icon()
        else:
            icon = chevron_left_icon() if self._collapsed \
                else FIF.CHEVRON_RIGHT_MED.icon()
        self._collapse_btn.setIcon(icon)
        self._collapse_btn.setToolTip('展开面板' if self._collapsed else '收起面板')

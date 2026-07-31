# -*- coding: utf-8 -*-
"""CollapsiblePanel — 可折叠侧栏容器。

把任意内容 QWidget 包裹成可横向折叠/展开的侧栏。
折叠后只保留一个窄按钮条，点击可重新展开。

与全局 LogPanel 的折叠动画风格保持一致：
- QPropertyAnimation(maximumWidth) 220ms OutCubic
- 折叠按钮 QSS 去圆角去边框
"""

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon as FIF, PushButton


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

        # 折叠按钮：明显底色 + 描边 + hover 高亮，窄条也能一眼发现
        self._collapse_btn = PushButton(self)
        self._collapse_btn.setFixedSize(22, 64)
        self._update_button_icon()
        self._collapse_btn.setToolTip('收起 / 展开面板')
        self._collapse_btn.setStyleSheet(
            'PushButton {'
            ' background-color: rgba(0, 120, 215, 0.10);'
            ' border: 1px solid rgba(0, 120, 215, 0.45);'
            ' border-radius: 4px; padding: 0; }'
            'PushButton:hover { background-color: rgba(0, 120, 215, 0.25); }'
            'PushButton:pressed { background-color: rgba(0, 120, 215, 0.35); }')
        self._collapse_btn.clicked.connect(self.toggle)

        # 布局：按钮 + 内容（或 内容 + 按钮）
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if self._side == 'left':
            layout.addWidget(self._collapse_btn, 0, Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(self._content, 1)
        else:
            layout.addWidget(self._content, 1)
            layout.addWidget(self._collapse_btn, 0, Qt.AlignmentFlag.AlignVCenter)

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
            icon = FIF.RIGHT_ARROW if self._collapsed else FIF.LEFT_ARROW
        else:
            icon = FIF.LEFT_ARROW if self._collapsed else FIF.RIGHT_ARROW
        self._collapse_btn.setIcon(icon)

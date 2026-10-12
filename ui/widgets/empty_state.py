# -*- coding: utf-8 -*-
"""EmptyStateOverlay — 画布空态引导浮层（评审 P0-1 统一组件）。

数据类视图（B-Scan / 地图 / 成果表）零数据时原本只剩空画布或孤立表头，
无任何引导。本组件以浮层覆盖在宿主控件之上：图标 + 主句 + 副句，随宿主
resize 自动铺满，主题切换自刷新（主窗口遍历的鸭子类型 ``apply_theme``）。

用法::

    self._empty = EmptyStateOverlay(
        self._canvas, icon=FIF.PHOTO, title='暂无数据',
        hint='导入测线后此处显示剖面')
    # 数据到达时隐藏：
    self._empty.setVisible(False)

浮层不接管宿主生命周期（宿主即父控件），且始终鼠标事件穿透，不挡宿主
右键菜单等交互。
"""
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel
from qfluentwidgets import FluentIconBase
from qfluentwidgets import isDarkTheme

from ui.theme_helpers import control_palette, hint_qss, status_color

__all__ = ['EmptyStateOverlay']

_ICON_SIZE = 40


class EmptyStateOverlay(QWidget):
    """画布空态浮层：图标 + 主句 + 副句。

    :param host: 被覆盖的宿主控件（画布/表格），浮层随其 resize 铺满；
    :param icon: FluentIconBase（以 secondary 色着色）或 None；
    :param title: 主句（一句话说清「什么数据会出现在这里」）；
    :param hint: 副句（可选，说明前置条件）。

    早期版本带一个可选行动按钮（``action_text`` + ``sig_action``），但全部
    10 处调用点都不传按钮文案，该分支始终不可达，已移除。需要「引导用户去
    某处操作」时，请由宿主页面自己放按钮——浮层刻意保持鼠标穿透。
    """

    def __init__(self, host: QWidget, *, icon: FluentIconBase | None,
                 title: str, hint: str = ''):
        super().__init__(host)
        self._icon_base = icon

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)
        layout.addStretch(1)

        self._icon_label = QLabel(self)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._icon_label)

        self._title_label = BodyLabel(str(title), self)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title_label)

        self._hint_label = None
        if hint:
            self._hint_label = CaptionLabel(str(hint), self)
            self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._hint_label.setWordWrap(True)
            layout.addWidget(self._hint_label)

        layout.addStretch(1)

        # 浮层始终鼠标穿透：不挡宿主右键菜单 / 滚轮等交互
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        host.installEventFilter(self)
        self.setGeometry(host.rect())
        self.apply_theme(isDarkTheme())

    # ------------------------------------------------------------ 事件
    def eventFilter(self, obj, event) -> bool:
        """宿主 resize → 浮层同步铺满宿主客户区。"""
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parent().rect())
        return super().eventFilter(obj, event)

    def showEvent(self, event) -> None:
        """显示瞬间对齐宿主当前尺寸（首次显示前可能错过 resize）。"""
        self.setGeometry(self.parent().rect())
        super().showEvent(event)

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """图标与文字随主题（主窗口全量遍历鸭子类型调用）。"""
        palette = control_palette(dark)
        self._title_label.setStyleSheet(f'color: {palette["text"]};')
        if self._hint_label is not None:
            self._hint_label.setStyleSheet(hint_qss('secondary'))
        if self._icon_base is not None:
            icon_color = QColor(status_color('secondary'))
            self._icon_label.setPixmap(
                self._icon_base.icon(icon_color).pixmap(
                    _ICON_SIZE, _ICON_SIZE))

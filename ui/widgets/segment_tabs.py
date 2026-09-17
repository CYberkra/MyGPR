# -*- coding: utf-8 -*-
"""SlimSegment — 卡片内子标签（瘦版 SegmentedWidget）。

库版 SegmentedWidget 按页面级导航设计（14px 字、10px 纵向 padding、底部
主题色指示条），三件叠加在卡片头部过重。本控件差异：

- 字号跟随全局正文（``FONT_SIZE_BODY`` 10pt ≈ 13px，库版固定 14px）；
- 纵向 padding 收紧到 3px；
- ``paintEvent`` 只画滑块胶囊、不画底部指示条（滑块已足够表达选中态）。

轨道与滑块样式沿用库版并随主题切换。全 app 三处子标签统一用本控件：
输出面板（日志/任务）、处理页「数据预览」卡、空间页「空间视图」卡。
"""

from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import SegmentedWidget, isDarkTheme, setCustomStyleSheet

from ui import constants

__all__ = ['SlimSegment']

# 页签字号：全局正文 10pt ≈ 13px（96 DPI），库版 SegmentedItem 固定 14px
_TAB_FONT_PX = round(constants.FONT_SIZE_BODY * 4 / 3)


class SlimSegment(SegmentedWidget):
    """库版 SegmentedWidget 的瘦身版，适配卡片头部行（~30px）。"""

    _QSS = (
        'SegmentedItem { padding: 3px 12px; }'
        'SegmentedItem[isSelected=false] { padding: 3px 12px; margin: 2px 0px; }'
        'SegmentedItem[isSelected=true] { padding-top: 3px; padding-bottom: 3px; }'
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        setCustomStyleSheet(self, self._QSS, self._QSS)

    def addItem(self, routeKey, text, onClick=None, icon=None):
        item = super().addItem(routeKey, text, onClick, icon)
        if item is not None:
            self.setItemFontSize(_TAB_FONT_PX)
        return item

    def paintEvent(self, e):
        QWidget.paintEvent(self, e)   # 跳过库版的指示条/滑块绘制，下面只画滑块
        item = self.currentItem()
        if item is None:
            return
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        if isDarkTheme():
            painter.setPen(QColor(255, 255, 255, 14))
            painter.setBrush(QColor(255, 255, 255, 15))
        else:
            painter.setPen(QColor(0, 0, 0, 19))
            painter.setBrush(QColor(255, 255, 255, 179))
        rect = item.rect().adjusted(1, 1, -1, -1).translated(
            int(self.slideAni.value()), 0)
        painter.drawRoundedRect(rect, 5, 5)

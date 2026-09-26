# -*- coding: utf-8 -*-
"""BScanGallery — 总览墙：全部打开数据源的滚动网格画廊。

定位是**总览工具而非精读工具**（格内像素密度低于 0.45px/采样红线）：
跑完链且 tab 数 >4 时自动弹一次，平时经 tab 栏「总览墙」按钮唤起；
点格子的标题按钮把该源送回主区并关闭画廊。非模态、随宿主页销毁。
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QGridLayout, QScrollArea, QVBoxLayout, QWidget

from qfluentwidgets import PushButton

from ui.widgets.bscan_view import BScanView

_COLUMNS = 3


class BScanGallery(QDialog):
    """总览墙：每格一个迷你 B-Scan（无色标、轻量），标题即拾取按钮。"""

    def __init__(self, page) -> None:
        super().__init__(page)
        self._page = page
        self.setWindowTitle('总览墙 — 打开的数据源')
        self.setModal(False)
        self.setMinimumSize(900, 620)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        grid = QGridLayout(content)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setSpacing(10)

        for index, source in enumerate(page._preview_sources):
            row, col = divmod(index, _COLUMNS)
            pick = PushButton(source['title'], content)
            pick.clicked.connect(
                lambda _checked=False, key=source['key']: self._pick(key))
            grid.addWidget(pick, row * 2, col)
            view = BScanView(content, with_colorbar=False)
            view.setMinimumSize(380, 280)
            if source['bundle'] is not None:
                view.set_bundle(source['bundle'])
            grid.addWidget(view, row * 2 + 1, col)

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _pick(self, key: str) -> None:
        """点格子：把该源送回主区（选中之 → 可见性规则换入主面板）。"""
        self._page.on_gallery_pick(key)
        self.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)

# -*- coding: utf-8 -*-
"""FullscreenHost — B-Scan 视图的「独立窗口展开」宿主。

做法不是把画面复制一份（那要同步色标/ levels / 轴单位等一堆状态），而是
**临时改嫁**：把同一个 QWidget 实例挂到本窗口的布局里，关闭时按原 (父控件,
布局, 位置, stretch) 原位放回。同一实例意味着所有交互状态（缩放、漫游、
字号跟随）完全连续，不需要同步。

Esc 关闭；几何写到宿主自己的 QByteArray 由调用方持久化（本项目统一走
ui.settings_manager 的 JSON，本模块不碰文件）。
"""
from __future__ import annotations

from PyQt6.QtCore import QByteArray, Qt, pyqtSignal
from PyQt6.QtWidgets import QBoxLayout, QVBoxLayout, QWidget

__all__ = ['FullscreenHost']


class FullscreenHost(QWidget):
    """承载一个 QWidget 的临时独立窗口；take/release 成对使用。"""

    closed = pyqtSignal()   # 窗口关闭（宿主视图已还原原位）后发出

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent, Qt.WindowType.Window | Qt.WindowType.WindowMaximizeButtonHint)
        self.setWindowTitle('B-Scan 全屏浏览（Esc 退出）')
        self._guest: QWidget | None = None
        self._guest_parent: QWidget | None = None
        self._guest_layout: QBoxLayout | None = None
        self._guest_index = -1
        self._guest_stretch = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    # ------------------------------------------------------------ 接管 / 还原
    def take(self, guest: QWidget) -> None:
        """记住 guest 的原位信息后接管它（幂等：重复调用前先 release）。"""
        if self._guest is not None:
            self.release()
        self._guest = guest
        self._guest_parent = guest.parentWidget()
        self._guest_layout = None
        self._guest_index = -1
        self._guest_stretch = 0
        parent = self._guest_parent
        layout = parent.layout() if parent is not None else None
        if isinstance(layout, QBoxLayout):
            index = layout.indexOf(guest)
            if index >= 0:
                self._guest_layout = layout
                self._guest_index = index
                self._guest_stretch = layout.stretch(index)
        self.layout().addWidget(guest)
        guest.show()

    def release(self) -> None:
        """把 guest 放回原位（父控件 / 布局 / 位置 / stretch）。"""
        guest = self._guest
        if guest is None:
            return
        self._guest = None
        self.layout().removeWidget(guest)
        guest.setParent(self._guest_parent)
        layout = self._guest_layout
        if layout is not None and self._guest_index >= 0:
            layout.insertWidget(self._guest_index, guest, self._guest_stretch)
        guest.show()
        self._guest_layout = None
        self._guest_index = -1

    # ------------------------------------------------------------ 几何持久化
    def captured_geometry(self) -> QByteArray:
        """导出当前窗口几何（调用方按需持久化）。"""
        return self.saveGeometry()

    def restore_saved_geometry(self, data: QByteArray | bytes | None) -> bool:
        """回放几何；无历史几何返回 False（调用方据此改为全屏/最大化）。"""
        if data is None:
            return False
        blob = data if isinstance(data, QByteArray) else QByteArray(data)
        if blob.isEmpty():
            return False
        return bool(self.restoreGeometry(blob))

    # ------------------------------------------------------------ 事件
    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self.release()
        super().closeEvent(event)
        self.closed.emit()

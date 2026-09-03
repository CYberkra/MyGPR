"""AScanPopup — B-scan 点击取道的波形跟随浮窗。

非模态独立窗口，包装现有 AScanView；关闭仅隐藏（实例由 BScanView
持有复用），窗口几何持久化到 ui settings 由调用方负责。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ui.widgets.ascan_view import AScanView


class AScanPopup(QWidget):
    """单道波形跟随浮窗：show_trace 更新曲线与标题，close 隐藏不销毁。"""

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setWindowTitle('A-Scan 波形跟随')
        self.resize(420, 320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._ascan_view = AScanView(self)
        layout.addWidget(self._ascan_view)

    def show_trace(self, samples, *, trace_index: int,
                   distance_m: float | None = None) -> None:
        """显示指定道的波形；标题带道号与里程（有则）。"""
        title = f'A-Scan 波形 — 道 {trace_index}'
        if distance_m is not None:
            title += f'（{distance_m:.2f} m）'
        self._ascan_view.set_trace(samples, title=title)
        self.show()
        self.raise_()

    def clear(self) -> None:
        self._ascan_view.clear()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        # 关闭=隐藏，实例复用；通知宿主同步菜单勾选态
        self.hide()
        event.accept()

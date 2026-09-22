# -*- coding: utf-8 -*-
"""LevelsDialog — B-Scan 显示动态范围（色阶）设置弹窗。

处理页工具条已有低%/高% 两个 SpinBox，但主页 / 解释页的 B-Scan 没有；
与其每页再摆一排控件，不如收进视图右键菜单里的一枚弹窗（用户心智：
「这图的明暗不对 → 右键 → 色阶设置」）。

只改**显示**的 vmin/vmax（百分位裁切），绝不动数据本身。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFormLayout, QLabel, QVBoxLayout)

from ui import constants

__all__ = ['LevelsDialog']

_HINT = '在图内取像素灰度的低/高百分位作为显示上下限（不改数据）'


class LevelsDialog(QDialog):
    """低/高百分位输入弹窗；``percentiles()`` 取结果，取消返回 None。"""

    def __init__(self, parent=None, p_low: float = 2.0,
                 p_high: float = 98.0) -> None:
        super().__init__(parent)
        self.setWindowTitle('色阶设置')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        root = QVBoxLayout(self)
        root.setContentsMargins(constants.PAGE_MARGINS[0] or 16, 16, 16, 12)
        root.setSpacing(10)

        form = QFormLayout()
        self._low_spin = QDoubleSpinBox(self)
        self._high_spin = QDoubleSpinBox(self)
        for spin, value in ((self._low_spin, p_low), (self._high_spin, p_high)):
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setValue(float(value))
            spin.setMinimumWidth(120)
        self._low_spin.setToolTip('低于该百分位的灰度被压到色阶最暗端')
        self._high_spin.setToolTip('高于该百分位的灰度被压到色阶最亮端')
        form.addRow('低百分位 (%):', self._low_spin)
        form.addRow('高百分位 (%):', self._high_spin)
        root.addLayout(form)

        self._hint = QLabel(_HINT, self)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            f'color: gray; font-size: {constants.FONT_SIZE_SECONDARY}pt;')
        root.addWidget(self._hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('确定')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ---------------------------------------------------------------- 结果
    def percentiles(self) -> tuple[float, float] | None:
        """确定的百分位；用户取消返回 None。"""
        if self.result() != QDialog.DialogCode.Accepted:
            return None
        return float(self._low_spin.value()), float(self._high_spin.value())

    def _accept_if_valid(self) -> None:
        """低百分位必须严格小于高百分位，否则提示而非静默应用。"""
        low = float(self._low_spin.value())
        high = float(self._high_spin.value())
        if low >= high:
            self._hint.setText('低百分位必须小于高百分位（例如 2 / 98）')
            self._hint.setStyleSheet(
                f'color: #c8000a; font-size: {constants.FONT_SIZE_SECONDARY}pt;')
            return
        self.accept()

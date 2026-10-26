"""Per-view time zoom and local display levels; never change the data matrix."""

import math
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout
from ui.widgets.context_menus import RoundMenu, add_action


class TimeWindowDialog(QDialog):
    def __init__(self, parent, bounds, current):
        super().__init__(parent)
        self.setWindowTitle("查看时窗")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.bounds = bounds
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.start_spin = QDoubleSpinBox(self)
        self.end_spin = QDoubleSpinBox(self)
        for spin, value in zip((self.start_spin, self.end_spin), current):
            spin.setDecimals(6)
            spin.setRange(*bounds)
            spin.setSuffix(" ns")
            spin.setValue(value)
            spin.setMinimumWidth(170)
        form.addRow("起始时间:", self.start_spin)
        form.addRow("结束时间:", self.end_spin)
        layout.addLayout(form)
        label = QLabel("仅缩放当前 B 扫；保留横向视野，不裁剪或处理数据。", self)
        label.setWordWrap(True)
        layout.addWidget(label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.reset_button = self.buttons.addButton("完整时窗", QDialogButtonBox.ButtonRole.ResetRole)
        self.reset_button.clicked.connect(self._reset)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.start_spin.valueChanged.connect(self._validate)
        self.end_spin.valueChanged.connect(self._validate)
        layout.addWidget(self.buttons)
        self._validate()

    def _reset(self):
        self.start_spin.setValue(self.bounds[0])
        self.end_spin.setValue(self.bounds[1])

    def _validate(self):
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.start_spin.value() < self.end_spin.value()
        )


class BScanTimeWindowMixin:
    def _time_axis_for_view(self):
        if self._matrix is None or self._showing_elevation or "ns" not in self._sample_axis_label.lower():
            return None
        axis = np.asarray(self._sample_axis, dtype=float)
        if axis.ndim != 1 or len(axis) != self._matrix.shape[0] or len(axis) < 2:
            return None
        if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
            return None
        return axis

    def time_window_available(self):
        return self._time_axis_for_view() is not None

    def view_time_window(self):
        return self._view_time_window

    def set_view_time_window(self, start, end):
        axis = self._time_axis_for_view()
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            return False
        if axis is None or not (math.isfinite(start) and math.isfinite(end) and axis[0] <= start < end <= axis[-1]):
            return False
        self._selection_tool.cancel()
        self._view_selection = None
        rows = np.interp([start, end], axis, np.arange(len(axis)) + 0.5)
        self._plot.vb.setAspectLocked(False)
        self._plot.vb.setYRange(float(rows[0]), float(rows[1]), padding=0)
        self._view_time_window = (start, end)
        self._apply_levels_to_render()
        return True

    def _restore_view_time_window(self):
        if self._restore_rectangle_view():
            return
        if self._view_time_window is not None:
            self.set_view_time_window(*self._view_time_window)

    def restore_full_time_window(self):
        self._clear_view_selection()
        if self._image_shape is not None:
            self._plot.vb.setAspectLocked(False)
            self._plot.vb.setYRange(0, self._image_shape[1], padding=0)
            self._apply_levels_to_render()

    def _edit_view_time_window(self):
        axis = self._time_axis_for_view()
        if axis is None:
            return
        bounds = (float(axis[0]), float(axis[-1]))
        dialog = TimeWindowDialog(self, bounds, self._view_time_window or bounds)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_view_time_window(
                max(bounds[0], dialog.start_spin.value()), min(bounds[1], dialog.end_spin.value())
            )
        dialog.deleteLater()

    def _add_menu_time_window(self, menu):
        axis = self._time_axis_for_view()
        sub = RoundMenu("查看时窗" if axis is not None else "查看时窗（需 ns 时间轴）", menu)
        add_action(sub, None, "自定义时窗…", self._edit_view_time_window, enabled=axis is not None)
        for end in (100, 200, 400):
            add_action(
                sub,
                None,
                f"0–{end} ns",
                lambda _checked=False, value=end: self.set_view_time_window(0, value),
                enabled=axis is not None and axis[0] <= 0 and axis[-1] >= end,
            )
        sub.addSeparator()
        add_action(sub, None, "恢复完整时窗", self.restore_full_time_window, enabled=self._image_shape is not None)
        menu.addMenu(sub)

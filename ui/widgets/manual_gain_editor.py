"""Small parameter button and a node editor for saved manual curve gain."""
from __future__ import annotations

import math

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget
from qfluentwidgets import DoubleSpinBox, PrimaryPushButton, PushButton, isDarkTheme

from ui.desktop_backend_facade import manual_gain_nodes, manual_gain_nodes_json
from ui.widgets.manual_gain_preview import ManualGainPreview


class _GainNodes(pg.ScatterPlotItem):
    selected = pyqtSignal(int)
    moved = pyqtSignal(int, float, float)
    removed = pyqtSignal(int)

    def __init__(self):
        super().__init__(pxMode=True)
        self._drag_index = None

    def mouseClickEvent(self, event):
        points = self.pointsAt(event.pos())
        if not len(points):
            event.ignore()
            return
        index = int(points[0].data())
        if event.button() == Qt.MouseButton.RightButton:
            self.removed.emit(index)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(index)
        event.accept()

    def mouseDragEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        if event.isStart():
            points = self.pointsAt(event.buttonDownPos())
            self._drag_index = int(points[0].data()) if len(points) else None
        if self._drag_index is None:
            event.ignore()
            return
        event.accept()
        self.selected.emit(self._drag_index)
        self.moved.emit(self._drag_index, float(event.pos().y()) / 100.0, float(event.pos().x()))
        if event.isFinish():
            self._drag_index = None


class GainCurveDialog(QDialog):
    """Edit a private draft; only accepting the dialog updates its parameter."""

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动曲线增益")
        self.resize(1180, 650)
        self._nodes = manual_gain_nodes(value)
        self._selected = 0
        root = QVBoxLayout(self)
        body = QSplitter(Qt.Orientation.Horizontal, self)
        editor_panel = QWidget(body)
        editor_panel.setMinimumWidth(280)
        layout = QVBoxLayout(editor_panel)
        layout.setContentsMargins(0, 0, 8, 0)
        root.addWidget(body, 1)
        hint = QLabel("位置以本步骤输入为准：0% 为首采样点，100% 为末采样点；预览会先计算支持的前置步骤。", self)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "增益", units="dB")
        self.plot.setLabel("left", "输入采样位置", units="%")
        self.plot.invertY(True)
        self.plot.setXRange(-60, 60, padding=0.02)
        self.plot.setYRange(0, 100, padding=0.02)
        self.plot.setMenuEnabled(False)
        self.plot.addItem(pg.InfiniteLine(0, angle=90, pen=(130, 130, 130)))
        self.curve = self.plot.plot([], [])
        self.scatter = _GainNodes()
        self.plot.addItem(self.scatter)
        layout.addWidget(self.plot, 1)
        row = QHBoxLayout()
        self.add_button = PushButton("添加节点")
        self.remove_button = PushButton("删除节点")
        self.reset_button = PushButton("还原为 0 dB")
        for button in (self.add_button, self.remove_button, self.reset_button):
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        form = QFormLayout()
        self.position_spin = DoubleSpinBox()
        self.position_spin.setRange(0, 100)
        self.position_spin.setDecimals(4)
        self.position_spin.setSuffix(" %")
        self.gain_spin = DoubleSpinBox()
        self.gain_spin.setRange(-60, 60)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSuffix(" dB")
        form.addRow("选中节点位置", self.position_spin)
        form.addRow("选中节点增益", self.gain_spin)
        layout.addLayout(form)
        self.status_label = QLabel("拖动节点调节；双击空白添加；右键节点或 Delete 删除。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.preview = ManualGainPreview(body)
        body.addWidget(editor_panel)
        body.addWidget(self.preview)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([320, 820])
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.ok_button = PrimaryPushButton("应用曲线")
        self.cancel_button = PushButton("取消")
        buttons.addWidget(self.ok_button)
        buttons.addWidget(self.cancel_button)
        root.addLayout(buttons)
        self.scatter.selected.connect(self.select_node)
        self.scatter.moved.connect(self.move_node)
        self.scatter.removed.connect(self.remove_node)
        self.plot.viewport().installEventFilter(self)
        self.add_button.clicked.connect(self._add_between)
        self.remove_button.clicked.connect(lambda: self.remove_node(self._selected))
        self.reset_button.clicked.connect(self.reset_curve)
        self.position_spin.setAccessibleName("选中节点位置百分比")
        self.gain_spin.setAccessibleName("选中节点增益 dB")
        self.position_spin.valueChanged.connect(self._edit_position)
        self.gain_spin.valueChanged.connect(self._edit_gain)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.preview.readyChanged.connect(self.ok_button.setEnabled)
        self.finished.connect(self.preview.shutdown)
        self.apply_theme(isDarkTheme())
        self._refresh()

    def apply_theme(self, dark):
        palette = self.palette()
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Button):
            palette.setColor(role, QColor("#202020" if dark else "#ffffff"))
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
            palette.setColor(role, QColor("#eeeeee" if dark else "#222222"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.plot.setBackground("#202020" if dark else "#ffffff")
        color = "#eeeeee" if dark else "#222222"
        for name in ("left", "bottom"):
            self.plot.getAxis(name).setPen(color)
            self.plot.getAxis(name).setTextPen(color)
        self.curve.setPen("#74aaff" if dark else "#2563eb", width=2)
        self.preview.view.apply_theme(dark)

    def value(self):
        return manual_gain_nodes_json(self._nodes)

    def _refresh(self):
        self.curve.setData([n["db"] for n in self._nodes], [n["tau"] * 100 for n in self._nodes])
        self.scatter.setData(
            x=[n["db"] for n in self._nodes], y=[n["tau"] * 100 for n in self._nodes],
            data=list(range(len(self._nodes))), size=12, brush="#e9a23b", pen="#80541b",
        )
        self.select_node(min(self._selected, len(self._nodes) - 1))
        self.add_button.setEnabled(len(self._nodes) < 32)
        self.preview.set_curve(self.value())

    def select_node(self, index):
        if not 0 <= index < len(self._nodes):
            return
        self._selected = index
        node = self._nodes[index]
        self.position_spin.blockSignals(True)
        self.gain_spin.blockSignals(True)
        self.position_spin.setValue(node["tau"] * 100)
        self.gain_spin.setValue(node["db"])
        self.position_spin.setEnabled(0 < index < len(self._nodes) - 1)
        self.remove_button.setEnabled(0 < index < len(self._nodes) - 1)
        self.position_spin.blockSignals(False)
        self.gain_spin.blockSignals(False)

    def move_node(self, index, tau, db):
        if not 0 <= index < len(self._nodes) or not math.isfinite(tau) or not math.isfinite(db):
            return
        if index == 0:
            tau = 0.0
        elif index == len(self._nodes) - 1:
            tau = 1.0
        else:
            left, right = self._nodes[index - 1]["tau"], self._nodes[index + 1]["tau"]
            tau = max(math.nextafter(left, right), min(math.nextafter(right, left), tau))
        self._nodes[index] = {"tau": tau, "db": max(-60.0, min(60.0, db))}
        self._selected = index
        self._refresh()

    def _edit_position(self, *_args):
        self.move_node(self._selected, self.position_spin.value() / 100, self.gain_spin.value())

    def _edit_gain(self, *_args):
        # Numeric display precision must not shift a restored high-precision position.
        self.move_node(self._selected, self._nodes[self._selected]["tau"], self.gain_spin.value())

    def add_node(self, tau, db=None):
        try:
            if len(self._nodes) >= 32 or not 0 < tau < 1:
                raise ValueError("最多 32 个节点；新节点须位于 0%～100% 之间")
            if any(abs(node["tau"] - tau) <= 1e-6 for node in self._nodes):
                raise ValueError("新节点与现有节点过近")
            index = next(i for i, node in enumerate(self._nodes) if node["tau"] > tau)
            if db is None:
                left, right = self._nodes[index - 1], self._nodes[index]
                ratio = (tau - left["tau"]) / (right["tau"] - left["tau"])
                db = left["db"] + ratio * (right["db"] - left["db"])
            nodes = [*self._nodes[:index], {"tau": tau, "db": db}, *self._nodes[index:]]
            self._nodes = manual_gain_nodes(nodes)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return False
        self._selected = index
        self._refresh()
        return True

    def _add_between(self):
        index = min(self._selected + 1, len(self._nodes) - 1)
        self.add_node((self._nodes[index - 1]["tau"] + self._nodes[index]["tau"]) / 2)

    def eventFilter(self, watched, event):
        if (watched is self.plot.viewport() and event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton):
            view = self.plot.getViewBox()
            scene_position = self.plot.mapToScene(event.position().toPoint())
            if view.sceneBoundingRect().contains(scene_position):
                position = view.mapSceneToView(scene_position)
                if not len(self.scatter.pointsAt(position)):
                    self.add_node(float(position.y()) / 100, max(-60.0, min(60.0, float(position.x()))))
                return True
        return super().eventFilter(watched, event)

    def remove_node(self, index):
        if index <= 0 or index >= len(self._nodes) - 1:
            return False
        del self._nodes[index]
        self._refresh()
        return True

    def reset_curve(self):
        self._nodes = manual_gain_nodes()
        self._refresh()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.remove_node(self._selected)
            event.accept()
        else:
            super().keyPressEvent(event)


class GainCurveEditor(QWidget):
    valueChanged = pyqtSignal()
    previewRequested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = manual_gain_nodes_json()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = PushButton(self)
        self.button.setMinimumHeight(34)
        self.button.clicked.connect(self._edit)
        layout.addWidget(self.button)
        self._update_summary()

    def value(self):
        return self._value

    def setValue(self, value):
        encoded = manual_gain_nodes_json(value)
        if encoded == self._value:
            return
        self._value = encoded
        self._update_summary()
        self.valueChanged.emit()

    def _update_summary(self):
        nodes = manual_gain_nodes(self._value)
        lo, hi = min(n["db"] for n in nodes), max(n["db"] for n in nodes)
        self.button.setText(f"编辑曲线 · {len(nodes)} 节点")
        self.button.setToolTip(f"{lo:g}～{hi:g} dB。沿当前输入范围按百分比设置增益；处理后保存为新成果。")

    def _edit(self):
        dialog = GainCurveDialog(self._value, self)
        self.previewRequested.emit(dialog)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.setValue(dialog.value())
        dialog.deleteLater()

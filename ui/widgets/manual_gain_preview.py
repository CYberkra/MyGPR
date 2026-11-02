"""Full-resolution live gain preview with fixed color levels and latest-only work."""
from __future__ import annotations

from dataclasses import replace
from threading import Event

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import ComboBox, PushButton

from ui.controllers.backend_controller import run_command
from ui.desktop_backend_facade import compute_display_levels, evaluate_manual_gain_preview
from ui.widgets.bscan_view import BScanView


class _PreviewSignals(QObject):
    ready = pyqtSignal(int, int, object, str)


class _GainPreviewCommand:
    def __init__(self, matrix, nodes, generation, input_generation, signals, cancelled):
        self.matrix, self.nodes, self.generation = matrix, nodes, generation
        self.input_generation = input_generation
        self.signals, self.cancelled = signals, cancelled

    def execute(self):
        if self.cancelled.is_set():
            return
        try:
            matrix = evaluate_manual_gain_preview(self.matrix, self.nodes)
            result, error = (matrix, compute_display_levels(matrix, p_low=2, p_high=98)), ""
        except (ValueError, TypeError, RuntimeError, MemoryError, OverflowError) as exc:
            result = None
            error = "完整预览内存不足，请先裁剪输入范围" if isinstance(exc, MemoryError) else str(exc)
        if not self.cancelled.is_set():
            self.signals.ready.emit(self.generation, self.input_generation, result, error)


class ManualGainPreview(QWidget):
    readyChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bundle = None
        self._result = None
        self._result_levels = None
        self._nodes = None
        self._generation = 0
        self._input_generation = 0
        self._active = False
        self._cancelled = Event()
        # Unparented emitter is held by the command too, so a closing dialog
        # cannot delete the emitter while its NumPy calculation is finishing.
        self._signals = _PreviewSignals()
        self._signals.ready.connect(self._received)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._launch)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.mode = ComboBox(self)
        self.mode.addItems(["增益后", "增益前"])
        self.mode.setAccessibleName("手动增益预览对比")
        row.addWidget(self.mode)
        self.rescale_button = PushButton("按当前图定标", self)
        self.rescale_button.setToolTip("以当前图重新确定色阶，再锁定；之后调节点不自动定标。")
        row.addWidget(self.rescale_button)
        self.fit_button = PushButton("全览", self)
        row.addWidget(self.fit_button)
        row.addStretch()
        layout.addLayout(row)
        self.source_label = QLabel("尚未关联处理输入", self)
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self.view = BScanView(self)
        self.view.setMinimumSize(360, 250)
        self.view._toggle_selection_levels(False)
        # The preview has one gain and one fixed reference scale. Additional
        # display gain/normalization would make before/after comparisons false.
        self.view._glw.scene().sigMouseClicked.disconnect(self.view._on_mouse_clicked)
        self.view.setToolTip("预览不叠加其他显示增益；支持滚轮缩放、平移和全览。")
        layout.addWidget(self.view, 1)
        self.status_label = QLabel("选定测线输入后，拖动曲线即可预览；此窗口不保存处理成果。", self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.mode.currentIndexChanged.connect(self._render)
        self.rescale_button.clicked.connect(self._rescale)
        self.fit_button.clicked.connect(self.view.fit_to_data)
        self.rescale_button.setEnabled(False)
        self.fit_button.setEnabled(False)

    def set_loading(self, description):
        self.source_label.setText(description)
        self.status_label.setText("正在后台准备输入与前置步骤…")
        self.readyChanged.emit(False)

    def set_input(self, bundle, description):
        if self._cancelled.is_set():
            return
        self._bundle = bundle
        self._input_generation += 1
        self._result = None
        self._result_levels = None
        self.source_label.setText(description)
        self.view.set_levels_override((bundle.vmin, bundle.vmax))
        self._render()
        self.fit_button.setEnabled(True)
        self.set_curve(self._nodes)

    def set_error(self, message):
        self.status_label.setText("预览不可用：" + str(message))
        self.readyChanged.emit(False)

    def set_curve(self, nodes):
        self._nodes = nodes
        self._generation += 1
        if self._bundle is not None and not self._cancelled.is_set():
            self.readyChanged.emit(False)
            self.status_label.setText("正在更新增益预览…（色阶固定）")
            if not self._active and not self._timer.isActive():
                self._timer.start()

    def _launch(self):
        if self._active or self._bundle is None or self._cancelled.is_set():
            return
        self._active = True
        run_command(_GainPreviewCommand(self._bundle.matrix, self._nodes, self._generation,
                                       self._input_generation, self._signals, self._cancelled), name="mygpr-live-gain")

    def _received(self, generation, input_generation, result, error):
        self._active = False
        if self._cancelled.is_set():
            return
        if input_generation != self._input_generation:
            self._timer.start()
            return
        if error:
            if generation != self._generation:
                self._timer.start()
                return
            self.set_error(error)
            return
        self._result, self._result_levels = result
        self._render()
        if generation != self._generation:
            self.status_label.setText("正在更新…（图像为最近完成的预览帧，色阶固定）")
            self._timer.start()
            return
        self.status_label.setText("已更新 · 色阶固定 · 应用曲线只更新参数，运行处理链后才保存成果。")
        self.rescale_button.setEnabled(True)
        self.readyChanged.emit(True)

    def _render(self, *_):
        if self._bundle is None:
            return
        matrix = (self._result if self.mode.currentIndex() == 0 and self._result is not None
                  else self._bundle.matrix)
        state = self.view.export_display_state() if self.view._matrix is not None else {}
        self.view.set_bundle(replace(self._bundle, matrix=matrix))
        if state:
            self.view.restore_display_state(state, viewport=True)

    def _rescale(self):
        if self._bundle is None:
            return
        levels = ((self._bundle.vmin, self._bundle.vmax) if self.mode.currentIndex() == 1
                  else self._result_levels)
        if levels is not None:
            self.view.set_levels_override(levels)

    def shutdown(self, *_):
        self._cancelled.set()
        self._timer.stop()
        self._generation += 1

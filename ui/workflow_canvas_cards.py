#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComfyUI-like workflow canvas cards for MyGPR."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QEvent, QPoint, QPointF, QSignalBlocker, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.methods_registry import PROCESSING_METHODS, get_method_display_name
from core.workflow_data import METHOD_CATEGORIES, WORKFLOW_STAGE_BY_ID, WorkflowMethod
from ui.workflow_canvas_preview import BscanPreviewCard


class WorkflowNodeCard(QFrame):
    """Editable algorithm node embedded into the graphics canvas."""

    selected = pyqtSignal(int)
    changed = pyqtSignal(int)
    run_current_requested = pyqtSignal(int)
    run_from_requested = pyqtSignal(int)
    duplicate_requested = pyqtSignal(int)
    remove_requested = pyqtSignal(int)

    def __init__(self, row: int, method: WorkflowMethod, parent=None):
        super().__init__(parent)
        self.row = int(row)
        self.method = method
        self.expanded = False
        self.compact = False
        self._suppress = False
        self._param_getters: dict[str, Callable[[], object]] = {}
        self.setObjectName("workflowNodeCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(300)
        self.setMaximumWidth(360)
        self.setStyleSheet(
            """
            QFrame#workflowNodeCard {
                background: #ffffff;
                border: 1px solid #d7e2f0;
                border-radius: 14px;
            }
            QFrame#workflowNodeCard[current="true"] {
                border: 2px solid #3278ff;
                background: #f7faff;
            }
            QLabel#nodeTitle {
                font-weight: 800;
                color: #1f2d3d;
            }
            QLabel#nodeSubtitle {
                color: #52647a;
            }
            QLabel#nodeWarning {
                color: #a66a00;
                font-size: 12px;
            }
            QLabel#nodePortIn {
                color: #3278ff;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#nodePortOut {
                color: #7d4cff;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#nodeStatusChip {
                background: #eef4ff;
                color: #2457b8;
                border: 1px solid #cdddf8;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#nodeSectionLabel {
                color: #5f6f83;
                font-size: 12px;
                font-weight: 700;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #dbe5f2;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
                background: #3278ff;
            }
            QFrame#workflowNodeCard[compact="true"] {
                background: #f9fbff;
                border: 1px solid #8fb3ff;
            }
            QLabel#nodeCompactTitle {
                font-weight: 800;
                font-size: 15px;
                color: #0b4fd8;
            }
            QLabel#nodeCompactSubtitle {
                font-size: 13px;
                color: #1f2d3d;
            }
            QLabel#nodeCompactMeta {
                font-size: 12px;
                color: #52647a;
            }
            """
        )
        self._build()

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        self.selected.emit(self.row)
        super().mousePressEvent(event)

    def set_current(self, current: bool) -> None:
        self.setProperty("current", bool(current))
        self.style().unpolish(self)
        self.style().polish(self)

    def set_compact(self, compact: bool) -> None:
        compact = bool(compact)
        if self.compact == compact:
            return
        self.compact = compact
        self.setProperty("compact", compact)
        self._build()
        self.style().unpolish(self)
        self.style().polish(self)

    def _build(self) -> None:
        self._suppress = True
        try:
            self._param_getters.clear()
            old_layout = self.layout()
            if old_layout is not None:
                while old_layout.count():
                    item = old_layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
                QWidget().setLayout(old_layout)

            root = QVBoxLayout(self)
            root.setContentsMargins(9, 7, 9, 9)
            root.setSpacing(6)

            if self.compact:
                self._build_compact(root)
                return

            port_row = QHBoxLayout()
            port_row.setSpacing(4)
            input_port = QLabel("● 数据输入")
            input_port.setObjectName("nodePortIn")
            output_port = QLabel("数据输出 ●")
            output_port.setObjectName("nodePortOut")
            port_row.addWidget(input_port)
            port_row.addStretch(1)
            port_row.addWidget(output_port)
            root.addLayout(port_row)

            title_row = QHBoxLayout()
            title_row.setSpacing(8)
            title = QLabel(f"{self.row + 1:02d}. {self._stage_label()}")
            title.setObjectName("nodeTitle")
            title.setWordWrap(True)
            title_row.addWidget(title, 1)

            status_chip = QLabel(self._status_chip_text())
            status_chip.setObjectName("nodeStatusChip")
            title_row.addWidget(status_chip)

            self.enabled_check = QCheckBox("启用")
            self.enabled_check.setChecked(bool(self.method.enabled))
            self.enabled_check.toggled.connect(self._on_enabled_toggled)
            title_row.addWidget(self.enabled_check)
            root.addLayout(title_row)

            algorithm_label = QLabel("算法")
            algorithm_label.setObjectName("nodeSectionLabel")
            root.addWidget(algorithm_label)

            self.method_combo = QComboBox()
            for key in self._candidate_methods():
                if key in PROCESSING_METHODS:
                    self.method_combo.addItem(get_method_display_name(key), key)
            idx = self.method_combo.findData(self.method.method_id)
            self.method_combo.setCurrentIndex(max(idx, 0))
            self.method_combo.currentIndexChanged.connect(self._on_method_changed)
            self._install_wheel_guard(self.method_combo)
            root.addWidget(self.method_combo)

            utility_row = QHBoxLayout()
            utility_row.setSpacing(6)
            self.hidden_check = QCheckBox("隐藏")
            self.hidden_check.setChecked(bool(self.method.hidden))
            self.hidden_check.toggled.connect(self._on_hidden_toggled)
            utility_row.addWidget(self.hidden_check)
            utility_row.addStretch(1)
            root.addLayout(utility_row)

            warning = self._stage_warning()
            if warning:
                warning_label = QLabel(f"⚠ {warning}")
                warning_label.setObjectName("nodeWarning")
                warning_label.setWordWrap(True)
                root.addWidget(warning_label)

            param_metas = PROCESSING_METHODS.get(self.method.method_id, {}).get("params", [])
            max_params = len(param_metas) if self.expanded else min(2, len(param_metas))
            if max_params:
                param_label = QLabel("参数")
                param_label.setObjectName("nodeSectionLabel")
                root.addWidget(param_label)

                param_grid = QGridLayout()
                param_grid.setContentsMargins(0, 0, 0, 0)
                param_grid.setHorizontalSpacing(6)
                param_grid.setVerticalSpacing(5)
                for row, meta in enumerate(param_metas[:max_params]):
                    name = str(meta.get("name", ""))
                    label = QLabel(str(meta.get("label", name)))
                    label.setWordWrap(True)
                    control, getter = self._create_param_control(meta)
                    self._install_wheel_guard(control)
                    tooltip = str(meta.get("tooltip", ""))
                    if tooltip:
                        label.setToolTip(tooltip)
                        control.setToolTip(tooltip)
                    param_grid.addWidget(label, row, 0)
                    param_grid.addWidget(control, row, 1)
                    self._param_getters[name] = getter
                root.addLayout(param_grid)
            else:
                empty_label = QLabel("(无参数)")
                empty_label.setObjectName("nodeSubtitle")
                root.addWidget(empty_label)

            button_row = QHBoxLayout()
            button_row.setSpacing(6)
            for text, signal in [
                ("运行", self.run_current_requested),
                ("后续", self.run_from_requested),
            ]:
                button = QPushButton(text)
                button.setMinimumWidth(0)
                button.clicked.connect(lambda _=False, sig=signal: sig.emit(self.row))
                button_row.addWidget(button)
            if len(param_metas) > 2:
                self.more_button = QPushButton("收起" if self.expanded else "更多")
                self.more_button.clicked.connect(self._toggle_expanded)
                button_row.addWidget(self.more_button)
            for text, signal in [
                ("复制", self.duplicate_requested),
                ("删除", self.remove_requested),
            ]:
                button = QPushButton(text)
                button.setMinimumWidth(0)
                button.clicked.connect(lambda _=False, sig=signal: sig.emit(self.row))
                button_row.addWidget(button)
            root.addLayout(button_row)
        finally:
            self._suppress = False

    def _build_compact(self, root: QVBoxLayout) -> None:
        self.setMinimumWidth(210)
        self.setMaximumWidth(240)

        title = QLabel(f"{self.row + 1:02d}. {self._stage_label()}")
        title.setObjectName("nodeCompactTitle")
        title.setWordWrap(True)
        root.addWidget(title)

        subtitle = QLabel(self._short_text(get_method_display_name(self.method.method_id), 22))
        subtitle.setObjectName("nodeCompactSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        meta = QLabel(self._status_text())
        meta.setObjectName("nodeCompactMeta")
        root.addWidget(meta)

        summary = self._param_summary(max_items=2)
        if summary:
            params = QLabel(summary)
            params.setObjectName("nodeCompactMeta")
            params.setWordWrap(True)
            root.addWidget(params)

        hint = QLabel("Ctrl+滚轮放大后可编辑参数")
        hint.setObjectName("nodeCompactMeta")
        hint.setWordWrap(True)
        root.addWidget(hint)

    def _status_chip_text(self) -> str:
        if self.method.hidden:
            return "HIDE"
        if not self.method.enabled:
            return "OFF"
        return "ON"

    def _status_text(self) -> str:
        if self.method.hidden:
            return "状态: 隐藏"
        if not self.method.enabled:
            return "状态: 停用"
        return "状态: 启用"

    def _install_wheel_guard(self, control: QWidget) -> None:
        """Swallow wheel events on embedded editors inside the graphics proxy."""
        guarded_types = (QAbstractSpinBox, QComboBox, QSlider)
        targets: list[QWidget] = []
        if isinstance(control, guarded_types):
            targets.append(control)
        targets.extend(control.findChildren(guarded_types))
        for target in targets:
            target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            for widget in [target, *target.findChildren(QWidget)]:
                widget.setProperty("workflowWheelGuard", True)
                widget.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        if event.type() == QEvent.Type.Wheel and isinstance(watched, QWidget):
            if bool(watched.property("workflowWheelGuard")):
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _param_summary(self, max_items: int = 3) -> str:
        if not self.method.params:
            return ""
        tokens: list[str] = []
        for key, value in list(self.method.params.items())[:max_items]:
            tokens.append(f"{key}={self._short_text(str(value), 16)}")
        if len(self.method.params) > max_items:
            tokens.append("...")
        return " · ".join(tokens)

    @staticmethod
    def _short_text(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"

    def _candidate_methods(self) -> list[str]:
        stage = WORKFLOW_STAGE_BY_ID.get(self.method.stage_id, {})
        candidates = list(stage.get("candidate_methods") or [self.method.method_id])
        if self.method.method_id not in candidates:
            candidates.insert(0, self.method.method_id)
        return candidates

    def _stage_label(self) -> str:
        stage = WORKFLOW_STAGE_BY_ID.get(self.method.stage_id, {})
        category = METHOD_CATEGORIES.get(self.method.category, {})
        return str(stage.get("label") or category.get("name") or self.method.category or "未分组")

    def _stage_warning(self) -> str:
        return str(WORKFLOW_STAGE_BY_ID.get(self.method.stage_id, {}).get("warning", ""))

    def _toggle_expanded(self) -> None:
        self.expanded = not self.expanded
        self._build()
        self.changed.emit(self.row)

    def _on_enabled_toggled(self, checked: bool) -> None:
        if self._suppress:
            return
        self.method.enabled = bool(checked)
        self.changed.emit(self.row)

    def _on_hidden_toggled(self, checked: bool) -> None:
        if self._suppress:
            return
        self.method.hidden = bool(checked)
        self.changed.emit(self.row)

    def _on_method_changed(self) -> None:
        if self._suppress:
            return
        method_id = self.method_combo.currentData()
        if not method_id or method_id == self.method.method_id:
            return
        self.method.method_id = str(method_id)
        category = PROCESSING_METHODS.get(self.method.method_id, {}).get("category")
        if category:
            self.method.category = str(category)
        self.method.params = {
            str(meta.get("name")): meta.get("default", "")
            for meta in PROCESSING_METHODS.get(self.method.method_id, {}).get("params", [])
        }
        self._build()
        self.changed.emit(self.row)

    def _on_param_changed(self) -> None:
        if self._suppress:
            return
        for name, getter in self._param_getters.items():
            self.method.params[name] = getter()
        self.changed.emit(self.row)

    def _wrap_numeric_control(
        self,
        spin: QSpinBox | QDoubleSpinBox,
        *,
        is_float: bool,
    ) -> tuple[QWidget, Callable[[], object]]:
        """Pair a numeric editor with a compact slider strip."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setObjectName("nodeParamSlider")
        min_v = float(spin.minimum())
        max_v = float(spin.maximum())
        span = max_v - min_v

        def slider_from_value(value: float) -> int:
            if span <= 0:
                return 0
            ratio = max(0.0, min(1.0, (float(value) - min_v) / span))
            return int(round(ratio * 1000))

        def value_from_slider(position: int) -> float:
            if span <= 0:
                return min_v
            return min_v + (float(position) / 1000.0) * span

        slider.setRange(0, 1000)
        slider.setValue(slider_from_value(float(spin.value())))

        def sync_slider(value) -> None:
            blocker = QSignalBlocker(slider)
            try:
                slider.setValue(slider_from_value(float(value)))
            finally:
                del blocker

        def sync_spin(position: int) -> None:
            value = value_from_slider(position)
            blocker = QSignalBlocker(spin)
            try:
                if is_float:
                    spin.setValue(float(value))
                else:
                    spin.setValue(int(round(value)))
            finally:
                del blocker

        spin.valueChanged.connect(sync_slider)
        slider.valueChanged.connect(sync_spin)
        slider.sliderReleased.connect(self._on_param_changed)

        layout.addWidget(spin)
        layout.addWidget(slider)
        return container, spin.value

    def _create_param_control(self, meta: dict) -> tuple[QWidget, Callable[[], object]]:
        name = str(meta.get("name", ""))
        param_type = str(meta.get("type", "str"))
        value = self.method.params.get(name, meta.get("default", ""))

        if param_type == "bool":
            control = QCheckBox()
            control.setChecked(bool(value))
            control.toggled.connect(self._on_param_changed)
            return control, control.isChecked

        if param_type in {"str", "choice"} and meta.get("choices"):
            control = QComboBox()
            for choice in meta.get("choices", []):
                control.addItem(str(choice), choice)
            idx = control.findData(value)
            if idx < 0:
                idx = control.findText(str(value))
            control.setCurrentIndex(max(idx, 0))
            control.currentIndexChanged.connect(self._on_param_changed)
            return control, control.currentData

        if param_type == "int":
            control = QSpinBox()
            min_v = int(meta.get("min", -1000000))
            max_v = int(meta.get("max", 1000000))
            control.setRange(min_v, max_v)
            try:
                control.setValue(int(float(value)))
            except Exception:
                control.setValue(int(meta.get("default", min_v)))
            control.valueChanged.connect(self._on_param_changed)
            return self._wrap_numeric_control(control, is_float=False)

        if param_type == "float":
            control = QDoubleSpinBox()
            min_v = float(meta.get("min", -1.0e9))
            max_v = float(meta.get("max", 1.0e9))
            control.setRange(min_v, max_v)
            control.setDecimals(6 if abs(max_v) <= 1 else 3)
            try:
                control.setValue(float(value))
            except Exception:
                control.setValue(float(meta.get("default", min_v)))
            control.valueChanged.connect(self._on_param_changed)
            return self._wrap_numeric_control(control, is_float=True)

        control = QLineEdit(str(value))
        control.textEdited.connect(self._on_param_changed)
        return control, control.text


class WorkflowPortItem(QGraphicsEllipseItem):
    """Lightweight graphics port used as the real edge endpoint."""

    def __init__(self, kind: str, parent=None):
        super().__init__(-5.0, -5.0, 10.0, 10.0, parent)
        self.kind = str(kind)
        color = QColor("#3278ff") if self.kind == "input" else QColor("#7d4cff")
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor("#ffffff"), 1.5))
        self.setZValue(20)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def scene_anchor(self) -> QPointF:
        return self.sceneBoundingRect().center()


class WorkflowNodeProxy(QGraphicsProxyWidget):
    """Movable graphics proxy that notifies the scene when positions change."""

    def __init__(self, row: int, parent=None):
        super().__init__(parent)
        self.row = int(row)
        self.input_port = WorkflowPortItem("input", self)
        self.output_port = WorkflowPortItem("output", self)
        self.setFlags(
            QGraphicsProxyWidget.GraphicsItemFlag.ItemIsMovable
            | QGraphicsProxyWidget.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsProxyWidget.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    def setWidget(self, widget):  # noqa: N802 - Qt override
        super().setWidget(widget)
        self.update_port_positions()

    def update_port_positions(self) -> None:
        rect = self.boundingRect()
        if not rect.isValid():
            return
        self.input_port.setPos(rect.left(), rect.center().y())
        self.output_port.setPos(rect.right(), rect.center().y())

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        result = super().itemChange(change, value)
        if change == QGraphicsProxyWidget.GraphicsItemChange.ItemPositionHasChanged:
            self.update_port_positions()
            scene = self.scene()
            if isinstance(scene, WorkflowCanvasScene):
                scene.update_edges()
        return result


class WorkflowCanvasScene(QGraphicsScene):
    """Scene containing node proxy widgets and order edges."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proxies: list[WorkflowNodeProxy] = []
        self.edges: list[QGraphicsPathItem] = []

    def update_edges(self) -> None:
        target_count = max(0, len(self.proxies) - 1)
        while len(self.edges) > target_count:
            edge = self.edges.pop()
            self.removeItem(edge)
        while len(self.edges) < target_count:
            edge = QGraphicsPathItem()
            edge.setPen(QPen(QColor("#7b8794"), 2))
            edge.setZValue(-10)
            self.addItem(edge)
            self.edges.append(edge)

        for proxy in self.proxies:
            proxy.update_port_positions()

        for edge, (left, right) in zip(self.edges, zip(self.proxies, self.proxies[1:])):
            start = left.output_port.scene_anchor()
            end = right.input_port.scene_anchor()
            dx = max(80.0, (end.x() - start.x()) * 0.45)
            path = QPainterPath(start)
            path.cubicTo(
                QPointF(start.x() + dx, start.y()),
                QPointF(end.x() - dx, end.y()),
                end,
            )
            edge.setPath(path)


class WorkflowCanvasView(QGraphicsView):
    """Large workflow canvas with Ctrl+wheel zoom and middle-button pan."""

    node_selected = pyqtSignal(int)
    node_changed = pyqtSignal(int)
    run_node_requested = pyqtSignal(int)
    run_from_node_requested = pyqtSignal(int)
    duplicate_node_requested = pyqtSignal(int)
    remove_node_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = WorkflowCanvasScene(self)
        self.setScene(self._scene)
        self._methods: list[WorkflowMethod] = []
        self._current_row = -1
        self._panning = False
        self._last_pan_pos = QPoint()
        self._drag_proxy: WorkflowNodeProxy | None = None
        self._drag_scene_offset = QPointF()
        self._preview_data = None
        self._preview_label = "Workflow Output"
        self._preview_proxy: WorkflowNodeProxy | None = None
        self._rebuild_pending = False
        self._compact_cards = False
        self._compact_threshold = 0.62
        self._normal_threshold = 0.78
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSceneRect(-400, -300, 4200, 1800)
        self.setMinimumHeight(420)
        self.setStyleSheet(
            """
            QGraphicsView {
                background: #f3f7fc;
                border: 1px solid #d8e3f2;
                border-radius: 12px;
            }
            """
        )

    def set_methods(self, methods: list[WorkflowMethod]) -> None:
        self._methods = sorted(methods, key=lambda item: item.order)
        self._rebuild()

    def update_node(self, row: int) -> None:
        if 0 <= int(row) < len(self._methods):
            self._schedule_rebuild()

    def _schedule_rebuild(self) -> None:
        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        QTimer.singleShot(0, self._run_scheduled_rebuild)

    def _run_scheduled_rebuild(self) -> None:
        self._rebuild_pending = False
        self._rebuild()

    def set_preview_data(self, data, label: str = "Workflow Output") -> None:
        self._preview_data = data
        self._preview_label = label or "Workflow Output"
        card = self._preview_card()
        if card is not None:
            try:
                card.set_preview_data(self._preview_data, self._preview_label)
                card.set_compact(self._compact_cards)
                self._scene.update_edges()
            except RuntimeError:
                self._preview_proxy = None
                self._schedule_rebuild()

    def _preview_card(self) -> BscanPreviewCard | None:
        if self._preview_proxy is None:
            return None
        card = self._preview_proxy.widget()
        return card if isinstance(card, BscanPreviewCard) else None

    def set_selected_row(self, row: int) -> None:
        self._current_row = int(row)
        for proxy in self._scene.proxies:
            card = proxy.widget()
            if isinstance(card, WorkflowNodeCard):
                card.set_current(card.row == self._current_row)

    def _event_view_pos(self, event) -> QPoint:
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _proxy_at_view_pos(self, view_pos: QPoint) -> WorkflowNodeProxy | None:
        item = self.itemAt(view_pos)
        while item is not None and not isinstance(item, WorkflowNodeProxy):
            item = item.parentItem()
        return item if isinstance(item, WorkflowNodeProxy) else None

    def _is_interactive_widget(self, widget: QWidget) -> bool:
        return isinstance(
            widget,
            (
                QAbstractButton,
                QAbstractSpinBox,
                QComboBox,
                QLineEdit,
                QSlider,
            ),
        )

    def _is_interactive_card_target(self, proxy: WorkflowNodeProxy, scene_pos: QPointF) -> bool:
        card = proxy.widget()
        if isinstance(card, BscanPreviewCard):
            local_pos = proxy.mapFromScene(scene_pos)
            child = card.childAt(int(local_pos.x()), int(local_pos.y()))
            return isinstance(child, QAbstractButton)
        if not isinstance(card, QWidget):
            return False

        local_pos = proxy.mapFromScene(scene_pos)
        child = card.childAt(int(local_pos.x()), int(local_pos.y()))
        while child is not None and child is not card:
            if self._is_interactive_widget(child):
                return True
            child = child.parentWidget()
        return False

    def viewportEvent(self, event):  # noqa: N802 - Qt override
        event_type = event.type()

        if event_type == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
            view_pos = self._event_view_pos(event)
            proxy = self._proxy_at_view_pos(view_pos)
            if proxy is not None:
                card = proxy.widget()
                if isinstance(card, BscanPreviewCard):
                    card.open_large_view()
                    event.accept()
                    return True

        if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            view_pos = self._event_view_pos(event)
            scene_pos = self.mapToScene(view_pos)
            proxy = self._proxy_at_view_pos(view_pos)
            if proxy is not None and not self._is_interactive_card_target(proxy, scene_pos):
                self._drag_proxy = proxy
                self._drag_scene_offset = scene_pos - proxy.pos()
                card = proxy.widget()
                if isinstance(card, WorkflowNodeCard):
                    card.selected.emit(proxy.row)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return True

        if (
            event_type == QEvent.Type.MouseMove
            and self._drag_proxy is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            scene_pos = self.mapToScene(self._event_view_pos(event))
            try:
                self._drag_proxy.setPos(scene_pos - self._drag_scene_offset)
                self._scene.update_edges()
            except RuntimeError:
                self._clear_drag_state()
            event.accept()
            return True

        if event_type == QEvent.Type.MouseButtonRelease and self._drag_proxy is not None:
            self._clear_drag_state()
            event.accept()
            return True

        return super().viewportEvent(event)

    def _clear_drag_state(self) -> None:
        self._drag_proxy = None
        self._drag_scene_offset = QPointF()
        if not self._panning:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):  # noqa: N802 - Qt override
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            current_scale = self.transform().m11()
            next_scale = max(0.32, min(2.8, current_scale * factor))
            self.resetTransform()
            self.scale(next_scale, next_scale)
            self._apply_zoom_lod()
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        if self._panning:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def fit_nodes(self) -> None:
        rect = self._scene.itemsBoundingRect().adjusted(-80, -80, 80, 80)
        if rect.isValid():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._apply_zoom_lod(force=True)

    def _apply_zoom_lod(self, force: bool = False) -> None:
        scale = self.transform().m11()
        if scale < self._compact_threshold:
            compact = True
        elif scale > self._normal_threshold:
            compact = False
        else:
            compact = self._compact_cards

        if not force and compact == self._compact_cards:
            return

        self._compact_cards = compact
        for proxy in self._scene.proxies:
            card = proxy.widget()
            if isinstance(card, (WorkflowNodeCard, BscanPreviewCard)):
                card.set_compact(compact)
        self._scene.update_edges()

    def _rebuild(self) -> None:
        self._clear_drag_state()
        self._scene.clear()
        self._scene.proxies.clear()
        self._scene.edges.clear()
        self._preview_proxy = None

        x0, y0 = 40, 60
        x_step, y_step = 395, 255
        max_per_row = 3
        for row, method in enumerate(self._methods):
            card = WorkflowNodeCard(row, method)
            card.selected.connect(self._on_node_selected)
            card.changed.connect(self.node_changed)
            card.run_current_requested.connect(self.run_node_requested)
            card.run_from_requested.connect(self.run_from_node_requested)
            card.duplicate_requested.connect(self.duplicate_node_requested)
            card.remove_requested.connect(self.remove_node_requested)

            proxy = WorkflowNodeProxy(row)
            proxy.setWidget(card)
            lane = row // max_per_row
            col = row % max_per_row
            proxy.setPos(x0 + col * x_step, y0 + lane * y_step)
            self._scene.addItem(proxy)
            self._scene.proxies.append(proxy)

        preview_card = BscanPreviewCard()
        preview_card.set_preview_data(self._preview_data, self._preview_label)
        preview_proxy = WorkflowNodeProxy(-1)
        preview_proxy.setWidget(preview_card)
        preview_row = len(self._methods)
        preview_lane = preview_row // max_per_row
        preview_col = preview_row % max_per_row
        preview_proxy.setPos(x0 + preview_col * x_step, y0 + preview_lane * y_step)
        self._scene.addItem(preview_proxy)
        self._scene.proxies.append(preview_proxy)
        self._preview_proxy = preview_proxy

        self._scene.update_edges()
        self.set_selected_row(self._current_row)
        self._apply_zoom_lod(force=True)
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-200, -160, 400, 260))

    def _on_node_selected(self, row: int) -> None:
        self._current_row = int(row)
        self.set_selected_row(row)
        self.node_selected.emit(int(row))

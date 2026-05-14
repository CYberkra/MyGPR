#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComfyUI-like workflow canvas cards for MyGPR."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.methods_registry import PROCESSING_METHODS, get_method_display_name
from core.workflow_data import METHOD_CATEGORIES, WORKFLOW_STAGE_BY_ID, WorkflowMethod


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
        self.setMinimumWidth(270)
        self.setMaximumWidth(320)
        self.setStyleSheet(
            """
            QFrame#workflowNodeCard {
                background: #ffffff;
                border: 1px solid #d8e3f2;
                border-radius: 12px;
            }
            QFrame#workflowNodeCard[current="true"] {
                border: 2px solid #3278ff;
                background: #f6f9ff;
            }
            QLabel#nodeTitle {
                font-weight: 700;
                color: #1f2d3d;
            }
            QLabel#nodeSubtitle {
                color: #52647a;
            }
            QLabel#nodeWarning {
                color: #a66a00;
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
            root.setContentsMargins(10, 8, 10, 10)
            root.setSpacing(7)

            if self.compact:
                self._build_compact(root)
                return

            title_row = QHBoxLayout()
            title_row.setSpacing(8)
            title = QLabel(f"{self.row + 1:02d}. {self._stage_label()}")
            title.setObjectName("nodeTitle")
            title.setWordWrap(True)
            title_row.addWidget(title, 1)

            self.enabled_check = QCheckBox("启用")
            self.enabled_check.setChecked(bool(self.method.enabled))
            self.enabled_check.toggled.connect(self._on_enabled_toggled)
            title_row.addWidget(self.enabled_check)
            root.addLayout(title_row)

            subtitle = QLabel(get_method_display_name(self.method.method_id))
            subtitle.setObjectName("nodeSubtitle")
            subtitle.setWordWrap(True)
            root.addWidget(subtitle)

            method_row = QHBoxLayout()
            method_row.setSpacing(6)
            method_row.addWidget(QLabel("算法"))
            self.method_combo = QComboBox()
            for key in self._candidate_methods():
                if key in PROCESSING_METHODS:
                    self.method_combo.addItem(get_method_display_name(key), key)
            idx = self.method_combo.findData(self.method.method_id)
            self.method_combo.setCurrentIndex(max(idx, 0))
            self.method_combo.currentIndexChanged.connect(self._on_method_changed)
            method_row.addWidget(self.method_combo, 1)

            self.hidden_check = QCheckBox("隐藏")
            self.hidden_check.setChecked(bool(self.method.hidden))
            self.hidden_check.toggled.connect(self._on_hidden_toggled)
            method_row.addWidget(self.hidden_check)
            root.addLayout(method_row)

            warning = self._stage_warning()
            if warning:
                warning_label = QLabel(f"⚠ {warning}")
                warning_label.setObjectName("nodeWarning")
                warning_label.setWordWrap(True)
                root.addWidget(warning_label)

            param_metas = PROCESSING_METHODS.get(self.method.method_id, {}).get("params", [])
            max_params = len(param_metas) if self.expanded else min(3, len(param_metas))
            if max_params:
                param_grid = QGridLayout()
                param_grid.setContentsMargins(0, 0, 0, 0)
                param_grid.setHorizontalSpacing(6)
                param_grid.setVerticalSpacing(5)
                for row, meta in enumerate(param_metas[:max_params]):
                    name = str(meta.get("name", ""))
                    label = QLabel(str(meta.get("label", name)))
                    label.setWordWrap(True)
                    control, getter = self._create_param_control(meta)
                    tooltip = str(meta.get("tooltip", ""))
                    if tooltip:
                        label.setToolTip(tooltip)
                        control.setToolTip(tooltip)
                    param_grid.addWidget(label, row, 0)
                    param_grid.addWidget(control, row, 1)
                    self._param_getters[name] = getter
                root.addLayout(param_grid)
            else:
                root.addWidget(QLabel("(无参数)"))

            button_row = QHBoxLayout()
            button_row.setSpacing(6)
            if len(param_metas) > 3:
                self.more_button = QPushButton("收起" if self.expanded else "更多")
                self.more_button.clicked.connect(self._toggle_expanded)
                button_row.addWidget(self.more_button)
            for text, signal in [
                ("运行", self.run_current_requested),
                ("后续", self.run_from_requested),
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

    def _status_text(self) -> str:
        if self.method.hidden:
            return "状态: 隐藏"
        if not self.method.enabled:
            return "状态: 停用"
        return "状态: 启用"

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
            return control, control.value

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
            return control, control.value

        control = QLineEdit(str(value))
        control.textEdited.connect(self._on_param_changed)
        return control, control.text


class WorkflowNodeProxy(QGraphicsProxyWidget):
    """Movable graphics proxy that notifies the scene when positions change."""

    def __init__(self, row: int, parent=None):
        super().__init__(parent)
        self.row = int(row)
        self._dragging = False
        self._drag_scene_offset = QPointF()
        self._drag_filter_widgets: list[QWidget] = []
        self.setFlags(
            QGraphicsProxyWidget.GraphicsItemFlag.ItemIsMovable
            | QGraphicsProxyWidget.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsProxyWidget.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    def setWidget(self, widget):  # noqa: N802 - Qt override
        result = super().setWidget(widget)
        self._install_drag_event_filters(widget)
        return result

    def _install_drag_event_filters(self, widget: QWidget | None) -> None:
        self._drag_filter_widgets.clear()
        if widget is None:
            return
        widget.installEventFilter(self)
        self._drag_filter_widgets.append(widget)
        for child in widget.findChildren(QWidget):
            if not self._is_interactive_widget(child):
                child.installEventFilter(self)
                self._drag_filter_widgets.append(child)

    def _is_interactive_widget(self, widget: QWidget) -> bool:
        return isinstance(
            widget,
            (
                QAbstractButton,
                QAbstractSpinBox,
                QComboBox,
                QLineEdit,
            ),
        )

    def _event_scene_pos(self, event) -> QPointF:
        scene = self.scene()
        if scene is None or not scene.views():
            return QPointF(self.pos())

        view = scene.views()[0]
        if hasattr(event, "globalPosition"):
            global_pos = event.globalPosition().toPoint()
        else:
            global_pos = event.globalPos()
        viewport_pos = view.viewport().mapFromGlobal(global_pos)
        return view.mapToScene(viewport_pos)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        if isinstance(watched, QWidget) and self._is_interactive_widget(watched):
            return False

        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            scene_pos = self._event_scene_pos(event)
            self._drag_scene_offset = scene_pos - self.pos()
            card = self.widget()
            if isinstance(card, WorkflowNodeCard):
                card.selected.emit(self.row)
            event.accept()
            return True

        if (
            event_type == QEvent.Type.MouseMove
            and self._dragging
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            scene_pos = self._event_scene_pos(event)
            self.setPos(scene_pos - self._drag_scene_offset)
            scene = self.scene()
            if isinstance(scene, WorkflowCanvasScene):
                scene.update_edges()
            event.accept()
            return True

        if event_type == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()
            return True

        return super().eventFilter(watched, event)

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        result = super().itemChange(change, value)
        if change == QGraphicsProxyWidget.GraphicsItemChange.ItemPositionHasChanged:
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
        for edge in self.edges:
            self.removeItem(edge)
        self.edges.clear()
        pen = QPen(Qt.GlobalColor.darkGray, 2)
        for left, right in zip(self.proxies, self.proxies[1:]):
            a = left.sceneBoundingRect()
            b = right.sceneBoundingRect()
            start = QPointF(a.right(), a.center().y())
            end = QPointF(b.left(), b.center().y())
            dx = max(80.0, (end.x() - start.x()) * 0.45)
            path = QPainterPath(start)
            path.cubicTo(
                QPointF(start.x() + dx, start.y()),
                QPointF(end.x() - dx, end.y()),
                end,
            )
            edge = QGraphicsPathItem(path)
            edge.setPen(pen)
            edge.setZValue(-10)
            self.addItem(edge)
            self.edges.append(edge)


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
            self._rebuild()

    def set_selected_row(self, row: int) -> None:
        self._current_row = int(row)
        for proxy in self._scene.proxies:
            card = proxy.widget()
            if isinstance(card, WorkflowNodeCard):
                card.set_current(card.row == self._current_row)

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
            if isinstance(card, WorkflowNodeCard):
                card.set_compact(compact)
        self._scene.update_edges()

    def _rebuild(self) -> None:
        self._scene.clear()
        self._scene.proxies.clear()
        self._scene.edges.clear()

        x0, y0 = 40, 60
        x_step, y_step = 335, 255
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

        self._scene.update_edges()
        self.set_selected_row(self._current_row)
        self._apply_zoom_lod(force=True)
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-200, -160, 400, 260))

    def _on_node_selected(self, row: int) -> None:
        self._current_row = int(row)
        self.set_selected_row(row)
        self.node_selected.emit(int(row))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComfyUI-like workflow canvas for MyGPR."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QEvent, QMimeData, QPoint, QPointF, QRectF, QSize, QSignalBlocker, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QFontMetrics, QKeyEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QAction

from core.methods_registry import PROCESSING_METHODS, get_method_display_name
from core.workflow_data import (
    METHOD_CATEGORIES,
    WORKFLOW_STAGE_BY_ID,
    WorkflowLink,
    WorkflowMethod,
    ensure_workflow_method_ids,
)
from ui.workflow_canvas_preview import BscanPreviewCard


PREVIEW_NODE_ID = "__workflow_preview__"
MIN_NODE_WIDTH = 280
MIN_NODE_HEIGHT = 150
MAX_NODE_WIDTH = 520
MAX_NODE_HEIGHT = 720
NODE_PORT_ROW_Y = 50.0


def default_params_for_method(method_id: str) -> dict[str, object]:
    """Return registry defaults for a workflow method."""
    params: dict[str, object] = {}
    for meta in PROCESSING_METHODS.get(method_id, {}).get("params", []):
        params[str(meta.get("name"))] = meta.get("default", "")
    return params


def candidate_methods_for_workflow_method(method: WorkflowMethod) -> list[str]:
    """Return valid candidate algorithm ids for a workflow node."""
    stage = WORKFLOW_STAGE_BY_ID.get(method.stage_id, {})
    candidates = [
        str(item)
        for item in stage.get("candidate_methods", [])
        if str(item) in PROCESSING_METHODS
    ]
    if not candidates:
        candidates = [str(method.method_id)] if method.method_id in PROCESSING_METHODS else []
        category = METHOD_CATEGORIES.get(method.category, {})
        for key in category.get("methods", []):
            if key in PROCESSING_METHODS and key not in candidates:
                candidates.append(str(key))
    if method.method_id in PROCESSING_METHODS and method.method_id not in candidates:
        candidates.insert(0, str(method.method_id))
    return candidates


def update_workflow_method_algorithm(method: WorkflowMethod, method_id: str) -> bool:
    """Switch a node to a new algorithm while preserving node identity/layout state."""
    method_id = str(method_id)
    if method_id == method.method_id or method_id not in PROCESSING_METHODS:
        return False

    old_params = dict(method.params or {})
    new_params = default_params_for_method(method_id)
    valid_names = set(new_params)
    for name, value in old_params.items():
        if name in valid_names:
            new_params[name] = value

    method.method_id = method_id
    category = PROCESSING_METHODS.get(method_id, {}).get("category")
    if category:
        method.category = str(category)
    method.params = new_params
    method.status = "idle"
    method.input_shape = None
    method.output_shape = None
    method.error_message = ""
    method.elapsed_ms = 0.0
    return True


def _elided(text: str, width: int, widget: QWidget) -> str:
    metrics = QFontMetrics(widget.font())
    return metrics.elidedText(str(text), Qt.TextElideMode.ElideRight, int(width))


def workflow_port_labels(method: WorkflowMethod) -> tuple[str, str]:
    """Return compact visual input/output labels for current first-pass ports."""
    method_id = str(getattr(method, "method_id", ""))
    stage_id = str(getattr(method, "stage_id", ""))
    category = str(getattr(method, "category", ""))
    if method_id == "raw_input":
        return "source", "data"
    if method_id == "bscan_preview":
        return "data", "preview"
    if method_id in {"manual_velocity_model"} or stage_id == "velocity_model":
        return "data", "velocity"
    if method_id in {"geometry_depth_context", "time_to_depth"} or stage_id == "geometry_depth":
        return "data / velocity", "depth"
    if method_id in {"kirchhoff_migration", "stolt_migration"} or category == "migration":
        return "data / velocity", "image"
    if category in {"export", "evidence"}:
        return "data", "evidence"
    return "data", "data"


class ParamRowWidget(QWidget):
    """Compact parameter block used inside workflow nodes."""

    def __init__(self, label: str, control: QWidget, tooltip: str = "", parent=None):
        super().__init__(parent)
        self._slider: QSlider | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setObjectName("paramRow")
        self.setStyleSheet(
            """
            QWidget#paramRow {
                min-height: 28px;
            }
            QLabel#paramName {
                color: #4b5563;
                font-size: 12px;
                font-weight: 600;
            }
            QAbstractSpinBox, QComboBox, QLineEdit, QToolButton#boolChip {
                background: #f8fafc;
                border: 1px solid #d8e3f2;
                border-radius: 10px;
                padding: 3px 8px;
                min-height: 22px;
                max-height: 26px;
            }
            QAbstractSpinBox {
                qproperty-alignment: AlignRight;
            }
            QToolButton#boolChip {
                min-width: 68px;
                max-width: 68px;
                font-weight: 700;
                font-size: 12px;
            }
            QToolButton#boolChip[checked="true"] {
                background: #e8f2ff;
                color: #2457b8;
                border-color: #9ec5ff;
            }
            QToolButton#boolChip[checked="false"] {
                background: #f3f4f6;
                color: #6b7280;
                border-color: #d1d5db;
            }
            """
        )

        short_label = _elided(str(label), 142, self)
        name = QLabel(short_label)
        name.setObjectName("paramName")
        name.setWordWrap(False)
        name.setMinimumWidth(80)
        name.setMaximumWidth(150)
        name.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        name.setToolTip(str(label))

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.addWidget(name, 1)

        spin = control.findChild(QAbstractSpinBox)
        slider = control.findChild(QSlider)
        if spin is not None and slider is not None:
            spin.setParent(self)
            slider.setParent(self)
            spin.setMinimumWidth(80)
            spin.setMaximumWidth(120)
            spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            top_row.addWidget(spin, 0)
            layout.addLayout(top_row)
            layout.addWidget(slider)
            self._slider = slider
            slider.hide()
            control.deleteLater()
        else:
            if isinstance(control, QComboBox):
                control.setMinimumWidth(80)
                control.setMaximumWidth(140)
            elif isinstance(control, QLineEdit):
                control.setMinimumWidth(80)
                control.setMaximumWidth(140)
            control.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            top_row.addWidget(control, 0)
            layout.addLayout(top_row)

        if tooltip:
            self.setToolTip(tooltip)
            name.setToolTip(tooltip)
            control.setToolTip(tooltip)
            if spin is not None:
                spin.setToolTip(tooltip)
            if slider is not None:
                slider.setToolTip(tooltip)

    def enterEvent(self, event):  # noqa: N802 - Qt override
        if self._slider is not None:
            self._slider.show()
        event.accept()

    def leaveEvent(self, event):  # noqa: N802 - Qt override
        if self._slider is not None:
            self._slider.hide()
        event.accept()


class BoolChipButton(QToolButton):
    """ON/OFF chip with stable size hint independent of label text."""

    def sizeHint(self):  # noqa: N802 - Qt override
        return QSize(68, 26)


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
        self._suppress = False
        self._param_getters: dict[str, Callable[[], object]] = {}
        self.setObjectName("workflowNodeCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(MIN_NODE_WIDTH, MIN_NODE_HEIGHT)
        self.setMaximumSize(MAX_NODE_WIDTH, MAX_NODE_HEIGHT)
        self.setStyleSheet(
            """
            QFrame#workflowNodeCard {
                background: #ffffff;
                border: 1px solid #d7e2f0;
                border-radius: 8px;
            }
            QFrame#workflowNodeCard[current="true"] {
                border: 2px solid #3278ff;
                background: #f7faff;
            }
            QFrame#workflowNodeCard[hiddenState="true"] {
                border: 1px dashed #9aa8ba;
                background: #fafbfc;
            }
            QFrame#workflowNodeCard[status="success"] {
                border: 1px solid #36d399;
            }
            QFrame#workflowNodeCard[status="success_stale"] {
                border: 1px dashed #9ca3af;
                background: #f9fafb;
            }
            QFrame#workflowNodeCard[status="running"] {
                border: 2px solid #3278ff;
                background: #f0f7ff;
            }
            QFrame#workflowNodeCard[status="queued"] {
                border: 1px dashed #93c5fd;
                background: #eff6ff;
            }
            QFrame#workflowNodeCard[status="failed"] {
                border: 2px solid #ef4444;
                background: #fff1f1;
            }
            QFrame#workflowNodeCard[status="skipped"] {
                border: 1px solid #d1d5db;
                background: #f9fafb;
            }
            QLabel#nodeTitle {
                font-weight: 800;
                color: #1f2d3d;
                font-size: 14px;
            }
            QLabel#portLabel {
                color: #52647a;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#portDotIn {
                color: #3278ff;
                font-size: 15px;
                font-weight: 900;
            }
            QLabel#portDotOut {
                color: #7d4cff;
                font-size: 15px;
                font-weight: 900;
            }
            QLabel#nodeSubtitle {
                color: #52647a;
                font-size: 13px;
            }
            QLabel#nodeWarning {
                color: #a66a00;
                font-size: 12px;
            }
            QLabel#nodeError {
                color: #dc2626;
                font-size: 11px;
                word-wrap: break-word;
            }
            QLabel#nodeStatusChip {
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#nodeStatusChip[state="on"] {
                background: #eef4ff;
                color: #2457b8;
                border: 1px solid #cdddf8;
            }
            QLabel#nodeStatusChip[state="off"] {
                background: #f0f2f5;
                color: #697386;
                border: 1px solid #d7dce4;
            }
            QLabel#nodeStatusChip[state="hide"] {
                background: #fff4de;
                color: #9a6100;
                border: 1px solid #f2ce8c;
            }
            QLabel#nodeStatusChip[state="success"] {
                background: #ecfdf5;
                color: #047857;
                border: 1px solid #6ee7b7;
            }
            QLabel#nodeStatusChip[state="running"] {
                background: #dbeafe;
                color: #1d4ed8;
                border: 1px solid #60a5fa;
            }
            QLabel#nodeStatusChip[state="failed"] {
                background: #fef2f2;
                color: #dc2626;
                border: 1px solid #fca5a5;
            }
            QLabel#nodeStatusChip[state="skipped"] {
                background: #f3f4f6;
                color: #6b7280;
                border: 1px solid #d1d5db;
            }
            QLabel#nodeStatusChip[state="success_stale"] {
                background: #f3f4f6;
                color: #6b7280;
                border: 1px dashed #9ca3af;
            }
            QLabel#nodeStatusChip[state="queued"] {
                background: #dbeafe;
                color: #1d4ed8;
                border: 1px dashed #60a5fa;
            }
            QLabel#nodeStatusChip[state="idle"] {
                background: #f0f2f5;
                color: #697386;
                border: 1px solid #d7dce4;
            }
            QLabel#nodeMetaInfo {
                color: #6b7280;
                font-size: 11px;
            }
            QLabel#paramName {
                color: #64748b;
                font-size: 12px;
                font-weight: 700;
            }
            QToolButton#eyeButton {
                border: none;
                color: #52647a;
                font-size: 16px;
                padding: 0px;
            }
            QSlider::groove:horizontal {
                height: 3px;
                background: #dbe5f2;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 10px;
                margin: -4px 0;
                border-radius: 5px;
                background: #3278ff;
            }
            """
        )
        self._build()

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        self.selected.emit(self.row)
        super().mousePressEvent(event)

    def set_current(self, current: bool) -> None:
        self.setProperty("current", bool(current))
        self._refresh_state_properties()

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if self.expanded == expanded:
            return
        self.expanded = expanded
        self._build()

    def toggle_expanded(self) -> None:
        self.set_expanded(not self.expanded)

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

            # 初始化控件引用列表
            self.param_rows: list[ParamRowWidget] = []
            self.title_label: QLabel | None = None
            self.subtitle_label: QLabel | None = None
            self.meta_label: QLabel | None = None
            self.more_button: QToolButton | None = None
            self.warning_label: QLabel | None = None
            self.error_label: QLabel | None = None
            self.algorithm_row_widget: QWidget | None = None
            self.algorithm_combo: QComboBox | None = None
            self.algorithm_button: QToolButton | None = None
            self.algorithm_menu: QMenu | None = None

            root = QVBoxLayout(self)
            root.setContentsMargins(10, 8, 10, 10)
            root.setSpacing(5)

            title_row = QHBoxLayout()
            title_row.setSpacing(6)
            full_title = f"{self.row + 1:02d} {self._stage_label()}"
            self.title_label = QLabel(_elided(full_title, 205, self))
            self.title_label.setObjectName("nodeTitle")
            self.title_label.setWordWrap(False)
            self.title_label.setToolTip(full_title)
            title_row.addWidget(self.title_label, 1)

            status_chip = QLabel(self._status_chip_text())
            status_chip.setObjectName("nodeStatusChip")
            status_chip.setProperty("state", self._status_chip_state())
            self.status_chip = status_chip
            title_row.addWidget(status_chip)

            self.eye_button = QToolButton()
            self.eye_button.setObjectName("eyeButton")
            self.eye_button.setText("◎" if self.method.hidden else "◉")
            self.eye_button.setToolTip("显示/隐藏节点")
            self.eye_button.clicked.connect(self._on_eye_clicked)
            title_row.addWidget(self.eye_button)
            root.addLayout(title_row)

            in_label, out_label = self.port_labels()
            self.input_port_label = QLabel(in_label)
            self.input_port_label.setObjectName("portLabel")
            self.input_port_label.setToolTip(f"输入: {in_label}")
            self.output_port_label = QLabel(out_label)
            self.output_port_label.setObjectName("portLabel")
            self.output_port_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.output_port_label.setToolTip(f"输出: {out_label}")
            self._port_row = QHBoxLayout()
            self._port_row.setContentsMargins(0, 4, 0, 4)
            self._port_row.setSpacing(6)
            self._port_row.addWidget(self.input_port_label, 0)
            self._port_row.addStretch(1)
            self._port_row.addWidget(self.output_port_label, 0)
            root.addLayout(self._port_row)

            candidates = self._candidate_methods()
            self.algorithm_row_widget = QWidget()
            # QComboBox 本身已经能被 _is_interactive_widget 识别，不需要 workflowInteractiveRegion
            
            algorithm_layout = QHBoxLayout(self.algorithm_row_widget)
            algorithm_layout.setContentsMargins(0, 0, 0, 0)
            algorithm_layout.setSpacing(8)
            algorithm_label = QLabel("算法")
            algorithm_label.setObjectName("paramName")
            algorithm_layout.addWidget(algorithm_label)
            
            if len(candidates) > 1:
                # 使用 QComboBox 替代 QToolButton + QMenu，保证稳定性
                self.algorithm_combo = QComboBox()
                self.algorithm_combo.setObjectName("nodeAlgorithmCombo")
                self.algorithm_combo.setMinimumWidth(170)
                self.algorithm_combo.setMinimumHeight(32)
                self.algorithm_combo.setCursor(Qt.CursorShape.PointingHandCursor)
                # 设置样式
                self.algorithm_combo.setStyleSheet("""
                    QComboBox {
                        background: #f8fafc;
                        border: 1px solid #d8e3f2;
                        border-radius: 8px;
                        padding: 4px 10px;
                    }
                    QComboBox:hover {
                        background: #eef3fa;
                        border-color: #a8c4e8;
                    }
                    QComboBox::drop-down {
                        border: none;
                        width: 20px;
                    }
                    QComboBox::down-arrow {
                        image: none;
                        border-left: 4px solid transparent;
                        border-right: 4px solid transparent;
                        border-top: 5px solid #666;
                        margin-right: 4px;
                    }
                """)
                # 添加候选算法
                for key in candidates:
                    self.algorithm_combo.addItem(get_method_display_name(key), key)
                # 设置当前项
                index = self.algorithm_combo.findData(self.method.method_id)
                if index >= 0:
                    self.algorithm_combo.setCurrentIndex(index)
                self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_combo_changed)
                algorithm_layout.addWidget(self.algorithm_combo, 1)
            else:
                readonly_method = QLabel(_elided(get_method_display_name(self.method.method_id), 170, self))
                readonly_method.setObjectName("nodeSubtitle")
                readonly_method.setToolTip(get_method_display_name(self.method.method_id))
                algorithm_layout.addWidget(readonly_method, 1)
                
            root.addWidget(self.algorithm_row_widget)

            self.subtitle_label = QLabel(_elided(get_method_display_name(self.method.method_id), 245, self))
            if self.method.method_id == "raw_input":
                self.subtitle_label.setText("Raw Input")
            self.subtitle_label.setObjectName("nodeSubtitle")
            self.subtitle_label.setWordWrap(False)
            self.subtitle_label.setToolTip(get_method_display_name(self.method.method_id))
            root.addWidget(self.subtitle_label)

            if self.method.method_id == "raw_input":
                shape = self.method.params.get("shape", "--")
                file_name = self.method.params.get("file", "--")
                raw_status = self.method.params.get("raw", "missing")
                rtk_status = self.method.params.get("rtk", "missing")
                imu_status = self.method.params.get("imu", "missing")
                agl_status = self.method.params.get("agl", "missing")
                status_text = f"Raw: {raw_status} | RTK: {rtk_status} | IMU: {imu_status} | AGL: {agl_status}"
                input_label = QLabel(f"shape: {shape}\nfile: {file_name}\n{status_text}")
                input_label.setObjectName("nodeSubtitle")
                input_label.setWordWrap(True)
                root.addWidget(input_label)
                root.addStretch(1)
                self._refresh_state_properties()
                return

            # Show meta info (input/output shape, elapsed) if available
            meta_lines = []
            input_shape = getattr(self.method, "input_shape", None)
            output_shape = getattr(self.method, "output_shape", None)
            elapsed_ms = getattr(self.method, "elapsed_ms", 0)
            if input_shape:
                meta_lines.append(f"in: {input_shape}")
            if output_shape:
                meta_lines.append(f"out: {output_shape}")
            if elapsed_ms > 0:
                meta_lines.append(f"time: {elapsed_ms:.1f}ms")
            if meta_lines:
                self.meta_label = QLabel(" | ".join(meta_lines))
                self.meta_label.setObjectName("nodeMetaInfo")
                root.addWidget(self.meta_label)

            # Show error message if failed
            error_msg = getattr(self.method, "error_message", "")
            if error_msg:
                self.error_label = QLabel(f"⚠️ {error_msg}")
                self.error_label.setObjectName("nodeError")
                self.error_label.setWordWrap(True)
                self.error_label.setMinimumWidth(200)
                root.addWidget(self.error_label)

            warning = self._stage_warning()
            if warning:
                self.warning_label = QLabel(warning)
                self.warning_label.setObjectName("nodeWarning")
                self.warning_label.setWordWrap(True)
                root.addWidget(self.warning_label)

            param_metas = PROCESSING_METHODS.get(self.method.method_id, {}).get("params", [])
            max_params = len(param_metas) if self.expanded else min(2, len(param_metas))
            for meta in param_metas[:max_params]:
                name = str(meta.get("name", ""))
                control, getter = self._create_param_control(meta)
                self._install_wheel_guard(control)
                row_widget = ParamRowWidget(
                    str(meta.get("label", name)),
                    control,
                    str(meta.get("tooltip", "")),
                )
                root.addWidget(row_widget)
                self.param_rows.append(row_widget)
                self._param_getters[name] = getter

            if not max_params:
                empty_label = QLabel("(无参数)")
                empty_label.setObjectName("nodeSubtitle")
                root.addWidget(empty_label)

            if len(param_metas) > 2:
                self.more_button = QToolButton()
                self.more_button.setText("收起" if self.expanded else f"更多({len(param_metas) - 2})")
                self.more_button.clicked.connect(self._toggle_expanded)
                root.addWidget(self.more_button)

            root.addStretch(1)
            
            # 恢复 full 模式（默认）
            self._lod_mode = 'full'
            self._apply_full_mode()
            
            self._refresh_state_properties()
            self._refresh_state_properties()
        finally:
            self._suppress = False

    def port_anchor_y(self) -> float:
        if hasattr(self, '_port_row') and self._port_row is not None:
            try:
                geom = self._port_row.geometry()
                if geom.isValid() and geom.height() > 0:
                    return float(geom.center().y())
            except Exception:
                pass
        return NODE_PORT_ROW_Y

    def port_labels(self) -> tuple[str, str]:
        return workflow_port_labels(self.method)

    def set_lod_mode(self, mode: str) -> None:
        """Set level-of-detail display mode: full, compact, or mini."""
        mode = mode if mode in {"full", "compact", "mini"} else "full"
        self._lod_mode = mode
        
        if mode == "full":
            self.setMinimumSize(MIN_NODE_WIDTH, MIN_NODE_HEIGHT)
            self.setMaximumSize(MAX_NODE_WIDTH, MAX_NODE_HEIGHT)
            self._apply_full_mode()
        elif mode == "compact":
            self.setMinimumSize(200, 80)
            self.setMaximumSize(260, 120)
            self._apply_compact_mode()
        elif mode == "mini":
            self.setMinimumSize(160, 60)
            self.setMaximumSize(200, 80)
            self._apply_mini_mode()
        
        self.updateGeometry()

    def _apply_full_mode(self) -> None:
        """Apply full LOD mode with normal fonts."""
        from PyQt6.QtGui import QFont
        
        # 恢复所有控件可见性
        if self.title_label:
            self.title_label.show()
            self.title_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        if self.status_chip:
            self.status_chip.show()
            self.status_chip.setStyleSheet("font-size: 11px;")
        if self.eye_button:
            self.eye_button.show()
        if self.input_port_label:
            self.input_port_label.show()
            self.input_port_label.setStyleSheet("font-size: 12px;")
        if self.output_port_label:
            self.output_port_label.show()
            self.output_port_label.setStyleSheet("font-size: 12px;")
        if self.algorithm_row_widget:
            self.algorithm_row_widget.show()
        if self.subtitle_label:
            self.subtitle_label.hide()
            self.subtitle_label.setStyleSheet("font-size: 11px;")
        if self.meta_label:
            self.meta_label.show()
            self.meta_label.setStyleSheet("font-size: 11px;")
        if self.error_label:
            self.error_label.show()
        if self.warning_label:
            self.warning_label.show()
        if self.more_button:
            self.more_button.show()
        for row in self.param_rows:
            row.show()
        self._restore_layout()

    def _apply_compact_mode(self) -> None:
        """Apply compact LOD mode with larger fonts for readability."""
        from PyQt6.QtGui import QFont
        
        # 隐藏参数行，更多按钮，长错误
        for row in self.param_rows:
            row.hide()
        if self.more_button:
            self.more_button.hide()
        if self.error_label:
            self.error_label.hide()
        
        # 增大字号
        if self.title_label:
            self.title_label.show()
            self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        if self.status_chip:
            self.status_chip.show()
            self.status_chip.setStyleSheet("font-size: 14px; font-weight: bold; padding: 2px 6px;")
        if self.eye_button:
            self.eye_button.show()
        if self.input_port_label:
            self.input_port_label.show()
            self.input_port_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        if self.output_port_label:
            self.output_port_label.show()
            self.output_port_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        if self.algorithm_row_widget:
            self.algorithm_row_widget.hide()
        if self.subtitle_label:
            self.subtitle_label.show()
            self.subtitle_label.setStyleSheet("font-size: 14px;")
        if self.meta_label:
            self.meta_label.show()
            self.meta_label.setStyleSheet("font-size: 14px;")
        if self.warning_label:
            self.warning_label.show()
            self.warning_label.setStyleSheet("font-size: 13px;")
        
        self._simplify_layout()

    def _apply_mini_mode(self) -> None:
        """Apply mini LOD mode with largest fonts."""
        from PyQt6.QtGui import QFont
        
        # 只保留最核心信息
        for row in self.param_rows:
            row.hide()
        if self.more_button:
            self.more_button.hide()
        if self.meta_label:
            self.meta_label.hide()
        if self.error_label:
            self.error_label.hide()
        if self.warning_label:
            self.warning_label.hide()
        if self.subtitle_label:
            self.subtitle_label.hide()
        if self.algorithm_row_widget:
            self.algorithm_row_widget.hide()
        
        # 最大字号
        if self.title_label:
            self.title_label.show()
            self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        if self.status_chip:
            self.status_chip.show()
            self.status_chip.setStyleSheet("font-size: 16px; font-weight: bold; padding: 3px 8px;")
        if self.eye_button:
            self.eye_button.show()
        if self.input_port_label:
            self.input_port_label.hide()
            self.input_port_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        if self.output_port_label:
            self.output_port_label.hide()
            self.output_port_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self._simplify_layout()

    def _simplify_layout(self) -> None:
        """Simplify layout margins and spacing for compact views."""
        layout = self.layout()
        if layout is not None:
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(3)

    def _restore_layout(self) -> None:
        """Restore full layout margins and spacing."""
        layout = self.layout()
        if layout is not None:
            layout.setContentsMargins(10, 8, 10, 10)
            layout.setSpacing(5)

    def _status_chip_state(self) -> str:
        if self.method.method_id == "raw_input":
            return "on"
        if self.method.hidden:
            return "hide"
        if not self.method.enabled:
            return "off"
        status = getattr(self.method, "status", "idle")
        valid_states = ["idle", "queued", "running", "success", "success_stale", "failed", "skipped"]
        if status in valid_states:
            return status
        return "idle"

    def _status_chip_text(self) -> str:
        if self.method.method_id == "raw_input":
            return "SOURCE"
        if self.method.hidden:
            return "HIDE"
        if not self.method.enabled:
            return "OFF"
        status = getattr(self.method, "status", "idle")
        status_map = {
            "idle": "IDLE",
            "pending": "IDLE",
            "queued": "QUE",
            "running": "RUN",
            "success": "OK",
            "success_stale": "OLD",
            "failed": "FAIL",
            "skipped": "SKIP",
            "cancelled": "CANC",
        }
        return status_map.get(status, "IDLE")

    def _refresh_state_properties(self) -> None:
        self.setProperty("hiddenState", bool(self.method.hidden))
        status = getattr(self.method, "status", "idle")
        valid_statuses = ["queued", "running", "success", "success_stale", "failed", "skipped"]
        if status in valid_statuses:
            self.setProperty("status", status)
        else:
            self.setProperty("status", "")
        self.style().unpolish(self)
        self.style().polish(self)

    def refresh_status(self) -> None:
        chip = getattr(self, "status_chip", None)
        if isinstance(chip, QLabel):
            chip.setText(self._status_chip_text())
            chip.setProperty("state", self._status_chip_state())
            chip.style().unpolish(chip)
            chip.style().polish(chip)
        self._refresh_state_properties()

    def _install_wheel_guard(self, control: QWidget) -> None:
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
        # 处理滚轮事件
        if event.type() == QEvent.Type.Wheel and isinstance(watched, QWidget):
            if bool(watched.property("workflowWheelGuard")):
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _candidate_methods(self) -> list[str]:
        return candidate_methods_for_workflow_method(self.method)

    def _stage_label(self) -> str:
        stage = WORKFLOW_STAGE_BY_ID.get(self.method.stage_id, {})
        category = METHOD_CATEGORIES.get(self.method.category, {})
        return str(stage.get("label") or category.get("name") or self.method.category or "未分组")

    def _stage_warning(self) -> str:
        stage_warn = str(WORKFLOW_STAGE_BY_ID.get(self.method.stage_id, {}).get("warning", ""))
        # Check sidecar requirements
        if self.method.method_id in PROCESSING_METHODS:
            meta = PROCESSING_METHODS[self.method.method_id]
            missing_reqs = []
            # TODO: 获取项目数据是全局状态怎么获取？暂时先不检查具体值，至少能显示 warning
        return stage_warn

    def _toggle_expanded(self) -> None:
        self.toggle_expanded()
        self.changed.emit(self.row)

    def _on_eye_clicked(self) -> None:
        if self._suppress:
            return
        self.method.hidden = not bool(self.method.hidden)
        self.changed.emit(self.row)

    def _on_algorithm_combo_changed(self, index: int) -> None:
        """QComboBox 选择变化时切换算法，使用延迟避免重建自身。"""
        if self._suppress:
            return
        method_id = self.algorithm_combo.itemData(index)
        if not method_id or method_id == self.method.method_id:
            return
        QTimer.singleShot(0, lambda mid=str(method_id): self._switch_algorithm_from_card(mid))

    def _switch_algorithm_from_card(self, method_id: str) -> None:
        """Switch to a new algorithm and rebuild the card."""
        # 避免当前算法重复触发
        if method_id == self.method.method_id:
            return
        
        if update_workflow_method_algorithm(self.method, method_id):
            self._build()
            self.changed.emit(self.row)

    def _on_algorithm_changed(self) -> None:
        """Legacy handler for algorithm combo - no longer used for new QComboBox."""
        pass

    def set_method_enabled(self, enabled: bool) -> None:
        self.method.enabled = bool(enabled)
        self._build()

    def set_method_hidden(self, hidden: bool) -> None:
        self.method.hidden = bool(hidden)
        self._build()

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
        container = QWidget()
        container.setProperty("workflowNumericParam", True)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

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
        spin.valueChanged.connect(self._on_param_changed)
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
            control = BoolChipButton()
            control.setObjectName("boolChip")
            control.setCheckable(True)
            control.setFixedWidth(68)
            control.setChecked(bool(value))
            control.setText("ON" if control.isChecked() else "OFF")
            control.setProperty("checked", "true" if control.isChecked() else "false")

            def on_toggled(checked: bool) -> None:
                control.setText("ON" if checked else "OFF")
                control.setProperty("checked", "true" if checked else "false")
                control.style().unpolish(control)
                control.style().polish(control)
                self._on_param_changed()

            control.toggled.connect(on_toggled)
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
            control.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            control.setAlignment(Qt.AlignmentFlag.AlignRight)
            control.setMinimumWidth(96)
            control.setMaximumWidth(140)
            min_v = int(meta.get("min", -1000000))
            max_v = int(meta.get("max", 1000000))
            control.setRange(min_v, max_v)
            try:
                control.setValue(int(float(value)))
            except Exception:
                control.setValue(int(meta.get("default", min_v)))
            return self._wrap_numeric_control(control, is_float=False)

        if param_type == "float":
            control = QDoubleSpinBox()
            control.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            control.setAlignment(Qt.AlignmentFlag.AlignRight)
            control.setMinimumWidth(96)
            control.setMaximumWidth(140)
            min_v = float(meta.get("min", -1.0e9))
            max_v = float(meta.get("max", 1.0e9))
            control.setRange(min_v, max_v)
            control.setDecimals(6 if abs(max_v) <= 1 else 3)
            try:
                control.setValue(float(value))
            except Exception:
                control.setValue(float(meta.get("default", min_v)))
            return self._wrap_numeric_control(control, is_float=True)

        control = QLineEdit(str(value))
        control.textEdited.connect(self._on_param_changed)
        return control, control.text


class WorkflowPortItem(QGraphicsEllipseItem):
    """Lightweight graphics port used as an edge endpoint."""

    def __init__(self, port_name: str, owner: "WorkflowNodeProxy", label: str = "data"):
        super().__init__(-5.5, -5.5, 11.0, 11.0, owner)
        self.port_name = str(port_name)
        self.owner = owner
        self.label = str(label)
        self.setBrush(QBrush(QColor("#3278ff") if port_name == "input" else QColor("#7d4cff")))
        self.setPen(QPen(QColor("#ffffff"), 1.4))
        self.setZValue(30)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setAcceptHoverEvents(True)
        self.label_item = QGraphicsSimpleTextItem(self.label, owner)
        self.label_item.setBrush(QBrush(QColor("#64748b")))
        self.label_item.setZValue(29)
        self.label_item.hide()

    def scene_anchor(self) -> QPointF:
        return self.sceneBoundingRect().center()

    def set_label(self, label: str) -> None:
        self.label = str(label)
        self.label_item.setText(self.label)
        self.set_label_pos()

    def set_label_pos(self) -> None:
        if self.port_name == "input":
            self.label_item.setPos(self.pos().x() - 2.0, self.pos().y() - 24.0)
        else:
            self.label_item.setPos(self.pos().x() - 24.0, self.pos().y() + 6.0)

    def hoverEnterEvent(self, event):  # noqa: N802 - Qt override
        self.label_item.show()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # noqa: N802 - Qt override
        self.label_item.hide()
        super().hoverLeaveEvent(event)


class MiniNodeItem(QGraphicsRectItem):
    """Low-detail overlay shown when the canvas is zoomed out. Acts as visual summary only."""

    def __init__(self, proxy: "WorkflowNodeProxy"):
        super().__init__(proxy)  # 作为 proxy 的 child item
        self.proxy = proxy
        self.setBrush(QBrush(QColor("#f9fbff")))
        self.setPen(QPen(QColor("#8fb3ff"), 1.4))
        self.setZValue(15)  # 在端口之下，但可见
        # 不拦截鼠标事件，让事件穿透到父 proxy
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.title_item = QGraphicsSimpleTextItem("", self)
        self.subtitle_item = QGraphicsSimpleTextItem("", self)
        self.status_item = QGraphicsSimpleTextItem("", self)
        for item in [self.title_item, self.subtitle_item, self.status_item]:
            item.setBrush(QBrush(QColor("#1f2d3d")))
        self.hide()

    def refresh(self) -> None:
        rect = self.proxy.boundingRect()
        self.setRect(QRectF(0, 0, max(220.0, min(260.0, rect.width())), 92.0))
        # child item 使用本地坐标 (0,0)，自动跟随父 proxy
        method = self.proxy.method
        stage = WORKFLOW_STAGE_BY_ID.get(method.stage_id, {})
        stage_label = str(stage.get("label") or method.category or "节点")
        method_name = get_method_display_name(method.method_id)
        
        # 确定状态显示
        if method.hidden:
            status = "HIDE"
        elif not method.enabled:
            status = "OFF"
        elif self.proxy.row < 0:
            # Preview 节点的特殊状态显示
            method_status = getattr(method, "status", "idle")
            if method_status == "failed":
                status = "失败"
            elif method_status == "success_stale":
                status = "上次"
            elif method_status == "success":
                status = "最新"
            else:
                status = "暂无"
        else:
            # 普通节点使用 status_short
            status = self.proxy.status_short()
        
        in_label, out_label = workflow_port_labels(method)
        if self.proxy.row < 0:
            self.title_item.setText("B-scan Preview")
            self.subtitle_item.setText(status)
            self.status_item.setText("")
        else:
            self.title_item.setText(f"{self.proxy.row + 1:02d} {stage_label[:18]}")
            self.subtitle_item.setText(f"{method_name[:20]}   {status}")
            self.status_item.setText("")
        self.title_item.setPos(10, 8)
        self.subtitle_item.setPos(10, 32)
        self.status_item.setPos(10, 58)
        if method.hidden:
            self.setBrush(QBrush(QColor("#f4f6f8")))
            pen = QPen(QColor("#9aa8ba"), 1.2, Qt.PenStyle.DashLine)
        elif not method.enabled:
            self.setBrush(QBrush(QColor("#f1f3f6")))
            pen = QPen(QColor("#b8c1cd"), 1.2)
        else:
            self.setBrush(QBrush(QColor("#f9fbff")))
            pen = QPen(QColor("#8fb3ff"), 1.4)
        self.setPen(pen)


class CompactNodeItem(QGraphicsRectItem):
    """Medium-detail overlay shown at workflow zoom between full and mini. Acts as visual summary only."""

    def __init__(self, proxy: "WorkflowNodeProxy"):
        super().__init__(proxy)  # 作为 proxy 的 child item
        self.proxy = proxy
        self.setZValue(15)  # 在端口之下，但可见
        # 不拦截鼠标事件，让事件穿透到父 proxy
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.title_item = QGraphicsSimpleTextItem("", self)
        self.method_item = QGraphicsSimpleTextItem("", self)
        self.port_item = QGraphicsSimpleTextItem("", self)
        for item in [self.title_item, self.method_item, self.port_item]:
            item.setBrush(QBrush(QColor("#1f2d3d")))
        self.hide()

    def refresh(self) -> None:
        rect = self.proxy.boundingRect()
        self.setRect(QRectF(0, 0, max(240.0, min(300.0, rect.width())), 112.0))
        # child item 使用本地坐标 (0,0)，自动跟随父 proxy
        method = self.proxy.method
        stage = WORKFLOW_STAGE_BY_ID.get(method.stage_id, {})
        stage_label = str(stage.get("label") or method.category or "节点")
        method_name = get_method_display_name(method.method_id)
        in_label, out_label = workflow_port_labels(method)
        
        # 确定状态显示
        if method.hidden:
            status = "HIDE"
        elif not method.enabled:
            status = "OFF"
        elif self.proxy.row < 0:
            # Preview 节点的特殊状态显示
            method_status = getattr(method, "status", "idle")
            if method_status == "failed":
                status = "失败"
            elif method_status == "success_stale":
                status = "上次"
            elif method_status == "success":
                status = "最新"
            else:
                status = "暂无"
        else:
            # 普通节点使用 status_short
            status = self.proxy.status_short()
        
        if self.proxy.row < 0:
            output_shape = getattr(method, "output_shape", None)
            shape_text = "--" if output_shape is None else f"{output_shape[1]}×{output_shape[0]}"
            self.title_item.setText(f"B-scan Preview   {status}")
            self.method_item.setText(shape_text)
            self.port_item.setText(f"data → preview")
        else:
            self.title_item.setText(f"{self.proxy.row + 1:02d} {stage_label[:16]}   {status}")
            self.method_item.setText(method_name[:24])
            self.port_item.setText(f"{in_label} → {out_label}")
        self.title_item.setPos(10, 10)
        self.method_item.setPos(10, 40)
        self.port_item.setPos(10, 70)
        if method.hidden:
            brush = QBrush(QColor("#f4f6f8"))
            pen = QPen(QColor("#9aa8ba"), 1.2, Qt.PenStyle.DashLine)
        elif not method.enabled:
            brush = QBrush(QColor("#f1f3f6"))
            pen = QPen(QColor("#b8c1cd"), 1.2)
        else:
            brush = QBrush(QColor("#f9fbff"))
            pen = QPen(QColor("#8fb3ff"), 1.4)
        self.setBrush(brush)
        self.setPen(pen)


class WorkflowResizeHandleItem(QGraphicsRectItem):
    """Small bottom-right resize handle owned by a node proxy."""

    def __init__(self, owner: "WorkflowNodeProxy"):
        super().__init__(0, 0, 14, 14, owner)
        self.owner = owner
        self.setBrush(QBrush(QColor("#94a3b8")))
        self.setPen(QPen(QColor("#ffffff"), 1.0))
        self.setZValue(40)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setOpacity(0.22)

    def hoverEnterEvent(self, event):  # noqa: N802 - Qt override
        self.setOpacity(0.95)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # noqa: N802 - Qt override
        self.setOpacity(0.22)
        super().hoverLeaveEvent(event)


class WorkflowNodeProxy(QGraphicsProxyWidget):
    """Movable graphics proxy that owns ports, mini LOD, and resize handle."""

    def __init__(self, row: int, method: WorkflowMethod, parent=None):
        super().__init__(parent)
        self.row = int(row)
        self.method = method
        self.node_id = method.node_id if row >= 0 else PREVIEW_NODE_ID
        in_label, out_label = workflow_port_labels(method)
        self.input_port = WorkflowPortItem("input", self, in_label)
        self.output_port = WorkflowPortItem("output", self, out_label)
        self.compact_item = CompactNodeItem(self)
        self.mini_item = MiniNodeItem(self)
        self.resize_handle = WorkflowResizeHandleItem(self)
        self.setFlags(
            QGraphicsProxyWidget.GraphicsItemFlag.ItemIsMovable
            | QGraphicsProxyWidget.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsProxyWidget.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    def setWidget(self, widget):  # noqa: N802 - Qt override
        super().setWidget(widget)
        # 确保 layout 被激活
        if widget is not None:
            widget.layout().activate()
            widget.updateGeometry()
        self.update_visual_state()
        self.update_port_positions()

    def update_visual_state(self) -> None:
        if self.row < 0:
            self.setOpacity(1.0)
            return
        if self.method.hidden:
            self.setOpacity(0.45)
        elif not self.method.enabled:
            self.setOpacity(0.70)
        else:
            self.setOpacity(1.0)
        self.compact_item.refresh()
        self.mini_item.refresh()

    def set_lod_compact(self, compact: bool) -> None:
        self.set_lod_mode("compact" if compact else "full")

    def status_short(self) -> str:
        status = getattr(self.method, "status", "idle")
        return {
            "queued": "QUE",
            "running": "RUN",
            "success": "OK",
            "success_stale": "OLD",
            "failed": "FAIL",
            "skipped": "SKIP",
        }.get(status, "ON")

    def set_lod_mode(self, mode: str) -> None:
        """Set level-of-detail mode. Uses widget internal LOD only (no overlay)."""
        widget = self.widget()
        mode = mode if mode in {"full", "compact", "mini"} else "full"
        
        # 始终显示 widget，让卡片内部处理 LOD
        if widget is not None and hasattr(widget, "set_lod_mode"):
            widget.set_lod_mode(mode)
        
        # resize handle 只在 full 模式下显示
        self.resize_handle.setVisible(mode == "full" and self.row >= 0)
        
        # compact/mini overlay 不再显示，只依赖 widget 内部 LOD
        self.compact_item.setVisible(False)
        self.mini_item.setVisible(False)
        
        self.update_port_positions()

    def apply_size(self, width: float, height: float) -> None:
        width = max(MIN_NODE_WIDTH, min(MAX_NODE_WIDTH, float(width)))
        height = max(MIN_NODE_HEIGHT, min(MAX_NODE_HEIGHT, float(height)))
        widget = self.widget()
        if widget is not None:
            widget.setMinimumSize(QSize(int(width), int(height)))
            widget.setMaximumSize(QSize(int(width), int(height)))
            widget.resize(int(width), int(height))
        self.update_port_positions()

    def update_port_positions(self) -> None:
        rect = self.boundingRect()
        if not rect.isValid() or rect.width() <= 0:
            return
        anchor_y = rect.center().y()
        widget = self.widget()
        if widget is not None and hasattr(widget, "port_anchor_y"):
            try:
                anchor_y = float(widget.port_anchor_y())
            except Exception:
                anchor_y = rect.center().y()
        self.input_port.setPos(rect.left(), anchor_y)
        self.output_port.setPos(rect.right(), anchor_y)
        self.input_port.set_label_pos()
        self.output_port.set_label_pos()
        self.resize_handle.setPos(rect.right() - 16.0, rect.bottom() - 16.0)
        self.compact_item.refresh()
        self.mini_item.refresh()

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        result = super().itemChange(change, value)
        if change == QGraphicsProxyWidget.GraphicsItemChange.ItemPositionHasChanged:
            self.update_port_positions()
            # compact_item 和 mini_item 现在是 child item，会自动跟随父 item 的位置
            # 不需要手动设置位置
            scene = self.scene()
            if isinstance(scene, WorkflowCanvasScene):
                scene.update_edges()
        return result


class WorkflowEdgeItem(QGraphicsPathItem):
    """Interactive edge between two workflow ports."""

    def __init__(self, link: WorkflowLink, source_port: WorkflowPortItem, target_port: WorkflowPortItem):
        super().__init__()
        self.link = link
        self.source_port = source_port
        self.target_port = target_port
        self._hovered = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(-10)
        self._apply_pen()
        self.update_path()

    def _apply_pen(self) -> None:
        if self.isSelected():
            pen = QPen(QColor("#ff9f1a"), 3.0)
        elif self._hovered:
            pen = QPen(QColor("#3278ff"), 3.0)
        else:
            pen = QPen(QColor("#77808f"), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._apply_pen()
        return result

    def hoverEnterEvent(self, event):  # noqa: N802 - Qt override
        self._hovered = True
        self._apply_pen()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # noqa: N802 - Qt override
        self._hovered = False
        self._apply_pen()
        super().hoverLeaveEvent(event)

    def update_path(self) -> None:
        start = self.source_port.scene_anchor()
        end = self.target_port.scene_anchor()
        dx = max(80.0, abs(end.x() - start.x()) * 0.45)
        path = QPainterPath(start)
        path.cubicTo(
            QPointF(start.x() + dx, start.y()),
            QPointF(end.x() - dx, end.y()),
            end,
        )
        self.setPath(path)

    def contextMenuEvent(self, event):  # noqa: N802 - Qt override
        scene = self.scene()
        if isinstance(scene, WorkflowCanvasScene):
            menu = QMenu()
            delete_action = menu.addAction("删除连接")
            selected = menu.exec(event.screenPos())
            if selected == delete_action:
                scene.remove_edge(self)
                event.accept()
                return
        super().contextMenuEvent(event)


class WorkflowCanvasScene(QGraphicsScene):
    """Scene containing node proxy widgets and explicit workflow edges."""

    link_removed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proxies: list[WorkflowNodeProxy] = []
        self.proxy_by_id: dict[str, WorkflowNodeProxy] = {}
        self.links: list[WorkflowLink] = []
        self.edges: list[WorkflowEdgeItem] = []
        self.edge_by_key: dict[tuple[str, str, str, str, str], WorkflowEdgeItem] = {}

    def link_key(self, link: WorkflowLink) -> tuple[str, str, str, str, str]:
        return (link.from_node, link.from_port, link.to_node, link.to_port, link.kind)

    def set_links(self, links: list[WorkflowLink]) -> None:
        self.links = [WorkflowLink.from_dict(link.to_dict()) for link in links]
        self.update_edges()

    def all_links_with_preview(self) -> list[WorkflowLink]:
        links = list(self.links)
        method_proxies = [proxy for proxy in self.proxies if proxy.row >= 0]
        if method_proxies and PREVIEW_NODE_ID in self.proxy_by_id:
            last = method_proxies[-1]
            if not any(link.to_node == PREVIEW_NODE_ID for link in links):
                links.append(WorkflowLink(last.node_id, PREVIEW_NODE_ID, kind="preview"))
        return links

    def update_edges(self) -> None:
        valid_links = []
        for proxy in self.proxies:
            proxy.update_visual_state()
            proxy.update_port_positions()

        for link in self.all_links_with_preview():
            source = self.proxy_by_id.get(link.from_node)
            target = self.proxy_by_id.get(link.to_node)
            if source is None or target is None:
                continue
            if link.from_port != "output" or link.to_port != "input":
                continue
            valid_links.append(link)

        valid_keys = {self.link_key(link) for link in valid_links}
        for key, edge in list(self.edge_by_key.items()):
            if key not in valid_keys:
                self.removeItem(edge)
                self.edge_by_key.pop(key, None)

        for link in valid_links:
            key = self.link_key(link)
            source = self.proxy_by_id[link.from_node].output_port
            target = self.proxy_by_id[link.to_node].input_port
            edge = self.edge_by_key.get(key)
            if edge is None:
                edge = WorkflowEdgeItem(link, source, target)
                self.addItem(edge)
                self.edge_by_key[key] = edge
            else:
                edge.source_port = source
                edge.target_port = target
                edge.update_path()

        self.edges = list(self.edge_by_key.values())

    def remove_edge(self, edge: WorkflowEdgeItem) -> None:
        key = self.link_key(edge.link)
        self.links = [link for link in self.links if self.link_key(link) != key]
        self.removeItem(edge)
        self.edge_by_key.pop(key, None)
        self.edges = list(self.edge_by_key.values())
        self.link_removed.emit(edge.link)


class WorkflowCanvasView(QGraphicsView):
    """Workflow Studio canvas with nodes, ports, links, and preview output."""

    node_selected = pyqtSignal(int)
    node_changed = pyqtSignal(int)
    run_node_requested = pyqtSignal(int)
    run_from_node_requested = pyqtSignal(int)
    duplicate_node_requested = pyqtSignal(int)
    remove_node_requested = pyqtSignal(int)
    add_node_requested = pyqtSignal(str, QPointF)
    tuning_lab_requested = pyqtSignal(int)
    apply_best_params_requested = pyqtSignal(int)
    benchmark_node_requested = pyqtSignal(int)
    preview_large_requested = pyqtSignal(object, str)
    preview_settings_requested = pyqtSignal()
    preview_compare_requested = pyqtSignal()
    preview_snapshot_requested = pyqtSignal()
    links_changed = pyqtSignal(object)
    layout_changed = pyqtSignal(object)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = WorkflowCanvasScene(self)
        self.setScene(self._scene)
        self._scene.link_removed.connect(self._on_scene_link_removed)
        self._methods: list[WorkflowMethod] = []
        self._links: list[WorkflowLink] = []
        self._canvas_layout: dict = {"nodes": {}}
        self._current_row = -1
        self._panning = False
        self._left_panning = False
        self._space_pressed = False
        self._last_pan_pos = QPoint()
        self._drag_proxy: WorkflowNodeProxy | None = None
        self._drag_scene_offset = QPointF()
        self._resize_proxy: WorkflowNodeProxy | None = None
        self._resize_start_scene = QPointF()
        self._resize_start_size = QSize()
        self._drag_source_port: WorkflowPortItem | None = None
        self._temp_edge: QGraphicsPathItem | None = None
        self._preview_data = None
        self._preview_label = "Workflow Output"
        self._last_preview_source_node = None
        self._last_preview_shape = None
        self._last_preview_is_stale = False
        self._last_run_success = False
        self._preview_proxy: WorkflowNodeProxy | None = None
        self._rebuild_pending = False
        self._compact_cards = False
        self._lod_state = "full"
        self._compact_threshold = 0.58
        self._normal_threshold = 0.74
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSceneRect(-600, -400, 5200, 2600)
        self.setMinimumHeight(520)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            """
            QGraphicsView {
                background: #f4f7fb;
                border: 1px solid #d8e3f2;
                border-radius: 10px;
            }
            """
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt override
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802 - Qt override
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt override
        method_id = event.mimeData().text()
        scene_pos = self.mapToScene(event.pos())
        self.add_node_requested.emit(method_id, scene_pos)
        event.acceptProposedAction()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802 - Qt override
        super().drawBackground(painter, rect)
        grid = 32
        pen = QPen(QColor("#dfe7f1"), 1)
        painter.setPen(pen)
        left = int(rect.left()) - (int(rect.left()) % grid)
        top = int(rect.top()) - (int(rect.top()) % grid)
        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                painter.drawPoint(x, y)
                y += grid
            x += grid

    def set_methods(self, methods: list[WorkflowMethod]) -> None:
        self.set_workflow(methods, None, None)

    def set_workflow(
        self,
        methods: list[WorkflowMethod],
        links: list[WorkflowLink] | None = None,
        canvas_layout: dict | None = None,
    ) -> None:
        ensure_workflow_method_ids(methods)
        self._methods = sorted(methods, key=lambda item: item.order)
        self._links = self._default_links(self._methods) if links is None else [WorkflowLink.from_dict(link.to_dict()) for link in links]
        self._canvas_layout = canvas_layout if isinstance(canvas_layout, dict) else {"nodes": {}}
        self._rebuild()

    def current_links(self) -> list[WorkflowLink]:
        return [WorkflowLink.from_dict(link.to_dict()) for link in self._links]

    def update_node(self, row: int) -> None:
        """Update a single node without rebuilding the entire scene."""
        row = int(row)
        if 0 <= row < len(self._methods):
            # 查找对应的 proxy
            proxy = self._scene.proxy_by_id.get(self._methods[row].node_id)
            if proxy is not None and isinstance(proxy, WorkflowNodeProxy):
                # 保存所有状态
                old_widget = proxy.widget()
                current_lod_mode = old_widget._lod_mode if hasattr(old_widget, '_lod_mode') else 'full'
                old_width = old_widget.width() if old_widget is not None else None
                old_height = old_widget.height() if old_widget is not None else None
                old_pos = proxy.pos()
                is_current = (proxy.row == self._current_row)
                
                # 创建新的卡片
                card = WorkflowNodeCard(row, self._methods[row])
                card.selected.connect(self._on_node_selected)
                card.changed.connect(self.node_changed)
                card.run_current_requested.connect(self.run_node_requested)
                card.run_from_requested.connect(self.run_from_node_requested)
                card.duplicate_requested.connect(self.duplicate_node_requested)
                card.remove_requested.connect(self.remove_node_requested)
                
                # 应用保存的状态
                card.set_lod_mode(current_lod_mode)
                if old_width and old_height:
                    card.setFixedSize(old_width, old_height)
                
                # 更新 proxy
                proxy.setWidget(card)
                
                # 清理旧的 widget
                if old_widget is not None:
                    old_widget.deleteLater()
                
                # 恢复位置
                proxy.setPos(old_pos)
                
                # 更新端口
                in_label, out_label = workflow_port_labels(self._methods[row])
                proxy.input_port.set_label(in_label)
                proxy.output_port.set_label(out_label)
                
                # 更新视觉状态和选中态
                proxy.update_visual_state()
                proxy.update_port_positions()
                card.set_current(is_current)
                
                # 更新边
                self._scene.update_edges()
            else:
                # 如果找不到 proxy，回退到完整重建
                self._schedule_rebuild()

    def _schedule_rebuild(self) -> None:
        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        QTimer.singleShot(0, self._run_scheduled_rebuild)

    def _run_scheduled_rebuild(self) -> None:
        self._rebuild_pending = False
        self._rebuild()

    def set_preview_data(self, data, label: str = "Workflow Output", source_node=None, success: bool = True) -> None:
        self._last_run_success = bool(success)
        if success:
            self._preview_data = data
            self._preview_label = label or "Workflow Output"
            self._last_preview_source_node = source_node
            self._last_preview_is_stale = False
            if data is not None and hasattr(data, "shape"):
                self._last_preview_shape = data.shape
            else:
                self._last_preview_shape = None
        else:
            self._last_preview_is_stale = True
        card = self._preview_card()
        if card is not None:
            try:
                card.set_preview_data(self._preview_data, self._preview_label, success=not self._last_preview_is_stale)
                self._scene.update_edges()
            except RuntimeError:
                self._preview_proxy = None
                self._schedule_rebuild()

    def _preview_card(self) -> BscanPreviewCard | None:
        if self._preview_proxy is None:
            return None
        card = self._preview_proxy.widget()
        return card if isinstance(card, BscanPreviewCard) else None

    def get_preview_data(self):
        """Return the current preview data."""
        return self._preview_data

    def set_stale_preview(self) -> None:
        """Mark the current preview as stale."""
        self._last_preview_is_stale = True
        card = self._preview_card()
        if card is not None:
            try:
                card.set_stale()
            except Exception:
                pass

    def viewport_scene_center(self) -> QPointF:
        """Return the current viewport center in scene coordinates."""
        return self.mapToScene(self.viewport().rect().center())

    def set_selected_row(self, row: int) -> None:
        self._current_row = int(row)
        for proxy in self._scene.proxies:
            card = proxy.widget()
            if isinstance(card, WorkflowNodeCard):
                card.set_current(card.row == self._current_row)

    def refresh_all_nodes(self) -> None:
        for proxy in self._scene.proxies:
            card = proxy.widget()
            if isinstance(card, WorkflowNodeCard):
                card.refresh_status()
                proxy.update_visual_state()
        self._scene.update_edges()

    def _event_view_pos(self, event) -> QPoint:
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _proxy_at_view_pos(self, view_pos: QPoint) -> WorkflowNodeProxy | None:
        for item in self.items(view_pos):
            while item is not None and not isinstance(item, WorkflowNodeProxy):
                item = item.parentItem()
            if isinstance(item, WorkflowNodeProxy):
                return item
        return None

    def _port_at_view_pos(self, view_pos: QPoint) -> WorkflowPortItem | None:
        for item in self.items(view_pos):
            current = item
            while current is not None and not isinstance(current, WorkflowPortItem):
                current = current.parentItem()
            if isinstance(current, WorkflowPortItem):
                return current
        return None

    def _edge_at_view_pos(self, view_pos: QPoint) -> WorkflowEdgeItem | None:
        for item in self.items(view_pos):
            if isinstance(item, WorkflowEdgeItem):
                return item
        return None

    def _resize_handle_at_view_pos(self, view_pos: QPoint) -> WorkflowResizeHandleItem | None:
        for item in self.items(view_pos):
            if isinstance(item, WorkflowResizeHandleItem):
                return item
        return None

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
            # 检查是否是交互控件 (QComboBox 已被包含)
            if self._is_interactive_widget(child):
                return True
            child = child.parentWidget()
        return False

    def contextMenuEvent(self, event):  # noqa: N802 - Qt override
        view_pos = event.pos()
        edge = self._edge_at_view_pos(view_pos)
        if edge is not None:
            menu = QMenu(self)
            delete_action = menu.addAction("删除连接")
            selected = menu.exec(event.globalPos())
            if selected == delete_action:
                self._scene.remove_edge(edge)
            return

        proxy = self._proxy_at_view_pos(view_pos)
        if proxy is not None:
            menu = self._build_node_context_menu(proxy)
            menu.exec(event.globalPos())
            return

        scene_pos = self.mapToScene(view_pos)
        menu = self._build_canvas_context_menu(scene_pos)
        menu.exec(event.globalPos())

    def _build_node_context_menu(self, proxy: WorkflowNodeProxy) -> QMenu:
        menu = QMenu(self)
        if proxy.row >= 0:
            menu.addAction("运行此节点", lambda row=proxy.row: self.run_node_requested.emit(row))
            menu.addAction("从此节点运行", lambda row=proxy.row: self.run_from_node_requested.emit(row))
            menu.addAction("打开调参", lambda row=proxy.row: self.tuning_lab_requested.emit(row))
            menu.addAction("应用最佳参数", lambda row=proxy.row: self.apply_best_params_requested.emit(row))
            menu.addAction("评估此节点", lambda row=proxy.row: self.benchmark_node_requested.emit(row))
            candidates = candidate_methods_for_workflow_method(proxy.method)
            if len(candidates) > 1:
                switch_menu = menu.addMenu("切换算法")
                for key in candidates:
                    action = switch_menu.addAction(get_method_display_name(key))
                    action.setCheckable(True)
                    action.setChecked(key == proxy.method.method_id)
                    action.triggered.connect(
                        lambda _checked=False, p=proxy, method_key=key: self._switch_proxy_algorithm(p, method_key)
                    )
            menu.addSeparator()
            enabled_text = "停用" if proxy.method.enabled else "启用"
            menu.addAction(enabled_text, lambda p=proxy: self._toggle_proxy_enabled(p))
            hidden_text = "显示" if proxy.method.hidden else "隐藏"
            menu.addAction(hidden_text, lambda p=proxy: self._toggle_proxy_hidden(p))
            menu.addSeparator()
            menu.addAction("复制节点", lambda row=proxy.row: self.duplicate_node_requested.emit(row))
            menu.addAction("删除节点", lambda row=proxy.row: self.remove_node_requested.emit(row))
            menu.addAction("添加预览节点", lambda: self._ensure_preview_visible())
            rename_action = menu.addAction("重命名节点")
            rename_action.setEnabled(False)
            menu.addAction(
                "折叠" if self._card_expanded(proxy) else "展开",
                lambda p=proxy: self._toggle_proxy_expanded(p),
            )
            menu.addAction("适配到此节点", lambda p=proxy: self.fit_proxy(p))
        else:
            menu.addAction("打开大图", lambda p=proxy: self._open_preview_proxy(p))
            menu.addAction("添加前后对比", self.preview_compare_requested.emit)
            menu.addAction("保存快照", self.preview_snapshot_requested.emit)
            menu.addAction("预览设置", self.preview_settings_requested.emit)
            menu.addAction("适配到此节点", lambda p=proxy: self.fit_proxy(p))
        return menu

    def _build_canvas_context_menu(self, scene_pos: QPointF) -> QMenu:
        menu = QMenu(self)
        add_menu = menu.addMenu("添加节点")
        quick_groups = {
            "Input": [],
            "QC": ["trace_qc"],
            "Preprocess": ["set_zero_time", "dc_shift", "dewow", "frequency_filter_1d"],
            "Geometry": ["motion_compensation_v2", "geometry_depth_context"],
            "Clutter": ["subtracting_average_2D", "median_background_2D", "svd_bg", "fk_filter"],
            "Denoise": ["svd_subspace", "wavelet_2d", "wavelet_svd"],
            "Imaging": ["kirchhoff_migration", "stolt_migration", "time_to_depth"],
            "Preview": [],
            "Export": [],
        }
        for label, method_ids in quick_groups.items():
            submenu = add_menu.addMenu(label)
            if not method_ids:
                placeholder = submenu.addAction("暂未接入")
                placeholder.setEnabled(False)
            for method_id in method_ids:
                if method_id in PROCESSING_METHODS:
                    submenu.addAction(
                        get_method_display_name(method_id),
                        lambda _=False, key=method_id, pos=QPointF(scene_pos): self.add_node_requested.emit(key, pos),
                    )
        menu.addAction("粘贴节点").setEnabled(False)
        menu.addSeparator()
        menu.addAction("适配全部节点", self.fit_nodes)
        menu.addAction("重排节点", self.auto_layout)
        menu.addAction("恢复 100% 缩放", self.reset_zoom)
        return menu

    def _card_expanded(self, proxy: WorkflowNodeProxy) -> bool:
        card = proxy.widget()
        return isinstance(card, WorkflowNodeCard) and bool(card.expanded)

    def _toggle_proxy_expanded(self, proxy: WorkflowNodeProxy) -> None:
        card = proxy.widget()
        if isinstance(card, WorkflowNodeCard):
            card.toggle_expanded()
            self._scene.update_edges()

    def _toggle_proxy_enabled(self, proxy: WorkflowNodeProxy) -> None:
        proxy.method.enabled = not bool(proxy.method.enabled)
        self.node_changed.emit(proxy.row)

    def _toggle_proxy_hidden(self, proxy: WorkflowNodeProxy) -> None:
        proxy.method.hidden = not bool(proxy.method.hidden)
        self.node_changed.emit(proxy.row)

    def _switch_proxy_algorithm(self, proxy: WorkflowNodeProxy, method_id: str) -> None:
        if update_workflow_method_algorithm(proxy.method, method_id):
            self.node_changed.emit(proxy.row)

    def _ensure_preview_visible(self) -> None:
        if self._preview_proxy is not None:
            self.fit_proxy(self._preview_proxy)

    def _open_preview_proxy(self, proxy: WorkflowNodeProxy) -> None:
        card = proxy.widget()
        if isinstance(card, BscanPreviewCard):
            self.preview_large_requested.emit(card._data, card._label)

    def viewportEvent(self, event):  # noqa: N802 - Qt override
        event_type = event.type()

        if event_type == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
            proxy = self._proxy_at_view_pos(self._event_view_pos(event))
            if proxy is not None:
                card = proxy.widget()
                if isinstance(card, BscanPreviewCard):
                    self.preview_large_requested.emit(card._data, card._label)
                    event.accept()
                    return True

        if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            view_pos = self._event_view_pos(event)
            scene_pos = self.mapToScene(view_pos)

            handle = self._resize_handle_at_view_pos(view_pos)
            if handle is not None:
                self._resize_proxy = handle.owner
                self._resize_start_scene = scene_pos
                widget = self._resize_proxy.widget()
                self._resize_start_size = widget.size() if widget is not None else QSize(MIN_NODE_WIDTH, MIN_NODE_HEIGHT)
                event.accept()
                return True

            port = self._port_at_view_pos(view_pos)
            if port is not None and port.port_name == "output":
                self._start_temp_edge(port)
                event.accept()
                return True

            proxy = self._proxy_at_view_pos(view_pos)
            if proxy is not None and not self._is_interactive_card_target(proxy, scene_pos):
                if self._space_pressed:
                    self._start_pan(view_pos, left=True)
                    event.accept()
                    return True
                self._drag_proxy = proxy
                self._drag_scene_offset = scene_pos - proxy.pos()
                if proxy.row >= 0:
                    self.node_selected.emit(proxy.row)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return True

            if proxy is None and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._start_pan(view_pos, left=True)
                event.accept()
                return True

        if event_type == QEvent.Type.MouseMove:
            view_pos = self._event_view_pos(event)
            scene_pos = self.mapToScene(view_pos)

            if self._resize_proxy is not None and event.buttons() & Qt.MouseButton.LeftButton:
                delta = scene_pos - self._resize_start_scene
                self._resize_proxy.apply_size(
                    self._resize_start_size.width() + delta.x(),
                    self._resize_start_size.height() + delta.y(),
                )
                self._record_node_layout(self._resize_proxy)
                self._scene.update_edges()
                event.accept()
                return True

            if self._drag_source_port is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self._update_temp_edge(scene_pos)
                event.accept()
                return True

            if self._drag_proxy is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self._drag_proxy.setPos(scene_pos - self._drag_scene_offset)
                self._record_node_layout(self._drag_proxy)
                self._scene.update_edges()
                event.accept()
                return True

            if (self._left_panning or self._panning) and event.buttons() & (
                Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton
            ):
                self._pan_to(view_pos)
                event.accept()
                return True

        if event_type == QEvent.Type.MouseButtonRelease:
            view_pos = self._event_view_pos(event)
            if self._resize_proxy is not None:
                self._record_node_layout(self._resize_proxy)
                self._resize_proxy = None
                self.layout_changed.emit(self._canvas_layout)
                event.accept()
                return True

            if self._drag_source_port is not None:
                target = self._port_at_view_pos(view_pos)
                if target is not None:
                    self._finish_temp_edge(target)
                self._clear_temp_edge()
                event.accept()
                return True

            if self._drag_proxy is not None:
                self._record_node_layout(self._drag_proxy)
                self.layout_changed.emit(self._canvas_layout)
                self._clear_drag_state()
                event.accept()
                return True

            if self._left_panning:
                self._left_panning = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return True

        return super().viewportEvent(event)

    def _start_pan(self, view_pos: QPoint, *, left: bool = False) -> None:
        self._left_panning = bool(left)
        self._panning = not left
        self._last_pan_pos = view_pos
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _pan_to(self, view_pos: QPoint) -> None:
        delta = view_pos - self._last_pan_pos
        self._last_pan_pos = view_pos
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def _clear_drag_state(self) -> None:
        self._drag_proxy = None
        self._drag_scene_offset = QPointF()
        if not self._panning and not self._left_panning:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _start_temp_edge(self, port: WorkflowPortItem) -> None:
        self._drag_source_port = port
        self._temp_edge = QGraphicsPathItem()
        pen = QPen(QColor("#3278ff"), 2.0, Qt.PenStyle.DashLine)
        self._temp_edge.setPen(pen)
        self._temp_edge.setZValue(-5)
        self._scene.addItem(self._temp_edge)
        self._update_temp_edge(port.scene_anchor())

    def _update_temp_edge(self, end: QPointF) -> None:
        if self._drag_source_port is None or self._temp_edge is None:
            return
        start = self._drag_source_port.scene_anchor()
        dx = max(80.0, abs(end.x() - start.x()) * 0.45)
        path = QPainterPath(start)
        path.cubicTo(QPointF(start.x() + dx, start.y()), QPointF(end.x() - dx, end.y()), end)
        self._temp_edge.setPath(path)

    def _finish_temp_edge(self, target: WorkflowPortItem) -> None:
        source = self._drag_source_port
        if source is None:
            return
        if not self._is_valid_connection(source, target):
            return
        new_link = WorkflowLink(source.owner.node_id, target.owner.node_id, "output", "input", "data")
        self._links = [
            link
            for link in self._links
            if not (link.to_node == new_link.to_node and link.to_port == "input" and link.kind == "data")
        ]
        self._links.append(new_link)
        self._scene.set_links(self._links)
        self.links_changed.emit(self.current_links())

    def _is_valid_connection(self, source: WorkflowPortItem, target: WorkflowPortItem) -> bool:
        return (
            source.port_name == "output"
            and target.port_name == "input"
            and source.owner is not target.owner
            and source.owner.node_id != target.owner.node_id
        )

    def _clear_temp_edge(self) -> None:
        if self._temp_edge is not None:
            self._scene.removeItem(self._temp_edge)
        self._temp_edge = None
        self._drag_source_port = None

    def wheelEvent(self, event):  # noqa: N802 - Qt override
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - event.angleDelta().y())
            event.accept()
            return
        # 以鼠标位置为中心缩放 - 使用 scrollbar 补偿方式
        mouse_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        old_scene_pos = self.mapToScene(mouse_pos)
        # 计算缩放因子
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        old_scale = self.transform().m11()
        new_scale = max(0.32, min(2.8, old_scale * factor))
        # 设置新缩放
        self._set_zoom_scale(new_scale)
        # 计算并应用 scrollbar 偏移，使鼠标下的点保持不变
        new_mouse_pos = self.mapFromScene(old_scene_pos)
        scroll_delta = new_mouse_pos - mouse_pos
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + scroll_delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + scroll_delta.y())
        event.accept()

    def _set_zoom_scale(self, scale: float) -> None:
        next_scale = max(0.32, min(2.8, float(scale)))
        self.resetTransform()
        self.scale(next_scale, next_scale)
        self._apply_zoom_lod(force=True)
        self.zoom_changed.emit(next_scale)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_pan(event.pos(), left=False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        if self._panning:
            self._pan_to(event.pos())
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

    def keyPressEvent(self, event: QKeyEvent):  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = True
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            self.delete_selected_items()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._clear_temp_edge()
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_D and self._current_row >= 0:
                self.duplicate_node_requested.emit(self._current_row)
                event.accept()
                return
            if event.key() == Qt.Key.Key_0:
                self.fit_nodes()
                event.accept()
                return
            if event.key() == Qt.Key.Key_1:
                self.reset_zoom()
                event.accept()
                return
            if event.key() == Qt.Key.Key_A:
                for proxy in self._scene.proxies:
                    proxy.setSelected(True)
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = False
            event.accept()
            return
        super().keyReleaseEvent(event)

    def delete_selected_items(self) -> None:
        selected_edges = [item for item in self._scene.selectedItems() if isinstance(item, WorkflowEdgeItem)]
        if selected_edges:
            for edge in selected_edges:
                self._scene.remove_edge(edge)
            return
        selected_nodes = [item for item in self._scene.selectedItems() if isinstance(item, WorkflowNodeProxy) and item.row >= 0]
        if selected_nodes:
            self.remove_node_requested.emit(selected_nodes[0].row)

    def fit_proxy(self, proxy: WorkflowNodeProxy) -> None:
        rect = proxy.sceneBoundingRect().adjusted(-80, -80, 80, 80)
        if rect.isValid():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._apply_zoom_lod(force=True)
            self.zoom_changed.emit(self.transform().m11())

    def fit_nodes(self) -> None:
        rect = self._scene.itemsBoundingRect().adjusted(-80, -80, 80, 80)
        if rect.isValid():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._apply_zoom_lod(force=True)
            self.zoom_changed.emit(self.transform().m11())

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._apply_zoom_lod(force=True)
        self.zoom_changed.emit(self.transform().m11())

    def auto_layout(self) -> None:
        x0, y0 = 60, 70
        x_step, y_step = 430, 280
        max_per_row = 3
        for index, proxy in enumerate([proxy for proxy in self._scene.proxies if proxy.row >= 0]):
            lane = index // max_per_row
            col = index % max_per_row
            proxy.setPos(x0 + col * x_step, y0 + lane * y_step)
            self._record_node_layout(proxy)
        if self._preview_proxy is not None:
            preview_row = len(self._methods)
            self._preview_proxy.setPos(
                x0 + (preview_row % max_per_row) * x_step,
                y0 + (preview_row // max_per_row) * y_step,
            )
        self._scene.update_edges()
        self.layout_changed.emit(self._canvas_layout)

    def _lod_mode(self) -> str:
        scale = self.transform().m11()
        if scale >= 0.85:
            return "full"
        if scale >= 0.50:
            return "compact"
        return "mini"

    def _apply_zoom_lod(self, force: bool = False) -> None:
        mode = self._lod_mode()
        if not force and mode == self._lod_state:
            return

        self._lod_state = mode
        self._compact_cards = mode != "full"
        for proxy in self._scene.proxies:
            proxy.set_lod_mode(mode)
        self._scene.update_edges()

    def _rebuild(self) -> None:
        self._clear_drag_state()
        self._clear_temp_edge()
        self._scene.clear()
        self._scene.proxies.clear()
        self._scene.proxy_by_id.clear()
        self._scene.edges.clear()
        self._scene.edge_by_key.clear()
        self._preview_proxy = None

        x0, y0 = 60, 70
        x_step, y_step = 430, 280
        max_per_row = 3
        nodes_layout = self._canvas_layout.setdefault("nodes", {})
        for row, method in enumerate(self._methods):
            card = WorkflowNodeCard(row, method)
            card.selected.connect(self._on_node_selected)
            card.changed.connect(self.node_changed)
            card.run_current_requested.connect(self.run_node_requested)
            card.run_from_requested.connect(self.run_from_node_requested)
            card.duplicate_requested.connect(self.duplicate_node_requested)
            card.remove_requested.connect(self.remove_node_requested)

            proxy = WorkflowNodeProxy(row, method)
            proxy.setWidget(card)
            layout = nodes_layout.get(method.node_id, {}) if isinstance(nodes_layout, dict) else {}
            lane = row // max_per_row
            col = row % max_per_row
            x = float(layout.get("x", x0 + col * x_step))
            y = float(layout.get("y", y0 + lane * y_step))
            proxy.setPos(x, y)
            if "width" in layout or "height" in layout:
                proxy.apply_size(float(layout.get("width", MIN_NODE_WIDTH)), float(layout.get("height", MIN_NODE_HEIGHT)))
            self._scene.addItem(proxy)
            # compact_item 和 mini_item 现在是 proxy 的 child item，不需要单独添加到 scene
            self._scene.proxies.append(proxy)
            self._scene.proxy_by_id[proxy.node_id] = proxy

        preview_card = BscanPreviewCard()
        preview_card.set_preview_data(self._preview_data, self._preview_label, success=not self._last_preview_is_stale)
        preview_card.large_view_requested.connect(self.preview_large_requested)
        preview_method = WorkflowMethod("preview", "bscan_preview", enabled=True, order=len(self._methods), node_id=PREVIEW_NODE_ID)
        
        # 设置预览状态，用于 compact/mini 模式显示
        if not self._last_run_success:
            preview_method.status = "failed"
        elif self._last_preview_is_stale:
            preview_method.status = "success_stale"
        elif self._preview_data is not None:
            preview_method.status = "success"
        else:
            preview_method.status = "idle"
        
        # 设置预览形状
        if self._last_preview_shape is not None:
            preview_method.output_shape = self._last_preview_shape
        
        preview_proxy = WorkflowNodeProxy(-1, preview_method)
        preview_proxy.output_port.hide()
        preview_proxy.output_port.label_item.hide()
        preview_proxy.resize_handle.hide()
        preview_proxy.setWidget(preview_card)
        preview_row = len(self._methods)
        preview_layout = nodes_layout.get(PREVIEW_NODE_ID, {}) if isinstance(nodes_layout, dict) else {}
        preview_proxy.setPos(
            float(preview_layout.get("x", x0 + (preview_row % max_per_row) * x_step)),
            float(preview_layout.get("y", y0 + (preview_row // max_per_row) * y_step)),
        )
        self._scene.addItem(preview_proxy)
        # compact_item 和 mini_item 现在是 proxy 的 child item，不需要单独添加到 scene
        self._scene.proxies.append(preview_proxy)
        self._scene.proxy_by_id[PREVIEW_NODE_ID] = preview_proxy
        self._preview_proxy = preview_proxy

        self._scene.set_links(self._links)
        self.set_selected_row(self._current_row)
        self._apply_zoom_lod(force=True)
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-240, -180, 480, 320))

    def _default_links(self, methods: list[WorkflowMethod]) -> list[WorkflowLink]:
        return [
            WorkflowLink(left.node_id, right.node_id)
            for left, right in zip(methods, methods[1:])
        ]

    def _record_node_layout(self, proxy: WorkflowNodeProxy) -> None:
        nodes = self._canvas_layout.setdefault("nodes", {})
        widget = proxy.widget()
        width = widget.width() if widget is not None else proxy.boundingRect().width()
        height = widget.height() if widget is not None else proxy.boundingRect().height()
        nodes[proxy.node_id] = {
            "x": float(proxy.pos().x()),
            "y": float(proxy.pos().y()),
            "width": int(width),
            "height": int(height),
            "collapsed": bool(self._compact_cards),
        }

    def _on_node_selected(self, row: int) -> None:
        self._current_row = int(row)
        self.set_selected_row(row)
        self.node_selected.emit(int(row))

    def _on_scene_link_removed(self, _link: WorkflowLink) -> None:
        self._links = self._scene.links
        self.links_changed.emit(self.current_links())

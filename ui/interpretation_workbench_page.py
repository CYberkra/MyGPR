#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured target marking and positioning workspace."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.interpretation_service import InterpretationService
from core.processing_session import ProcessingSessionService
from core.project_models import InterpretationFeatureV1
from core.project_service import ProjectService


class InterpretationWorkbenchPage(QWidget):
    interpretation_changed = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: ProjectService | None = None
        self.line_id: str | None = None
        self.source_result_id: str | None = None
        self.data: np.ndarray | None = None
        self.service: InterpretationService | None = None
        self.features: list[InterpretationFeatureV1] = []
        self._build_ui()
        self._sync_controls()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QFrame()
        top_layout = QHBoxLayout(top)
        self.title = QLabel("未加载目标定位测线")
        self.source_combo = QComboBox()
        self.source_combo.addItem("原始测线", None)
        self.refresh_button = QPushButton("刷新标注")
        self.delete_button = QPushButton("删除选中")
        top_layout.addWidget(self.title)
        top_layout.addStretch(1)
        top_layout.addWidget(QLabel("来源"))
        top_layout.addWidget(self.source_combo)
        top_layout.addWidget(self.refresh_button)
        top_layout.addWidget(self.delete_button)
        layout.addWidget(top)

        splitter = QSplitter()
        canvas_panel = QFrame()
        canvas_layout = QVBoxLayout(canvas_panel)
        self.figure = Figure(facecolor="#101820")
        self.canvas = FigureCanvas(self.figure)
        canvas_layout.addWidget(self.canvas)
        splitter.addWidget(canvas_panel)

        editor = QFrame()
        editor_layout = QVBoxLayout(editor)
        editor_layout.addWidget(QLabel("标注内容"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("目标点", "point")
        self.type_combo.addItem("界面线", "interface_line")
        self.type_combo.addItem("异常范围", "interval")
        self.label_edit = QLineEdit("疑似目标")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.8)
        self.coordinates_editor = QTextEdit()
        self.coordinates_editor.setPlainText("[10, 20]")
        self.add_button = QPushButton("新增标注")
        editor_layout.addWidget(self.type_combo)
        editor_layout.addWidget(QLabel("名称"))
        editor_layout.addWidget(self.label_edit)
        editor_layout.addWidget(QLabel("置信度"))
        editor_layout.addWidget(self.confidence_spin)
        editor_layout.addWidget(QLabel("坐标（道号, 采样点）"))
        editor_layout.addWidget(self.coordinates_editor)
        editor_layout.addWidget(self.add_button)
        splitter.addWidget(editor)
        splitter.setSizes([820, 320])
        layout.addWidget(splitter, 1)

        self.feature_table = QTableWidget(0, 5)
        self.feature_table.setHorizontalHeaderLabels(["类型", "名称", "置信度", "来源结果", "编号"])
        self.feature_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.feature_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.feature_table)

        self.type_combo.currentIndexChanged.connect(self._type_changed)
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.refresh_button.clicked.connect(self.refresh)
        self.delete_button.clicked.connect(self.delete_selected)
        self.add_button.clicked.connect(self.add_from_editor)

    def open_line(self, project: ProjectService, line_id: str) -> None:
        self.project = project
        self.line_id = line_id
        self.service = InterpretationService(project)
        session = ProcessingSessionService.open_line(
            project,
            line_id,
            enforce_processing_gate=False,
        )
        self.data = np.array(session.original_data, copy=True)
        self.title.setText(f"目标定位 · {session.line.name}")
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("原始测线", None)
        for result in project.list_processing_results(line_id):
            self.source_combo.addItem(f"处理结果 · {result.name}", result.result_id)
        self.source_combo.blockSignals(False)
        self.source_result_id = None
        self.refresh()
        self._sync_controls()

    def add_point(
        self,
        *,
        trace: float,
        sample: float,
        confidence: float,
        label: str,
    ) -> InterpretationFeatureV1:
        self._require_service()
        feature = self.service.add_point(
            self.line_id,
            trace=trace,
            sample=sample,
            confidence=confidence,
            result_id=self.source_result_id,
            label=label,
        )
        self.refresh()
        self.interpretation_changed.emit(self.line_id)
        return feature

    def add_interface_line(
        self,
        *,
        points: list[tuple[float, float]],
        confidence: float,
        label: str,
    ) -> InterpretationFeatureV1:
        self._require_service()
        feature = self.service.add_interface_line(
            self.line_id,
            points=points,
            confidence=confidence,
            result_id=self.source_result_id,
            label=label,
        )
        self.refresh()
        self.interpretation_changed.emit(self.line_id)
        return feature

    def add_interval(
        self,
        *,
        trace_start: float,
        trace_end: float,
        sample_start: float,
        sample_end: float,
        confidence: float,
        label: str,
    ) -> InterpretationFeatureV1:
        self._require_service()
        feature = self.service.add_interval(
            self.line_id,
            trace_start=trace_start,
            trace_end=trace_end,
            sample_start=sample_start,
            sample_end=sample_end,
            confidence=confidence,
            result_id=self.source_result_id,
            label=label,
        )
        self.refresh()
        self.interpretation_changed.emit(self.line_id)
        return feature

    def add_from_editor(self) -> None:
        try:
            coordinates = json.loads(self.coordinates_editor.toPlainText())
            feature_type = str(self.type_combo.currentData())
            confidence = self.confidence_spin.value()
            label = self.label_edit.text()
            if feature_type == "point":
                self.add_point(
                    trace=coordinates[0],
                    sample=coordinates[1],
                    confidence=confidence,
                    label=label,
                )
            elif feature_type == "interface_line":
                self.add_interface_line(
                    points=[tuple(item) for item in coordinates],
                    confidence=confidence,
                    label=label,
                )
            else:
                self.add_interval(
                    trace_start=coordinates[0],
                    trace_end=coordinates[1],
                    sample_start=coordinates[2],
                    sample_end=coordinates[3],
                    confidence=confidence,
                    label=label,
                )
        except Exception as exc:
            QMessageBox.critical(self, "标注内容无效", f"请按示例填写坐标：点 [道号, 采样点]；线 [[道号, 采样点], ...]；范围 [起始道, 结束道, 起始采样点, 结束采样点]。\n\n详细信息：{exc}")

    def delete_selected(self) -> None:
        if self.service is None or self.line_id is None:
            return
        row = self.feature_table.currentRow()
        if row < 0:
            return
        feature_id = self.feature_table.item(row, 4).text()
        if self.service.delete_feature(self.line_id, feature_id):
            self.refresh()
            self.interpretation_changed.emit(self.line_id)

    def refresh(self) -> None:
        if self.service is None or self.line_id is None:
            return
        self.features = self.service.list_features(self.line_id)
        self.feature_table.setRowCount(len(self.features))
        for row, feature in enumerate(self.features):
            values = (
                feature.feature_type,
                feature.properties.get("label", ""),
                f"{feature.confidence:.2f}",
                feature.result_id or "原始测线",
                feature.feature_id,
            )
            for column, value in enumerate(values):
                self.feature_table.setItem(row, column, QTableWidgetItem(str(value)))
        self._draw()

    def _source_changed(self) -> None:
        if self.project is None or self.line_id is None:
            return
        result_id = self.source_combo.currentData()
        self.source_result_id = str(result_id) if result_id else None
        if self.source_result_id:
            payload = self.project.load_processing_result(
                self.source_result_id, line_id=self.line_id
            )
            self.data = np.array(payload["data"], copy=True)
        else:
            self.data = ProcessingSessionService.open_line(
                self.project,
                self.line_id,
                enforce_processing_gate=False,
            ).original_data
        self._draw()

    def _draw(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#101820")
        axis.tick_params(colors="#AFC2CF")
        axis.set_xlabel("道号", color="#AFC2CF")
        axis.set_ylabel("采样点", color="#AFC2CF")
        if self.data is not None:
            axis.imshow(self.data, cmap="gray", aspect="auto", interpolation="nearest")
        for feature in self.features:
            geometry = feature.geometry
            if geometry.get("type") == "Point":
                x, y = geometry["coordinates"]
                axis.scatter([x], [y], c="#FFB000", s=35)
            elif geometry.get("type") == "LineString":
                points = np.asarray(geometry["coordinates"], dtype=float)
                axis.plot(points[:, 0], points[:, 1], color="#39D0D8", linewidth=2)
            elif geometry.get("type") == "Polygon":
                ring = np.asarray(geometry["coordinates"][0], dtype=float)
                axis.fill(ring[:, 0], ring[:, 1], color="#E45756", alpha=0.25)
                axis.plot(ring[:, 0], ring[:, 1], color="#E45756", linewidth=1.5)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _type_changed(self) -> None:
        feature_type = str(self.type_combo.currentData())
        templates: dict[str, Any] = {
            "point": [10, 20],
            "interface_line": [[0, 20], [10, 22], [20, 25]],
            "interval": [10, 30, 20, 60],
        }
        self.coordinates_editor.setPlainText(
            json.dumps(templates[feature_type], ensure_ascii=False)
        )

    def _sync_controls(self) -> None:
        enabled = self.service is not None and self.line_id is not None
        self.add_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)

    def _require_service(self) -> None:
        if self.service is None or self.line_id is None:
            raise RuntimeError("未加载目标定位测线")


__all__ = ["InterpretationWorkbenchPage"]

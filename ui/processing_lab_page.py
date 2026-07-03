#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native project-first processing laboratory."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.methods_registry import (
    PROCESSING_METHODS,
    get_method_category_label,
    get_method_display_name,
)
from core.preset_profiles import RECOMMENDED_RUN_PROFILES
from core.processing_session import ProcessingSessionService
from core.project_models import ProcessingResultV1
from core.project_service import ProjectService
from ui.workbench_tasks import WorkbenchTaskWorker


class ProcessingLabPage(QWidget):
    """Processing canvas, chain editor, parameters, and field-facing recommendations."""

    result_saved = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: ProcessingSessionService | None = None
        self.last_result: ProcessingResultV1 | None = None
        self._task_thread: QThread | None = None
        self._task_worker: WorkbenchTaskWorker | None = None
        self._saved_versions: dict[str, dict[str, Any]] = {}
        self._method_ids = ProcessingSessionService.public_method_ids()
        self._build_ui()
        self._populate_methods()
        self._sync_controls()

    def _selected_recommendation(self) -> dict[str, Any] | None:
        """Return the recommendation that is safe to apply for this method."""
        if self.session is None or not self.session.last_recommendation:
            return None
        recommendation = self.session.last_recommendation
        if str(recommendation.get("method_key")) != self.selected_method_id():
            return None
        return recommendation

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setObjectName("processingToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        self.line_title = QLabel("未加载测线")
        self.line_title.setObjectName("processingLineTitle")
        self.preview_button = QPushButton("预览")
        self.apply_button = QPushButton("应用方法")
        self.undo_button = QPushButton("撤销")
        self.redo_button = QPushButton("重做")
        self.reset_button = QPushButton("重置原始")
        self.autotune_button = QPushButton("自动推荐")
        self.apply_recommendation_button = QPushButton("应用推荐")
        self.compare_autotune_button = QPushButton("手动/推荐对比")
        self.export_comparison_button = QPushButton("导出对比报告")
        self.save_button = QPushButton("保存处理结果")
        toolbar_layout.addWidget(self.line_title)
        toolbar_layout.addStretch(1)
        for button in (
            self.preview_button,
            self.apply_button,
            self.undo_button,
            self.redo_button,
            self.reset_button,
            self.autotune_button,
            self.apply_recommendation_button,
            self.compare_autotune_button,
            self.export_comparison_button,
            self.save_button,
        ):
            toolbar_layout.addWidget(button)
        layout.addWidget(toolbar)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        chain_panel = QFrame()
        chain_layout = QVBoxLayout(chain_panel)
        chain_layout.addWidget(QLabel("处理步骤"))
        self.chain_table = QTableWidget(0, 3)
        self.chain_table.setHorizontalHeaderLabels(["状态", "方法", "参数"])
        self.chain_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.chain_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        chain_layout.addWidget(self.chain_table)
        chain_actions = QHBoxLayout()
        self.toggle_step_button = QPushButton("启用/停用")
        self.move_up_button = QPushButton("上移")
        self.move_down_button = QPushButton("下移")
        self.remove_step_button = QPushButton("删除")
        for button in (
            self.toggle_step_button,
            self.move_up_button,
            self.move_down_button,
            self.remove_step_button,
        ):
            chain_actions.addWidget(button)
        chain_layout.addLayout(chain_actions)
        splitter.addWidget(chain_panel)

        canvas_panel = QFrame()
        canvas_layout = QVBoxLayout(canvas_panel)
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("显示"))
        self.compare_combo = QComboBox()
        self.compare_combo.addItem("当前结果", "current")
        self.compare_combo.addItem("原始数据", "original")
        self.compare_combo.addItem("差值图", "difference")
        compare_row.addWidget(self.compare_combo)
        compare_row.addWidget(QLabel("色图"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(
            ["gray", "viridis", "plasma", "inferno", "magma", "jet", "seismic"]
        )
        self.cmap_combo.setCurrentText("gray")
        compare_row.addWidget(self.cmap_combo)
        self.symmetric_scale_check = QCheckBox("对称")
        self.symmetric_scale_check.setToolTip("显示层以 0 为中心锁定正负振幅范围")
        self.percentile_clip_check = QCheckBox("百分位")
        self.percentile_clip_check.setToolTip("显示层使用 0.5/99.5 百分位裁剪离群值")
        self.colorbar_check = QCheckBox("色标")
        self.colorbar_check.setToolTip("显示当前画布色标")
        self.lock_scale_check = QCheckBox("锁定")
        self.lock_scale_check.setChecked(True)
        self.lock_scale_check.setToolTip("按原始、当前和已保存版本共享色标，便于对比")
        for widget in (
            self.symmetric_scale_check,
            self.percentile_clip_check,
            self.colorbar_check,
            self.lock_scale_check,
        ):
            compare_row.addWidget(widget)
        compare_row.addStretch(1)
        canvas_layout.addLayout(compare_row)
        self.figure = Figure(facecolor="#101820")
        self.canvas = FigureCanvas(self.figure)
        canvas_layout.addWidget(self.canvas, 1)
        splitter.addWidget(canvas_panel)

        method_panel = QFrame()
        method_layout = QVBoxLayout(method_panel)
        method_layout.addWidget(QLabel("处理方法与参数"))
        self.method_combo = QComboBox()
        method_layout.addWidget(self.method_combo)
        self.method_summary = QLabel()
        self.method_summary.setWordWrap(True)
        method_layout.addWidget(self.method_summary)
        method_layout.addWidget(QLabel("参数设置（高级）"))
        self.params_editor = QTextEdit()
        self.params_editor.setMinimumWidth(260)
        method_layout.addWidget(self.params_editor, 1)
        method_layout.addWidget(QLabel("推荐处理流程"))
        self.profile_combo = QComboBox()
        for key, profile in RECOMMENDED_RUN_PROFILES.items():
            self.profile_combo.addItem(str(profile.get("label", key)), key)
        self.run_profile_button = QPushButton("运行推荐流程")
        method_layout.addWidget(self.profile_combo)
        method_layout.addWidget(self.run_profile_button)
        self.recommendation_text = QLabel("尚未生成参数推荐。")
        self.recommendation_text.setWordWrap(True)
        method_layout.addWidget(self.recommendation_text)
        splitter.addWidget(method_panel)
        splitter.setSizes([260, 760, 320])
        layout.addWidget(splitter, 1)

        self.method_combo.currentIndexChanged.connect(self._method_changed)
        self.preview_button.clicked.connect(self.preview_selected_method_async)
        self.apply_button.clicked.connect(self.apply_selected_method_async)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.reset_button.clicked.connect(self.reset)
        self.autotune_button.clicked.connect(self.recommend_selected_method_async)
        self.apply_recommendation_button.clicked.connect(self.apply_recommendation)
        self.compare_autotune_button.clicked.connect(self.run_manual_auto_comparison_async)
        self.export_comparison_button.clicked.connect(self.export_manual_auto_comparison)
        self.run_profile_button.clicked.connect(self.run_selected_profile_async)
        self.save_button.clicked.connect(lambda: self.save_version("测线处理结果"))
        self.compare_combo.currentIndexChanged.connect(self.refresh_plot)
        self.cmap_combo.currentIndexChanged.connect(self.refresh_plot)
        self.symmetric_scale_check.stateChanged.connect(self.refresh_plot)
        self.percentile_clip_check.stateChanged.connect(self.refresh_plot)
        self.colorbar_check.stateChanged.connect(self.refresh_plot)
        self.lock_scale_check.stateChanged.connect(self.refresh_plot)
        self.toggle_step_button.clicked.connect(self.toggle_step)
        self.move_up_button.clicked.connect(lambda: self.move_step(-1))
        self.move_down_button.clicked.connect(lambda: self.move_step(1))
        self.remove_step_button.clicked.connect(self.remove_step)

    def _populate_methods(self) -> None:
        for method_id in self._method_ids:
            self.method_combo.addItem(get_method_display_name(method_id), method_id)
        self._method_changed()

    def open_line(self, project: ProjectService, line_id: str) -> None:
        self.session = ProcessingSessionService.open_line(project, line_id)
        self._saved_versions = {
            result.result_id: project.load_processing_result(
                result.result_id, line_id=line_id
            )
            for result in project.list_processing_results(line_id)
        }
        self._refresh_compare_sources()
        self.line_title.setText(self.session.line.name)
        self.recommendation_text.setText("尚未生成参数推荐。")
        self._refresh_chain()
        self.refresh_plot()
        self._sync_controls()
        self.status_changed.emit(f"测线处理已加载：{self.session.line.name}")

    def select_method(self, method_id: str) -> None:
        index = self.method_combo.findData(method_id)
        if index < 0:
            raise KeyError(method_id)
        self.method_combo.setCurrentIndex(index)

    def selected_method_id(self) -> str:
        return str(self.method_combo.currentData())

    def selected_params(self) -> dict[str, Any]:
        payload = json.loads(self.params_editor.toPlainText() or "{}")
        if not isinstance(payload, dict):
            raise ValueError("参数 JSON 必须是对象")
        return payload

    def preview_selected_method(self) -> None:
        if self.session is None:
            return
        try:
            preview = self.session.preview_method(
                self.selected_method_id(), self.selected_params()
            )
            self._draw_data(preview.data, title=f"预览 · {get_method_display_name(preview.method_id)}")
            self.status_changed.emit("预览完成；正式处理状态未改变。")
        except Exception as exc:
            self._show_error("预览失败", exc)

    def preview_selected_method_async(self) -> None:
        if self.session is None:
            return
        try:
            method_id = self.selected_method_id()
            params = self.selected_params()
        except Exception as exc:
            self._show_error("参数错误", exc)
            return
        self._start_task(
            self.session.preview_method,
            method_id,
            params,
            task_name="预览",
            on_success=lambda preview: self._finish_preview(preview),
        )

    def _finish_preview(self, preview) -> None:
        self._draw_data(
            preview.data,
            title=f"预览 · {get_method_display_name(preview.method_id)}",
        )
        self.status_changed.emit("预览完成；正式处理状态未改变。")

    def apply_selected_method(self) -> None:
        if self.session is None:
            return
        try:
            step = self.session.apply_method(
                self.selected_method_id(), self.selected_params()
            )
            self._refresh_chain()
            self.refresh_plot()
            self._sync_controls()
            self.status_changed.emit(f"已应用：{step.display_name}")
        except Exception as exc:
            self._show_error("处理失败", exc)

    def apply_selected_method_async(self) -> None:
        if self.session is None:
            return
        try:
            method_id = self.selected_method_id()
            params = self.selected_params()
        except Exception as exc:
            self._show_error("参数错误", exc)
            return
        self._start_task(
            self.session.apply_method,
            method_id,
            params,
            task_name="应用方法",
            on_success=lambda _step: self._finish_session_change(),
        )

    def recommend_selected_method(self) -> None:
        if self.session is None:
            return
        method_id = self.selected_method_id()
        if not PROCESSING_METHODS.get(method_id, {}).get("auto_tune_enabled"):
            self.recommendation_text.setText("当前方法暂不支持自动推荐。")
            self._sync_controls()
            return
        try:
            result = self.session.recommend_method(method_id, search_mode="fast")
            params = result.get("recommended_params") or {}
            self.recommendation_text.setText(
                f"推荐：{get_method_display_name(method_id)}\n"
                f"参数：{json.dumps(params, ensure_ascii=False)}\n"
                f"风险：{result.get('risk_level', 'low')} · {result.get('risk_reason', '')}"
            )
            self._sync_controls()
        except Exception as exc:
            self._show_error("自动推荐失败", exc)

    def recommend_selected_method_async(self) -> None:
        if self.session is None:
            return
        method_id = self.selected_method_id()
        if not PROCESSING_METHODS.get(method_id, {}).get("auto_tune_enabled"):
            self.recommendation_text.setText("当前方法暂不支持自动推荐。")
            self._sync_controls()
            return
        self._start_task(
            self.session.recommend_method,
            method_id,
            search_mode="fast",
            task_name="自动推荐",
            on_success=self._finish_recommendation,
        )

    def _finish_recommendation(self, result: dict[str, Any]) -> None:
        method_id = str(result.get("method_key"))
        params = result.get("recommended_params") or {}
        self.recommendation_text.setText(
            f"推荐：{get_method_display_name(method_id)}\n"
            f"参数：{json.dumps(params, ensure_ascii=False)}\n"
            f"风险：{result.get('risk_level', 'low')} · {result.get('risk_reason', '')}"
        )
        self._sync_controls()

    def apply_recommendation(self) -> None:
        if self.session is None:
            return
        if self._selected_recommendation() is None:
            self.recommendation_text.setText("当前推荐与所选方法不一致；请重新生成参数推荐。")
            self._sync_controls()
            return
        try:
            self.session.apply_recommendation()
            self._refresh_chain()
            self.refresh_plot()
            self._sync_controls()
        except Exception as exc:
            self._show_error("应用推荐失败", exc)

    def run_manual_auto_comparison_async(self) -> None:
        if self.session is None:
            return
        try:
            method_id = self.selected_method_id()
            params = self.selected_params()
        except Exception as exc:
            self._show_error("参数错误", exc)
            return
        if not PROCESSING_METHODS.get(method_id, {}).get("auto_tune_enabled"):
            self.recommendation_text.setText("当前方法暂不支持手动/推荐对比。")
            return
        self._start_task(
            self.session.run_manual_auto_comparison,
            pipeline=[method_id],
            manual_params_by_method={method_id: params},
            search_mode="fast",
            task_name="人工/自动对比",
            on_success=self._finish_manual_auto_comparison,
        )

    def _finish_manual_auto_comparison(self, comparison) -> None:
        delta = comparison.metric_delta.get("comparison_score")
        delta_text = "无可用分数"
        if isinstance(delta, (int, float)) and np.isfinite(delta):
            delta_text = f"comparison_score Δ={float(delta):.4f}"
        self.recommendation_text.setText(
            "人工/自动对比完成\n"
            f"结论：{comparison.verdict}\n"
            f"{delta_text}\n"
            "可导出对比报告。"
        )
        self._draw_data(comparison.automatic.result, title="推荐参数对比")
        self._sync_controls()

    def export_manual_auto_comparison(self) -> None:
        if self.session is None:
            return
        try:
            bundle = self.session.export_last_manual_auto_comparison(
                bundle_name=f"manual_auto_{self.session.line_id}",
            )
            self.recommendation_text.setText(
                "对比报告已导出\n"
                f"{bundle.get('output_dir', '')}"
            )
            self.status_changed.emit("手动/推荐对比报告已导出。")
        except Exception as exc:
            self._show_error("导出对比报告失败", exc)

    def run_selected_profile(self) -> None:
        if self.session is None:
            return
        key = str(self.profile_combo.currentData())
        profile = RECOMMENDED_RUN_PROFILES[key]
        params = dict(profile.get("method_params") or {})
        steps = [
            {
                "method_id": method_id,
                "params": dict(params.get(method_id) or {}),
            }
            for method_id in profile.get("order", [])
        ]
        try:
            self.session.apply_pipeline(steps)
            self._refresh_chain()
            self.refresh_plot()
            self._sync_controls()
            self.status_changed.emit(f"已运行推荐流程：{profile.get('label', key)}")
        except Exception as exc:
            self._show_error("推荐流程失败", exc)

    def run_selected_profile_async(self) -> None:
        if self.session is None:
            return
        key = str(self.profile_combo.currentData())
        profile = RECOMMENDED_RUN_PROFILES[key]
        params = dict(profile.get("method_params") or {})
        steps = [
            {"method_id": method_id, "params": dict(params.get(method_id) or {})}
            for method_id in profile.get("order", [])
        ]
        self._start_task(
            self.session.apply_pipeline,
            steps,
            task_name=f"推荐流程 · {profile.get('label', key)}",
            on_success=lambda _steps: self._finish_session_change(),
        )

    def undo(self) -> None:
        if self.session and self.session.undo():
            self._refresh_chain()
            self.refresh_plot()
            self._sync_controls()

    def redo(self) -> None:
        if self.session and self.session.redo():
            self._refresh_chain()
            self.refresh_plot()
            self._sync_controls()

    def reset(self) -> None:
        if self.session is not None:
            self.session.reset()
            self._refresh_chain()
            self.refresh_plot()
            self._sync_controls()

    def selected_step_index(self) -> int:
        return self.chain_table.currentRow()

    def toggle_step(self) -> None:
        if self.session is None:
            return
        index = self.selected_step_index()
        if index < 0:
            return
        self.session.set_step_enabled(index, not self.session.steps[index].enabled)
        self._refresh_chain()
        self.refresh_plot()
        self._sync_controls()

    def move_step(self, offset: int) -> None:
        if self.session is None:
            return
        source = self.selected_step_index()
        target = source + int(offset)
        if source < 0 or target < 0 or target >= len(self.session.steps):
            return
        self.session.move_step(source, target)
        self._refresh_chain()
        self.chain_table.selectRow(target)
        self.refresh_plot()
        self._sync_controls()

    def remove_step(self) -> None:
        if self.session is None:
            return
        index = self.selected_step_index()
        if index < 0:
            return
        self.session.remove_step(index)
        self._refresh_chain()
        self.refresh_plot()
        self._sync_controls()

    def save_version(self, name: str) -> ProcessingResultV1:
        if self.session is None:
            raise RuntimeError("未加载处理会话")
        result = self.session.save_version(name)
        self.last_result = result
        self._saved_versions[result.result_id] = self.session.project.load_processing_result(
            result.result_id,
            line_id=self.session.line_id,
        )
        self._refresh_compare_sources(select=f"version:{result.result_id}")
        self.result_saved.emit(result)
        self.status_changed.emit(f"已保存处理结果：{result.result_id}")
        return result

    def refresh_plot(self) -> None:
        if self.session is None:
            self._draw_empty()
            return
        mode = str(self.compare_combo.currentData())
        if mode == "original":
            self._draw_data(self.session.original_data, title="原始数据")
        elif mode == "difference":
            current = np.asarray(self.session.current_data)
            original = np.asarray(self.session.original_data)
            if current.shape == original.shape:
                self._draw_data(current - original, title="当前 - 原始", cmap="coolwarm")
            else:
                self._draw_data(current, title="当前结果（尺寸变化，无法差值）")
        elif mode.startswith("version:"):
            result_id = mode.split(":", 1)[1]
            payload = self._saved_versions.get(result_id)
            if payload is None:
                self._draw_empty()
            else:
                record = payload["record"]
                self._draw_data(
                    payload["data"],
                    title=f"处理结果 · {record.name}",
                )
        else:
            self._draw_data(self.session.current_data, title="当前处理结果")

    def _draw_data(self, data: np.ndarray, *, title: str, cmap: str | None = None) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#101820")
        axis.tick_params(colors="#AFC2CF")
        axis.set_title(title, color="#EAF4F4")
        axis.set_xlabel("道号", color="#AFC2CF")
        axis.set_ylabel("采样点", color="#AFC2CF")
        vmin, vmax = self._display_limits(np.asarray(data))
        image = axis.imshow(
            np.asarray(data),
            cmap=cmap or str(self.cmap_combo.currentText() or "gray"),
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        if self.colorbar_check.isChecked():
            colorbar = self.figure.colorbar(image, ax=axis, shrink=0.82)
            colorbar.ax.tick_params(colors="#AFC2CF")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _display_limits(self, data: np.ndarray) -> tuple[float | None, float | None]:
        source = np.asarray(data, dtype=np.float32)
        arrays = [source]
        if self.lock_scale_check.isChecked():
            arrays = self._display_reference_arrays(source.shape) or arrays
        values = _finite_values(arrays)
        if values.size == 0:
            return None, None
        symmetric = bool(self.symmetric_scale_check.isChecked())
        if self.percentile_clip_check.isChecked():
            if symmetric:
                limit = float(np.percentile(np.abs(values), 99.5))
                return _safe_symmetric_limits(limit)
            lo, hi = np.percentile(values, [0.5, 99.5])
            return _safe_limits(float(lo), float(hi))
        if symmetric:
            limit = float(np.max(np.abs(values)))
            return _safe_symmetric_limits(limit)
        return _safe_limits(float(np.min(values)), float(np.max(values)))

    def _display_reference_arrays(self, shape: tuple[int, ...]) -> list[np.ndarray]:
        if self.session is None:
            return []
        arrays = [
            np.asarray(self.session.original_data),
            np.asarray(self.session.current_data),
        ]
        for payload in self._saved_versions.values():
            arrays.append(np.asarray(payload.get("data")))
        return [
            np.asarray(item, dtype=np.float32)
            for item in arrays
            if tuple(np.asarray(item).shape) == tuple(shape)
        ]

    def _draw_empty(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#101820")
        axis.text(0.5, 0.5, "选择正式、质控就绪的测线进入测线处理", ha="center", color="#EAF4F4")
        axis.set_axis_off()
        self.canvas.draw_idle()

    def _refresh_chain(self) -> None:
        steps = self.session.steps if self.session else []
        self.chain_table.setRowCount(len(steps))
        for row, step in enumerate(steps):
            self.chain_table.setItem(row, 0, QTableWidgetItem("启用" if step.enabled else "停用"))
            self.chain_table.setItem(row, 1, QTableWidgetItem(step.display_name))
            self.chain_table.setItem(
                row,
                2,
                QTableWidgetItem(json.dumps(step.params, ensure_ascii=False)),
            )

    def _method_changed(self) -> None:
        method_id = self.selected_method_id()
        if not method_id:
            return
        self.params_editor.setPlainText(
            json.dumps(
                ProcessingSessionService.default_params(method_id),
                ensure_ascii=False,
                indent=2,
            )
        )
        self.method_summary.setText(
            f"{get_method_category_label(method_id)} · "
            f"{PROCESSING_METHODS.get(method_id, {}).get('maturity', 'stable')}"
        )
        if self.session is not None and self.session.last_recommendation:
            recommended_method = str(self.session.last_recommendation.get("method_key"))
            if recommended_method and recommended_method != method_id:
                self.recommendation_text.setText("已切换方法；请重新生成参数推荐。")
        self._sync_controls()

    def _sync_controls(self) -> None:
        loaded = self.session is not None
        busy = self._task_thread is not None
        for button in (
            self.preview_button,
            self.apply_button,
            self.reset_button,
            self.autotune_button,
            self.run_profile_button,
            self.compare_autotune_button,
            self.save_button,
        ):
            button.setEnabled(loaded and not busy)
        self.undo_button.setEnabled(bool(loaded and not busy and self.session and self.session.can_undo))
        self.redo_button.setEnabled(bool(loaded and not busy and self.session and self.session.can_redo))
        self.apply_recommendation_button.setEnabled(
            bool(loaded and not busy and self._selected_recommendation() is not None)
        )
        self.export_comparison_button.setEnabled(
            bool(
                loaded
                and not busy
                and self.session
                and self.session.last_manual_auto_comparison is not None
            )
        )
        has_step = bool(loaded and not busy and self.session and self.session.steps)
        for button in (
            self.toggle_step_button,
            self.move_up_button,
            self.move_down_button,
            self.remove_step_button,
        ):
            button.setEnabled(has_step)

    def _refresh_compare_sources(self, *, select: str | None = None) -> None:
        current = select or str(self.compare_combo.currentData() or "current")
        self.compare_combo.blockSignals(True)
        self.compare_combo.clear()
        self.compare_combo.addItem("当前结果", "current")
        self.compare_combo.addItem("原始数据", "original")
        self.compare_combo.addItem("差值图", "difference")
        for result_id, payload in self._saved_versions.items():
            record = payload["record"]
            self.compare_combo.addItem(f"处理结果 · {record.name}", f"version:{result_id}")
        index = self.compare_combo.findData(current)
        self.compare_combo.setCurrentIndex(max(0, index))
        self.compare_combo.blockSignals(False)

    def _finish_session_change(self) -> None:
        self._refresh_chain()
        self.refresh_plot()
        self._sync_controls()

    def _start_task(
        self,
        operation,
        *args,
        task_name: str,
        on_success,
        **kwargs,
    ) -> None:
        if self._task_thread is not None:
            return
        thread = QThread(self)
        worker = WorkbenchTaskWorker(operation, *args, **kwargs)
        self._task_thread = thread
        self._task_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.finished.connect(lambda _value: self._task_done(task_name))
        worker.failed.connect(lambda message: self._task_failed(task_name, message))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.status_changed.emit(f"{task_name}运行中…")
        self._sync_controls()
        thread.start()

    def _task_done(self, task_name: str) -> None:
        self.status_changed.emit(f"{task_name}完成。")
        self._task_thread = None
        self._task_worker = None
        self._sync_controls()

    def _task_failed(self, task_name: str, message: str) -> None:
        self._task_thread = None
        self._task_worker = None
        self._sync_controls()
        self._show_error(f"{task_name}失败", RuntimeError(message))

    def shutdown_background_task(self) -> None:
        """Stop the page-owned worker thread when the workbench is closing."""
        thread = self._task_thread
        if thread is None:
            return
        thread.requestInterruption()
        thread.quit()
        if not thread.wait(3000):
            thread.terminate()
            thread.wait(1000)
        self._task_thread = None
        self._task_worker = None
        self._sync_controls()

    def release_plot_resources(self) -> None:
        """Release Matplotlib/Qt canvas resources for long GUI test batches."""
        try:
            self.figure.clear()
        except Exception:
            pass
        try:
            self.canvas.close()
            self.canvas.deleteLater()
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.shutdown_background_task()
        self.release_plot_resources()
        super().closeEvent(event)


__all__ = ["ProcessingLabPage"]


def _finite_values(arrays: list[np.ndarray]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for array in arrays:
        values = np.ravel(np.asarray(array, dtype=np.float32))
        if values.size == 0:
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(finite.astype(np.float64, copy=False))
    if not chunks:
        return np.asarray([], dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _safe_limits(vmin: float, vmax: float) -> tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if abs(vmax - vmin) < 1.0e-12:
        pad = max(abs(vmax) * 0.05, 1.0)
        return float(vmin - pad), float(vmax + pad)
    return float(vmin), float(vmax)


def _safe_symmetric_limits(limit: float) -> tuple[float, float]:
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    return -float(limit), float(limit)

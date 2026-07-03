#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing page and line-processing callbacks for the MyGPR field workbench.

This mixin keeps algorithm preview/apply/save UI logic outside the main window.
It assumes the host window provides project/status helpers such as
``_refresh_project_widgets``, ``_selected_line_record`` and drawing helpers.
"""

from __future__ import annotations

from pathlib import Path
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.project_events import ProjectEventType
from core.field_processing_bridge import (
    PARAM_LABELS,
    display_name as field_method_display_name,
    get_method_params_schema,
    recommended_params as recommend_field_method_params,
)
from core.manual_processing_chain import ManualProcessingSession
from core.app_errors import error_info_from_exception
from ui.field_panels.processing_panel import (
    PROCESSING_CATEGORY_LABEL,
    PROCESSING_METHOD_LABEL,
    PROCESSING_PARAMS_TITLE,
)
from ui.field_panels.layout_metrics import layout_metrics_for
from ui.field_panels.plots import draw_bscan
from ui.field_panels.widgets import Card, MetricCard, PlotCard, open_plot_viewer

logger = logging.getLogger(__name__)


class ProcessingPageMixin:
    """Build and operate the line-processing workspace."""

    def _page_processing(self) -> QWidget:
        widget = QWidget()
        v = QVBoxLayout(widget)
        v.setContentsMargins(0, 0, 0, 0)
        lm = layout_metrics_for(self)
        v.setSpacing(lm.spacing)

        metrics = QHBoxLayout()
        metrics.setSpacing(lm.spacing)
        st = self.project_status
        metric_defs = [
            ("⌘", "测线", str(st.line_count), "条", ""),
            ("✓", "已处理", str(st.processed_line_count), "条", ""),
            ("◎", "当前处理链", str(self._processing_step_count()), "步", self._processing_chain_metric_note()),
            ("▤", "报告状态", st.report_status, "", ""),
        ]
        self.processing_metric_cards = []
        for icon, title, value, suffix, note in metric_defs:
            card = MetricCard(icon, title, value, suffix, note)
            metrics.addWidget(card)
            self.processing_metric_cards.append(card)
        v.addLayout(metrics)

        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        self.processing_status_label = QLabel("当前显示：-- / Step 00 原始 B-scan")
        self.processing_status_label.setObjectName("activityTitle")
        status_row.addWidget(self.processing_status_label, 1)
        v.addLayout(status_row)

        work = QHBoxLayout()
        work.setSpacing(lm.spacing)
        current_card = PlotCard(
            "当前 B-scan 结果",
            height=lm.processing_bscan_h,
            expand_title="当前 B-scan 放大查看",
            expand_callback=self._draw_processing_processed_bscan,
            expand_parent=self,
        )
        current_card.setProperty("layoutKey", "processingProcessedBscanCard")
        current_card.canvas.setObjectName("processingProcessedBscanCanvas")
        current_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.processing_bscan_canvas = current_card.canvas
        self.processing_diff_canvas = None
        self._draw_processing_processed_bscan(current_card.canvas)
        # --- Compare button in the B-scan title bar ---
        compare_btn = QPushButton("⊞")
        compare_btn.setObjectName("smallButton")
        compare_btn.setFixedWidth(28)
        compare_btn.setToolTip("原始 / 当前 / 差异图对比查看")
        compare_btn.clicked.connect(self._open_processing_compare_viewer)
        self.processing_compare_button = compare_btn
        current_card.add_title_button(compare_btn)
        work.addWidget(current_card, 84, Qt.AlignmentFlag.AlignTop)

        params_card = self._processing_params_card()
        params_card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        params_card.setMaximumHeight(lm.processing_bscan_h + 40)
        work.addWidget(params_card, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(work, 3)

        v.addWidget(self._processing_messages_card(), 1)
        return widget

    def _processing_step_count(self) -> int:
        session = getattr(self, "processing_session", None)
        return int(session.step_count) if session is not None else 0

    def _processing_chain_metric_note(self) -> str:
        if self.processing_last_failed:
            return "执行失败"
        count = self._processing_step_count()
        return "当前链" if count else "原始"

    def _current_processing_dataset(self):
        session = getattr(self, "processing_session", None)
        if session is not None and session.line_id == getattr(self, "selected_line", ""):
            return session.current_dataset
        return self.processed_gpr_dataset or self.active_gpr_dataset

    def _ensure_processing_session(self) -> ManualProcessingSession | None:
        raw = self.active_gpr_dataset
        if raw is None:
            return None
        session = getattr(self, "processing_session", None)
        if session is None or session.line_id != raw.line_id:
            session = ManualProcessingSession(raw)
            self.processing_session = session
        return session

    def _draw_processing_raw_bscan(self, canvas) -> None:
        raw = self.active_gpr_dataset
        if raw is None:
            self._draw_empty_plot(canvas, "暂无原始 B-scan\n请先导入当前测线数据")
            return
        draw_bscan(
            canvas,
            title="",
            diff=False,
            gain=1.0,
            data_matrix=raw.matrix,
            distance_axis_m=raw.distance_axis_m,
            depth_axis_m=raw.depth_axis_m,
        )

    def _draw_processing_processed_bscan(self, canvas) -> None:
        target = self._current_processing_dataset()
        if target is None:
            self._draw_empty_plot(canvas, "暂无处理结果\n请先导入当前测线数据")
            return
        draw_bscan(
            canvas,
            title="",
            diff=False,
            gain=1.0,
            data_matrix=target.matrix,
            distance_axis_m=target.distance_axis_m,
            depth_axis_m=target.depth_axis_m,
        )

    def _refresh_processing_preview(self) -> None:
        raw = self.active_gpr_dataset
        target = self._current_processing_dataset()
        session = getattr(self, "processing_session", None)
        if self.processing_bscan_canvas is not None:
            self._draw_processing_processed_bscan(self.processing_bscan_canvas)
        if self.processing_diff_canvas is not None:
            self._draw_processing_processed_bscan(self.processing_diff_canvas)
        if self.processing_status_label is not None:
            if self.processing_last_failed:
                label = f"{self.selected_line} / 执行失败"
            elif session is not None:
                label = f"{self.selected_line} / {session.current_step_label()}"
            else:
                label = f"{self.selected_line} / Step 00 原始 B-scan"
            shape = f"{target.sample_count}×{target.trace_count}" if target is not None else "preview"
            self.processing_status_label.setText(f"当前显示：{label}｜矩阵 {shape}")
        if self.processing_info_label is not None:
            if self.processing_last_failed and self.last_processing_error:
                self.processing_info_label.setText(
                    f"算法：{field_method_display_name(self.selected_processing_method_id)}    状态：执行失败    原因：{self.last_processing_error}"
                )
            elif session is not None and session.steps:
                step = session.steps[-1]
                shape_note = "    尺寸变化：需复核" if step.input_shape != step.output_shape else ""
                self.processing_info_label.setText(
                    f"当前链：第 {step.index} 步    算法：{step.method_name}    输入：{step.input_shape}    输出：{step.output_shape}    耗时：{step.elapsed_s:.4f} s{shape_note}"
                )
            else:
                self.processing_info_label.setText("算法：未执行    输入：--    输出：--    耗时：--")
        if self.processing_save_button is not None:
            self.processing_save_button.setEnabled(False)
        if self.processing_undo_step_button is not None:
            self.processing_undo_step_button.setEnabled(bool(session is not None and session.step_count))
        if self.processing_reset_button is not None:
            self.processing_reset_button.setEnabled(bool(session is not None and session.step_count))
        if self.processing_compare_button is not None:
            self.processing_compare_button.setEnabled(bool(raw is not None and target is not None and raw is not target))
        if self.processing_execute_button is not None:
            self.processing_execute_button.setEnabled(raw is not None)
        if getattr(self, "processing_batch_button", None) is not None:
            self.processing_batch_button.setEnabled(raw is not None)
        if getattr(self, "processing_chain_status_label", None) is not None:
            if self.processing_last_failed:
                self.processing_chain_status_label.setText("执行失败")
            elif session is not None and session.step_count:
                self.processing_chain_status_label.setText(f"当前 Step {session.current_step_index:02d}")
            else:
                self.processing_chain_status_label.setText("尚未执行")
        self._refresh_processing_metrics()
        self._refresh_processing_history_table()
        if hasattr(self, "_update_project_tree"):
            self._update_project_tree()

    def _refresh_processing_metrics(self) -> None:
        cards = getattr(self, "processing_metric_cards", [])
        if len(cards) < 4:
            return
        cards[0].value_label.setText(str(self.project_status.line_count))
        cards[1].value_label.setText(str(self.project_status.processed_line_count))
        cards[2].value_label.setText(str(self._processing_step_count()))
        if cards[2].note_label is not None:
            cards[2].note_label.setText(self._processing_chain_metric_note())
        cards[3].value_label.setText(str(self.project_status.report_status))

    def _refresh_processing_history_table(self) -> None:
        if self.processing_history_table is None:
            return
        session = getattr(self, "processing_session", None)
        rows = [("00", "原始 B-scan", "当前" if session is None or getattr(session, "current_step_index", 0) == 0 else "可回看", "--", "项目树")]
        if session is not None:
            rows.extend(step.to_history_row() for step in session.steps)
            highlight = int(getattr(session, "current_step_index", 0))
        else:
            highlight = 0
        self._fill_table(self.processing_history_table, rows, highlight_row=highlight)
        for row in range(self.processing_history_table.rowCount()):
            self.processing_history_table.setRowHeight(row, 20)
        if getattr(self, "processing_history_hint_label", None) is not None:
            if rows:
                self.processing_history_hint_label.setText((session.summary_text() if session is not None else "") + "。步骤已同步到左侧项目树，撤回会同步删除最后一步。")
            else:
                self.processing_history_hint_label.setText("当前显示 Step 00 原始 B-scan。执行后会在项目树生成 Step 01。")

    def _collect_processing_params(self) -> dict:
        params: dict = {}
        for spec in get_method_params_schema(self.selected_processing_method_id):
            name = str(spec.get("name", ""))
            if not name:
                continue
            widget = self.processing_param_widgets.get(name)
            if widget is None:
                if "default" in spec:
                    params[name] = spec.get("default")
                continue
            if isinstance(widget, QSpinBox):
                params[name] = int(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                params[name] = float(widget.value())
            elif isinstance(widget, QCheckBox):
                params[name] = bool(widget.isChecked())
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentData() if widget.currentData() is not None else widget.currentText()
            elif isinstance(widget, QLineEdit):
                params[name] = widget.text()
        return params

    def _run_selected_processing(self, *, apply_result: bool) -> bool:
        if self.active_gpr_dataset is None:
            if self.processing_log_label is not None:
                self.processing_log_label.setText("⚠  当前测线没有可处理的 GPR 矩阵，请先导入或选择测线。")
            return False
        session = self._ensure_processing_session()
        if session is None:
            return False
        method_id = self.selected_processing_method_id
        params = self._collect_processing_params()
        try:
            result_dataset, manifest = session.append_step(method_id, params, trajectory=self.trajectory_model)
        except Exception as exc:
            error_info = error_info_from_exception(exc)
            self.processing_applied = False
            self.processing_last_failed = True
            self.last_processing_error = error_info.user_message
            if self.processing_log_label is not None:
                self.processing_log_label.setText(
                    f"⚠  {field_method_display_name(method_id)} 执行失败：{error_info.error_type}: {error_info.user_message}；已保留上一步结果。"
                )
            self._refresh_processing_preview()
            return False
        self.processed_gpr_dataset = result_dataset
        self.last_processing_manifest = manifest
        self.processing_last_failed = False
        self.last_processing_error = ""
        self.processing_applied = bool(apply_result)
        self._refresh_processing_preview()
        if self.processing_log_label is not None:
            shape_note = "；输出尺寸变化，保存前建议复核" if (manifest.get("sample_count_changed") or manifest.get("trace_count_changed")) else ""
            self.processing_log_label.setText(
                f"✓  已执行第 {session.step_count} 步：{field_method_display_name(method_id)}；输入 {manifest['input_shape']} → 输出 {manifest['output_shape']}；耗时 {manifest['elapsed_s']} s{shape_note}。"
            )
        return True

    def _select_processing_tree_step(self, step_index: int) -> None:
        session = self._ensure_processing_session()
        if session is None or not session.select_step(int(step_index)):
            return
        self.processed_gpr_dataset = session.current_dataset if session.current_step_index else None
        self.processing_last_failed = False
        self.last_processing_error = ""
        if session.current_step_index > 0:
            self.last_processing_manifest = session.steps[session.current_step_index - 1].manifest
        else:
            self.last_processing_manifest = None
        self._refresh_processing_preview()
        self._refresh_project_widgets()
        if self.processing_log_label is not None:
            self.processing_log_label.setText(f"↳  已切换到 {session.current_step_label()}。")

    def _preview_processing(self) -> None:
        self._run_selected_processing(apply_result=True)

    def _apply_processing(self) -> None:
        if not self._run_selected_processing(apply_result=True):
            return
        for line in self.line_records:
            if line["id"] == self.selected_line:
                line["status"] = f"◌ 连续处理 {self._processing_step_count()} 步"
                line["updated"] = "刚刚"
                break
        self._refresh_project_widgets()

    def _undo_processing(self) -> None:
        session = getattr(self, "processing_session", None)
        if session is None or not session.undo_last_step():
            if self.processing_log_label is not None:
                self.processing_log_label.setText("ℹ  当前没有可撤回的处理步骤。")
            return
        self.processing_applied = False
        self.processing_last_failed = False
        self.last_processing_error = ""
        self.last_processing_manifest = session.last_manifest
        self.processed_gpr_dataset = session.current_dataset if session.step_count else None
        self._refresh_processing_preview()
        self._refresh_project_widgets()
        if self.processing_log_label is not None:
            if session.step_count:
                self.processing_log_label.setText(f"↶  已撤回最后一步，项目树已同步到 Step {session.current_step_index:02d}。")
            else:
                self.processing_log_label.setText("↶  已撤回全部处理步骤，项目树仅保留原始 B-scan。")

    def _reset_processing_chain(self) -> None:
        session = getattr(self, "processing_session", None)
        if session is None or not session.reset_to_original():
            if self.processing_log_label is not None:
                self.processing_log_label.setText("ℹ  当前没有可重置的连续处理步骤。")
            return
        self.processing_applied = False
        self.processing_last_failed = False
        self.last_processing_error = ""
        self.last_processing_manifest = None
        self.processed_gpr_dataset = None
        self._refresh_processing_preview()
        if self.processing_log_label is not None:
            self.processing_log_label.setText("↺  已重置到原始 B-scan，连续处理链已清空。")

    def _batch_apply_processing(self) -> None:
        """Apply current processing step to all imported lines."""
        method_id = self.selected_processing_method_id
        params = self._collect_processing_params()
        if self.processing_log_label is not None:
            self.processing_log_label.setText(
                f"批量处理：正在对 {len(self.line_records)} 条测线执行 "
                f"{field_method_display_name(method_id)}..."
            )
        ok_count = 0
        fail_count = 0
        for line in self.line_records:
            line_id = str(line.get("id", ""))
            if not line_id or self.project_store is None:
                fail_count += 1
                continue
            try:
                raw_dataset = self.project_store.load_line_data(line_id)
                if raw_dataset is None:
                    fail_count += 1
                    continue
                session = ManualProcessingSession(raw_dataset)
                session.append_step(
                    method_id, params, trajectory=self.trajectory_model
                )
                self.project_store.save_processed_line(
                    line_id,
                    session.current_dataset.matrix,
                    session.build_save_payload(method_id, params),
                )
                ok_count += 1
            except Exception:
                fail_count += 1
        if self.processing_log_label is not None:
            self.processing_log_label.setText(
                f"批量处理完成：{ok_count} 条成功，{fail_count} 条失败。"
            )

    def _open_processing_compare_viewer(self) -> None:
        raw = self.active_gpr_dataset
        target = self._current_processing_dataset()
        session = getattr(self, "processing_session", None)
        step_label = ""
        if session is not None and session.step_count:
            step_label = f"  (Step {session.current_step_index:02d})"

        if raw is None or target is None or raw is target:
            if self.processing_log_label is not None:
                if raw is None:
                    self.processing_log_label.setText("ℹ  请先导入测线数据再进行对比。")
                elif target is raw:
                    self.processing_log_label.setText("ℹ  当前尚未执行处理步骤，无法对比。请先点击「执行当前步骤」。")
                else:
                    self.processing_log_label.setText("ℹ  当前没有可对比的处理结果。")
            return

        def _draw(canvas) -> None:
            fig = canvas.figure
            fig.clear()
            fig.suptitle(
                f"{self.selected_line}{step_label} 处理前后对比",
                fontsize=13, fontweight="bold", y=0.98,
            )
            axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
            raw_mat = raw.normalized_matrix
            target_mat = target.normalized_matrix
            rows = min(raw_mat.shape[0], target_mat.shape[0])
            cols = min(raw_mat.shape[1], target_mat.shape[1])
            diff_mat = target_mat[:rows, :cols] - raw_mat[:rows, :cols]
            extent = [
                float(target.distance_axis_m[0]),
                float(target.distance_axis_m[-1]),
                float(target.depth_axis_m[-1]),
                float(target.depth_axis_m[0]),
            ]
            panels = (
                (axes[0], raw_mat, "原始 B-scan", "gray", None),
                (axes[1], target_mat, "当前步骤结果", "gray", None),
                (axes[2], diff_mat, "差异图（当前 − 原始）", "RdBu_r", "差异幅值"),
            )
            for ax, mat, title, cmap, cbar_label in panels:
                image = ax.imshow(mat, cmap=cmap, aspect="auto", origin="upper", extent=extent)
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.set_xlabel("距离 (m)", fontsize=8)
                ax.set_ylabel("深度 (m)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.15, linewidth=0.4)
                for spine in ax.spines.values():
                    spine.set_color("#D5DEE7")
                    spine.set_linewidth(0.7)
                if cbar_label:
                    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
                    cb.set_label(cbar_label, fontsize=8)
                    cb.ax.tick_params(labelsize=7)
            fig.tight_layout(rect=[0, 0, 1, 0.94], pad=1.0)
            canvas.draw_idle()

        open_plot_viewer(self, title=f"{self.selected_line} 原始/当前/差异对比", draw_callback=_draw)

    def _save_processing_result(self) -> None:
        if self.processing_last_failed:
            if self.processing_log_label is not None:
                self.processing_log_label.setText("⚠  当前处理执行失败，已禁止保存。请调整参数或更换算法后重新执行。")
            return
        session = getattr(self, "processing_session", None)
        if session is None or session.step_count == 0:
            if self.processing_log_label is not None:
                self.processing_log_label.setText("⚠  当前还没有连续处理结果。请先点击“执行当前步骤”。")
            return
        data_path: Path | None = None
        params_path: Path | None = None
        if self.project_store is not None:
            if self.processed_gpr_dataset is None or self.active_gpr_dataset is None:
                if self.processing_log_label is not None:
                    self.processing_log_label.setText("⚠  无有效处理结果，禁止保存。请先执行处理步骤。")
                return
            if self.active_gpr_dataset.line_id != self.selected_line or self.processed_gpr_dataset.line_id != self.selected_line:
                if self.processing_log_label is not None:
                    self.processing_log_label.setText("⚠  当前处理结果与所选测线不一致，已禁止保存。")
                return
            payload = session.build_save_payload(self.selected_processing_method_id, self._collect_processing_params())
            try:
                data_path, params_path = self.project_store.save_processed_line(self.selected_line, self.processed_gpr_dataset.matrix, payload)
                self._sync_project_lines_to_ui()
                self._refresh_target_source_options(preferred_source_id=data_path.stem)
            except Exception as exc:
                error_info = error_info_from_exception(exc)
                logger.warning("Save processing result failed: %s [%s]", error_info.user_message, error_info.error_code)
                if self.processing_log_label is not None:
                    self.processing_log_label.setText(f"⚠  处理结果写入项目失败：{error_info.user_message}")
                return
        for line in self.line_records:
            if line["id"] == self.selected_line:
                line["status"] = "● 已完成"
                line["quality"] = "★★★★★" if line["quality"] != "--" else "★★★★☆"
                line["updated"] = "刚刚"
                break
        self.processing_applied = True
        if getattr(self, "linkage_controller", None) is not None:
            self.linkage_controller.emit(
                ProjectEventType.PROCESSING_RESULT_SAVED,
                line_id=self.selected_line,
                reason=f"{self.selected_line} 连续处理结果已保存",
                changed_paths=[data_path, params_path] if data_path and params_path else [],
                refresh=False,
            )
        self._refresh_project_widgets()
        if self.processing_log_label is not None:
            if data_path and params_path:
                self.processing_log_label.setText(
                    f"✓  {self.selected_line} 连续处理结果已保存：{data_path.name}；参数：{params_path.name}；已同步到目标来源与项目树。"
                )
            else:
                self.processing_log_label.setText(f"✓  {self.selected_line}_处理结果 已保存，并同步到项目树/测线清单。")

    def _recommend_processing_params(self) -> None:
        values = recommend_field_method_params(self.selected_processing_method_id, self.active_gpr_dataset)
        self._set_processing_param_values(values)
        self.processing_last_failed = False
        self.last_processing_error = ""
        if self.processing_log_label is not None:
            names = ", ".join(f"{k}={v}" for k, v in values.items()) or "无可推荐参数"
            self.processing_log_label.setText(f"✣  已更新 {field_method_display_name(self.selected_processing_method_id)} 的当前参数：{names}")

    def _processing_params_card(self) -> Card:
        lm = layout_metrics_for(self)
        card = Card()
        card.setProperty("layoutKey", "processingParamsCard")
        card.setFixedWidth(lm.processing_params_w)

        # --- Algorithm selection section ---
        category_row = QHBoxLayout()
        category_row.setSpacing(6)
        category_row.addWidget(QLabel(PROCESSING_CATEGORY_LABEL))
        self.processing_category_combo = QComboBox()
        self.processing_category_combo.setMinimumHeight(26)
        for category in self.processing_categories:
            self.processing_category_combo.addItem(category, category)
        idx = self.processing_category_combo.findData(self.selected_processing_category)
        if idx >= 0:
            self.processing_category_combo.setCurrentIndex(idx)
        self.processing_category_combo.currentIndexChanged.connect(self._on_processing_category_changed)
        category_row.addWidget(self.processing_category_combo, 1)
        card.layout.addLayout(category_row)

        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_row.addWidget(QLabel(PROCESSING_METHOD_LABEL))
        self.processing_method_combo = QComboBox()
        self.processing_method_combo.setMinimumHeight(26)
        self.processing_method_combo.currentIndexChanged.connect(self._on_processing_method_changed)
        method_row.addWidget(self.processing_method_combo, 1)
        card.layout.addLayout(method_row)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setFixedHeight(2)
        sep1.setStyleSheet("background-color: #D5DEE7; margin: 2px 0;")
        card.layout.addWidget(sep1)

        # --- Parameter settings section ---
        group = QFrame()
        group.setObjectName("paramGroup")
        group.setProperty("layoutKey", "processingParameterGroup")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(5, 5, 5, 5)
        group_layout.setSpacing(3)
        title = QLabel(PROCESSING_PARAMS_TITLE)
        title.setObjectName("activityTitle")
        group_layout.addWidget(title)
        self.processing_params_body = QWidget()
        self.processing_params_body_layout = QVBoxLayout(self.processing_params_body)
        self.processing_params_body_layout.setContentsMargins(0, 0, 0, 0)
        self.processing_params_body_layout.setSpacing(3)
        group_layout.addWidget(self.processing_params_body)
        card.layout.addWidget(group)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFixedHeight(2)
        sep2.setStyleSheet("background-color: #D5DEE7; margin: 2px 0;")
        card.layout.addWidget(sep2)

        # --- Actions section ---
        actions_group = QFrame()
        actions_group.setObjectName("paramGroup")
        actions_group.setProperty("layoutKey", "processingContinuousCard")
        actions_group.setFixedHeight(174)
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setContentsMargins(5, 5, 5, 5)
        actions_layout.setSpacing(6)

        chain_row = QHBoxLayout()
        chain_label = QLabel("处理链")
        chain_label.setObjectName("activityTitle")
        chain_row.addWidget(chain_label)
        chain_row.addStretch(1)
        self.processing_chain_status_label = QLabel("尚未执行")
        self.processing_chain_status_label.setObjectName("activityDesc")
        self.processing_chain_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        chain_row.addWidget(self.processing_chain_status_label)
        actions_layout.addLayout(chain_row)

        recommend_btn = QPushButton("推荐当前参数")
        recommend_btn.setObjectName("smallButton")
        recommend_btn.setFixedHeight(22)
        recommend_btn.setProperty("layoutKey", "processingRecommendParamsButton")
        recommend_btn.clicked.connect(self._recommend_processing_params)
        actions_layout.addWidget(recommend_btn)

        self.processing_execute_button = QPushButton("执行当前步骤")
        self.processing_execute_button.setObjectName("primaryButton")
        self.processing_execute_button.setFixedHeight(30)
        self.processing_execute_button.setProperty("layoutKey", "processingExecuteStepButton")
        self.processing_execute_button.clicked.connect(self._apply_processing)
        actions_layout.addWidget(self.processing_execute_button)

        undo_reset_row = QHBoxLayout()
        undo_reset_row.setSpacing(6)
        self.processing_undo_step_button = QPushButton("撤回一步")
        self.processing_undo_step_button.setObjectName("smallButton")
        self.processing_undo_step_button.setFixedHeight(26)
        self.processing_undo_step_button.setProperty("layoutKey", "processingUndoStepButton")
        self.processing_undo_step_button.clicked.connect(self._undo_processing)
        undo_reset_row.addWidget(self.processing_undo_step_button, 1)
        self.processing_reset_button = QPushButton("重置到原始")
        self.processing_reset_button.setObjectName("smallButton")
        self.processing_reset_button.setFixedHeight(26)
        self.processing_reset_button.setProperty("layoutKey", "processingResetChainButton")
        self.processing_reset_button.clicked.connect(self._reset_processing_chain)
        undo_reset_row.addWidget(self.processing_reset_button, 1)
        actions_layout.addLayout(undo_reset_row)

        batch_btn = QPushButton("批量处理全部测线")
        batch_btn.setObjectName("smallButton")
        batch_btn.setFixedHeight(26)
        batch_btn.setToolTip("将当前算法和参数应用到所有已导入的测线")
        batch_btn.clicked.connect(self._batch_apply_processing)
        actions_layout.addWidget(batch_btn)
        self.processing_batch_button = batch_btn

        self.processing_save_button = None
        card.layout.addWidget(actions_group)

        card.layout.addStretch(1)
        self._populate_processing_methods()
        return card

    def _on_processing_category_changed(self, _index: int) -> None:
        if self.processing_category_combo is None:
            return
        value = self.processing_category_combo.currentData() or self.processing_category_combo.currentText()
        self.selected_processing_category = str(value)
        self._populate_processing_methods()

    def _populate_processing_methods(self) -> None:
        if self.processing_method_combo is None:
            return
        self.processing_method_combo.blockSignals(True)
        self.processing_method_combo.clear()
        methods = self.processing_categories.get(self.selected_processing_category, [])
        if not methods and self.processing_categories:
            self.selected_processing_category = next(iter(self.processing_categories))
            methods = self.processing_categories[self.selected_processing_category]
        for method in methods:
            self.processing_method_combo.addItem(method.display_name, method.method_id)
        idx = self.processing_method_combo.findData(self.selected_processing_method_id)
        if idx < 0:
            idx = 0
        if self.processing_method_combo.count():
            self.processing_method_combo.setCurrentIndex(idx)
            self.selected_processing_method_id = str(self.processing_method_combo.currentData())
        self.processing_method_combo.blockSignals(False)
        self._rebuild_processing_params_panel()
        self._refresh_processing_preview()

    def _on_processing_method_changed(self, _index: int) -> None:
        if self.processing_method_combo is None or self.processing_method_combo.count() == 0:
            return
        value = self.processing_method_combo.currentData() or self.processing_method_combo.currentText()
        self.selected_processing_method_id = str(value)
        self.processing_last_failed = False
        self.last_processing_error = ""
        self._rebuild_processing_params_panel()
        self._refresh_processing_preview()

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.deleteLater()
            child_layout = item.layout()
            if child_layout is not None:
                while child_layout.count():
                    sub = child_layout.takeAt(0)
                    if sub.widget() is not None:
                        sub.widget().deleteLater()

    def _rebuild_processing_params_panel(self) -> None:
        if self.processing_params_body_layout is None:
            return
        self._clear_layout(self.processing_params_body_layout)
        self.processing_param_widgets = {}
        schema = get_method_params_schema(self.selected_processing_method_id)
        if not schema:
            label = QLabel("当前算法无需手动参数。")
            label.setObjectName("activityDesc")
            self.processing_params_body_layout.addWidget(label)
            return
        for spec in schema:
            name = str(spec.get("name", ""))
            if not name:
                continue
            row = QHBoxLayout()
            row.setSpacing(6)
            label_text = PARAM_LABELS.get(name, str(spec.get("label") or name))
            lab = QLabel(label_text)
            lab.setMinimumWidth(layout_metrics_for(self).processing_param_label_w)
            lab.setMaximumWidth(layout_metrics_for(self).processing_param_label_w + 8)
            row.addWidget(lab)
            widget = self._make_param_widget(spec)
            widget.setMinimumWidth(104)
            widget.setMinimumHeight(24)
            row.addWidget(widget, 1)
            self.processing_params_body_layout.addLayout(row)
            self.processing_param_widgets[name] = widget

    def _make_param_widget(self, spec: dict) -> QWidget:
        typ = str(spec.get("type", "str")).lower()
        default = spec.get("default", "")
        if typ in {"int", "integer"}:
            spin = QSpinBox()
            spin.setButtonSymbols(QSpinBox.ButtonSymbols.PlusMinus)
            spin.setMinimumWidth(104)
            spin.setMinimumHeight(24)
            spin.setRange(int(spec.get("min", -1000000)), int(spec.get("max", 1000000)))
            spin.setValue(int(default if default not in (None, "") else 0))
            return spin
        if typ in {"float", "double"}:
            spin = QDoubleSpinBox()
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.PlusMinus)
            spin.setMinimumWidth(104)
            spin.setMinimumHeight(24)
            spin.setRange(float(spec.get("min", -1_000_000.0)), float(spec.get("max", 1_000_000.0)))
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setValue(float(default if default not in (None, "") else 0.0))
            return spin
        if typ in {"bool", "boolean"}:
            chk = QCheckBox("启用")
            chk.setChecked(bool(default))
            return chk
        if typ in {"choice", "enum"} or spec.get("choices"):
            combo = QComboBox()
            combo.setMinimumWidth(104)
            combo.setMinimumHeight(24)
            choices = spec.get("choices") or []
            for choice in choices:
                combo.addItem(str(choice), choice)
            idx = combo.findData(default)
            if idx < 0:
                idx = combo.findText(str(default))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            return combo
        edit = QLineEdit()
        edit.setMinimumWidth(104)
        edit.setMinimumHeight(24)
        edit.setText(str(default))
        return edit

    def _set_processing_param_values(self, values: dict) -> None:
        for name, value in values.items():
            widget = self.processing_param_widgets.get(str(name))
            if widget is None:
                continue
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findData(value)
                if idx < 0:
                    idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    def _on_gain_slider_changed(self, value: int) -> None:
        self.processing_gain = max(0.7, min(2.6, value / 40.0 if value > 10 else 1.0 + value / 10.0))
        self._refresh_processing_preview()

    def _processing_messages_card(self) -> Card:
        lm = layout_metrics_for(self)
        card = Card(title="处理日志")
        card.setProperty("layoutKey", "processingMessagesCard")
        card.setMinimumHeight(lm.processing_bottom_max_h)
        card.setMaximumHeight(lm.processing_bottom_max_h)
        tabs = QTabWidget()
        tabs.setObjectName("innerTabs")

        history = QWidget()
        vh = QVBoxLayout(history)
        vh.setContentsMargins(0, 0, 0, 0)
        self.processing_history_table = self._table(["步次", "算法 / 参数", "状态", "时间", "操作"], 0)
        vh.addWidget(self.processing_history_table, 1)
        self.processing_history_hint_label = QLabel("尚未执行连续处理步骤。点击“执行当前步骤”后，会按当前算法继续叠加处理。")
        self.processing_history_hint_label.setObjectName("activityDesc")
        self.processing_history_hint_label.setWordWrap(False)
        self.processing_history_hint_label.setMaximumHeight(22)
        vh.addWidget(self.processing_history_hint_label)
        tabs.addTab(history, "处理历史")

        warn = QWidget()
        v = QVBoxLayout(warn)
        for icon, text, time in [
            ("⚠", "L03   辅助定位文件缺少高密度数据", "10:15:32"),
            ("✓", "L01   处理结果已保存", "10:12:08"),
            ("ℹ", "L02   建议检查里程连续性（存在 1 处里程跳变）", "10:10:45"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(icon))
            row.addWidget(QLabel(text), 1)
            row.addWidget(QLabel(time))
            v.addLayout(row)
        self.processing_log_label = QLabel("ℹ  连续处理模式：每执行一次，左侧项目树自动新增一个 Step；撤回会同步移除最后一步。")
        self.processing_log_label.setObjectName("activityDesc")
        self.processing_log_label.setWordWrap(True)
        v.addWidget(self.processing_log_label)
        v.addStretch(1)
        tabs.addTab(warn, "日志")
        card.layout.addWidget(tabs)
        return card


__all__ = ["ProcessingPageMixin"]

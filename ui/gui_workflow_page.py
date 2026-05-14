#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Realtime UAV-GPR workflow editor page."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import FluentIcon, PushButton

from core.methods_registry import PROCESSING_METHODS, get_method_display_name
from core.workflow_data import (
    METHOD_CATEGORIES,
    QUICK_PRESETS,
    WORKFLOW_STAGE_BY_ID,
    WORKFLOW_STAGE_DEFINITIONS,
    WorkflowConfig,
    WorkflowMethod,
    build_default_workflow_config,
    get_config_manager,
)


HEAVY_REALTIME_METHODS = {
    "kirchhoff_migration",
    "stolt_migration",
    "time_to_depth",
}


class WorkflowStepList(QListWidget):
    """QListWidget with an explicit signal after internal drag/drop ordering."""

    order_changed = pyqtSignal()

    def dropEvent(self, event):  # noqa: N802 - Qt override
        super().dropEvent(event)
        self.order_changed.emit()


class WorkflowPage(QWidget):
    """Visual workflow editor for MyGPR's UAV-GPR processing chain."""

    workflow_run_requested = pyqtSignal(object, bool)
    save_live_result_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._config_manager = get_config_manager()
        self.config = build_default_workflow_config("high_quality_uav_gpr")
        self._param_getters: dict[str, Callable[[], Any]] = {}
        self._param_controls: dict[str, QWidget] = {}
        self._data_shape: tuple[int, int] | None = None
        self._suppress_change = False
        self._slider_dragging = False
        self._last_run_methods: list[WorkflowMethod] = []
        self._live_result_available = False
        self._setup_ui()
        self.load_config(self.config)

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("workflowHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(10, 10, 10, 8)
        header_layout.setSpacing(6)
        title = QLabel("工作流")
        title.setProperty("class", "sectionTitle")
        subtitle = QLabel("按 UAV-GPR 标准链路组织处理步骤；拖动步骤、切换算法或调整参数后可实时刷新预览。")
        subtitle.setWordWrap(True)
        subtitle.setProperty("class", "hintText")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        outer.addWidget(header)

        # Split template controls and run controls so the left panel stays usable
        # on narrower screens.
        toolbar = QWidget()
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 0, 10, 8)
        toolbar_layout.setSpacing(6)

        template_row = QWidget()
        template_layout = QHBoxLayout(template_row)
        template_layout.setContentsMargins(0, 0, 0, 0)
        template_layout.setSpacing(8)
        self.template_combo = QComboBox()
        self.template_combo.setToolTip("选择内置或已保存的工作流模板")
        self.template_combo.setMinimumWidth(160)
        self._reload_template_combo()
        template_layout.addWidget(QLabel("模板"))
        template_layout.addWidget(self.template_combo, 1)

        self.realtime_check = QCheckBox("实时预览")
        self.realtime_check.setToolTip("参数或顺序变化后自动计算当前工作流实时结果")
        template_layout.addWidget(self.realtime_check)
        toolbar_layout.addWidget(template_row)

        run_row = QWidget()
        run_layout = QHBoxLayout(run_row)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(6)
        self.btn_run_all = PushButton(FluentIcon.PLAY_SOLID, "全链")
        self.btn_run_all.setToolTip("按当前步骤顺序运行工作流")
        self.btn_run_from_current = PushButton("后续")
        self.btn_run_from_current.setToolTip("从选中步骤开始运行到工作流末尾")
        self.btn_run_selected = PushButton("当前")
        self.btn_run_selected.setToolTip("只运行选中的单个步骤，便于逐步验证")
        self.btn_save_live = PushButton(FluentIcon.SAVE, "保存")
        self.btn_save_live.setToolTip("将实时预览或最近一次工作流结果写入正式历史")
        self.btn_save_live.setEnabled(False)

        template_action_row = QWidget()
        template_action_layout = QHBoxLayout(template_action_row)
        template_action_layout.setContentsMargins(0, 0, 0, 0)
        template_action_layout.setSpacing(6)

        self.btn_new_template = PushButton(FluentIcon.ADD, "新建")
        self.btn_new_template.setToolTip("从内置高质量 UAV-GPR 模板创建一个用户模板")
        self.btn_duplicate_template = PushButton(FluentIcon.COPY, "复制")
        self.btn_save_template = PushButton(FluentIcon.SAVE, "存模板")
        self.btn_import_template = PushButton(FluentIcon.FOLDER, "导入")
        self.btn_export_template = PushButton(FluentIcon.SAVE_AS, "导出")
        self.btn_restore_default = PushButton(FluentIcon.SYNC, "默认")

        for btn in [
            self.btn_run_all,
            self.btn_run_from_current,
            self.btn_run_selected,
            self.btn_save_live,
        ]:
            btn.setMinimumWidth(0)
            run_layout.addWidget(btn)
        run_layout.addStretch(1)
        toolbar_layout.addWidget(run_row)

        for btn in [
            self.btn_new_template,
            self.btn_duplicate_template,
            self.btn_save_template,
            self.btn_import_template,
            self.btn_export_template,
            self.btn_restore_default,
        ]:
            btn.setMinimumWidth(0)
            template_action_layout.addWidget(btn)
        template_action_layout.addStretch(1)
        toolbar_layout.addWidget(template_action_row)
        outer.addWidget(toolbar)

        body_scroll = QScrollArea()
        body_scroll.setWidgetResizable(True)
        body_scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(10, 0, 10, 10)
        body_layout.setSpacing(10)
        body_scroll.setWidget(body)
        outer.addWidget(body_scroll, 1)

        self.step_list = WorkflowStepList()
        self.step_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.step_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.step_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.step_list.setMinimumWidth(0)
        self.step_list.setMinimumHeight(260)
        self.step_list.setToolTip("拖拽调整处理顺序；隐藏的步骤不会执行")

        step_panel = QWidget()
        step_panel_layout = QVBoxLayout(step_panel)
        step_panel_layout.setContentsMargins(0, 0, 0, 0)
        step_panel_layout.setSpacing(8)
        step_panel_layout.addWidget(self.step_list, 1)
        step_action_row = QWidget()
        step_action_layout = QHBoxLayout(step_action_row)
        step_action_layout.setContentsMargins(0, 0, 0, 0)
        step_action_layout.setSpacing(6)
        self.btn_add_step = PushButton("添加")
        self.btn_duplicate_step = PushButton("复制")
        self.btn_remove_step = PushButton("删除")
        self.btn_add_step.setToolTip("在当前步骤后插入同阶段默认步骤")
        self.btn_duplicate_step.setToolTip("复制当前步骤及其参数")
        self.btn_remove_step.setToolTip("删除当前步骤")
        for btn in [self.btn_add_step, self.btn_duplicate_step, self.btn_remove_step]:
            step_action_layout.addWidget(btn)
        step_action_layout.addStretch(1)
        step_panel_layout.addWidget(step_action_row)
        body_layout.addWidget(self._wrap_group("流程步骤", step_panel))

        self.detail_box = QGroupBox("当前步骤")
        detail_layout = QVBoxLayout(self.detail_box)
        detail_layout.setContentsMargins(10, 14, 10, 10)
        detail_layout.setSpacing(8)

        self.stage_label = QLabel("--")
        self.stage_label.setProperty("class", "titleSmall")
        self.stage_warning = QLabel("")
        self.stage_warning.setWordWrap(True)
        self.stage_warning.setProperty("class", "hintText")

        method_row = QWidget()
        method_layout = QHBoxLayout(method_row)
        method_layout.setContentsMargins(0, 0, 0, 0)
        method_layout.setSpacing(8)
        method_layout.addWidget(QLabel("算法"))
        self.method_combo = QComboBox()
        method_layout.addWidget(self.method_combo, 1)
        self.enabled_check = QCheckBox("启用")
        self.hidden_check = QCheckBox("隐藏")
        method_layout.addWidget(self.enabled_check)
        method_layout.addWidget(self.hidden_check)

        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.param_host = QWidget()
        self.param_layout = QFormLayout(self.param_host)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_layout.setHorizontalSpacing(10)
        self.param_layout.setVerticalSpacing(8)
        self.param_scroll.setWidget(self.param_host)
        self.param_scroll.setMinimumHeight(170)

        detail_layout.addWidget(self.stage_label)
        detail_layout.addWidget(self.stage_warning)
        detail_layout.addWidget(method_row)
        detail_layout.addWidget(self.param_scroll, 1)
        body_layout.addWidget(self.detail_box)

        self.log_box = QGroupBox("预览与质量提示")
        log_layout = QVBoxLayout(self.log_box)
        log_layout.setContentsMargins(10, 14, 10, 10)
        log_layout.setSpacing(8)
        self.status_label = QLabel("未运行")
        self.status_label.setProperty("class", "hintText")
        self.workflow_log = QTextEdit()
        self.workflow_log.setReadOnly(True)
        self.workflow_log.setMinimumHeight(160)
        self.workflow_log.setPlaceholderText("工作流运行状态、风险提示和最近步骤日志")
        log_layout.addWidget(self.status_label)
        log_layout.addWidget(self.workflow_log, 1)
        body_layout.addWidget(self.log_box)
        body_layout.addStretch(1)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(700)
        self._debounce_timer.timeout.connect(self._emit_realtime_run)

        self.step_list.currentRowChanged.connect(self._on_step_selected)
        self.step_list.order_changed.connect(self._on_order_changed)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.enabled_check.stateChanged.connect(self._on_step_flags_changed)
        self.hidden_check.stateChanged.connect(self._on_step_flags_changed)
        self.realtime_check.stateChanged.connect(self._on_realtime_changed)
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        self.btn_run_all.clicked.connect(self.request_manual_run)
        self.btn_run_from_current.clicked.connect(self.request_run_from_current)
        self.btn_run_selected.clicked.connect(self.request_selected_run)
        self.btn_save_live.clicked.connect(self.save_live_result_requested)
        self.btn_new_template.clicked.connect(self.new_user_template)
        self.btn_duplicate_template.clicked.connect(self.duplicate_current_template)
        self.btn_save_template.clicked.connect(self.save_current_template)
        self.btn_import_template.clicked.connect(self.import_template)
        self.btn_export_template.clicked.connect(self.export_template)
        self.btn_restore_default.clicked.connect(self.restore_default_template)
        self.btn_add_step.clicked.connect(self.add_step_after_current)
        self.btn_duplicate_step.clicked.connect(self.duplicate_current_step)
        self.btn_remove_step.clicked.connect(self.remove_current_step)

    def _wrap_group(self, title: str, widget: QWidget) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.addWidget(widget)
        return box

    def _reload_template_combo(self) -> None:
        if not hasattr(self, "template_combo"):
            return
        current_name = self.config.name if hasattr(self, "config") else ""
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        for key, preset in QUICK_PRESETS.items():
            self.template_combo.addItem(f"系统模板 - {preset['name']}", ("system", key))
        for item in self._config_manager.list_configs():
            self.template_combo.addItem(f"用户模板 - {item['name']}", ("user", item["filename"]))
        for index in range(self.template_combo.count()):
            if current_name and current_name in self.template_combo.itemText(index):
                self.template_combo.setCurrentIndex(index)
                break
        self.template_combo.blockSignals(False)

    def load_config(self, config: WorkflowConfig) -> None:
        self.config = config
        self.config.methods = sorted(self.config.methods, key=lambda item: item.order)
        self.realtime_check.blockSignals(True)
        self.realtime_check.setChecked(bool(self.config.realtime_enabled))
        self.realtime_check.blockSignals(False)
        self._render_steps()
        if self.step_list.count() > 0:
            self.step_list.setCurrentRow(0)
        self._log(f"已加载模板: {self.config.name}")

    def _render_steps(self) -> None:
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for index, method in enumerate(sorted(self.config.methods, key=lambda item: item.order)):
            method.order = index
            item = QListWidgetItem(self._format_step_text(method))
            item.setData(Qt.ItemDataRole.UserRole, method)
            self.step_list.addItem(item)
        self.step_list.blockSignals(False)
        self._update_step_buttons()

    def _format_step_text(self, method: WorkflowMethod) -> str:
        stage = WORKFLOW_STAGE_BY_ID.get(method.stage_id, {})
        category = METHOD_CATEGORIES.get(method.category, {})
        stage_label = stage.get("label") or category.get("name") or method.category or "未分组"
        method_name = get_method_display_name(method.method_id)
        if method.hidden:
            state = "隐藏"
        elif not method.enabled:
            state = "停用"
        else:
            state = "启用"
        return f"{method.order + 1:02d}. {stage_label}\n{method_name} | {state}"

    def _param_summary(self, method: WorkflowMethod) -> str:
        if not method.params:
            return ""
        tokens = []
        for key, value in list(method.params.items())[:3]:
            tokens.append(f"{key}={value}")
        if len(method.params) > 3:
            tokens.append("...")
        return " | " + ", ".join(tokens)

    def _selected_method(self) -> WorkflowMethod | None:
        row = self.step_list.currentRow()
        if row < 0 or row >= self.step_list.count():
            return None
        item = self.step_list.item(row)
        method = item.data(Qt.ItemDataRole.UserRole)
        return method if isinstance(method, WorkflowMethod) else None

    def _on_step_selected(self, row: int) -> None:
        method = self._selected_method()
        self._update_step_buttons()
        if method is None:
            return
        self._suppress_change = True
        try:
            stage = WORKFLOW_STAGE_BY_ID.get(method.stage_id, {})
            category = METHOD_CATEGORIES.get(method.category, {})
            stage_label = stage.get("label") or category.get("name") or method.category or "--"
            self.stage_label.setText(stage_label)
            self.stage_warning.setText(stage.get("warning", ""))
            self.enabled_check.setChecked(bool(method.enabled))
            self.hidden_check.setChecked(bool(method.hidden))
            self._render_method_combo(method)
            self._render_params(method)
        finally:
            self._suppress_change = False

    def _render_method_combo(self, method: WorkflowMethod) -> None:
        stage = WORKFLOW_STAGE_BY_ID.get(method.stage_id, {})
        candidates = list(stage.get("candidate_methods") or [method.method_id])
        if method.method_id not in candidates:
            candidates.insert(0, method.method_id)
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        for key in candidates:
            if key in PROCESSING_METHODS:
                self.method_combo.addItem(get_method_display_name(key), key)
        idx = self.method_combo.findData(method.method_id)
        self.method_combo.setCurrentIndex(max(idx, 0))
        self.method_combo.blockSignals(False)

    def _render_params(self, method: WorkflowMethod) -> None:
        while self.param_layout.rowCount():
            self.param_layout.removeRow(0)
        self._param_getters.clear()
        self._param_controls.clear()

        params = PROCESSING_METHODS.get(method.method_id, {}).get("params", [])
        if not params:
            self.param_layout.addRow(QLabel("(无参数)"))
            return
        for meta in params:
            name = str(meta.get("name"))
            value = method.params.get(name, meta.get("default", ""))
            row_widget, getter = self._create_param_control(meta, value)
            label = QLabel(str(meta.get("label", name)))
            label.setWordWrap(True)
            tooltip = str(meta.get("tooltip", ""))
            if tooltip:
                label.setToolTip(tooltip)
                row_widget.setToolTip(tooltip)
            self.param_layout.addRow(label, row_widget)
            self._param_getters[name] = getter
            self._param_controls[name] = row_widget

    def _create_param_control(self, meta: dict[str, Any], value: Any) -> tuple[QWidget, Callable[[], Any]]:
        param_type = str(meta.get("type", "str"))
        if param_type == "bool":
            checkbox = QCheckBox()
            checkbox.setChecked(bool(value))
            checkbox.stateChanged.connect(self._on_param_changed)
            return checkbox, checkbox.isChecked

        if param_type in {"str", "choice"} and meta.get("choices"):
            combo = QComboBox()
            for choice in meta.get("choices", []):
                combo.addItem(str(choice), choice)
            idx = combo.findData(value)
            if idx < 0:
                idx = combo.findText(str(value))
            combo.setCurrentIndex(max(idx, 0))
            combo.currentIndexChanged.connect(self._on_param_changed)
            return combo, combo.currentData

        if param_type == "int":
            return self._create_int_control(meta, value)
        if param_type == "float":
            return self._create_float_control(meta, value)

        edit = QLineEdit(str(value))
        edit.textEdited.connect(self._on_param_changed)
        return edit, edit.text

    def _create_int_control(self, meta: dict[str, Any], value: Any) -> tuple[QWidget, Callable[[], Any]]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        min_v = int(meta.get("min", 0))
        max_v = int(self._dynamic_int_max(meta, int(meta.get("max", 1000))))
        spin = QSpinBox()
        spin.setRange(min_v, max(min_v, max_v))
        spin.setValue(max(min_v, min(max_v, int(float(value or meta.get("default", min_v))))))
        layout.addWidget(spin)

        if max_v - min_v <= 5000:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(spin.value())
            slider.sliderPressed.connect(self._on_slider_pressed)
            slider.sliderReleased.connect(self._on_slider_released)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            layout.addWidget(slider, 1)
        spin.valueChanged.connect(self._on_param_changed)
        return row, spin.value

    def _create_float_control(self, meta: dict[str, Any], value: Any) -> tuple[QWidget, Callable[[], Any]]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        min_v = float(meta.get("min", 0.0))
        max_v = float(meta.get("max", 1.0))
        current = float(value if value not in (None, "") else meta.get("default", min_v))
        current = max(min_v, min(max_v, current))
        spin = QDoubleSpinBox()
        spin.setDecimals(6 if max_v <= 1.0 else 3)
        spin.setRange(min_v, max_v)
        spin.setValue(current)
        spin.setSingleStep(_float_step(min_v, max_v))
        layout.addWidget(spin)

        if max_v > min_v and max_v / max(min_v, 1.0e-9) < 10000:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(_float_to_slider(current, min_v, max_v))
            slider.sliderPressed.connect(self._on_slider_pressed)
            slider.sliderReleased.connect(self._on_slider_released)
            slider.valueChanged.connect(
                lambda raw, s=spin, lo=min_v, hi=max_v: s.setValue(_slider_to_float(raw, lo, hi))
            )
            spin.valueChanged.connect(
                lambda raw, sl=slider, lo=min_v, hi=max_v: sl.setValue(_float_to_slider(raw, lo, hi))
            )
            layout.addWidget(slider, 1)
        spin.valueChanged.connect(self._on_param_changed)
        return row, spin.value

    def _dynamic_int_max(self, meta: dict[str, Any], fallback: int) -> int:
        if self._data_shape is None:
            return fallback
        samples, traces = self._data_shape
        name = str(meta.get("name", "")).lower()
        if "ntrace" in name or "trace" in name and "index" not in name:
            return max(1, min(fallback, int(traces)))
        if "rank" in name:
            return max(1, min(fallback, int(min(samples, traces))))
        if "window" in name:
            return max(1, min(fallback, int(samples)))
        return fallback

    def _on_method_changed(self) -> None:
        if self._suppress_change:
            return
        method = self._selected_method()
        if method is None:
            return
        new_key = self.method_combo.currentData()
        if not new_key or new_key == method.method_id:
            return
        method.method_id = str(new_key)
        category = PROCESSING_METHODS.get(method.method_id, {}).get("category")
        if category:
            method.category = str(category)
        method.params = self._default_params_for(method.method_id)
        self._render_params(method)
        self._refresh_selected_item()
        self._queue_realtime_run()

    def _on_step_flags_changed(self) -> None:
        if self._suppress_change:
            return
        method = self._selected_method()
        if method is None:
            return
        method.enabled = bool(self.enabled_check.isChecked())
        method.hidden = bool(self.hidden_check.isChecked())
        self._refresh_selected_item()
        self._queue_realtime_run()

    def _on_param_changed(self) -> None:
        if self._suppress_change:
            return
        method = self._selected_method()
        if method is None:
            return
        try:
            method.params = self._read_params()
            self.status_label.setText("参数已更新，等待实时预览")
        except ValueError as exc:
            self.status_label.setText(f"参数错误: {exc}")
            return
        self._refresh_selected_item()
        self._queue_realtime_run()

    def _read_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, getter in self._param_getters.items():
            params[name] = getter()
        self._validate_current_params(params)
        return params

    def _validate_current_params(self, params: dict[str, Any]) -> None:
        method = self._selected_method()
        if method is None:
            return
        metadata = {
            item["name"]: item
            for item in PROCESSING_METHODS.get(method.method_id, {}).get("params", [])
        }
        for name, value in params.items():
            meta = metadata.get(name, {})
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                min_v = meta.get("min")
                max_v = meta.get("max")
                if min_v is not None and value < min_v:
                    raise ValueError(f"{name} 低于 {min_v}")
                if max_v is not None and value > max_v:
                    raise ValueError(f"{name} 高于 {max_v}")

    def _refresh_selected_item(self) -> None:
        row = self.step_list.currentRow()
        method = self._selected_method()
        if row >= 0 and method is not None:
            method.order = row
            self.step_list.item(row).setText(self._format_step_text(method))
            self._sync_order_from_list()

    def _on_order_changed(self) -> None:
        self._sync_order_from_list()
        self._queue_realtime_run()

    def _sync_order_from_list(self) -> None:
        methods = []
        for row in range(self.step_list.count()):
            method = self.step_list.item(row).data(Qt.ItemDataRole.UserRole)
            if isinstance(method, WorkflowMethod):
                method.order = row
                methods.append(method)
                self.step_list.item(row).setText(self._format_step_text(method))
        self.config.methods = methods
        self._update_step_buttons()

    def _on_realtime_changed(self) -> None:
        self.config.realtime_enabled = bool(self.realtime_check.isChecked())
        if self.config.realtime_enabled:
            self._queue_realtime_run()

    def _queue_realtime_run(self) -> None:
        if self._suppress_change or not self.realtime_check.isChecked():
            return
        method = self._selected_method()
        if (
            self._slider_dragging
            and method is not None
            and method.method_id in HEAVY_REALTIME_METHODS
        ):
            return
        self._debounce_timer.start()

    def _emit_realtime_run(self) -> None:
        methods = self.get_enabled_methods()
        self._emit_run(
            methods,
            realtime=True,
            status="实时预览计算中",
            log_text="实时预览请求已发出",
        )

    def request_manual_run(self) -> None:
        methods = self.get_enabled_methods()
        if not methods:
            QMessageBox.information(self, "无步骤", "当前工作流没有启用的步骤。")
            return
        self._emit_run(
            methods,
            realtime=False,
            status="工作流运行中",
            log_text="手动运行工作流",
        )

    def request_selected_run(self) -> None:
        method = self._selected_method()
        if method is None:
            QMessageBox.information(self, "无步骤", "请先选择一个步骤。")
            return
        if method.hidden or not method.enabled:
            QMessageBox.information(self, "步骤未启用", "当前步骤被停用或隐藏，不会运行。")
            return
        self._emit_run(
            [deepcopy(method)],
            realtime=False,
            status="当前步骤运行中",
            log_text=f"运行当前步骤: {get_method_display_name(method.method_id)}",
        )

    def request_run_from_current(self) -> None:
        row = self.step_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "无步骤", "请先选择一个起始步骤。")
            return
        self._sync_order_from_list()
        methods = [
            deepcopy(method)
            for method in self.config.methods[row:]
            if method.enabled and not method.hidden
        ]
        if not methods:
            QMessageBox.information(self, "无步骤", "从当前步骤到末尾没有启用的步骤。")
            return
        self._emit_run(
            methods,
            realtime=False,
            status="从当前步骤运行中",
            log_text=f"从第 {row + 1} 步运行，共 {len(methods)} 步",
        )

    def _emit_run(
        self,
        methods: list[WorkflowMethod],
        *,
        realtime: bool,
        status: str,
        log_text: str,
    ) -> None:
        if not methods:
            self.status_label.setText("没有启用的步骤")
            return
        self._last_run_methods = [deepcopy(method) for method in methods]
        self.workflow_run_requested.emit(self._last_run_methods, realtime)
        self.status_label.setText(status)
        self._log(log_text)

    def get_enabled_methods(self) -> list[WorkflowMethod]:
        self._sync_order_from_list()
        return [deepcopy(method) for method in self.config.get_enabled_methods()]

    def set_data_shape(self, shape: tuple[int, int] | None) -> None:
        self._data_shape = shape
        method = self._selected_method()
        if method is not None:
            self._suppress_change = True
            try:
                self._render_params(method)
            finally:
                self._suppress_change = False

    def set_running(self, message: str) -> None:
        self.status_label.setText(message)

    def set_run_result(self, outputs: list[dict[str, Any]], realtime: bool) -> None:
        self._live_result_available = True
        self.btn_save_live.setEnabled(True)
        label = "实时预览完成" if realtime else "工作流运行完成"
        self.status_label.setText(label)
        self._log(f"{label}: {len(outputs)} 步")
        for index, output in enumerate(outputs, start=1):
            name = output.get("method_name") or output.get("method_key") or f"step-{index}"
            shape = output.get("data").shape if output.get("data") is not None else "--"
            self._log(f"  [{index}] {name} -> {shape}")

    def set_run_error(self, error_message: str) -> None:
        self.status_label.setText("运行失败")
        self._log(f"运行失败: {error_message}")

    def _log(self, text: str) -> None:
        self.workflow_log.append(str(text))
        self.workflow_log.ensureCursorVisible()

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self._on_param_changed()

    def _on_template_changed(self) -> None:
        if self._suppress_change:
            return
        payload = self.template_combo.currentData()
        if not payload:
            return
        kind, key = payload
        if kind == "system":
            config = build_default_workflow_config(str(key), template_type="system")
            self.load_config(config)
            return
        config = self._config_manager.load_config(str(key))
        if config is not None:
            self.load_config(config)

    def new_user_template(self) -> None:
        config = build_default_workflow_config("high_quality_uav_gpr", template_type="user")
        config.name = "用户工作流"
        self.load_config(config)

    def duplicate_current_template(self) -> None:
        duplicated = WorkflowConfig.from_dict(self.config.to_dict())
        duplicated.name = f"{self.config.name} 副本"
        duplicated.template_type = "user"
        duplicated.realtime_enabled = True
        self.load_config(duplicated)

    def save_current_template(self) -> None:
        self._on_order_changed()
        self.config.template_type = "user"
        self.config.realtime_enabled = self.realtime_check.isChecked()
        path = self._config_manager.save_config(self.config)
        self._reload_template_combo()
        self._log(f"模板已保存: {path}")

    def import_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入工作流模板", "", "JSON (*.json)")
        if not path:
            return
        config = self._config_manager.load_config(path)
        if config is None:
            try:
                import json

                with open(path, "r", encoding="utf-8") as handle:
                    config = WorkflowConfig.from_dict(json.load(handle))
            except Exception as exc:
                QMessageBox.warning(self, "导入失败", str(exc))
                return
        self.load_config(config)

    def export_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出工作流模板", "workflow.json", "JSON (*.json)")
        if not path:
            return
        self._on_order_changed()
        if not path.lower().endswith(".json"):
            path += ".json"
        import json

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, ensure_ascii=False, indent=2)
        self._log(f"模板已导出: {path}")

    def restore_default_template(self) -> None:
        self.load_config(build_default_workflow_config("high_quality_uav_gpr"))

    def add_step_after_current(self) -> None:
        row = self.step_list.currentRow()
        selected = self._selected_method()
        if selected is not None:
            stage_id = selected.stage_id
            category = selected.category
        else:
            first_stage = WORKFLOW_STAGE_DEFINITIONS[0] if WORKFLOW_STAGE_DEFINITIONS else {}
            stage_id = str(first_stage.get("id", ""))
            category = "preprocessing"
        new_step = self._make_default_step(stage_id=stage_id, category=category)
        self._insert_step(new_step, row + 1 if row >= 0 else self.step_list.count())
        self._log(f"已添加步骤: {get_method_display_name(new_step.method_id)}")

    def duplicate_current_step(self) -> None:
        row = self.step_list.currentRow()
        method = self._selected_method()
        if method is None:
            QMessageBox.information(self, "无步骤", "请先选择要复制的步骤。")
            return
        new_step = WorkflowMethod.from_dict(method.to_dict())
        new_step.status = "pending"
        self._insert_step(new_step, row + 1)
        self._log(f"已复制步骤: {get_method_display_name(new_step.method_id)}")

    def remove_current_step(self) -> None:
        row = self.step_list.currentRow()
        if row < 0 or row >= self.step_list.count():
            QMessageBox.information(self, "无步骤", "请先选择要删除的步骤。")
            return
        method = self._selected_method()
        label = get_method_display_name(method.method_id) if method else "步骤"
        self.step_list.takeItem(row)
        self._sync_order_from_list()
        if self.step_list.count() > 0:
            self.step_list.setCurrentRow(min(row, self.step_list.count() - 1))
        self._queue_realtime_run()
        self._log(f"已删除步骤: {label}")

    def _insert_step(self, method: WorkflowMethod, row: int) -> None:
        row = max(0, min(row, self.step_list.count()))
        self._sync_order_from_list()
        methods = list(self.config.methods)
        methods.insert(row, method)
        for index, item in enumerate(methods):
            item.order = index
        self.config.methods = methods
        self._render_steps()
        self.step_list.setCurrentRow(row)
        self._queue_realtime_run()

    def _make_default_step(self, stage_id: str = "", category: str = "") -> WorkflowMethod:
        stage = WORKFLOW_STAGE_BY_ID.get(stage_id, {})
        method_id = str(stage.get("default_method") or "dewow")
        if method_id not in PROCESSING_METHODS:
            method_id = next(iter(PROCESSING_METHODS.keys()))
        return WorkflowMethod(
            category=category or self._category_for_stage(stage_id, method_id),
            stage_id=stage_id,
            method_id=method_id,
            enabled=True,
            hidden=False,
            order=self.step_list.count(),
            params=self._default_params_for(method_id),
        )

    def _category_for_stage(self, stage_id: str, method_id: str) -> str:
        method_category = PROCESSING_METHODS.get(method_id, {}).get("category")
        if method_category:
            return str(method_category)
        if stage_id in {"zero_time", "trace_correction"}:
            return "preprocessing"
        if stage_id == "motion_compensation":
            return "motion_compensation"
        if stage_id in {"background_clutter", "spatial_denoise"}:
            return "background_removal"
        if stage_id == "gain":
            return "gain"
        if stage_id in {"velocity_model", "geometry_depth", "migration"}:
            return "migration"
        return "custom"

    def _update_step_buttons(self) -> None:
        has_selection = self._selected_method() is not None
        for attr in (
            "btn_duplicate_step",
            "btn_remove_step",
            "btn_run_selected",
            "btn_run_from_current",
        ):
            button = getattr(self, attr, None)
            if button is not None:
                button.setEnabled(has_selection)

    def _default_params_for(self, method_id: str) -> dict[str, Any]:
        params = {}
        for meta in PROCESSING_METHODS.get(method_id, {}).get("params", []):
            params[str(meta.get("name"))] = meta.get("default", "")
        return params


def _float_step(min_v: float, max_v: float) -> float:
    span = max(abs(max_v - min_v), 1.0e-9)
    return span / 100.0


def _float_to_slider(value: float, min_v: float, max_v: float) -> int:
    if max_v <= min_v:
        return 0
    ratio = (float(value) - min_v) / (max_v - min_v)
    return max(0, min(1000, int(round(ratio * 1000))))


def _slider_to_float(value: int, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return min_v
    return min_v + (max_v - min_v) * (float(value) / 1000.0)

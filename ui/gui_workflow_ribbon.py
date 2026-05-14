#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact workflow ribbon shown above the main B-scan view."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.methods_registry import get_method_display_name
from core.workflow_data import (
    METHOD_CATEGORIES,
    WORKFLOW_STAGE_BY_ID,
    WorkflowConfig,
    WorkflowMethod,
)


class WorkflowRibbon(QWidget):
    """Always-visible processing-chain overview above the main plot."""

    run_all_requested = pyqtSignal()
    run_selected_requested = pyqtSignal()
    run_from_current_requested = pyqtSignal()
    save_requested = pyqtSignal()
    step_selected = pyqtSignal(int)
    realtime_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config: WorkflowConfig | None = None
        self._current_step = -1
        self._step_buttons: list[QPushButton] = []
        self._suppress_realtime_signal = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("workflowRibbon")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        self.title_label = QLabel("处理链")
        self.title_label.setProperty("class", "titleSmall")
        top_layout.addWidget(self.title_label)

        self.template_label = QLabel("未加载工作流")
        self.template_label.setProperty("class", "hintText")
        self.template_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top_layout.addWidget(self.template_label, 1)

        self.realtime_check = QCheckBox("实时")
        self.realtime_check.setToolTip("同步右侧工作流页的实时预览开关")
        self.realtime_check.toggled.connect(self._on_realtime_toggled)
        top_layout.addWidget(self.realtime_check)

        self.btn_run_all = QPushButton("全链 ▶")
        self.btn_run_all.setToolTip("运行当前启用的完整工作流")
        self.btn_run_current = QPushButton("当前 ▶")
        self.btn_run_current.setToolTip("运行当前选中的步骤")
        self.btn_run_tail = QPushButton("后续 ▶")
        self.btn_run_tail.setToolTip("从当前步骤运行到末尾")
        self.btn_save = QPushButton("保存")
        self.btn_save.setToolTip("保存实时预览或最近一次工作流结果")

        self.btn_run_all.clicked.connect(self.run_all_requested)
        self.btn_run_current.clicked.connect(self.run_selected_requested)
        self.btn_run_tail.clicked.connect(self.run_from_current_requested)
        self.btn_save.clicked.connect(self.save_requested)

        for button in [
            self.btn_run_all,
            self.btn_run_current,
            self.btn_run_tail,
            self.btn_save,
        ]:
            button.setMinimumWidth(0)
            top_layout.addWidget(button)

        root.addWidget(top_row)

        self.step_scroll = QScrollArea()
        self.step_scroll.setWidgetResizable(True)
        self.step_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.step_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.step_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.step_scroll.setFixedHeight(58)

        self.step_host = QWidget()
        self.step_layout = QHBoxLayout(self.step_host)
        self.step_layout.setContentsMargins(0, 0, 0, 0)
        self.step_layout.setSpacing(6)
        self.step_scroll.setWidget(self.step_host)
        root.addWidget(self.step_scroll)

    def set_config(self, config: WorkflowConfig | None) -> None:
        self._config = config
        self._update_template_label()
        self._rebuild_steps()

    def set_current_step(self, row: int) -> None:
        self._current_step = int(row)
        self._update_step_selection()

    def set_realtime_checked(self, checked: bool) -> None:
        self._suppress_realtime_signal = True
        try:
            self.realtime_check.setChecked(bool(checked))
        finally:
            self._suppress_realtime_signal = False

    def set_save_enabled(self, enabled: bool) -> None:
        self.btn_save.setEnabled(bool(enabled))

    def _on_realtime_toggled(self, checked: bool) -> None:
        if self._suppress_realtime_signal:
            return
        self.realtime_toggled.emit(bool(checked))

    def _rebuild_steps(self) -> None:
        self._clear_step_layout()

        methods = sorted(getattr(self._config, "methods", []) or [], key=lambda item: item.order)
        visible_methods = [method for method in methods if not method.hidden]
        if not visible_methods:
            placeholder = QPushButton("未配置步骤")
            placeholder.setEnabled(False)
            placeholder.setProperty("workflow_row", -1)
            self.step_layout.addWidget(placeholder)
            self._step_buttons.append(placeholder)
            return

        for visible_index, method in enumerate(visible_methods):
            original_index = methods.index(method)
            button = QPushButton(self._format_step(method, original_index))
            button.setObjectName("workflowRibbonStep")
            button.setCheckable(True)
            button.setProperty("workflow_row", original_index)
            button.setMinimumWidth(118)
            button.setMaximumWidth(180)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.setToolTip(self._build_tooltip(method, original_index))
            button.clicked.connect(lambda _checked=False, row=original_index: self.step_selected.emit(row))
            self.step_layout.addWidget(button)
            self._step_buttons.append(button)

            if visible_index < len(visible_methods) - 1:
                arrow = QLabel("→")
                arrow.setProperty("class", "hintText")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.step_layout.addWidget(arrow)

        self.step_layout.addStretch(1)
        self._update_step_selection()

    def _clear_step_layout(self) -> None:
        while self.step_layout.count():
            item = self.step_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._step_buttons.clear()

    def _update_step_selection(self) -> None:
        methods = sorted(getattr(self._config, "methods", []) or [], key=lambda item: item.order)
        current_method = None
        if 0 <= self._current_step < len(methods):
            current_method = methods[self._current_step]

        for button in self._step_buttons:
            row = button.property("workflow_row")
            is_current = row == self._current_step
            button.setChecked(is_current)
            button.setProperty("current", is_current)
            button.style().unpolish(button)
            button.style().polish(button)

        if current_method is None:
            self._update_template_label()
            return
        self.template_label.setText(
            f"{getattr(self._config, 'name', '工作流')} · 当前："
            f"{self._stage_label(current_method)} / {get_method_display_name(current_method.method_id)}"
        )

    def _update_template_label(self) -> None:
        name = getattr(self._config, "name", "") or "未命名工作流"
        self.template_label.setText(name if self._config else "未加载工作流")

    def _format_step(self, method: WorkflowMethod, row: int) -> str:
        state = "停用" if not method.enabled else "启用"
        method_name = get_method_display_name(method.method_id)
        if len(method_name) > 14:
            method_name = method_name[:13] + "…"
        return f"{row + 1:02d}. {self._stage_label(method)}\n{method_name} · {state}"

    def _stage_label(self, method: WorkflowMethod) -> str:
        stage = WORKFLOW_STAGE_BY_ID.get(method.stage_id, {})
        category = METHOD_CATEGORIES.get(method.category, {})
        return str(stage.get("label") or category.get("name") or method.category or "未分组")

    def _build_tooltip(self, method: WorkflowMethod, row: int) -> str:
        params = ", ".join(f"{key}={value}" for key, value in list(method.params.items())[:6])
        parts = [
            f"步骤 {row + 1}",
            f"阶段: {self._stage_label(method)}",
            f"算法: {get_method_display_name(method.method_id)}",
            f"状态: {'启用' if method.enabled else '停用'}",
        ]
        if params:
            parts.append(f"参数: {params}")
        return "\n".join(parts)

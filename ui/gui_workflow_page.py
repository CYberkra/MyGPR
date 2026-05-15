#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Realtime UAV-GPR workflow editor page."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt, QTimer, pyqtSignal, QSettings
from PyQt6.QtGui import QDrag
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
    QMenu,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
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
from core.workflow_validation import to_text_with_suggestions, validate_workflow_config
from ui.workflow_canvas_cards import (
    WorkflowCanvasView,
    candidate_methods_for_workflow_method,
    update_workflow_method_algorithm,
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


class CollapsibleNodeListWidget(QListWidget):
    """Collapsible node library list widget with drag support."""

    node_dragged = pyqtSignal(str)
    node_double_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._collapsed_categories: set[str] = set()
        self._category_items: dict[str, QListWidgetItem] = {}
        self._method_items: dict[str, list[QListWidgetItem]] = {}
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.itemClicked.connect(self._on_item_clicked)

    def startDrag(self, supportedActions):  # noqa: N802
        """Start drag event for selected node."""
        item = self.currentItem()
        if not item:
            return
        method_id = item.data(Qt.ItemDataRole.UserRole)
        if not method_id:
            return

        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(str(method_id))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)

    def set_category_collapsed(self, category_key: str, collapsed: bool):
        """Set category collapsed state."""
        if collapsed:
            self._collapsed_categories.add(category_key)
        else:
            self._collapsed_categories.discard(category_key)
        self._update_category_visibility(category_key)

    def _update_category_visibility(self, category_key: str):
        """Update visibility of items under a category."""
        items = self._method_items.get(category_key, [])
        is_collapsed = category_key in self._collapsed_categories
        for item in items:
            item.setHidden(is_collapsed)

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click to toggle category collapse."""
        category_key = item.data(Qt.ItemDataRole.UserRole + 1)
        if category_key:
            new_state = category_key not in self._collapsed_categories
            self.set_category_collapsed(category_key, new_state)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle item double click for method nodes."""
        method_id = item.data(Qt.ItemDataRole.UserRole)
        if method_id:
            self.node_double_clicked.emit(str(method_id))

    def populate(self, query: str = ""):
        """Populate the list with methods organized by category."""
        self.clear()
        self._category_items = {}
        self._method_items = {}

        for category_key, category in METHOD_CATEGORIES.items():
            methods = [
                method_id
                for method_id in category.get("methods", [])
                if method_id in PROCESSING_METHODS
                and (
                    not query
                    or query in method_id.lower()
                    or query in get_method_display_name(method_id).lower()
                    or query in str(category.get("name", "")).lower()
                )
            ]
            if not methods:
                continue

            # Add category header
            category_name = str(category.get("name", category_key))
            header_text = f"▼ {category_name}" if category_key not in self._collapsed_categories else f"▶ {category_name}"
            header_item = QListWidgetItem(header_text)
            header_item.setData(Qt.ItemDataRole.UserRole + 1, category_key)
            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.addItem(header_item)
            self._category_items[category_key] = header_item

            # Add method items
            self._method_items[category_key] = []
            for method_id in methods:
                display_name = get_method_display_name(method_id)
                item = QListWidgetItem(f"  {display_name}")
                item.setData(Qt.ItemDataRole.UserRole, method_id)
                item.setData(Qt.ItemDataRole.UserRole + 1, None)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled)
                item.setHidden(category_key in self._collapsed_categories)
                item.setToolTip(f"{display_name}\n({method_id})")
                self.addItem(item)
                self._method_items[category_key].append(item)


class WorkflowPage(QWidget):
    """Visual workflow editor for MyGPR's UAV-GPR processing chain."""

    workflow_run_requested = pyqtSignal(object, bool, str)
    save_live_result_requested = pyqtSignal()
    import_raw_requested = pyqtSignal()
    import_sidecar_requested = pyqtSignal(str)
    tuning_lab_requested = pyqtSignal(object)
    preview_settings_requested = pyqtSignal()
    preview_large_requested = pyqtSignal(object, str)
    export_evidence_requested = pyqtSignal()
    validation_report_requested = pyqtSignal(str)
    log_message_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._config_manager = get_config_manager()
        self.config = build_default_workflow_config("high_quality_uav_gpr")
        self._param_getters: dict[str, Callable[[], Any]] = {}
        self._param_controls: dict[str, QWidget] = {}
        self._data_shape: tuple[int, int] | None = None
        self._current_file: str = ""
        self._metadata_status: str = "未加载"
        self._sidecar_files: dict[str, str | None] = {}
        self._suppress_change = False
        self._slider_dragging = False
        self._last_run_methods: list[WorkflowMethod] = []
        self._live_result_available = False
        self._run_history: list[dict[str, Any]] = []
        self._run_history_index = 0
        self._setup_ui()
        self._restore_workspace_state()
        self.load_config(self.config)

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        studio_bar = QFrame()
        studio_bar.setObjectName("workflowStudioBar")
        studio_row = QHBoxLayout(studio_bar)
        studio_row.setContentsMargins(10, 7, 10, 7)
        studio_row.setSpacing(8)

        title = QLabel("MyGPR 工作流")
        title.setObjectName("workflowStudioTitle")
        title.setMinimumWidth(96)
        studio_row.addWidget(title)

        template_label = QLabel("模板")
        template_label.setObjectName("workflowTopHint")
        studio_row.addWidget(template_label)
        self.template_combo = QComboBox()
        self.template_combo.setToolTip("选择内置或已保存的工作流模板")
        self.template_combo.setMinimumWidth(170)
        self.template_combo.setMaximumWidth(260)
        self._reload_template_combo()
        studio_row.addWidget(self.template_combo, 0)

        self.btn_run_all = PushButton(FluentIcon.PLAY_SOLID, "运行")
        self.btn_run_all.setObjectName("workflowRunButton")
        self.btn_run_all.setToolTip("运行全部")
        self.run_menu_button = QToolButton()
        self.run_menu_button.setText("▾")
        self.run_menu_button.setToolTip("选择运行模式")
        self.run_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.run_menu = QMenu(self.run_menu_button)
        self.run_menu.addAction("运行全部", self.request_manual_run)
        self.run_menu.addAction("从当前节点运行", self.request_run_from_current)
        self.run_menu.addAction("只运行选中节点", self.request_selected_run)
        self.run_menu.addAction("验证工作流", self._validate_workflow_ui)
        self.stop_action = self.run_menu.addAction("停止 / 取消")
        self.stop_action.setEnabled(False)
        self.run_menu_button.setMenu(self.run_menu)
        self.btn_run_from_current = PushButton("后续")
        self.btn_run_from_current.setToolTip("从选中步骤运行到末尾")
        self.btn_run_selected = PushButton("选中")
        self.btn_run_selected.setToolTip("只运行选中的单个步骤")
        self.btn_validate = PushButton("验证")
        self.btn_open_tuning_lab = PushButton("调参")
        self.btn_open_tuning_lab.setToolTip("打开选中节点的自动选参与实验室")
        self.btn_save_live = PushButton(FluentIcon.SAVE, "保存")
        self.btn_save_live.setToolTip("将实时预览或最近一次工作流结果写入正式历史")
        self.btn_save_live.setEnabled(False)

        self.btn_toggle_project = PushButton("项目")
        self.btn_toggle_project.setToolTip("展开或收起左侧 Project / Data 与节点库")
        self.btn_toggle_project.setCheckable(True)
        self.btn_toggle_project.setChecked(True)
        self.btn_toggle_inspector = PushButton("属性")
        self.btn_toggle_inspector.setToolTip("展开或收起右侧 Inspector")
        self.btn_toggle_inspector.setCheckable(True)
        self.btn_toggle_inspector.setChecked(True)
        self.btn_run_from_current.hide()
        self.btn_run_selected.hide()
        self.btn_validate.hide()
        self.btn_open_tuning_lab.hide()
        self.btn_save_live.hide()
        self.btn_toggle_project.hide()
        # 恢复 Inspector 切换按钮的显示
        self.btn_toggle_inspector.setMinimumWidth(60)
        self.btn_toggle_inspector.setMaximumWidth(80)
        studio_row.addWidget(self.btn_toggle_inspector)

        self.btn_run_all.setMinimumWidth(88)
        self.btn_run_all.setMaximumWidth(118)
        self.run_menu_button.setFixedWidth(30)
        studio_row.addWidget(self.btn_run_all)
        studio_row.addWidget(self.run_menu_button)
        studio_row.addStretch(1)

        self.realtime_check = QCheckBox("实时")
        self.realtime_check.setToolTip("参数或顺序变化后自动计算当前工作流实时结果")
        self.safe_check = QCheckBox("安全")
        self.safe_check.setChecked(True)
        self.execution_mode_label = QLabel("执行：顺序")
        self.execution_mode_label.setToolTip("当前执行语义仍以步骤顺序为准；画布连接用于可视化和验证提示")
        self.zoom_label = QLabel("缩放 100%")
        self.btn_fit_canvas = PushButton("适配")
        self.btn_auto_layout = PushButton("自动布局")
        self.btn_reset_zoom = PushButton("100%")
        self.btn_fit_canvas.setMaximumWidth(88)
        self.btn_auto_layout.setMaximumWidth(110)
        self.btn_reset_zoom.setMaximumWidth(76)
        for widget in [
            self.realtime_check,
            self.safe_check,
            self.execution_mode_label,
            self.zoom_label,
        ]:
            studio_row.addWidget(widget)
        self.btn_fit_canvas.hide()
        self.btn_auto_layout.hide()
        self.btn_reset_zoom.hide()

        self.template_menu_button = QToolButton()
        self.template_menu_button.setText("⋯")
        self.template_menu_button.setToolTip("模板、视图和工作区设置")
        self.template_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        template_menu = QMenu(self.template_menu_button)
        self.template_menu_button.setMenu(template_menu)
        studio_row.addWidget(self.template_menu_button)
        outer.addWidget(studio_bar)

        self.btn_new_template = PushButton(FluentIcon.ADD, "新建模板")
        self.btn_new_template.setToolTip("从内置高质量 UAV-GPR 模板创建一个用户模板")
        self.btn_duplicate_template = PushButton(FluentIcon.COPY, "复制模板")
        self.btn_save_template = PushButton(FluentIcon.SAVE, "保存模板")
        self.btn_import_template = PushButton(FluentIcon.FOLDER, "导入")
        self.btn_export_template = PushButton(FluentIcon.SAVE_AS, "导出")
        self.btn_restore_default = PushButton(FluentIcon.SYNC, "恢复默认")
        template_menu.addAction("新建模板", self.new_user_template)
        template_menu.addAction("复制模板", self.duplicate_current_template)
        template_menu.addAction("保存模板", self.save_current_template)
        template_menu.addAction("导入模板", self.import_template)
        template_menu.addAction("导出模板", self.export_template)
        template_menu.addAction("恢复默认", self.restore_default_template)
        template_menu.addSeparator()
        template_menu.addAction("自动布局", lambda: self.workflow_canvas.auto_layout())
        template_menu.addAction("适配画布", lambda: self.workflow_canvas.fit_nodes())
        template_menu.addAction("重置缩放", lambda: self.workflow_canvas.reset_zoom())
        template_menu.addSeparator()
        template_menu.addAction("显示设置", self.preview_settings_requested.emit)
        workspace_settings_action = template_menu.addAction("工作区设置")
        workspace_settings_action.setEnabled(False)

        workspace = QWidget()
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(10, 8, 10, 10)
        workspace_layout.setSpacing(0)
        outer.addWidget(workspace, 1)

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(True)
        self.workspace_splitter.setHandleWidth(8)
        workspace_layout.addWidget(self.workspace_splitter, 1)

        self.step_list = WorkflowStepList()
        self.step_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.step_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.step_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.step_list.setMinimumWidth(0)
        self.step_list.setMinimumHeight(260)
        self.step_list.setToolTip("拖拽调整处理顺序；隐藏的步骤不会执行")

        left_sidebar = QWidget()
        self.left_sidebar = left_sidebar
        left_sidebar.setObjectName("workflowLeftDock")
        left_sidebar.setMinimumWidth(52)
        left_sidebar.setMaximumWidth(620)
        left_sidebar.setMinimumHeight(0)
        left_sidebar_layout = QHBoxLayout(left_sidebar)
        left_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        left_sidebar_layout.setSpacing(6)

        self.left_icon_rail = QFrame()
        self.left_icon_rail.setObjectName("workflowIconRail")
        self.left_icon_rail.setFixedWidth(52)
        rail_layout = QVBoxLayout(self.left_icon_rail)
        rail_layout.setContentsMargins(5, 6, 5, 6)
        rail_layout.setSpacing(6)
        self.rail_buttons: dict[str, QToolButton] = {}
        self.left_panel_keys: list[str] = []

        self.left_panel_stack = QStackedWidget()
        self.left_panel_stack.setObjectName("workflowLeftPanelStack")
        self.left_panel_stack.setMinimumWidth(340)
        self.left_panel_stack.setMaximumWidth(560)
        self.left_panel_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        left_sidebar_layout.addWidget(self.left_icon_rail)
        left_sidebar_layout.addWidget(self.left_panel_stack, 1)

        def add_rail_button(key: str, icon: FluentIcon, tooltip: str, index: int) -> None:
            button = QToolButton()
            button.setIcon(icon.icon())
            button.setToolTip(tooltip)
            button.setCheckable(True)
            button.setFixedSize(40, 40)
            button.clicked.connect(lambda _checked=False, panel_key=key: self._toggle_left_panel(panel_key))
            self.rail_buttons[key] = button
            rail_layout.addWidget(button)
            self.left_panel_keys.insert(index, key)

        self.project_panel = QGroupBox("项目 / 数据")
        project_layout = QVBoxLayout(self.project_panel)
        project_layout.setContentsMargins(8, 14, 8, 8)
        project_layout.setSpacing(6)
        import_row = QWidget()
        import_layout = QHBoxLayout(import_row)
        import_layout.setContentsMargins(0, 0, 0, 0)
        import_layout.setSpacing(5)
        self.btn_import_raw = PushButton("导入")
        self.btn_import_rtk = PushButton("RTK")
        self.btn_import_imu = PushButton("IMU")
        self.btn_import_agl = PushButton("AGL")
        for btn in [
            self.btn_import_raw,
            self.btn_import_rtk,
            self.btn_import_imu,
            self.btn_import_agl,
        ]:
            btn.setMinimumWidth(0)
            import_layout.addWidget(btn)
        project_layout.addWidget(import_row)
        self.project_file_label = QLabel("当前文件：--")
        self.project_shape_label = QLabel("数据尺寸：--")
        self.project_metadata_label = QLabel("元数据：未加载")
        self.raw_status_label = QLabel("Raw：missing")
        self.rtk_status_label = QLabel("RTK：missing")
        self.imu_status_label = QLabel("IMU：missing")
        self.agl_status_label = QLabel("AGL：missing")
        for label in [
            self.project_file_label,
            self.project_shape_label,
            self.project_metadata_label,
            self.raw_status_label,
            self.rtk_status_label,
            self.imu_status_label,
            self.agl_status_label,
        ]:
            label.setWordWrap(True)
            label.setProperty("class", "hintText")
            project_layout.addWidget(label)
        self.btn_create_raw_input = PushButton("创建 / 更新输入节点")
        self.btn_create_raw_input.setToolTip("在画布中创建或更新原始数据输入节点占位")
        project_layout.addWidget(self.btn_create_raw_input)
        self.left_panel_stack.addWidget(self.project_panel)

        self.palette_panel = QGroupBox("节点库")
        palette_layout = QVBoxLayout(self.palette_panel)
        palette_layout.setContentsMargins(8, 14, 8, 8)
        palette_layout.setSpacing(6)
        self.palette_search = QLineEdit()
        self.palette_search.setPlaceholderText("搜索节点")
        self.palette_list = CollapsibleNodeListWidget()
        self.palette_list.setMinimumWidth(320)
        self.palette_list.setMaximumWidth(520)
        palette_layout.addWidget(self.palette_search)
        palette_layout.addWidget(self.palette_list, 1)
        self.left_panel_stack.addWidget(self.palette_panel)

        # 运行历史面板
        self.run_panel = QGroupBox("运行记录")
        run_layout = QVBoxLayout(self.run_panel)
        run_layout.setContentsMargins(8, 14, 8, 8)
        run_layout.setSpacing(6)
        self.left_run_history_list = QListWidget()
        self.left_run_history_list.setAlternatingRowColors(True)
        self.left_run_history_list.itemClicked.connect(self._on_history_item_clicked)
        run_layout.addWidget(self.left_run_history_list, 1)
        self.left_panel_stack.addWidget(self.run_panel)

        # 调参与实验面板
        self.tuning_panel = QGroupBox("调参与实验")
        tuning_layout = QVBoxLayout(self.tuning_panel)
        tuning_layout.setContentsMargins(8, 14, 8, 8)
        tuning_layout.setSpacing(6)
        self.tuning_node_info = QLabel("请先在画布中选择一个节点")
        self.tuning_node_info.setWordWrap(True)
        self.tuning_node_info.setProperty("class", "hintText")
        self.btn_open_tuning = PushButton("打开自动选参")
        self.btn_open_tuning.clicked.connect(self.request_tuning_lab_for_current)
        self.btn_apply_best = PushButton("应用最佳参数")
        self.btn_apply_best.setEnabled(False)
        self.btn_apply_best.setToolTip("完成自动选参后可用")
        self.btn_apply_best.clicked.connect(self._apply_best_params)
        tuning_layout.addWidget(self.tuning_node_info)
        tuning_layout.addWidget(self.btn_open_tuning)
        tuning_layout.addWidget(self.btn_apply_best)
        tuning_layout.addStretch(1)
        self.left_panel_stack.addWidget(self.tuning_panel)

        # 验证/QC面板
        self.validation_panel = QGroupBox("验证 / QC")
        validation_layout = QVBoxLayout(self.validation_panel)
        validation_layout.setContentsMargins(8, 14, 8, 8)
        validation_layout.setSpacing(6)
        self.validation_info_label = QLabel("点击\"运行验证\"检查工作流")
        self.validation_info_label.setWordWrap(True)
        self.validation_info_label.setProperty("class", "hintText")
        self.sidecar_status_label = QLabel(self._current_sidecar_status_text())
        self.sidecar_status_label.setWordWrap(True)
        self.sidecar_status_label.setProperty("class", "hintText")
        self.btn_run_validation = PushButton("运行验证")
        self.btn_run_validation.clicked.connect(self._validate_workflow_ui)
        validation_layout.addWidget(self.validation_info_label)
        validation_layout.addWidget(self.sidecar_status_label)
        validation_layout.addWidget(self.btn_run_validation)
        validation_layout.addStretch(1)
        self.left_panel_stack.addWidget(self.validation_panel)

        # 导出/Evidence面板
        self.export_panel = QGroupBox("导出 / Evidence")
        export_layout = QVBoxLayout(self.export_panel)
        export_layout.setContentsMargins(8, 14, 8, 8)
        export_layout.setSpacing(6)
        self.btn_export_evidence = PushButton("导出 Evidence Package")
        self.btn_export_evidence.clicked.connect(self.export_evidence_requested.emit)
        self.btn_preview_snapshot = PushButton("预览快照")
        self.btn_preview_snapshot.clicked.connect(self._request_preview_snapshot)
        export_layout.addWidget(self.btn_export_evidence)
        export_layout.addWidget(self.btn_preview_snapshot)
        export_layout.addStretch(1)
        self.left_panel_stack.addWidget(self.export_panel)

        # 设置面板
        self.settings_panel = QGroupBox("设置")
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(8, 14, 8, 8)
        settings_layout.setSpacing(6)
        self.btn_fit_canvas_settings = PushButton("适配画布")
        self.btn_auto_layout_settings = PushButton("自动布局")
        self.btn_reset_zoom_settings = PushButton("重置缩放")
        self.btn_preview_settings = PushButton("预览设置")
        self.btn_preview_settings.clicked.connect(self.preview_settings_requested.emit)
        settings_layout.addWidget(self.btn_fit_canvas_settings)
        settings_layout.addWidget(self.btn_auto_layout_settings)
        settings_layout.addWidget(self.btn_reset_zoom_settings)
        settings_layout.addWidget(self.btn_preview_settings)
        settings_layout.addStretch(1)
        self.left_panel_stack.addWidget(self.settings_panel)

        rail_items = [
            ("project", FluentIcon.FOLDER, "项目 / 数据", 0),
            ("nodes", FluentIcon.APPLICATION, "节点库", 1),
            ("runs", FluentIcon.HISTORY, "运行记录", 2),
            ("tuning", FluentIcon.EDIT, "调参与实验", 3),
            ("validation", FluentIcon.COMPLETED, "验证 / QC", 4),
            ("export", FluentIcon.SHARE, "导出 / Evidence", 5),
            ("settings", FluentIcon.SETTING, "设置", 6),
        ]
        for key, icon, tooltip, index in rail_items:
            add_rail_button(key, icon, tooltip, index)
        rail_layout.addStretch(1)
        self.left_panel_stack.setCurrentIndex(1)
        self.rail_buttons["nodes"].setChecked(True)
        self.workspace_splitter.addWidget(left_sidebar)

        self.step_panel = QWidget()
        self.step_panel.setMinimumWidth(520)
        step_panel_layout = QVBoxLayout(self.step_panel)
        step_panel_layout.setContentsMargins(0, 0, 0, 0)
        step_panel_layout.setSpacing(0)
        self.step_list.hide()
        self.workflow_canvas = WorkflowCanvasView()
        self.workflow_canvas.setToolTip("空白左键拖动画布；滚轮缩放；节点拖动移动；端口拖动连线。")
        step_panel_layout.addWidget(self.workflow_canvas, 1)
        self.workspace_splitter.addWidget(self.step_panel)

        self.btn_add_step = PushButton("添加")
        self.btn_duplicate_step = PushButton("复制")
        self.btn_remove_step = PushButton("删除")
        self.btn_add_step.setToolTip("在当前步骤后插入同阶段默认步骤")
        self.btn_duplicate_step.setToolTip("复制当前步骤及其参数")
        self.btn_remove_step.setToolTip("删除当前步骤")


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

        self.inspector_box = QGroupBox("属性 / 检查")
        self.inspector_box.setMinimumWidth(280)
        self.inspector_box.setMaximumWidth(520)
        inspector_layout = QVBoxLayout(self.inspector_box)
        inspector_layout.setContentsMargins(8, 14, 8, 8)
        inspector_layout.setSpacing(6)
        self.inspector_label = QLabel("未选择节点")
        self.inspector_label.setWordWrap(True)
        self.qc_label = QLabel("QC\n数据尺寸：--\n告警：--\n元数据：--")
        self.qc_label.setWordWrap(True)
        self.qc_label.setProperty("class", "hintText")
        self.export_label = QLabel("Export\n保存：将最新结果写入历史\n证据：导出 Evidence Package\n导出：报告 / 快照 / 数据")
        self.export_label.setWordWrap(True)
        self.export_label.setProperty("class", "hintText")
        self.status_label = QLabel("未运行")
        self.status_label.setProperty("class", "hintText")
        self.run_history_list = QListWidget()
        self.run_history_list.setMaximumHeight(100)
        self.run_history_list.setAlternatingRowColors(True)
        self.run_history_list.setToolTip("点击查看历史运行详情")
        self.run_history_list.itemClicked.connect(self._on_history_item_clicked)
        inspector_layout.addWidget(self.inspector_label)
        inspector_layout.addWidget(self.stage_label)
        inspector_layout.addWidget(self.stage_warning)
        inspector_layout.addWidget(method_row)
        inspector_layout.addWidget(self.param_scroll, 1)
        inspector_layout.addWidget(self.qc_label)
        inspector_layout.addWidget(self.export_label)
        inspector_layout.addWidget(self.status_label)
        inspector_layout.addWidget(QLabel("运行历史"))
        inspector_layout.addWidget(self.run_history_list)
        self.workspace_splitter.addWidget(self.inspector_box)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setStretchFactor(2, 0)
        self.workspace_splitter.setSizes([420, 720, 300])
        self.log_box = self.inspector_box

        self.bottom_drawer = QFrame()
        self.bottom_drawer.setObjectName("workflowBottomDrawer")
        drawer_layout = QVBoxLayout(self.bottom_drawer)
        drawer_layout.setContentsMargins(10, 4, 10, 8)
        drawer_layout.setSpacing(4)
        drawer_header = QHBoxLayout()
        drawer_header.setSpacing(6)
        self.bottom_drawer_toggle = QToolButton()
        self.bottom_drawer_toggle.setText("日志")
        self.bottom_drawer_toggle.setCheckable(True)
        self.bottom_drawer_toggle.setToolTip("展开或收起底部运行抽屉")
        self.bottom_drawer_toggle.clicked.connect(lambda checked: self._set_bottom_drawer_expanded(bool(checked)))
        drawer_header.addWidget(self.bottom_drawer_toggle)
        self.bottom_drawer_buttons: dict[str, QToolButton] = {}
        for key, text in [
            ("logs", "日志"),
            ("validation", "验证"),
            ("qc", "QC / 告警"),
            ("evidence", "Evidence"),
            ("export", "导出"),
        ]:
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked=False, page_key=key: self._select_bottom_drawer_page(page_key, expand=True))
            drawer_header.addWidget(btn)
            self.bottom_drawer_buttons[key] = btn
        drawer_header.addStretch(1)
        # 添加高度控制按钮
        self.btn_height_small = PushButton("小")
        self.btn_height_small.setMaximumWidth(45)
        self.btn_height_small.setToolTip("设置抽屉为小高度")
        self.btn_height_small.clicked.connect(lambda: self._set_bottom_drawer_expanded(True, height_mode="small"))
        self.btn_height_medium = PushButton("中")
        self.btn_height_medium.setMaximumWidth(45)
        self.btn_height_medium.setToolTip("设置抽屉为中等高度")
        self.btn_height_medium.clicked.connect(lambda: self._set_bottom_drawer_expanded(True, height_mode="medium"))
        self.btn_height_large = PushButton("大")
        self.btn_height_large.setMaximumWidth(45)
        self.btn_height_large.setToolTip("设置抽屉为大高度")
        self.btn_height_large.clicked.connect(lambda: self._set_bottom_drawer_expanded(True, height_mode="large"))
        drawer_header.addWidget(self.btn_height_small)
        drawer_header.addWidget(self.btn_height_medium)
        drawer_header.addWidget(self.btn_height_large)
        # 添加清空和复制按钮
        self.btn_clear_log = PushButton("清空")
        self.btn_clear_log.setMaximumWidth(60)
        self.btn_clear_log.clicked.connect(self._clear_current_drawer_content)
        self.btn_copy_log = PushButton("复制")
        self.btn_copy_log.setMaximumWidth(60)
        self.btn_copy_log.clicked.connect(self._copy_current_drawer_content)
        drawer_header.addWidget(self.btn_clear_log)
        drawer_header.addWidget(self.btn_copy_log)
        drawer_layout.addLayout(drawer_header)
        self.bottom_drawer_stack = QStackedWidget()
        self.runtime_log_view = QTextEdit()
        self.runtime_log_view.setReadOnly(True)
        self.workflow_validation_view = QTextEdit()
        self.workflow_validation_view.setReadOnly(True)
        self.workflow_qc_view = QTextEdit("暂无 QC / 告警。")
        self.workflow_qc_view.setReadOnly(True)
        self.workflow_evidence_view = QTextEdit("Evidence 导出结果会显示在这里。")
        self.workflow_evidence_view.setReadOnly(True)
        self.workflow_export_view = QTextEdit("导出图像、数据和报告的结果会显示在这里。")
        self.workflow_export_view.setReadOnly(True)
        for widget in [
            self.runtime_log_view,
            self.workflow_validation_view,
            self.workflow_qc_view,
            self.workflow_evidence_view,
            self.workflow_export_view,
        ]:
            widget.setMinimumHeight(120)
            self.bottom_drawer_stack.addWidget(widget)
        drawer_layout.addWidget(self.bottom_drawer_stack)
        self.bottom_drawer_stack.hide()
        self.bottom_drawer.setMaximumHeight(42)
        self._select_bottom_drawer_page("logs", expand=False)
        outer.addWidget(self.bottom_drawer)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(700)
        self._debounce_timer.timeout.connect(self._emit_realtime_run)

        self.step_list.currentRowChanged.connect(self._on_step_selected)
        self.step_list.order_changed.connect(self._on_order_changed)
        self.workflow_canvas.node_selected.connect(self._select_step_row)
        self.workflow_canvas.node_changed.connect(self._on_canvas_node_changed)
        self.workflow_canvas.run_node_requested.connect(self._run_canvas_node)
        self.workflow_canvas.run_from_node_requested.connect(self._run_from_canvas_node)
        self.workflow_canvas.duplicate_node_requested.connect(self._duplicate_canvas_node)
        self.workflow_canvas.remove_node_requested.connect(self._remove_canvas_node)
        self.workflow_canvas.add_node_requested.connect(self._add_canvas_node)
        self.workflow_canvas.tuning_lab_requested.connect(self._request_tuning_lab_for_row)
        self.workflow_canvas.apply_best_params_requested.connect(self._request_apply_best_params_for_row)
        self.workflow_canvas.benchmark_node_requested.connect(self._request_benchmark_for_row)
        self.workflow_canvas.preview_large_requested.connect(self.preview_large_requested.emit)
        self.workflow_canvas.preview_settings_requested.connect(self.preview_settings_requested.emit)
        self.workflow_canvas.preview_compare_requested.connect(self._request_preview_compare)
        self.workflow_canvas.preview_snapshot_requested.connect(self._request_preview_snapshot)
        self.workflow_canvas.links_changed.connect(self._on_canvas_links_changed)
        self.workflow_canvas.layout_changed.connect(self._on_canvas_layout_changed)
        self.workflow_canvas.zoom_changed.connect(self._update_zoom_label)
        self.btn_toggle_project.clicked.connect(self._toggle_project_panel)
        self.btn_toggle_inspector.clicked.connect(self._toggle_inspector_panel)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.enabled_check.stateChanged.connect(self._on_step_flags_changed)
        self.hidden_check.stateChanged.connect(self._on_step_flags_changed)
        self.realtime_check.stateChanged.connect(self._on_realtime_changed)
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        self.btn_run_all.clicked.connect(self.request_manual_run)
        self.btn_run_from_current.clicked.connect(self.request_run_from_current)
        self.btn_run_selected.clicked.connect(self.request_selected_run)
        self.btn_validate.clicked.connect(self._validate_workflow_ui)
        self.btn_open_tuning_lab.clicked.connect(self.request_tuning_lab_for_current)
        self.btn_save_live.clicked.connect(self.save_live_result_requested)
        self.btn_import_raw.clicked.connect(self.import_raw_requested)
        self.btn_import_rtk.clicked.connect(lambda: self.import_sidecar_requested.emit("rtk"))
        self.btn_import_imu.clicked.connect(lambda: self.import_sidecar_requested.emit("imu"))
        self.btn_import_agl.clicked.connect(lambda: self.import_sidecar_requested.emit("altimeter"))
        self.btn_create_raw_input.clicked.connect(self._create_or_update_raw_input_node)
        self.btn_new_template.clicked.connect(self.new_user_template)
        self.btn_duplicate_template.clicked.connect(self.duplicate_current_template)
        self.btn_save_template.clicked.connect(self.save_current_template)
        self.btn_import_template.clicked.connect(self.import_template)
        self.btn_export_template.clicked.connect(self.export_template)
        self.btn_restore_default.clicked.connect(self.restore_default_template)
        self.btn_add_step.clicked.connect(self.add_step_after_current)
        self.btn_duplicate_step.clicked.connect(self.duplicate_current_step)
        self.btn_remove_step.clicked.connect(self.remove_current_step)
        self.btn_fit_canvas.clicked.connect(self.workflow_canvas.fit_nodes)
        self.btn_auto_layout.clicked.connect(self.workflow_canvas.auto_layout)
        self.btn_reset_zoom.clicked.connect(self.workflow_canvas.reset_zoom)
        self.palette_search.textChanged.connect(self._populate_palette)
        self.palette_list.node_double_clicked.connect(lambda method_id: self._add_canvas_node(method_id, self.workflow_canvas.viewport_scene_center()))
        self.workspace_splitter.splitterMoved.connect(self._save_workspace_state)
        
        # 连接左侧设置面板按钮
        self.btn_fit_canvas_settings.clicked.connect(self.workflow_canvas.fit_nodes)
        self.btn_auto_layout_settings.clicked.connect(self.workflow_canvas.auto_layout)
        self.btn_reset_zoom_settings.clicked.connect(self.workflow_canvas.reset_zoom)
        
        self._populate_palette()
        # 初始化时更新一次调参面板信息
        self._update_tuning_panel_info()

    def _make_text_panel(self, title: str, text: str) -> QGroupBox:
        panel = QGroupBox(title)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 14, 10, 10)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("class", "hintText")
        layout.addWidget(label)
        layout.addStretch(1)
        return panel

    def _make_action_panel(self, title: str, text: str, action: Callable[[], None]) -> QGroupBox:
        panel = QGroupBox(title)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 14, 10, 10)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("class", "hintText")
        button = PushButton(title)
        button.clicked.connect(action)
        layout.addWidget(label)
        layout.addWidget(button)
        layout.addStretch(1)
        return panel

    def _toggle_left_panel(self, key: str) -> None:
        if key not in self.left_panel_keys:
            return
        index = self.left_panel_keys.index(key)
        is_same_visible = self.left_panel_stack.isVisible() and self.left_panel_stack.currentIndex() == index
        self.left_panel_stack.setVisible(not is_same_visible)
        if not is_same_visible:
            self.left_panel_stack.setCurrentIndex(index)
        for panel_key, button in self.rail_buttons.items():
            button.setChecked(panel_key == key and self.left_panel_stack.isVisible())
        self.btn_toggle_project.setChecked(self.left_panel_stack.isVisible())
        self._save_workspace_state()

    def _select_bottom_drawer_page(self, key: str, *, expand: bool) -> None:
        keys = ["logs", "validation", "qc", "evidence", "export"]
        if key not in keys:
            key = "logs"
        self.bottom_drawer_stack.setCurrentIndex(keys.index(key))
        for page_key, button in self.bottom_drawer_buttons.items():
            button.setChecked(page_key == key)
        if expand:
            self._set_bottom_drawer_expanded(True, height_mode="medium")

    def _set_bottom_drawer_expanded(self, expanded: bool, *, height_mode: str = "medium") -> None:
        expanded = bool(expanded)
        self.bottom_drawer_toggle.setChecked(expanded)
        self.bottom_drawer_stack.setVisible(expanded)
        # 小/中/大/错误四档高度 - 符合需求
        max_height = 42  # 默认收起高度
        if expanded:
            # 安全处理 height_mode 参数
            if height_mode == "small":
                max_height = 120  # 小
            elif height_mode == "large":
                max_height = 250  # 大 (原280改为符合需求)
            elif height_mode == "error":
                max_height = 240  # 错误展开高度
            else:  # medium 或其他
                max_height = 160  # 普通展开高度（符合需求）
        self.bottom_drawer.setMaximumHeight(max_height)
        
    def _expand_drawer_for_error(self) -> None:
        """当有错误时，展开底部抽屉到较大高度"""
        self._set_bottom_drawer_expanded(True, height_mode="large")
    
    def _clear_current_drawer_content(self) -> None:
        """清空当前底部抽屉页面的内容"""
        current_widget = self.bottom_drawer_stack.currentWidget()
        if hasattr(current_widget, 'clear'):
            current_widget.clear()
    
    def _copy_current_drawer_content(self) -> None:
        """复制当前底部抽屉页面的内容到剪贴板"""
        from PyQt6.QtWidgets import QApplication
        current_widget = self.bottom_drawer_stack.currentWidget()
        if hasattr(current_widget, 'toPlainText'):
            text = current_widget.toPlainText()
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text)

    def _append_runtime_log(self, text: str) -> None:
        if not hasattr(self, "runtime_log_view"):
            return
        self.runtime_log_view.append(str(text))

    def _update_zoom_label(self, scale: float) -> None:
        self.zoom_label.setText(f"缩放 {int(round(float(scale) * 100))}%")

    def _populate_palette(self) -> None:
        if not hasattr(self, "palette_list"):
            return
        query = self.palette_search.text().strip().lower() if hasattr(self, "palette_search") else ""
        self.palette_list.populate(query)

    def _toggle_project_panel(self, checked: bool) -> None:
        self.left_sidebar.setVisible(True)
        self.left_panel_stack.setVisible(bool(checked))
        if checked and self.left_panel_stack.currentIndex() < 0:
            self.left_panel_stack.setCurrentIndex(0)
        for key, button in self.rail_buttons.items():
            button.setChecked(self.left_panel_stack.isVisible() and self.left_panel_keys[self.left_panel_stack.currentIndex()] == key)
        if checked:
            sizes = self.workspace_splitter.sizes()
            if sizes and sizes[0] <= 8:
                self.workspace_splitter.setSizes([420, max(640, sizes[1] if len(sizes) > 1 else 720), sizes[2] if len(sizes) > 2 else 300])
        self._save_workspace_state()

    def _toggle_inspector_panel(self, checked: bool) -> None:
        self.inspector_box.setVisible(bool(checked))
        if checked:
            sizes = self.workspace_splitter.sizes()
            if sizes and len(sizes) > 2 and sizes[2] <= 8:
                self.workspace_splitter.setSizes([sizes[0] if sizes else 260, max(640, sizes[1] if len(sizes) > 1 else 720), 300])
        self._save_workspace_state()
        
    def _save_workspace_state(self) -> None:
        settings = QSettings("MyGPR", "WorkflowStudio")
        sizes = self.workspace_splitter.sizes()
        settings.setValue("workspace_sizes", sizes)
        settings.setValue("project_panel_visible", self.left_panel_stack.isVisible())
        settings.setValue("left_panel_index", self.left_panel_stack.currentIndex())
        settings.setValue("inspector_panel_visible", self.inspector_box.isVisible())
        settings.sync()
        
    def _restore_workspace_state(self) -> None:
        settings = QSettings("MyGPR", "WorkflowStudio")
        if settings.contains("workspace_sizes"):
            try:
                sizes = settings.value("workspace_sizes")
                if sizes is not None:
                    parsed_sizes = []
                    for s in sizes:
                        try:
                            parsed_sizes.append(int(s))
                        except (ValueError, TypeError):
                            parsed_sizes = []
                            break
                    if len(parsed_sizes) >= 3:
                        self.workspace_splitter.setSizes(parsed_sizes)
            except Exception:
                pass
        if settings.contains("project_panel_visible"):
            try:
                visible = settings.value("project_panel_visible", True, type=bool)
                self.btn_toggle_project.setChecked(visible)
                self.left_sidebar.setVisible(True)
                self.left_panel_stack.setVisible(visible)
            except Exception:
                pass
        if settings.contains("left_panel_index"):
            try:
                index = int(settings.value("left_panel_index", 1))
                if 0 <= index < self.left_panel_stack.count():
                    self.left_panel_stack.setCurrentIndex(index)
            except Exception:
                pass
        if hasattr(self, "rail_buttons"):
            current_key = self.left_panel_keys[self.left_panel_stack.currentIndex()]
            for key, button in self.rail_buttons.items():
                button.setChecked(self.left_panel_stack.isVisible() and key == current_key)
        if settings.contains("inspector_panel_visible"):
            try:
                visible = settings.value("inspector_panel_visible", True, type=bool)
                self.btn_toggle_inspector.setChecked(visible)
                self.inspector_box.setVisible(visible)
            except Exception:
                pass

    def _validate_workflow_ui(self) -> None:
        report, text = self._build_validation_report_text()

        self.status_label.setText(report.summary())
        self._log(text)
        self.workflow_validation_view.setPlainText(text)
        # 有错误时使用 large 高度，警告时使用 medium
        has_errors = bool(report.errors)
        has_warnings = bool(report.warnings) and not has_errors
        self._select_bottom_drawer_page("validation", expand=has_errors or has_warnings)
        if has_errors:
            self._set_bottom_drawer_expanded(True, height_mode="large")
        # 同时切到左侧验证/QC面板，高亮验证图标
        self._toggle_left_panel("validation")

        if hasattr(self, "qc_label") and self.qc_label is not None:
            lines = [
                "QC / Validation",
                f"errors: {len(report.errors)}",
                f"warnings: {len(report.warnings)}",
                f"info: {len(report.infos)}",
            ]
            lines.extend(f"- {issue.code}" for issue in report.issues[:6])
            self.qc_label.setText("\n".join(lines))
            self.workflow_qc_view.setPlainText("\n".join(lines))
        
        # 更新左侧验证面板
        if hasattr(self, "validation_info_label"):
            summary_lines = [
                "验证结果:",
                f"✅ 通过: {len(report.infos)}",
                f"⚠️ 警告: {len(report.warnings)}",
                f"❌ 错误: {len(report.errors)}",
            ]
            if report.issues:
                summary_lines.append("\n最近问题:")
                for issue in report.issues[:3]:
                    summary_lines.append(f"• {issue.code}")
            self.validation_info_label.setText("\n".join(summary_lines))
        
        # 更新 sidecar 状态
        if hasattr(self, "sidecar_status_label"):
            self.sidecar_status_label.setText(self._current_sidecar_status_text())
        
        self.validation_report_requested.emit(text)

    def _current_sidecar_status(self) -> dict[str, bool]:
        return {
            "rtk": bool(self._sidecar_files.get("rtk")),
            "imu": bool(self._sidecar_files.get("imu")),
            "agl": bool(self._sidecar_files.get("altimeter")),
        }

    def _build_validation_report_text(self):
        sidecar_status = self._current_sidecar_status()
        report = validate_workflow_config(
            self.config,
            sidecar_status=sidecar_status,
            execution_mode="order",
        )
        text = to_text_with_suggestions(
            report,
            sidecar_status=sidecar_status,
            execution_mode="order",
        )
        return report, f"{self._graph_order_mismatch_text()}\n{text}"

    def _graph_order_mismatch_text(self) -> str:
        ordered = [method.node_id for method in sorted(self.config.methods, key=lambda item: item.order)]
        expected = {(left, right) for left, right in zip(ordered, ordered[1:])}
        actual = {
            (link.from_node, link.to_node)
            for link in self.config.canvas_links
            if getattr(link, "kind", "data") == "data"
        }
        if actual == expected:
            return "graph/order mismatch：无"
        missing = len(expected - actual)
        extra = len(actual - expected)
        return f"graph/order mismatch：存在（缺少 {missing} 条顺序连接，额外 {extra} 条画布连接）"

    def _on_canvas_links_changed(self, links: object) -> None:
        self.config.canvas_links = list(links) if isinstance(links, list) else []
        self._on_step_selected(self.step_list.currentRow())

    def _on_canvas_layout_changed(self, layout: object) -> None:
        if isinstance(layout, dict):
            self.config.canvas_layout = layout

    def _add_canvas_node(self, method_id: str, scene_pos: QPointF) -> None:
        if method_id not in PROCESSING_METHODS:
            return
        metadata = PROCESSING_METHODS.get(method_id, {})
        category = str(metadata.get("category") or "custom")
        method = WorkflowMethod(
            category=category,
            stage_id=self._stage_for_method(method_id, category),
            method_id=method_id,
            enabled=True,
            hidden=False,
            order=self.step_list.count(),
            params=self._default_params_for(method_id),
        )
        self._sync_order_from_list(rebuild_canvas=False)
        self.config.methods.append(method)
        method.order = len(self.config.methods) - 1
        self.config.canvas_layout.setdefault("nodes", {})[method.node_id] = {
            "x": float(scene_pos.x()),
            "y": float(scene_pos.y()),
            "width": 300,
            "height": 180,
            "collapsed": False,
        }
        self.config.canvas_links = self.workflow_canvas.current_links()
        self._render_steps()
        self.step_list.setCurrentRow(len(self.config.methods) - 1)
        self._queue_realtime_run()

    def _make_link(self, from_node: str, to_node: str):
        from core.workflow_data import WorkflowLink

        return WorkflowLink(from_node, to_node)

    def _stage_for_method(self, method_id: str, category: str) -> str:
        for stage in WORKFLOW_STAGE_DEFINITIONS:
            if method_id in stage.get("candidate_methods", []):
                return str(stage.get("id", ""))
        return self._category_for_stage("", method_id) if category == "custom" else ""

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
        QTimer.singleShot(0, self.workflow_canvas.fit_nodes)

    def _render_steps(self) -> None:
        self.config.ensure_canvas_links()
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for index, method in enumerate(sorted(self.config.methods, key=lambda item: item.order)):
            method.order = index
            item = QListWidgetItem(self._format_step_text(method))
            item.setData(Qt.ItemDataRole.UserRole, method)
            self.step_list.addItem(item)
        self.step_list.blockSignals(False)
        self.workflow_canvas.set_workflow(
            self.config.methods,
            self.config.canvas_links,
            self.config.canvas_layout,
        )
        self.workflow_canvas.set_selected_row(self.step_list.currentRow())
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

    def _node_display_label(self, node_id: str) -> str:
        for index, method in enumerate(self.config.methods):
            if method.node_id == node_id:
                stage = WORKFLOW_STAGE_BY_ID.get(method.stage_id, {})
                category = METHOD_CATEGORIES.get(method.category, {})
                stage_label = (
                    stage.get("label")
                    or category.get("name")
                    or method.category
                    or "节点"
                )
                return f"{index + 1:02d} {stage_label}"
        if node_id == "__workflow_preview__":
            return "B-scan Preview"
        return "--"

    def _on_step_selected(self, row: int) -> None:
        method = self._selected_method()
        self.workflow_canvas.set_selected_row(int(row))
        self._update_step_buttons()
        self._update_tuning_panel_info()
        if method is None:
            self.inspector_label.setText("未选择节点")
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
            state = "隐藏" if method.hidden else ("停用" if not method.enabled else "启用")
            inbound = [
                self._node_display_label(link.from_node)
                for link in self.config.canvas_links
                if link.to_node == method.node_id
            ]
            outbound = [
                self._node_display_label(link.to_node)
                for link in self.config.canvas_links
                if link.from_node == method.node_id
            ]
            self.inspector_label.setText(
                f"当前节点\n{method.order + 1:02d} {stage_label}\n"
                f"算法：{get_method_display_name(method.method_id)}\n"
                f"状态：{state}\n\n"
                f"输入：{', '.join(inbound) if inbound else '--'}\n"
                f"输出：{', '.join(outbound) if outbound else '--'}\n\n"
                f"高级信息\nmethod_id: {method.method_id}"
            )
        finally:
            self._suppress_change = False

    def _render_method_combo(self, method: WorkflowMethod) -> None:
        candidates = candidate_methods_for_workflow_method(method)
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
        update_workflow_method_algorithm(method, str(new_key))
        self._render_method_combo(method)
        self._render_params(method)
        self._on_step_selected(self.step_list.currentRow())
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
            self.workflow_canvas.update_node(row)
            self._update_step_buttons()

    def _on_order_changed(self) -> None:
        self._sync_order_from_list()
        self._queue_realtime_run()

    def _sync_order_from_list(self, *, rebuild_canvas: bool = True) -> None:
        methods = []
        for row in range(self.step_list.count()):
            method = self.step_list.item(row).data(Qt.ItemDataRole.UserRole)
            if isinstance(method, WorkflowMethod):
                method.order = row
                methods.append(method)
                self.step_list.item(row).setText(self._format_step_text(method))
        self.config.methods = methods
        if rebuild_canvas:
            self.config.ensure_canvas_links()
            self.workflow_canvas.set_workflow(
                self.config.methods,
                self.config.canvas_links,
                self.config.canvas_layout,
            )
            self.workflow_canvas.set_selected_row(self.step_list.currentRow())
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
            run_mode="Realtime",
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
            run_mode="Run All",
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
            run_mode="Run Selected",
        )

    def request_run_from_current(self) -> None:
        row = self.step_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "无步骤", "请先选择一个起始步骤。")
            return
        self._sync_order_from_list(rebuild_canvas=False)
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
            run_mode="Run From",
        )

    def _emit_run(
        self,
        methods: list[WorkflowMethod],
        *,
        realtime: bool,
        status: str,
        log_text: str,
        run_mode: str = "",
    ) -> None:
        if not methods:
            self.status_label.setText("没有启用的步骤")
            return
        report, report_text = self._build_validation_report_text()
        self.validation_report_requested.emit(report_text)
        self.workflow_validation_view.setPlainText(report_text)
        if report.errors:
            self._select_bottom_drawer_page("validation", expand=True)
            error_text = "\n".join(
                f"- {issue.code}: {issue.message}"
                for issue in report.errors
            )
            reply = QMessageBox.question(
                self,
                "工作流验证失败",
                f"运行前发现以下错误，是否继续运行？\n\n{error_text}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return
        elif report.warnings:
            self._select_bottom_drawer_page("validation", expand=True)
            self._log("运行前发现警告：")
            for issue in report.warnings:
                self._log(f"  [{issue.code}] {issue.message}")
        run_node_ids = {m.node_id for m in methods}
        for method in self.config.methods:
            if method.hidden or not method.enabled:
                method.status = "skipped"
            elif method.node_id in run_node_ids:
                method.status = "queued"
            elif method.status == "success":
                method.status = "success_stale"
        if methods:
            first_node_id = methods[0].node_id
            for method in self.config.methods:
                if method.node_id == first_node_id:
                    method.status = "running"
                    break
        self.workflow_canvas.refresh_all_nodes()
        self._last_run_methods = [deepcopy(method) for method in methods]
        self.workflow_run_requested.emit(self._last_run_methods, realtime, run_mode)
        self.status_label.setText(status)
        self._log(log_text)

    def get_enabled_methods(self) -> list[WorkflowMethod]:
        self._sync_order_from_list(rebuild_canvas=False)
        return [deepcopy(method) for method in self.config.get_enabled_methods()]

    def _select_step_row(self, row: int) -> None:
        row = int(row)
        if 0 <= row < self.step_list.count():
            self.step_list.setCurrentRow(row)

    def _on_canvas_node_changed(self, row: int) -> None:
        self._select_step_row(row)
        self._refresh_selected_item()
        self._on_step_selected(row)
        self._queue_realtime_run()

    def _run_canvas_node(self, row: int) -> None:
        self._select_step_row(row)
        self.request_selected_run()

    def _run_from_canvas_node(self, row: int) -> None:
        self._select_step_row(row)
        self.request_run_from_current()

    def _duplicate_canvas_node(self, row: int) -> None:
        self._select_step_row(row)
        self.duplicate_current_step()

    def _remove_canvas_node(self, row: int) -> None:
        self._select_step_row(row)
        self.remove_current_step()

    def request_tuning_lab_for_current(self) -> None:
        row = self.step_list.currentRow()
        self._request_tuning_lab_for_row(row)

    def _request_tuning_lab_for_row(self, row: int) -> None:
        self._select_step_row(row)
        method = self._selected_method()
        if method is None:
            QMessageBox.information(self, "无节点", "请先选择一个工作流节点。")
            return
        self.tuning_lab_requested.emit(deepcopy(method))
        self._log(f"调参: {get_method_display_name(method.method_id)}")

    def _request_apply_best_params_for_row(self, row: int) -> None:
        self._select_step_row(row)
        method = self._selected_method()
        if method is None:
            return
        self.tuning_lab_requested.emit(deepcopy(method))
        self._log(f"应用最佳参数: 已打开 {get_method_display_name(method.method_id)} 调参入口")

    def _request_benchmark_for_row(self, row: int) -> None:
        self._select_step_row(row)
        method = self._selected_method()
        if method is None:
            return
        self.tuning_lab_requested.emit(deepcopy(method))
        self._log(f"评估此节点: {get_method_display_name(method.method_id)}")

    def _request_preview_compare(self) -> None:
        if hasattr(self.workflow_canvas, "_preview_data"):
            self.preview_large_requested.emit(
                self.workflow_canvas._preview_data,
                self.workflow_canvas._preview_label
            )
        self._log("Preview: 打开大图/对比查看入口")

    def _request_preview_snapshot(self) -> None:
        self.export_evidence_requested.emit()
        self.workflow_evidence_view.append("已请求 Evidence / 快照导出。")
        self._select_bottom_drawer_page("evidence", expand=True)
        self._log("Preview: 快照导出请使用 Evidence / Export 入口")

    def _create_or_update_raw_input_node(self) -> None:
        shape_text = (
            f"{self._data_shape[0]} samples × {self._data_shape[1]} traces"
            if self._data_shape
            else "--"
        )
        params = {
            "file": self._current_file or "--",
            "shape": shape_text,
            "metadata": self._metadata_status,
            "raw": "loaded" if self._current_file else "missing",
            "rtk": "loaded" if self._sidecar_files.get("rtk") else "missing",
            "imu": "loaded" if self._sidecar_files.get("imu") else "missing",
            "agl": "loaded" if self._sidecar_files.get("altimeter") else "missing",
        }
        existing = next(
            (method for method in self.config.methods if method.method_id == "raw_input"),
            None,
        )
        if existing is None:
            existing = WorkflowMethod(
                category="输入",
                stage_id="raw_input",
                method_id="raw_input",
                enabled=False,
                hidden=False,
                order=0,
                params=params,
                node_id="node_raw_input",
            )
            self.config.methods.insert(0, existing)
            self.config.canvas_layout.setdefault("nodes", {})[existing.node_id] = {
                "x": 40.0,
                "y": 50.0,
                "width": 300,
                "height": 170,
                "collapsed": False,
            }
        else:
            existing.params = params
        for index, method in enumerate(self.config.methods):
            method.order = index
        self.config.canvas_links = self.workflow_canvas.current_links()
        self._render_steps()
        self.step_list.setCurrentRow(self.config.methods.index(existing))
        self._log(f"输入节点已更新：{shape_text}")
        self.status_label.setText("Raw Input 节点已更新")

    def set_project_data_state(
        self,
        *,
        file_path: str | None = None,
        shape: tuple[int, int] | None = None,
        metadata_status: str | None = None,
        sidecar_files: dict[str, str | None] | None = None,
    ) -> None:
        if file_path is not None:
            self._current_file = str(file_path)
        if shape is not None:
            self._data_shape = shape
        if metadata_status is not None:
            self._metadata_status = str(metadata_status)
        if sidecar_files is not None:
            self._sidecar_files = dict(sidecar_files)
        file_text = self._current_file or "--"
        shape_text = (
            f"{self._data_shape[0]} samples × {self._data_shape[1]} traces"
            if self._data_shape
            else "--"
        )
        self.project_file_label.setText(f"当前文件：{file_text}")
        self.project_shape_label.setText(f"数据尺寸：{shape_text}")
        self.project_metadata_label.setText(f"元数据：{self._metadata_status}")
        raw_name = Path(self._current_file).name if self._current_file else "missing"
        self.raw_status_label.setText(f"Raw：{'loaded ' + raw_name if self._current_file else 'missing'}")
        self.rtk_status_label.setText(self._sidecar_status_text("RTK", "rtk"))
        self.imu_status_label.setText(self._sidecar_status_text("IMU", "imu"))
        self.agl_status_label.setText(self._sidecar_status_text("AGL", "altimeter"))
        self.qc_label.setText(
            f"QC\n数据尺寸：{shape_text}\n告警：--\n元数据：{self._metadata_status}"
        )
        if hasattr(self, "workflow_qc_view"):
            self.workflow_qc_view.setPlainText(
                "\n".join(
                    [
                        "Project / Data",
                        f"Raw: {'loaded' if self._current_file else 'missing'}",
                        self._sidecar_status_text("RTK", "rtk"),
                        self._sidecar_status_text("IMU", "imu"),
                        self._sidecar_status_text("AGL", "altimeter"),
                        f"shape: {shape_text}",
                        f"metadata: {self._metadata_status}",
                    ]
                )
            )

    def _sidecar_status_text(self, label: str, key: str) -> str:
        path = self._sidecar_files.get(key)
        return f"{label}：loaded {Path(path).name}" if path else f"{label}：missing"

    def set_data_shape(self, shape: tuple[int, int] | None) -> None:
        self._data_shape = shape
        self.set_project_data_state(shape=shape)
        method = self._selected_method()
        if method is not None:
            self._suppress_change = True
            try:
                self._render_params(method)
            finally:
                self._suppress_change = False

    def set_running(self, message: str) -> None:
        self.status_label.setText(message)

    def apply_best_params_to_node(
        self, node_id: str, params: dict, result: dict, reason: str
    ):
        """将最佳参数应用到指定的工作流节点。"""
        # 查找匹配的方法
        target_method = None
        for method in self.config.methods:
            if method.node_id == node_id:
                target_method = method
                break
        if target_method is None:
            self._log(f"无法找到节点 {node_id} 来应用参数")
            return
        # 更新方法的参数
        target_method.params = dict(params)
        self._log(f"已将最佳参数应用到节点 {target_method.method_id} ({node_id})")
        if reason:
            self._log(f"推荐原因：{reason}")
        # 标记配置已更改
        self._config_changed = True
        # 重新渲染工作流
        self._render_steps()
        # 如果启用了实时预览，触发运行
        self._queue_realtime_run()
        self.status_label.setText(f"已更新 {target_method.method_id} 的参数")
    
    def set_run_result(self, outputs: list[dict[str, Any]], realtime: bool, run_mode: str = "") -> None:
        self._live_result_available = True
        self.btn_save_live.setEnabled(True)
        label = "实时预览完成" if realtime else "工作流运行完成"
        self.status_label.setText(label)
        self._log(f"{label}: {len(outputs)} 步")
        self._select_bottom_drawer_page("logs", expand=not realtime)

        output_by_node_id = {o.get("node_id"): o for o in outputs if o.get("node_id")}
        run_node_ids = set(output_by_node_id.keys())

        for method in self.config.methods:
            if method.hidden or not method.enabled:
                method.status = "skipped"
                continue
            if method.node_id in run_node_ids:
                method.status = "success"
                output = output_by_node_id[method.node_id]
                method.elapsed_ms = output.get("elapsed_ms", 0.0)
                method.error_message = ""
                data = output.get("data")
                method.output_shape = data.shape if data is not None else None
            else:
                if method.status in ("queued", "running"):
                    method.status = "idle"
                elif method.status == "success":
                    method.status = "success_stale"

        outputs_without_node_id = [o for o in outputs if not o.get("node_id")]
        if outputs_without_node_id and hasattr(self, "_last_run_methods") and self._last_run_methods:
            for last_run_method, output in zip(self._last_run_methods, outputs_without_node_id):
                for config_method in self.config.methods:
                    if config_method.node_id == last_run_method.node_id:
                        config_method.status = "success"
                        config_method.elapsed_ms = output.get("elapsed_ms", 0.0)
                        config_method.error_message = ""
                        data = output.get("data")
                        config_method.output_shape = data.shape if data is not None else None
                        break

        self.workflow_canvas.refresh_all_nodes()

        run_record = self._create_run_record(outputs, realtime, run_mode)
        self._run_history.insert(0, run_record)
        self._run_history_index = 0
        self._update_history_list()

        if outputs:
            final_output = outputs[-1]
            final_name = (
                final_output.get("method_name")
                or final_output.get("method_key")
                or "Workflow Output"
            )
            self.workflow_canvas.set_preview_data(
                final_output.get("data"),
                label=f"{final_name} · {label}",
                success=True,
            )
        for index, output in enumerate(outputs, start=1):
            name = output.get("method_name") or output.get("method_key") or f"step-{index}"
            shape = output.get("data").shape if output.get("data") is not None else "--"
            self._log(f"  [{index}] {name} -> {shape}")

    def set_run_error(self, error_message: str, failed_node_id: str = "") -> None:
        self.status_label.setText("运行失败")
        self._live_result_available = False
        self.btn_save_live.setEnabled(False)

        for method in self.config.methods:
            if method.hidden or not method.enabled:
                method.status = "skipped"
                continue
            if failed_node_id and method.node_id == failed_node_id:
                method.status = "failed"
                method.error_message = error_message[:200]
            elif method.status == "success":
                method.status = "success_stale"
            elif method.status == "running":
                method.status = "idle"
            else:
                method.error_message = ""

        self.workflow_canvas.refresh_all_nodes()

        old_preview_data = self.workflow_canvas.get_preview_data()
        if old_preview_data is not None:
            self.workflow_canvas.set_preview_data(
                old_preview_data,
                label="运行失败，显示上次结果",
                success=True,
            )
            self.workflow_canvas.set_stale_preview()
        else:
            self.workflow_canvas.set_preview_data(
                None,
                label="运行失败",
                success=False,
            )
        self._log(f"运行失败: {error_message}")
        self._select_bottom_drawer_page("logs", expand=True)
        # 有运行错误时使用 large 高度
        self._set_bottom_drawer_expanded(True, height_mode="large")
        if failed_node_id:
            self._log(f"失败节点: {failed_node_id}")

        error_record = {
            "index": len(self._run_history) + 1,
            "time": str(__import__("datetime").datetime.now().strftime("%H:%M:%S")),
            "mode": "error",
            "success_count": 0,
            "failed_count": 1,
            "skipped_count": 0,
            "elapsed_ms": 0.0,
            "final_shape": "--",
            "methods": [{"node_id": failed_node_id, "method_id": "--", "status": "failed", "elapsed_ms": 0.0}],
            "error_message": error_message,
            "realtime": False,
        }
        self._run_history.insert(0, error_record)
        self._update_history_list()

    def _log(self, text: str) -> None:
        self._append_runtime_log(str(text))
        self.log_message_requested.emit(str(text))

    def _create_run_record(self, outputs: list[dict[str, Any]], realtime: bool, run_mode: str = "") -> dict[str, Any]:
        from datetime import datetime
        total_elapsed = sum(o.get("elapsed_ms", 0.0) for o in outputs)
        final_shape = "--"
        if outputs and outputs[-1].get("data") is not None:
            try:
                shape = outputs[-1]["data"].shape
                final_shape = f"{shape[1]} x {shape[0]}"
            except Exception:
                pass

        success_count = 0
        failed_count = 0
        skipped_count = 0
        success_stale_count = 0
        for method in self.config.methods:
            s = getattr(method, "status", "idle")
            if s == "success":
                success_count += 1
            elif s == "failed":
                failed_count += 1
            elif s == "skipped":
                skipped_count += 1
            elif s == "success_stale":
                success_stale_count += 1

        if run_mode:
            mode = run_mode
        elif realtime:
            mode = "实时"
        else:
            mode = "Run All"

        record = {
            "index": len(self._run_history) + 1,
            "time": datetime.now().strftime("%H:%M:%S"),
            "mode": mode,
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "success_stale_count": success_stale_count,
            "elapsed_ms": total_elapsed,
            "final_shape": final_shape,
            "methods": [
                {
                    "node_id": o.get("node_id", ""),
                    "method_id": o.get("method_key", ""),
                    "status": "success",
                    "elapsed_ms": o.get("elapsed_ms", 0.0),
                }
                for o in outputs
            ],
            "realtime": realtime,
        }
        return record

    def _update_history_list(self) -> None:
        # 更新 inspector 中的历史列表
        self.run_history_list.clear()
        # 更新左侧运行历史面板
        if hasattr(self, 'left_run_history_list'):
            self.left_run_history_list.clear()
        
        for record in self._run_history[:20]:
            time = record.get("time", "--")
            mode = record.get("mode", "--")
            success = record.get("success_count", 0)
            failed = record.get("failed_count", 0)
            skipped = record.get("skipped_count", 0)
            stale = record.get("success_stale_count", 0)
            elapsed = record.get("elapsed_ms", 0.0)
            shape = record.get("final_shape", "--")

            if record.get("mode") == "error":
                text = f"#{record.get('index', '?')} {time} | 失败 | {record.get('error_message', '')[:30]}"
            else:
                if elapsed >= 1000:
                    elapsed_str = f"{elapsed / 1000:.1f}s"
                else:
                    elapsed_str = f"{elapsed:.0f}ms"
                text = f"#{record.get('index', '?')} {time} | {mode} | OK{success} FAIL{failed} SKIP{skipped} OLD{stale} | {elapsed_str} | {shape}"

            # 添加到 inspector 列表
            item1 = QListWidgetItem(text)
            if failed > 0 or record.get("mode") == "error":
                item1.setBackground(Qt.GlobalColor.red)
                item1.setForeground(Qt.GlobalColor.white)
            self.run_history_list.addItem(item1)
            
            # 同时添加到左侧面板列表
            if hasattr(self, 'left_run_history_list'):
                item2 = QListWidgetItem(text)
                if failed > 0 or record.get("mode") == "error":
                    item2.setBackground(Qt.GlobalColor.red)
                    item2.setForeground(Qt.GlobalColor.white)
                self.left_run_history_list.addItem(item2)

    def _current_sidecar_status_text(self) -> str:
        """生成当前 sidecar 文件状态的文本"""
        sidecar_status = self._current_sidecar_status()
        status_parts = []
        for key, available in sidecar_status.items():
            status = "✅ 可用" if available else "❌ 缺失"
            label_map = {"rtk": "RTK 定位", "imu": "IMU 惯性", "altimeter": "AGL 高度"}
            label = label_map.get(key, key.upper())
            status_parts.append(f"{label}: {status}")
        return "\n".join(status_parts)

    def _update_tuning_panel_info(self) -> None:
        """更新调参面板的信息"""
        if not hasattr(self, 'tuning_node_info'):
            return
        
        selected = self._selected_method()
        if selected:
            display_name = get_method_display_name(selected.method_id)
            self.tuning_node_info.setText(f"当前选中节点: {display_name}")
        else:
            self.tuning_node_info.setText("请先在画布中选择一个节点")

    def _apply_best_params(self) -> None:
        """应用最佳参数（打开调参实验室）"""
        self.request_tuning_lab_for_current()

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        # 从两个列表中查找对应的记录
        row = self.run_history_list.row(item)
        if row < 0 or row >= len(self._run_history):
            if hasattr(self, 'left_run_history_list'):
                row = self.left_run_history_list.row(item)
        
        if 0 <= row < len(self._run_history):
            record = self._run_history[row]
            details = []
            details.append(f"Run #{record.get('index', '?')} - {record.get('time', '--')} - {record.get('mode', '--')}")
            details.append(f"成功: {record.get('success_count', 0)}, 失败: {record.get('failed_count', 0)}, 跳过: {record.get('skipped_count', 0)}")
            details.append(f"总耗时: {record.get('elapsed_ms', 0.0):.1f} ms")
            details.append(f"最终输出: {record.get('final_shape', '--')}")
            details.append("---")
            details.append("节点详情:")
            for m in record.get("methods", []):
                status = m.get("status", "?")
                elapsed = m.get("elapsed_ms", 0.0)
                details.append(f"  [{status}] {m.get('method_id', '?')} ({elapsed:.1f}ms)")
            if record.get("error_message"):
                details.append(f"错误: {record.get('error_message', '')}")

            QMessageBox.information(self, "运行历史详情", "\n".join(details))

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
        payload = method.to_dict()
        payload["node_id"] = ""
        new_step = WorkflowMethod.from_dict(payload)
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
        removed_node_id = method.node_id if method else ""
        self.step_list.takeItem(row)
        self._sync_order_from_list(rebuild_canvas=False)
        if removed_node_id:
            self.config.canvas_links = [
                link
                for link in self.config.canvas_links
                if link.from_node != removed_node_id and link.to_node != removed_node_id
            ]
            self.config.canvas_layout.setdefault("nodes", {}).pop(removed_node_id, None)
        self._render_steps()
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
        if len(methods) > 1:
            left = methods[max(0, row - 1)]
            right = methods[min(len(methods) - 1, row + 1)]
            self.config.canvas_links = [
                link
                for link in self.config.canvas_links
                if not (link.from_node == left.node_id and link.to_node == right.node_id)
            ]
            if row > 0:
                self.config.canvas_links.append(self._make_link(left.node_id, method.node_id))
            if row + 1 < len(methods):
                self.config.canvas_links.append(self._make_link(method.node_id, right.node_id))
        self._render_steps()
        self.step_list.setCurrentRow(row)
        self.workflow_canvas.set_selected_row(row)
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

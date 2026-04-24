#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中间参数编辑区"""

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
)
from qfluentwidgets import (
    PushButton,
    PrimaryPushButton,
    FluentIcon,
    ScrollArea,
)

from core.methods_registry import PROCESSING_METHODS, get_method_display_name


class ParamEditorPanel(QWidget):
    """中间参数编辑区"""

    # 信号
    run_requested = pyqtSignal()  # 请求运行
    defaults_requested = pyqtSignal()  # 恢复默认
    params_changed = pyqtSignal()  # 参数已改变（用于实时预览）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_method_id = None
        self.param_widgets = {}  # 参数名 -> 控件
        self.default_params = {}  # 默认参数值

        # 实时预览防抖定时器
        from PyQt6.QtCore import QTimer

        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(500)  # 500ms 防抖
        self._preview_timer.timeout.connect(self.params_changed.emit)

        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("workbenchParamPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # ========== 方法头信息 ==========
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.method_name_label = QLabel("请选择一个方法")
        self.method_name_label.setProperty("class", "titleMedium")
        header_layout.addWidget(self.method_name_label)

        self.method_desc_label = QLabel("从左侧方法树选择一个处理方法")
        self.method_desc_label.setWordWrap(True)
        self.method_desc_label.setProperty("class", "textSecondary")
        header_layout.addWidget(self.method_desc_label)

        layout.addWidget(header)

        # ========== 输入来源选择 ==========
        source_group = QGroupBox("输入数据来源")
        source_group.setObjectName("workbenchSourceCard")
        source_layout = QVBoxLayout(source_group)
        source_layout.setContentsMargins(10, 14, 10, 10)
        source_layout.setSpacing(8)

        # 单选按钮行
        radio_row = QWidget()
        radio_layout = QHBoxLayout(radio_row)
        radio_layout.setContentsMargins(0, 0, 0, 0)
        radio_layout.setSpacing(10)

        self.source_button_group = QButtonGroup(self)

        self.radio_raw = QRadioButton("原始数据")
        self.radio_raw.setChecked(True)
        self.radio_raw.setToolTip("从原始加载的数据开始处理")
        self.source_button_group.addButton(self.radio_raw, 0)
        radio_layout.addWidget(self.radio_raw)

        self.radio_current = QRadioButton("当前结果")
        self.radio_current.setToolTip("从上一次处理结果继续")
        self.source_button_group.addButton(self.radio_current, 1)
        radio_layout.addWidget(self.radio_current)

        self.radio_history = QRadioButton("历史结果")
        self.radio_history.setToolTip("从历史结果中选择")
        self.source_button_group.addButton(self.radio_history, 2)
        radio_layout.addWidget(self.radio_history)

        radio_layout.addStretch()
        source_layout.addWidget(radio_row)

        # 历史结果下拉框
        from qfluentwidgets import ComboBox

        self.result_combo = ComboBox()
        self.result_combo.setFixedHeight(32)
        self.result_combo.setEnabled(False)
        self.result_combo.currentIndexChanged.connect(self._on_result_selected)
        source_layout.addWidget(self.result_combo)

        # 当前输入源信息标签
        self.source_info_label = QLabel("当前: 原始数据")
        self.source_info_label.setProperty("class", "hintText")
        source_layout.addWidget(self.source_info_label)

        # 连接信号，更新输入源标签
        self.source_button_group.buttonClicked.connect(self._update_source_info)
        self.source_button_group.buttonClicked.connect(
            lambda _: self.params_changed.emit()
        )

        layout.addWidget(source_group)

        # ========== 参数表单区 ==========
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)

        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)
        self.param_layout.setSpacing(6)

        scroll.setWidget(self.param_container)
        layout.addWidget(scroll, stretch=1)

        # ========== 操作按钮（三层布局）==========
        action_group = QWidget()
        action_group.setObjectName("workbenchActionPanel")
        action_layout = QVBoxLayout(action_group)
        action_layout.setSpacing(10)
        action_layout.setContentsMargins(0, 0, 0, 0)

        # 第一层：主操作按钮
        primary_group = QWidget()
        primary_layout = QVBoxLayout(primary_group)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        primary_layout.setSpacing(6)

        self.btn_run = PrimaryPushButton("▶ 应用当前方法")
        self.btn_run.setFixedHeight(42)
        self.btn_run.clicked.connect(self.run_requested.emit)
        primary_layout.addWidget(self.btn_run)

        action_layout.addWidget(primary_group)

        # 第二层：次操作按钮
        secondary_group = QWidget()
        secondary_layout = QHBoxLayout(secondary_group)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(8)

        self.btn_defaults = PushButton("🔄 恢复默认")
        self.btn_defaults.setFixedHeight(32)
        self.btn_defaults.clicked.connect(self._restore_defaults)
        secondary_layout.addWidget(self.btn_defaults)

        self.btn_favorite = PushButton("⭐ 收藏")
        self.btn_favorite.setFixedHeight(32)
        self.btn_favorite.clicked.connect(self._favorite_current)
        secondary_layout.addWidget(self.btn_favorite)

        action_layout.addWidget(secondary_group)

        # 第三层：收藏列表
        favorites_group = QGroupBox("收藏的参数组")
        favorites_group.setObjectName("workbenchFavoritesCard")
        favorites_layout = QVBoxLayout(favorites_group)
        favorites_layout.setContentsMargins(8, 16, 8, 8)
        favorites_layout.setSpacing(6)

        self.favorites_list = QWidget()
        self.favorites_list_layout = QVBoxLayout(self.favorites_list)
        self.favorites_list_layout.setSpacing(3)
        self.favorites_list_layout.setContentsMargins(0, 0, 0, 0)

        scroll_favorites = ScrollArea()
        scroll_favorites.setWidgetResizable(True)
        scroll_favorites.setWidget(self.favorites_list)
        scroll_favorites.setMaximumHeight(150)

        favorites_layout.addWidget(scroll_favorites)

        action_layout.addWidget(favorites_group, stretch=1)

        layout.addWidget(action_group)

    def load_method(self, method_id: str):
        """加载方法参数"""
        self.current_method_id = method_id

        method_info = PROCESSING_METHODS.get(method_id)
        if not method_info:
            self.method_name_label.setText(f"未知方法: {method_id}")
            self.method_desc_label.setText("")
            self._clear_params()
            return

        # 更新头信息
        name = get_method_display_name(method_id)
        self.method_name_label.setText(name)

        # 方法类型和描述
        method_type = method_info.get("type", "unknown")
        type_text = "CSV处理" if method_type == "core" else "数组处理"
        desc = f"类型: {type_text}"
        if "description" in method_info:
            desc += f" | {method_info['description']}"
        self.method_desc_label.setText(desc)

        # 清除旧参数
        self._clear_params()
        self.param_widgets.clear()
        self.default_params.clear()

        # 创建参数控件
        params = method_info.get("params", [])
        if not params:
            no_param_label = QLabel("此方法无需配置参数")
            no_param_label.setProperty("class", "hintText")
            no_param_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.param_layout.addWidget(no_param_label)
            return

        for param in params:
            param_name = param.get("name", "")
            param_label = param.get("label", param_name)
            param_type = param.get("type", "float")
            default = param.get("default", 0)
            min_val = param.get("min")
            max_val = param.get("max")
            tooltip = param.get("tooltip", "")

            # 保存默认值
            self.default_params[param_name] = default

            # 创建控件
            widget = self._create_param_widget(param_type, default, min_val, max_val)
            if widget:
                widget.setToolTip(tooltip)
                self.param_widgets[param_name] = widget

                # 添加到表单
                row = QWidget()
                row_layout = QVBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)

                label = QLabel(param_label)
                label.setWordWrap(True)
                label.setToolTip(tooltip)
                row_layout.addWidget(label)

                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setMinimumHeight(32)
                    widget.setMaximumWidth(180)
                row_layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignLeft)

                self.param_layout.addWidget(row)

        self.param_layout.addStretch()

    def _create_param_widget(self, param_type: str, default, min_val, max_val):
        """创建参数控件"""
        if param_type == "int":
            spin = QSpinBox()
            spin.setRange(
                int(min_val) if min_val is not None else -99999,
                int(max_val) if max_val is not None else 99999,
            )
            spin.setValue(int(default) if default is not None else 0)
            # 连接信号，参数改变时触发预览
            spin.valueChanged.connect(self._on_param_value_changed)
            return spin

        elif param_type == "float":
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(
                float(min_val) if min_val is not None else -99999.0,
                float(max_val) if max_val is not None else 99999.0,
            )
            spin.setValue(float(default) if default is not None else 0.0)
            # 连接信号，参数改变时触发预览
            spin.valueChanged.connect(self._on_param_value_changed)
            return spin

        elif param_type == "bool":
            check = QCheckBox()
            check.setChecked(bool(default) if default is not None else False)
            # 连接信号，参数改变时触发预览
            check.toggled.connect(self._on_param_value_changed)
            return check

        elif param_type == "str":
            edit = QLineEdit(str(default) if default is not None else "")
            edit.textChanged.connect(self._on_param_value_changed)
            return edit

        return None

    def _on_param_value_changed(self, *args):
        """参数值改变，启动防抖定时器"""
        self._preview_timer.start()

    def cancel_pending_preview(self):
        """取消尚未触发的参数预览请求。"""
        self._preview_timer.stop()

    def _clear_params(self):
        """清除参数控件"""
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _restore_defaults(self):
        """恢复默认参数"""
        for param_name, widget in self.param_widgets.items():
            if param_name in self.default_params:
                default = self.default_params[param_name]
                if isinstance(widget, QSpinBox):
                    widget.setValue(int(default) if default is not None else 0)
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(default) if default is not None else 0.0)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(default) if default is not None else False)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(default) if default is not None else "")

    def set_current_params(self, params: dict):
        """批量写入当前方法参数。"""
        changed = False
        for param_name, value in (params or {}).items():
            if param_name not in self.param_widgets:
                continue
            widget = self.param_widgets[param_name]
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value) if value is not None else 0)
                changed = True
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value) if value is not None else 0.0)
                changed = True
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value) if value is not None else False)
                changed = True
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
                changed = True
        if changed:
            self.params_changed.emit()

    def get_current_params(self) -> dict:
        """获取当前参数值"""
        params = {}
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                params[param_name] = widget.text().strip()
        return params

    def get_input_source(self) -> str:
        """获取输入来源"""
        if self.radio_raw.isChecked():
            return "raw"
        if self.radio_history.isChecked():
            return "history"
        return "current"

    def set_input_source(self, source: str):
        """主动设置输入来源。"""
        if source == "history" and self.radio_history.isEnabled():
            self.radio_history.setChecked(True)
            self._update_source_info(self.radio_history)
        elif source == "current" and self.radio_current.isEnabled():
            self.radio_current.setChecked(True)
            self._update_source_info(self.radio_current)
        else:
            self.radio_raw.setChecked(True)
            self._update_source_info(self.radio_raw)

    def _favorite_current(self):
        """收藏当前参数组"""
        if self.current_method_id is None:
            return

        params = self.get_current_params()

        # 发送收藏信号
        if hasattr(self, "parent_window") and hasattr(
            self.parent_window, "add_to_favorites"
        ):
            self.parent_window.add_to_favorites(self.current_method_id, params)

    def _load_favorite(self, favorite_data: dict):
        """加载收藏的参数组"""
        method_id = favorite_data.get("method_id")
        params = favorite_data.get("params", {})

        if method_id != self.current_method_id:
            # 需要先加载方法
            if hasattr(self, "parent_window") and hasattr(
                self.parent_window, "select_method"
            ):
                self.parent_window.select_method(method_id)

        # 设置参数值
        for param_name, value in params.items():
            if param_name in self.param_widgets:
                widget = self.param_widgets[param_name]
                if isinstance(widget, QSpinBox):
                    widget.setValue(int(value) if value is not None else 0)
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value) if value is not None else 0.0)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value) if value is not None else False)

    def update_favorites_list(self, favorites: list):
        """更新收藏列表显示"""
        # 清空现有列表
        while self.favorites_list_layout.count():
            item = self.favorites_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 添加收藏项
        for fav in favorites:
            fav_widget = QWidget()
            fav_layout = QHBoxLayout(fav_widget)
            fav_layout.setContentsMargins(4, 2, 4, 2)

            # 收藏名称
            name_label = QLabel(fav.get("name", "未命名"))
            name_label.setToolTip(f"方法: {fav.get('method_id', '未知')}")
            fav_layout.addWidget(name_label)

            # 使用次数
            count = fav.get("used_count", 0)
            if count > 0:
                count_label = QLabel(f"({count}次)")
                count_label.setProperty("class", "hintText")
                fav_layout.addWidget(count_label)

            fav_layout.addStretch()

            # 加载按钮
            btn_load = PushButton("加载")
            btn_load.setFixedHeight(24)
            btn_load.setFixedWidth(60)
            btn_load.clicked.connect(lambda checked, f=fav: self._load_favorite(f))
            fav_layout.addWidget(btn_load)

            self.favorites_list_layout.addWidget(fav_widget)

        if not favorites:
            no_fav_label = QLabel("暂无收藏")
            no_fav_label.setProperty("class", "hintText")
            no_fav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.favorites_list_layout.addWidget(no_fav_label)

    def _update_source_info(self, button):
        """更新输入源信息标签"""
        if button == self.radio_raw:
            self.source_info_label.setText("当前: 原始数据")
            self.source_info_label.setProperty("sourceState", "raw")
            self.result_combo.setEnabled(False)
        elif button == self.radio_current:
            self.source_info_label.setText("当前: 当前结果（基于上一次处理）")
            self.source_info_label.setProperty("sourceState", "current")
            self.result_combo.setEnabled(False)
        elif button == self.radio_history:
            # 启用下拉框
            self.result_combo.setEnabled(True)
            # 更新标签
            index = self.result_combo.currentIndex()
            if index >= 0:
                name = self.result_combo.currentText()
                self.source_info_label.setText(f"当前: {name}")
                self.source_info_label.setProperty("sourceState", "history")

        self.source_info_label.style().unpolish(self.source_info_label)
        self.source_info_label.style().polish(self.source_info_label)

    def set_buttons_for_no_data(self):
        """未加载数据时的按钮状态"""
        self.btn_run.setEnabled(False)
        self.btn_run.setText("▶ 请先加载数据")

    def set_buttons_for_no_method(self):
        """未选择方法时的按钮状态"""
        self.btn_run.setEnabled(False)
        self.btn_run.setText("▶ 请选择方法")

    def set_buttons_for_ready(self):
        """准备就绪的按钮状态"""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ 应用当前方法")

    def set_buttons_for_running(self):
        """运行中的按钮状态"""
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 处理中...")

    def set_buttons_for_preview(self):
        """有当前预览结果时的按钮状态。"""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ 应用当前方法")

    def set_buttons_for_committed_result(self):
        """已有正式结果，但当前没有新预览时的按钮状态"""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ 应用当前方法")

    def _on_result_selected(self, index: int):
        """历史结果被选择"""
        if index < 0:
            return

        # 结果列表顺序与 all_results 保持一致，直接用 index 即可
        if hasattr(self, "parent_window") and hasattr(
            self.parent_window, "select_history_result_by_index"
        ):
            self.parent_window.select_history_result_by_index(index)

    def update_result_list(self, results: list):
        """更新历史结果列表"""
        current_index = (
            self.result_combo.currentIndex() if self.result_combo.count() > 0 else -1
        )
        self.result_combo.clear()
        for i, (name, data) in enumerate(results):
            self.result_combo.addItem(name)

        if 0 <= current_index < self.result_combo.count():
            self.result_combo.setCurrentIndex(current_index)

        # 启用/禁用下拉框
        self.result_combo.setEnabled(
            len(results) > 0 and self.radio_history.isChecked()
        )

    def set_input_source_enabled(self, has_current: bool):
        """设置当前结果选项是否可用"""
        self.radio_current.setEnabled(has_current)
        self.radio_history.setEnabled(has_current)
        self.result_combo.setEnabled(has_current and self.radio_history.isChecked())
        if not has_current:
            self.set_input_source("raw")

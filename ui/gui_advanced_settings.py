#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 显示与对比页面 — 子标签页版（方案B）。"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QGroupBox,
    QCheckBox,
    QLineEdit,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
    QFrame,
    QStackedWidget,
)
from qfluentwidgets import PushButton, FluentIcon, SegmentedWidget

from core.page_operation_contract import get_page_contract


class AdvancedSettingsPage(QWidget):
    """显示与对比页面 — 顶部 SegmentedWidget + QStackedWidget。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.operation_contract = get_page_contract("display")
        self.allowed_operation_types = self.operation_contract.allowed_operation_types
        self.mutates_data = self.operation_contract.mutates_data
        self.setup_ui()

    def setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(12)

        # ========== 标题区 ==========
        title = QLabel("显示与对比")
        title.setProperty("class", "sectionTitle")
        outer_layout.addWidget(title)

        page_hint = QLabel(
            "本页集中控制主图显示、双图对比、聚焦裁剪与增强辅助，不改变正式处理结果。"
        )
        page_hint.setWordWrap(True)
        page_hint.setProperty("class", "hintText")
        outer_layout.addWidget(page_hint)

        compact_hint = QLabel("显示设置只改变主图呈现方式，不改变数组处理结果。")
        compact_hint.setWordWrap(True)
        compact_hint.setProperty("class", "hintText")
        outer_layout.addWidget(compact_hint)

        # ========== 顶部标签栏 ==========
        self.segmented = SegmentedWidget(self)
        self.segmented.addItem("mode", "模式")
        self.segmented.addItem("core", "色图")
        self.segmented.addItem("interact", "交互")
        self.segmented.addItem("enhance", "增强")
        outer_layout.addWidget(self.segmented)

        # ========== 内容栈 ==========
        self.stack = QStackedWidget(self)
        outer_layout.addWidget(self.stack, stretch=1)

        # ---- 页面1: 显示模式 ----
        self.page_mode = self._build_mode_page()
        self.stack.addWidget(self.page_mode)

        # ---- 页面2: 核心显示 ----
        self.page_core = self._build_core_page()
        self.stack.addWidget(self.page_core)

        # ---- 页面3: 聚焦交互 ----
        self.page_interact = self._build_interact_page()
        self.stack.addWidget(self.page_interact)

        # ---- 页面4: 显示增强 ----
        self.page_enhance = self._build_enhance_page()
        self.stack.addWidget(self.page_enhance)

        # 默认显示模式页
        self.segmented.setCurrentItem("mode")
        self.stack.setCurrentIndex(0)

        # ========== 内部兼容性控件（不显示，但保留给 app_qt.py 访问）==========
        self.compare_var = QCheckBox()
        self.compare_var.setChecked(False)
        self.diff_var = QCheckBox()
        self.diff_var.setChecked(False)
        self.slider_compare_var = QCheckBox()
        self.slider_compare_var.setChecked(False)
        self.compare_controls_row = self.compare_select_box
        self.compare_controls_row.setVisible(False)

        # ========== 信号连接 ==========
        self.segmented.currentItemChanged.connect(self._on_segment_changed)
        self.display_mode_group.buttonToggled.connect(self._on_display_mode_changed)
        self.view_style_combo.currentIndexChanged.connect(
            self._refresh_compare_select_visibility
        )
        self.view_style_combo.currentIndexChanged.connect(
            self._on_view_style_changed
        )
        self._refresh_view_style_ui_state()

    def _build_mode_page(self):
        """构建显示模式页面。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(16)

        mode_box = QGroupBox("显示模式")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setSpacing(10)

        self.display_mode_group = QButtonGroup(self)

        self.mode_single = QRadioButton("单图")
        self.mode_single.setChecked(True)
        self.mode_single.setToolTip("显示当前处理结果的单图")
        self.display_mode_group.addButton(self.mode_single, 0)
        mode_layout.addWidget(self.mode_single)

        self.mode_compare = QRadioButton("双视图对比")
        self.mode_compare.setToolTip("并排显示两个处理阶段的图像")
        self.display_mode_group.addButton(self.mode_compare, 1)
        mode_layout.addWidget(self.mode_compare)

        self.mode_diff = QRadioButton("差异图")
        self.mode_diff.setToolTip("显示两图差值")
        self.display_mode_group.addButton(self.mode_diff, 2)
        mode_layout.addWidget(self.mode_diff)

        self.mode_slider = QRadioButton("滑动对比")
        self.mode_slider.setToolTip("用拖动分隔线的方式查看两份结果")
        self.display_mode_group.addButton(self.mode_slider, 3)
        mode_layout.addWidget(self.mode_slider)

        layout.addWidget(mode_box)

        self.single_select_box = QGroupBox("单图查看")
        single_select_layout = QVBoxLayout(self.single_select_box)
        single_select_layout.setSpacing(8)

        single_row = QWidget()
        single_row_l = QHBoxLayout(single_row)
        single_row_l.setContentsMargins(0, 0, 0, 0)
        single_row_l.setSpacing(6)
        single_row_l.addWidget(QLabel("图像"))
        self.single_view_combo = QComboBox()
        self.single_view_combo.setToolTip("选择单图模式要查看的原始数据、步骤结果或当前结果")
        single_row_l.addWidget(self.single_view_combo)
        single_select_layout.addWidget(single_row)
        layout.addWidget(self.single_select_box)

        self.compare_select_box = QGroupBox("对比选择")
        compare_select_layout = QVBoxLayout(self.compare_select_box)
        compare_select_layout.setSpacing(8)

        left_row = QWidget()
        left_row_l = QHBoxLayout(left_row)
        left_row_l.setContentsMargins(0, 0, 0, 0)
        left_row_l.setSpacing(6)
        left_row_l.addWidget(QLabel("左图"))
        self.compare_left_combo = QComboBox()
        self.compare_left_combo.setToolTip("选择左侧对比图")
        left_row_l.addWidget(self.compare_left_combo)
        compare_select_layout.addWidget(left_row)

        right_row = QWidget()
        right_row_l = QHBoxLayout(right_row)
        right_row_l.setContentsMargins(0, 0, 0, 0)
        right_row_l.setSpacing(6)
        right_row_l.addWidget(QLabel("右图"))
        self.compare_right_combo = QComboBox()
        self.compare_right_combo.setToolTip("选择右侧对比图")
        right_row_l.addWidget(self.compare_right_combo)
        compare_select_layout.addWidget(right_row)

        self.compare_select_box.setVisible(False)
        layout.addWidget(self.compare_select_box)
        layout.addStretch(1)

        scroll.setWidget(content)
        return scroll

    def _build_core_page(self):
        """构建核心显示页面。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(16)

        view_box = QGroupBox("核心显示")
        view_layout = QVBoxLayout(view_box)
        view_layout.setSpacing(12)

        render_row = QWidget()
        render_l = QHBoxLayout(render_row)
        render_l.setContentsMargins(0, 0, 0, 0)
        render_l.setSpacing(12)
        render_l.addWidget(QLabel("显示形式"))
        self.view_style_combo = QComboBox()
        self.view_style_combo.addItem("图像", "image")
        self.view_style_combo.addItem("摆动图", "wiggle")
        self.view_style_combo.setToolTip("切换普通图像显示和摆动图显示")
        render_l.addWidget(self.view_style_combo)
        render_l.addSpacing(12)
        render_l.addWidget(QLabel("色图"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.setToolTip("选择色彩映射方案")
        self.cmap_combo.addItems(
            ["gray", "viridis", "plasma", "inferno", "magma", "jet", "seismic"]
        )
        self.cmap_combo.setCurrentText("gray")
        render_l.addWidget(self.cmap_combo)
        self.btn_toggle_theme = PushButton(FluentIcon.CONSTRACT, "切换深浅主题")
        self.btn_toggle_theme.setToolTip("切换全局深色/浅色主题")
        render_l.addWidget(self.btn_toggle_theme)
        render_l.addStretch(1)
        view_layout.addWidget(render_row)

        opts_row = QWidget()
        opts_l = QHBoxLayout(opts_row)
        opts_l.setContentsMargins(0, 0, 0, 0)
        opts_l.setSpacing(24)
        self.cmap_invert_var = QCheckBox("反转色图")
        self.cmap_invert_var.setToolTip("反转当前色图")
        self.show_cbar_var = QCheckBox("显示色标")
        self.show_cbar_var.setToolTip("在图像右侧显示色标")
        self.show_grid_var = QCheckBox("显示网格")
        self.show_grid_var.setToolTip("在图像上叠加参考网格")
        self.show_physical_y_axis_var = QCheckBox("显示物理纵轴（时间/深度）")
        self.show_physical_y_axis_var.setChecked(False)
        self.show_physical_y_axis_var.setToolTip(
            "默认按采样索引显示纵轴；开启后才使用 total_time_ns、深度或高程头信息显示物理纵轴。"
        )
        self.show_physical_x_axis_var = QCheckBox("显示物理横轴（距离）")
        self.show_physical_x_axis_var.setChecked(False)
        self.show_physical_x_axis_var.setToolTip(
            "默认按道索引显示横轴；开启后才使用 trace_distance_m 或 trace_interval_m 显示距离。"
        )
        opts_l.addWidget(self.cmap_invert_var)
        opts_l.addWidget(self.show_cbar_var)
        opts_l.addWidget(self.show_grid_var)
        opts_l.addWidget(self.show_physical_x_axis_var)
        opts_l.addWidget(self.show_physical_y_axis_var)
        opts_l.addStretch(1)
        view_layout.addWidget(opts_row)

        style_hint = QLabel(
            "默认横轴为道索引、纵轴为采样索引，不做显示层距离或时间/深度换算；需要物理坐标判读时，再开启对应物理轴。"
        )
        style_hint.setWordWrap(True)
        style_hint.setProperty("class", "hintText")
        view_layout.addWidget(style_hint)

        style_hint_2 = QLabel("摆动图适合看同相轴和波形结构；滑动对比可在主图中直接拖动分隔线，适合快速判读前后差异。")
        style_hint_2.setWordWrap(True)
        style_hint_2.setProperty("class", "hintText")
        view_layout.addWidget(style_hint_2)

        self.wiggle_sampling_hint = QLabel("Wiggle 显示为抽样显示，仅用于显示，不改变数据。")
        self.wiggle_sampling_hint.setWordWrap(True)
        self.wiggle_sampling_hint.setProperty("class", "hintText")
        view_layout.addWidget(self.wiggle_sampling_hint)

        layout.addWidget(view_box)
        layout.addStretch(1)

        scroll.setWidget(content)
        return scroll

    def _build_interact_page(self):
        """构建聚焦交互页面。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(16)

        # 裁剪
        crop_box = QGroupBox("聚焦裁剪")
        crop_layout = QVBoxLayout(crop_box)
        crop_layout.setSpacing(10)

        self.crop_enable_var = QCheckBox("启用聚焦裁剪")
        self.crop_enable_var.setToolTip("启用时间和距离范围裁剪")
        crop_layout.addWidget(self.crop_enable_var)

        crop_hint = QLabel(
            "时间优先按 ns 解释；若缺少头信息则按采样索引解释。距离优先按 m 解释；若缺少头信息则按道索引解释。"
        )
        crop_hint.setWordWrap(True)
        crop_hint.setProperty("class", "hintText")
        crop_layout.addWidget(crop_hint)

        self.time_start_edit = QLineEdit()
        self.time_start_edit.setToolTip("时间起始值（ns 或采样索引）")
        self.time_start_edit.setPlaceholderText("如 5.0 或 120")
        self.time_end_edit = QLineEdit()
        self.time_end_edit.setToolTip("时间结束值（ns 或采样索引）")
        self.time_end_edit.setPlaceholderText("如 45.0 或 800")
        self.dist_start_edit = QLineEdit()
        self.dist_start_edit.setToolTip("距离起始值（m 或道索引）")
        self.dist_start_edit.setPlaceholderText("如 0.5 或 20")
        self.dist_end_edit = QLineEdit()
        self.dist_end_edit.setToolTip("距离结束值（m 或道索引）")
        self.dist_end_edit.setPlaceholderText("如 3.2 或 180")
        crop_layout.addLayout(
            self._pair_row("时间起", self.time_start_edit, "止", self.time_end_edit)
        )
        crop_layout.addLayout(
            self._pair_row("距离起", self.dist_start_edit, "止", self.dist_end_edit)
        )

        crop_btn_row = QWidget()
        crop_btn_l = QHBoxLayout(crop_btn_row)
        crop_btn_l.setContentsMargins(0, 0, 0, 0)
        crop_btn_l.setSpacing(8)
        self.btn_apply_crop = PushButton(FluentIcon.CLIPPING_TOOL, "应用裁剪")
        self.btn_apply_crop.setToolTip("按当前裁剪范围刷新显示")
        self.btn_reset_crop = PushButton(FluentIcon.CANCEL, "重置裁剪")
        self.btn_reset_crop.setToolTip("恢复完整显示范围")
        crop_btn_l.addWidget(self.btn_apply_crop)
        crop_btn_l.addWidget(self.btn_reset_crop)
        crop_btn_l.addStretch(1)
        crop_layout.addWidget(crop_btn_row)

        layout.addWidget(crop_box)

        # ROI
        roi_box = QGroupBox("ROI 状态")
        roi_layout = QVBoxLayout(roi_box)
        roi_layout.setSpacing(10)

        roi_row = QWidget()
        roi_row_layout = QHBoxLayout(roi_row)
        roi_row_layout.setContentsMargins(0, 0, 0, 0)
        roi_row_layout.setSpacing(8)
        self.roi_status_label = QLabel("手动 ROI: 未设置")
        self.roi_status_label.setProperty("class", "hintText")
        self.btn_clear_manual_roi = PushButton(FluentIcon.CANCEL, "清除 ROI")
        self.btn_clear_manual_roi.setEnabled(False)
        self.btn_clear_manual_roi.setToolTip("清除当前手动框选 ROI")
        roi_row_layout.addWidget(self.roi_status_label)
        roi_row_layout.addStretch(1)
        roi_row_layout.addWidget(self.btn_clear_manual_roi)
        roi_layout.addWidget(roi_row)

        layout.addWidget(roi_box)
        layout.addStretch(1)

        scroll.setWidget(content)
        return scroll

    def _build_enhance_page(self):
        """构建显示增强页面。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(16)

        # 显示增强
        enhance_box = QGroupBox("显示增强")
        enhance_layout = QVBoxLayout(enhance_box)
        enhance_layout.setSpacing(10)

        enhance_hint = QLabel("仅调整当前主图的显示拉伸、归一化和对比度，不改变处理数组。")
        enhance_hint.setWordWrap(True)
        enhance_hint.setProperty("class", "hintText")
        enhance_layout.addWidget(enhance_hint)

        self.symmetric_var = QCheckBox("对称灰度拉伸（以零为中心）")
        self.symmetric_var.setToolTip("适合查看正负振幅对称的雷达数据")
        self.auto_contrast_var = QCheckBox("自动对比度（裁剪极值）")
        self.auto_contrast_var.setChecked(False)
        self.auto_contrast_var.setToolTip("自动裁剪极端值，增强主体反射")
        self.percentile_var = QCheckBox("百分位拉伸")
        self.percentile_var.setToolTip("基于百分位数拉伸对比度")
        enhance_layout.addWidget(self.symmetric_var)
        enhance_layout.addWidget(self.auto_contrast_var)
        enhance_layout.addWidget(self.percentile_var)

        perc_row = QWidget()
        perc_l = QHBoxLayout(perc_row)
        perc_l.setContentsMargins(0, 0, 0, 0)
        perc_l.setSpacing(6)
        perc_l.addWidget(QLabel("低百分位"))
        self.p_low_edit = QLineEdit("1")
        self.p_low_edit.setMinimumWidth(64)
        self.p_low_edit.setMaximumWidth(96)
        self.p_low_edit.setToolTip("低百分位阈值")
        perc_l.addWidget(self.p_low_edit)
        perc_l.addWidget(QLabel("高百分位"))
        self.p_high_edit = QLineEdit("99")
        self.p_high_edit.setMinimumWidth(64)
        self.p_high_edit.setMaximumWidth(96)
        self.p_high_edit.setToolTip("高百分位阈值")
        perc_l.addWidget(self.p_high_edit)
        perc_l.addStretch(1)
        enhance_layout.addWidget(perc_row)

        self.normalize_var = QCheckBox("归一化（最大绝对值）")
        self.normalize_var.setToolTip("将显示数据归一化到 [-1, 1]")
        self.demean_var = QCheckBox("显示去均值（逐道）")
        self.demean_var.setToolTip("仅在显示层逐道去均值；不写入处理结果")
        enhance_layout.addWidget(self.normalize_var)
        enhance_layout.addWidget(self.demean_var)

        layout.addWidget(enhance_box)

        # 空间辅助文件入口已迁移到“空间”页。这里仅保留隐藏兼容控件，
        # 使旧的 sidecar 状态同步/测试不会因为属性缺失而中断；显示页仍保持 display-only。
        self.sidecar_box = QGroupBox("空间辅助文件（已移至空间页）")
        self.sidecar_box.setVisible(False)
        self.rtk_sidecar_label = QLabel("未选择")
        self.rtk_sidecar_button = PushButton("选择 RTK")
        self.rtk_sidecar_clear_button = PushButton("清除")
        self.imu_sidecar_label = QLabel("未选择")
        self.imu_sidecar_button = PushButton("选择 IMU")
        self.imu_sidecar_clear_button = PushButton("清除")
        self.altimeter_sidecar_label = QLabel("未选择")
        self.altimeter_sidecar_button = PushButton("选择高度计")
        self.altimeter_sidecar_clear_button = PushButton("清除")

        layout.addStretch(1)

        scroll.setWidget(content)
        return scroll

    def _on_segment_changed(self, route_key: str):
        """切换标签页。"""
        mapping = {
            "mode": 0,
            "core": 1,
            "interact": 2,
            "enhance": 3,
        }
        self.stack.setCurrentIndex(mapping.get(route_key, 0))

    def _on_display_mode_changed(self, button, checked: bool):
        """同步显示模式到旧版 compare_var / diff_var。"""
        if not checked:
            return
        btn_id = self.display_mode_group.id(button)
        if btn_id == 0:   # 单图
            self.compare_var.setChecked(False)
            self.diff_var.setChecked(False)
            self.slider_compare_var.setChecked(False)
        elif btn_id == 1: # 双视图对比
            self.compare_var.setChecked(True)
            self.diff_var.setChecked(False)
            self.slider_compare_var.setChecked(False)
        elif btn_id == 2: # 差异图
            self.compare_var.setChecked(True)
            self.diff_var.setChecked(True)
            self.slider_compare_var.setChecked(False)
        elif btn_id == 3: # 滑动对比
            self.compare_var.setChecked(True)
            self.diff_var.setChecked(False)
            self.slider_compare_var.setChecked(True)
        self._refresh_compare_select_visibility()

    def _refresh_compare_select_visibility(self):
        """根据模式刷新对比选择区域可见性。"""
        show_compare_select = bool(
            self.compare_var.isChecked() or self.slider_compare_var.isChecked()
        )
        self.single_select_box.setVisible(not show_compare_select)
        self.compare_select_box.setVisible(show_compare_select)

    def get_view_style(self) -> str:
        """获取当前显示形式。"""
        return str(self.view_style_combo.currentData() or "image")

    def set_view_style(self, style_key: str) -> None:
        """设置显示形式。"""
        key = str(style_key or "").strip().lower()
        idx = self.view_style_combo.findData(key)
        if idx < 0:
            idx = self.view_style_combo.findData("image")
        if idx < 0:
            idx = 0
        old_block = self.view_style_combo.blockSignals(True)
        try:
            self.view_style_combo.setCurrentIndex(idx)
        finally:
            self.view_style_combo.blockSignals(old_block)
        self._refresh_view_style_ui_state()

    def update_wiggle_sampling_hint(self, shown_traces: int | None, total_traces: int | None):
        """更新 wiggle 抽样显示提示。"""
        if shown_traces is None or total_traces is None or total_traces <= 0:
            self.wiggle_sampling_hint.setText(
                "Wiggle 显示为抽样显示，仅用于显示，不改变数据。"
            )
            return
        self.wiggle_sampling_hint.setText(
            f"Wiggle 显示为抽样显示，当前约显示 {int(shown_traces)} / {int(total_traces)} 道，仅用于显示，不改变数据。"
        )

    @staticmethod
    def compute_wiggle_sampling_summary(n_traces: int, max_traces: int = 80) -> dict:
        """计算 wiggle 显示抽样统计。"""
        total = max(int(n_traces), 0)
        max_show = max(int(max_traces), 1)
        if total <= 0:
            return {"total": 0, "max_traces": max_show, "step": 1, "shown": 0}
        step = max(1, int((total + max_show - 1) // max_show))
        shown = (total + step - 1) // step
        return {"total": total, "max_traces": max_show, "step": step, "shown": shown}

    def _on_view_style_changed(self, _index: int):
        self._refresh_view_style_ui_state()

    def _refresh_view_style_ui_state(self):
        is_wiggle = self.get_view_style() == "wiggle"
        self.cmap_combo.setEnabled(not is_wiggle)
        if is_wiggle:
            self.cmap_combo.setToolTip("摆动图模式下不使用色图")
        else:
            self.cmap_combo.setToolTip("选择色彩映射方案")

    def set_manual_roi_status(self, text: str, has_roi: bool):
        """更新手动 ROI 状态显示。"""
        self.roi_status_label.setText(text)
        self.btn_clear_manual_roi.setEnabled(bool(has_roi))

    def _pair_row(self, label1, edit1, label2, edit2):
        """创建成对的输入行"""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label1))
        edit1.setMinimumWidth(72)
        edit1.setMaximumWidth(120)
        row.addWidget(edit1)
        row.addWidget(QLabel(label2))
        edit2.setMinimumWidth(72)
        edit2.setMaximumWidth(120)
        row.addWidget(edit2)
        row.addStretch(1)
        return row

    def _single_row(self, label, edit):
        """创建单行输入"""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        edit.setMinimumWidth(72)
        edit.setMaximumWidth(120)
        row.addWidget(edit)
        row.addStretch(1)
        return row

    def get_preset_key(self):
        """获取当前选中的预设key"""
        return None

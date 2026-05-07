#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 显示与对比页面 — 子标签页版（方案B）。"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
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
    QSizePolicy,
)
from qfluentwidgets import PushButton, FluentIcon, SegmentedWidget


class AdvancedSettingsPage(QWidget):
    """显示与对比页面 — 顶部 SegmentedWidget + QStackedWidget。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
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
            "本页集中控制主图显示、双图对比、聚焦裁剪与显示增强。"
        )
        page_hint.setWordWrap(True)
        page_hint.setProperty("class", "hintText")
        outer_layout.addWidget(page_hint)

        flow_box = QGroupBox("查看顺序")
        flow_box.setProperty("class", "calloutBox")
        flow_layout = QVBoxLayout(flow_box)
        flow_layout.setContentsMargins(10, 14, 10, 10)
        flow_layout.setSpacing(8)

        flow_hint = QLabel("推荐顺序：先确定显示模式，再设置聚焦范围，最后按需切到摆动图、滑动对比或显示增强。")
        flow_hint.setWordWrap(True)
        flow_hint.setProperty("class", "hintText")
        flow_layout.addWidget(flow_hint)

        flow_row = QWidget()
        flow_row_layout = QHBoxLayout(flow_row)
        flow_row_layout.setContentsMargins(0, 0, 0, 0)
        flow_row_layout.setSpacing(8)
        for text in ["① 显示模式", "② 聚焦交互", "③ 增强与性能"]:
            chip = QLabel(text)
            chip.setProperty("class", "statusChip")
            flow_row_layout.addWidget(chip)
        flow_row_layout.addStretch(1)
        flow_layout.addWidget(flow_row)
        outer_layout.addWidget(flow_box)

        # ========== 顶部标签栏 ==========
        self.segmented = SegmentedWidget(self)
        self.segmented.addItem("mode", "显示模式")
        self.segmented.addItem("core", "核心显示")
        self.segmented.addItem("interact", "聚焦交互")
        self.segmented.addItem("enhance", "增强与性能")
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

        # ---- 页面4: 增强与性能 ----
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
        mode_layout = QGridLayout(mode_box)
        mode_layout.setHorizontalSpacing(10)
        mode_layout.setVerticalSpacing(8)

        self.display_mode_group = QButtonGroup(self)

        self.mode_single = self._make_mode_radio("单图", "显示当前处理结果的单图")
        self.mode_single.setChecked(True)
        self.display_mode_group.addButton(self.mode_single, 0)
        mode_layout.addWidget(self.mode_single, 0, 0)

        self.mode_compare = self._make_mode_radio("双视图对比", "并排显示两个处理阶段的图像")
        self.display_mode_group.addButton(self.mode_compare, 1)
        mode_layout.addWidget(self.mode_compare, 0, 1)

        self.mode_diff = self._make_mode_radio("差异图", "显示两图差值")
        self.display_mode_group.addButton(self.mode_diff, 2)
        mode_layout.addWidget(self.mode_diff, 1, 0)

        self.mode_slider = self._make_mode_radio("滑动对比", "用拖动分隔线的方式查看两份结果")
        self.display_mode_group.addButton(self.mode_slider, 3)
        mode_layout.addWidget(self.mode_slider, 1, 1)

        mode_layout.setColumnStretch(0, 1)
        mode_layout.setColumnStretch(1, 1)

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

    @staticmethod
    def _make_mode_radio(text: str, tooltip: str) -> QRadioButton:
        radio = QRadioButton(text)
        radio.setToolTip(tooltip)
        radio.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return radio

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
        opts_l.addWidget(self.cmap_invert_var)
        opts_l.addWidget(self.show_cbar_var)
        opts_l.addWidget(self.show_grid_var)
        opts_l.addStretch(1)
        view_layout.addWidget(opts_row)

        style_hint = QLabel("摆动图适合看同相轴和波形结构；滑动对比可在主图中直接拖动分隔线，适合快速判读前后差异。")
        style_hint.setWordWrap(True)
        style_hint.setProperty("class", "hintText")
        view_layout.addWidget(style_hint)

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
        """构建增强与性能页面。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(16)

        # 增强
        enhance_box = QGroupBox("增强与对比辅助")
        enhance_layout = QVBoxLayout(enhance_box)
        enhance_layout.setSpacing(10)

        enhance_hint = QLabel("对当前显示做增强和轻量预处理，便于判读与对比。")
        enhance_hint.setWordWrap(True)
        enhance_hint.setProperty("class", "hintText")
        enhance_layout.addWidget(enhance_hint)

        self.symmetric_var = QCheckBox("对称灰度拉伸（以零为中心）")
        self.symmetric_var.setToolTip("适合查看正负振幅对称的雷达数据")
        self.chatgpt_style_var = QCheckBox("自动对比度（裁剪极值）")
        self.chatgpt_style_var.setChecked(False)
        self.chatgpt_style_var.setToolTip("自动裁剪极端值，增强主体反射")
        self.percentile_var = QCheckBox("百分位拉伸")
        self.percentile_var.setToolTip("基于百分位数拉伸对比度")
        enhance_layout.addWidget(self.symmetric_var)
        enhance_layout.addWidget(self.chatgpt_style_var)
        enhance_layout.addWidget(self.percentile_var)

        perc_row = QWidget()
        perc_l = QHBoxLayout(perc_row)
        perc_l.setContentsMargins(0, 0, 0, 0)
        perc_l.setSpacing(6)
        perc_l.addWidget(QLabel("低百分位"))
        self.p_low_edit = QLineEdit("1")
        self.p_low_edit.setFixedWidth(64)
        self.p_low_edit.setToolTip("低百分位阈值")
        perc_l.addWidget(self.p_low_edit)
        perc_l.addWidget(QLabel("高百分位"))
        self.p_high_edit = QLineEdit("99")
        self.p_high_edit.setFixedWidth(64)
        self.p_high_edit.setToolTip("高百分位阈值")
        perc_l.addWidget(self.p_high_edit)
        perc_l.addStretch(1)
        enhance_layout.addWidget(perc_row)

        self.normalize_var = QCheckBox("归一化（最大绝对值）")
        self.normalize_var.setToolTip("将显示数据归一化到 [-1, 1]")
        self.demean_var = QCheckBox("去均值（逐道）")
        self.demean_var.setToolTip("逐道去均值，减弱直流漂移")
        enhance_layout.addWidget(self.normalize_var)
        enhance_layout.addWidget(self.demean_var)

        layout.addWidget(enhance_box)

        # 可选 RTK/IMU 辅助文件：仅保存用户手动选择的路径，不影响普通 CSV 导入。
        self.sidecar_box = QGroupBox("可选 RTK/IMU 辅助文件")
        sidecar_layout = QVBoxLayout(self.sidecar_box)
        sidecar_layout.setSpacing(10)

        sidecar_hint = QLabel(
            "没有 RTK/IMU 数据时可保持未选择；普通 CSV 导入和处理流程不受影响。"
        )
        sidecar_hint.setWordWrap(True)
        sidecar_hint.setProperty("class", "hintText")
        sidecar_layout.addWidget(sidecar_hint)

        rtk_row = QWidget()
        rtk_layout = QHBoxLayout(rtk_row)
        rtk_layout.setContentsMargins(0, 0, 0, 0)
        rtk_layout.setSpacing(8)
        rtk_layout.addWidget(QLabel("RTK"))
        self.rtk_sidecar_label = QLabel("未选择")
        self.rtk_sidecar_label.setProperty("class", "hintText")
        self.rtk_sidecar_button = PushButton("选择 RTK")
        self.rtk_sidecar_button.setToolTip("选择可选 RTK CSV 辅助文件")
        self.rtk_sidecar_clear_button = PushButton("清除")
        self.rtk_sidecar_clear_button.setToolTip("清除当前 RTK 辅助文件选择")
        rtk_layout.addWidget(self.rtk_sidecar_label, stretch=1)
        rtk_layout.addWidget(self.rtk_sidecar_button)
        rtk_layout.addWidget(self.rtk_sidecar_clear_button)
        sidecar_layout.addWidget(rtk_row)

        imu_row = QWidget()
        imu_layout = QHBoxLayout(imu_row)
        imu_layout.setContentsMargins(0, 0, 0, 0)
        imu_layout.setSpacing(8)
        imu_layout.addWidget(QLabel("IMU"))
        self.imu_sidecar_label = QLabel("未选择")
        self.imu_sidecar_label.setProperty("class", "hintText")
        self.imu_sidecar_button = PushButton("选择 IMU")
        self.imu_sidecar_button.setToolTip("选择可选 IMU CSV 辅助文件")
        self.imu_sidecar_clear_button = PushButton("清除")
        self.imu_sidecar_clear_button.setToolTip("清除当前 IMU 辅助文件选择")
        imu_layout.addWidget(self.imu_sidecar_label, stretch=1)
        imu_layout.addWidget(self.imu_sidecar_button)
        imu_layout.addWidget(self.imu_sidecar_clear_button)
        sidecar_layout.addWidget(imu_row)

        layout.addWidget(self.sidecar_box)
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

    def set_manual_roi_status(self, text: str, has_roi: bool):
        """更新手动 ROI 状态显示。"""
        self.roi_status_label.setText(text)
        self.btn_clear_manual_roi.setEnabled(bool(has_roi))

    def _pair_row(self, label1, edit1, label2, edit2):
        """创建成对的输入行"""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label1))
        edit1.setFixedWidth(80)
        row.addWidget(edit1)
        row.addWidget(QLabel(label2))
        edit2.setFixedWidth(80)
        row.addWidget(edit2)
        row.addStretch(1)
        return row

    def _single_row(self, label, edit):
        """创建单行输入"""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        edit.setFixedWidth(80)
        row.addWidget(edit)
        row.addStretch(1)
        return row

    def get_preset_key(self):
        """获取当前选中的预设key"""
        return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""地形/三维成果页面。

该页面把 UAV-GPR 空间成果组织为工程软件常见的专业浏览器：
顶部轻状态条 + 主视图 Tab + 右侧属性面板 + 底部状态栏。
科研/说明性文字被压缩到空状态和 tooltip 中，避免在工程前端堆叠说明卡片。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QFrame,
    QSizePolicy,
    QTabWidget,
    QSplitter,
    QCheckBox,
)
from qfluentwidgets import PushButton, FluentIcon

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.gui_quality_log import QualityLogPage


class Terrain3DResultsPage(QualityLogPage):
    """地形/三维成果独立页面。

    该页复用 QualityLogPage 中已经稳定的 georef3d 绘制、弹窗和导出逻辑，
    但把前端组织为工程解释软件中的空间成果浏览器：

    1. 顶部状态条：坐标、高程、测线、C-scan、解释线状态。
    2. 主视图 Tab：剖面三维、测线轨迹、地形剖面、C-scan、基覆界面。
    3. 右侧属性面板：图层、当前点、解释对象。
    4. 底部状态栏：道数、采样点、测线长度和数据条件摘要。
    """

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ========== 顶部轻状态条 ==========
        header = QFrame()
        header.setObjectName("TerrainProfessionalHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(8)

        title = QLabel("空间成果")
        title.setProperty("class", "sectionTitle")
        header_layout.addWidget(title)

        self._terrain_status_chips: dict[str, QLabel] = {}
        for key, text in [
            ("coord", "坐标 —"),
            ("elev", "高程 —"),
            ("line", "测线 —"),
            ("cscan", "C-scan —"),
            ("interface", "解释线 —"),
        ]:
            chip = QLabel(text)
            chip.setObjectName("TerrainStatusChip")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setMinimumWidth(72)
            self._terrain_status_chips[key] = chip
            header_layout.addWidget(chip)

        header_layout.addStretch(1)
        self.btn_export_scene_image = PushButton(FluentIcon.PHOTO, "导出成果图")
        self.btn_export_scene_image.setEnabled(False)
        self.btn_export_scene_image.setToolTip("后续导出当前空间成果视图。当前版本请使用三维地理参考导出。")
        self.btn_export_georeference_3d = PushButton(FluentIcon.SAVE_AS, "导出三维地理参考")
        self.btn_export_georeference_3d.setToolTip("导出当前轨迹与剖面带的三维地理参考文件（VTK / CSV / JSON）")
        self.btn_open_log_dir = PushButton(FluentIcon.FOLDER, "打开输出目录")
        self.btn_open_log_dir.setToolTip("打开日志和输出目录")
        header_layout.addWidget(self.btn_export_scene_image)
        header_layout.addWidget(self.btn_export_georeference_3d)
        header_layout.addWidget(self.btn_open_log_dir)
        root.addWidget(header)

        # V0.8.43: explicit empty state for ordinary B-scan / profile data without spatial metadata.
        self.space_empty_state_card = QFrame()
        self.space_empty_state_card.setObjectName("SpatialEmptyStateCard")
        empty_layout = QHBoxLayout(self.space_empty_state_card)
        empty_layout.setContentsMargins(12, 8, 12, 8)
        empty_layout.setSpacing(10)
        empty_badge = QLabel("空间数据未接入")
        empty_badge.setObjectName("TerrainStatusChip")
        empty_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.space_empty_state_text = QLabel("当前数据可进行 B-scan 处理与显示；导入轨迹、高程、飞行高度或多测线网格后，空间成果页会启用测线轨迹、地形剖面、C-scan 和三维预览。")
        self.space_empty_state_text.setProperty("class", "hintText")
        self.space_empty_state_text.setWordWrap(True)
        empty_layout.addWidget(empty_badge)
        empty_layout.addWidget(self.space_empty_state_text, 1)
        root.addWidget(self.space_empty_state_card)

        # ========== 主工作区：主视图 + 属性面板 ==========
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("TerrainProfessionalSplitter")
        root.addWidget(splitter, 1)

        self.view_tabs = QTabWidget()
        self.view_tabs.setObjectName("TerrainResultsTabs")
        splitter.addWidget(self.view_tabs)

        # Tab 1: 剖面三维
        self.tab_3d = QWidget()
        tab_3d_layout = QVBoxLayout(self.tab_3d)
        tab_3d_layout.setContentsMargins(0, 0, 0, 0)
        tab_3d_layout.setSpacing(6)
        georef3d_panel = QWidget()
        self._georef3d_overlay_parent = georef3d_panel
        georef3d_panel_layout = QVBoxLayout(georef3d_panel)
        georef3d_panel_layout.setContentsMargins(0, 0, 0, 0)
        georef3d_panel_layout.setSpacing(0)
        self.georef3d_fig = Figure(figsize=(7.0, 5.2), dpi=100)
        self.georef3d_canvas = FigureCanvas(self.georef3d_fig)
        self.georef3d_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.georef3d_ax = self.georef3d_fig.add_subplot(111, projection="3d")
        georef3d_panel_layout.addWidget(self.georef3d_canvas)
        tab_3d_layout.addWidget(georef3d_panel, 1)
        self.view_tabs.addTab(self.tab_3d, "剖面三维")

        # 3D overlay buttons still exist for compatibility and direct manipulation.
        self.btn_georef3d_raw = self._create_georef3d_overlay_button("👁 原始", "显示原始三维航迹与剖面", checked=False)
        self.btn_georef3d_current = self._create_georef3d_overlay_button("👁 当前", "显示当前/处理后三维航迹与剖面", checked=True)
        self.btn_georef3d_bscan = self._create_georef3d_overlay_button("👁 B-scan", "显示或隐藏三维 B-scan 剖面带", checked=True)
        self.btn_georef3d_diff = self._create_georef3d_overlay_button("👁 差异", "显示当前减原始的差异剖面", checked=False)
        self.btn_georef3d_reset_view = self._create_georef3d_overlay_button("↺", "重置三维视角", checkable=False)
        self.btn_georef3d_expand = self._create_georef3d_overlay_button("⛶", "展开三维预览", checkable=False)
        for button in [self.btn_georef3d_raw, self.btn_georef3d_current, self.btn_georef3d_bscan, self.btn_georef3d_diff]:
            button.toggled.connect(self._schedule_georef3d_redraw)
        self.btn_georef3d_reset_view.clicked.connect(self._reset_georef3d_view)
        self.btn_georef3d_expand.clicked.connect(self._open_georef3d_dialog)
        self.georef3d_canvas.mpl_connect("button_press_event", self._on_georef3d_interaction_start)
        self.georef3d_canvas.mpl_connect("button_release_event", self._on_georef3d_interaction_end)
        self._position_georef3d_overlay_controls()

        # Tab 2: 测线轨迹
        self.tab_track = QWidget()
        track_layout = QVBoxLayout(self.tab_track)
        track_layout.setContentsMargins(0, 0, 0, 0)
        track_layout.setSpacing(6)
        self.track_fig = Figure(figsize=(6.8, 4.8), dpi=100)
        self.track_canvas = FigureCanvas(self.track_fig)
        self.track_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        track_layout.addWidget(self.track_canvas, 1)
        self.view_tabs.addTab(self.tab_track, "测线轨迹")

        # Tab 3: 地形剖面
        self.tab_terrain = QWidget()
        terrain_layout = QVBoxLayout(self.tab_terrain)
        terrain_layout.setContentsMargins(0, 0, 0, 0)
        terrain_layout.setSpacing(6)
        self.terrain_fig = Figure(figsize=(6.8, 4.8), dpi=100)
        self.terrain_canvas = FigureCanvas(self.terrain_fig)
        self.terrain_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        terrain_layout.addWidget(self.terrain_canvas, 1)
        self.view_tabs.addTab(self.tab_terrain, "地形剖面")

        # Tab 4: C-scan 空状态
        self.tab_cscan = QWidget()
        cscan_layout = QVBoxLayout(self.tab_cscan)
        cscan_layout.setContentsMargins(16, 16, 16, 16)
        cscan_layout.setSpacing(12)
        self.cscan_empty_label = QLabel("当前为单测线数据，C-scan / 深度切片需要多测线或网格数据。")
        self.cscan_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cscan_empty_label.setWordWrap(True)
        self.cscan_empty_label.setProperty("class", "hintText")
        self.btn_open_cscan = PushButton(FluentIcon.VIEW, "生成 C-scan / 深度切片")
        self.btn_open_cscan.setEnabled(False)
        self.btn_open_cscan.setToolTip("需要多测线/网格数据、空间坐标、测线间距和深度/时间窗切片条件。")
        cscan_layout.addStretch(1)
        cscan_layout.addWidget(self.cscan_empty_label)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_open_cscan)
        row.addStretch(1)
        cscan_layout.addLayout(row)
        cscan_layout.addStretch(1)
        self.view_tabs.addTab(self.tab_cscan, "C-scan")

        # Tab 5: 基覆界面 空状态
        self.tab_interface = QWidget()
        interface_layout = QVBoxLayout(self.tab_interface)
        interface_layout.setContentsMargins(16, 16, 16, 16)
        interface_layout.setSpacing(12)
        self.interface_empty_label = QLabel("尚未接入人工解释线 / 钻孔约束。后续用于基覆界面、裂隙/破碎带和地层分界面成果核查。")
        self.interface_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.interface_empty_label.setWordWrap(True)
        self.interface_empty_label.setProperty("class", "hintText")
        self.btn_import_interface = PushButton(FluentIcon.ADD, "导入解释线 / 钻孔约束")
        self.btn_import_interface.setEnabled(False)
        self.btn_import_interface.setToolTip("后续版本接入人工解释线、钻孔和基覆界面对比资料。")
        self.btn_export_interface = PushButton(FluentIcon.SAVE_AS, "导出解释成果")
        self.btn_export_interface.setEnabled(False)
        self.btn_export_interface.setToolTip("需要解释线或模型结果后启用。")
        interface_layout.addStretch(1)
        interface_layout.addWidget(self.interface_empty_label)
        row2 = QHBoxLayout()
        row2.addStretch(1)
        row2.addWidget(self.btn_import_interface)
        row2.addWidget(self.btn_export_interface)
        row2.addStretch(1)
        interface_layout.addLayout(row2)
        interface_layout.addStretch(1)
        self.view_tabs.addTab(self.tab_interface, "基覆界面")

        # ========== 右侧属性面板 ==========
        side_panel = QFrame()
        side_panel.setObjectName("TerrainPropertyPanel")
        side_panel.setMinimumWidth(260)
        side_panel.setMaximumWidth(360)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(8)

        layer_box = QGroupBox("图层")
        layer_layout = QVBoxLayout(layer_box)
        layer_layout.setContentsMargins(10, 14, 10, 10)
        layer_layout.setSpacing(6)
        self.chk_layer_bscan = QCheckBox("B-scan 剖面")
        self.chk_layer_bscan.setChecked(True)
        self.chk_layer_terrain = QCheckBox("地形曲线")
        self.chk_layer_terrain.setChecked(True)
        self.chk_layer_track = QCheckBox("测线轨迹")
        self.chk_layer_track.setChecked(True)
        self.chk_layer_height = QCheckBox("飞行高度")
        self.chk_layer_height.setChecked(True)
        self.chk_layer_interface = QCheckBox("基覆界面")
        self.chk_layer_interface.setEnabled(False)
        self.chk_layer_borehole = QCheckBox("钻孔")
        self.chk_layer_borehole.setEnabled(False)
        layer_layout.addWidget(self.chk_layer_bscan)
        layer_layout.addWidget(self.chk_layer_terrain)
        layer_layout.addWidget(self.chk_layer_track)
        layer_layout.addWidget(self.chk_layer_height)
        layer_layout.addWidget(self.chk_layer_interface)
        layer_layout.addWidget(self.chk_layer_borehole)
        side_layout.addWidget(layer_box)

        sidecar_box = QGroupBox("空间辅助文件")
        sidecar_box.setObjectName("SpatialSidecarBox")
        sidecar_box.setMinimumHeight(158)
        sidecar_layout = QVBoxLayout(sidecar_box)
        sidecar_layout.setContentsMargins(10, 14, 10, 10)
        sidecar_layout.setSpacing(8)
        sidecar_hint = QLabel("导入可选 RTK、IMU 和高度计文件，用于航迹、姿态、高度和三维成果联动。")
        sidecar_hint.setWordWrap(True)
        sidecar_hint.setProperty("class", "hintText")
        sidecar_layout.addWidget(sidecar_hint)

        def _sidecar_row(title: str, button_text: str):
            row = QWidget()
            row.setMinimumHeight(32)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            name = QLabel(title)
            name.setMinimumWidth(52)
            label = QLabel("未选择")
            label.setProperty("class", "hintText")
            label.setWordWrap(False)
            button = PushButton(button_text)
            clear_button = PushButton("清除")
            row_layout.addWidget(name)
            row_layout.addWidget(label, 1)
            row_layout.addWidget(button)
            row_layout.addWidget(clear_button)
            return row, label, button, clear_button

        rtk_row, self.rtk_sidecar_label, self.rtk_sidecar_button, self.rtk_sidecar_clear_button = _sidecar_row("RTK", "选择")
        imu_row, self.imu_sidecar_label, self.imu_sidecar_button, self.imu_sidecar_clear_button = _sidecar_row("IMU", "选择")
        alt_row, self.altimeter_sidecar_label, self.altimeter_sidecar_button, self.altimeter_sidecar_clear_button = _sidecar_row("高度计", "选择")
        self.rtk_sidecar_button.setToolTip("选择可选 RTK CSV 辅助文件")
        self.imu_sidecar_button.setToolTip("选择可选 IMU CSV 辅助文件")
        self.altimeter_sidecar_button.setToolTip("选择可选高度计 CSV 辅助文件")
        sidecar_layout.addWidget(rtk_row)
        sidecar_layout.addWidget(imu_row)
        sidecar_layout.addWidget(alt_row)
        side_layout.addWidget(sidecar_box)
        self.sidecar_box = sidecar_box

        current_box = QGroupBox("当前点")
        current_layout = QVBoxLayout(current_box)
        current_layout.setContentsMargins(10, 14, 10, 10)
        self.current_point_label = QLabel("在剖面或测线视图中移动鼠标后显示 trace、距离、坐标、高程和幅度。")
        self.current_point_label.setWordWrap(True)
        self.current_point_label.setProperty("class", "hintText")
        current_layout.addWidget(self.current_point_label)
        side_layout.addWidget(current_box)

        object_box = QGroupBox("解释对象")
        object_layout = QVBoxLayout(object_box)
        object_layout.setContentsMargins(10, 14, 10, 10)
        self.interpretation_object_label = QLabel("基覆界面：未接入\n裂隙/破碎带：未接入\n钻孔：未接入")
        self.interpretation_object_label.setWordWrap(True)
        self.interpretation_object_label.setProperty("class", "hintText")
        object_layout.addWidget(self.interpretation_object_label)
        side_layout.addWidget(object_box)
        side_layout.addStretch(1)
        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)

        # 图层联动：B-scan 开关直接映射到原有三维层按钮；其他开关刷新二维辅助图。
        self.chk_layer_bscan.toggled.connect(self.btn_georef3d_bscan.setChecked)
        self.chk_layer_terrain.toggled.connect(self._refresh_auxiliary_space_views)
        self.chk_layer_track.toggled.connect(self._refresh_auxiliary_space_views)
        self.chk_layer_height.toggled.connect(self._refresh_auxiliary_space_views)

        # ========== 底部状态栏 ==========
        self.terrain_bottom_status = QLabel("未加载空间成果数据")
        self.terrain_bottom_status.setObjectName("TerrainBottomStatus")
        self.terrain_bottom_status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(self.terrain_bottom_status)

        self.set_airborne_georeference_3d_visualization(None)

    def set_airborne_georeference_3d_visualization(self, payload: dict | None):
        """设置三维预览数据，并同步空间成果条件状态。"""
        super().set_airborne_georeference_3d_visualization(payload)
        self._update_terrain_condition_status()
        self._refresh_auxiliary_space_views()

    def _update_terrain_condition_status(self) -> None:
        """Update professional-view status chips and property text from current 3D payload."""
        if not hasattr(self, "_terrain_status_chips"):
            return
        payload = self._first_available_georef3d_payload()
        if not payload:
            if hasattr(self, "space_empty_state_card"):
                self.space_empty_state_card.setVisible(True)
            if hasattr(self, "space_empty_state_text"):
                self.space_empty_state_text.setText("当前数据未检测到空间元数据。B-scan 处理可正常使用；空间成果需要轨迹、坐标、高程、飞行高度或多测线网格数据。")
            self._set_status_chip("coord", "坐标 —")
            self._set_status_chip("elev", "高程 —")
            self._set_status_chip("line", "测线 —")
            self._set_status_chip("cscan", "C-scan 不可用")
            self._set_status_chip("interface", "解释线未接入")
            self.terrain_bottom_status.setText("空间数据未接入｜B-scan 处理不受影响｜空间成果需轨迹/高程/多测线数据")
            self.current_point_label.setText("未检测到空间元数据。可先在处理页/选参页完成 B-scan 处理；导入 RTK、IMU、高度计或飞行记录后，再查看空间联动结果。")
            self.interpretation_object_label.setText("基覆界面：未接入\n裂隙/破碎带：未接入\n钻孔：未接入\n状态：等待空间资料")
            self.cscan_empty_label.setText("当前未加载空间成果数据。C-scan / 深度切片需要多测线或网格数据。")
            self.interface_empty_label.setText("尚未接入人工解释线 / 钻孔约束。后续用于基覆界面、裂隙/破碎带和地层分界面成果核查。")
            self.btn_export_georeference_3d.setEnabled(False)
            self.btn_export_scene_image.setEnabled(False)
            self.btn_open_cscan.setEnabled(False)
            self.btn_import_interface.setEnabled(False)
            self.btn_export_interface.setEnabled(False)
            return

        trace_count = int(payload.get("trace_count") or 0)
        sample_count = int(payload.get("sample_count") or 0)
        has_coords = bool(payload.get("has_longitude_latitude"))
        has_ground = bool(payload.get("has_ground_elevation"))
        has_height = bool(payload.get("has_height_agl"))
        has_georef_ready = bool(has_coords and (has_ground or has_height))
        if hasattr(self, "space_empty_state_card"):
            self.space_empty_state_card.setVisible(not has_georef_ready)
        if hasattr(self, "space_empty_state_text"):
            if has_georef_ready:
                self.space_empty_state_text.setText("空间元数据已接入，可查看航迹、地形剖面和三维地理参考预览。")
            else:
                self.space_empty_state_text.setText("当前只有剖面/距离轴信息，空间坐标或高程条件不完整；下方为非地理参考剖面预览，不代表完整空间成果。")
        distance = payload.get("trace_distance_m")
        track_length = self._track_length_from_distance(distance)

        self._set_status_chip("coord", "坐标✓" if has_coords else "坐标缺失")
        self._set_status_chip("elev", "高程✓" if has_ground else "高程缺失")
        self._set_status_chip("line", "单测线" if trace_count else "测线 —")
        self._set_status_chip("cscan", "C-scan 不可用")
        self._set_status_chip("interface", "解释线未接入")

        length_text = f"｜测线长 {track_length:.1f} m" if track_length is not None else ""
        mode_text = "三维地理参考可用" if has_georef_ready else "仅剖面预览，非完整空间成果"
        self.terrain_bottom_status.setText(
            f"{trace_count} 道｜{sample_count} 采样点{length_text}｜"
            f"坐标{'可用' if has_coords else '缺失'}｜飞行高度{'可用' if has_height else '缺失'}｜"
            f"地表高程{'可用' if has_ground else '缺失'}｜{mode_text}｜单测线，不支持 C-scan"
        )
        length_line = f"测线长度：{track_length:.1f} m" if track_length is not None else "测线长度：未记录"
        self.current_point_label.setText(
            f"数据：{trace_count} 道 × {sample_count} 采样点\n"
            f"{length_line}\n"
            f"坐标：{'可用' if has_coords else '缺失'}\n"
            f"飞行高度：{'可用' if has_height else '缺失'}\n"
            f"地表高程：{'可用' if has_ground else '缺失'}\n"
            f"模式：{'三维地理参考' if has_georef_ready else '仅剖面预览'}"
        )
        self.interpretation_object_label.setText("基覆界面：未接入\n裂隙/破碎带：未接入\n钻孔：未接入\n状态：仅空间预览")
        self.cscan_empty_label.setText(
            "当前按单测线剖面处理，C-scan / 深度切片暂不可用。\n"
            "需要：多测线或网格数据、空间坐标、测线间距、深度/时间窗切片参数。"
        )
        self.interface_empty_label.setText(
            "尚未接入人工解释线 / 钻孔约束。\n"
            "后续将用于基覆界面深度误差、测线交叉一致性和裂隙/破碎带候选成果核查。"
        )
        self.btn_export_georeference_3d.setEnabled(has_georef_ready)
        self.btn_export_scene_image.setEnabled(False)
        self.btn_open_cscan.setEnabled(False)
        self.btn_import_interface.setEnabled(False)
        self.btn_export_interface.setEnabled(False)


    def _payload_has_georeferenced_space(self, payload: dict[str, Any] | None) -> bool:
        """Return True only when a payload is suitable for spatial/3D georeferenced display."""
        if not payload:
            return False
        return bool(
            payload.get("has_longitude_latitude")
            and (payload.get("has_ground_elevation") or payload.get("has_height_agl"))
        )

    def _visible_georef3d_entries(self) -> list[tuple[str, str, dict]]:
        """Hide non-georeferenced profile-only payloads from the 3D scene.

        The space page is for spatial products.  Plain B-scan/profile-only data
        remains valid for processing/display, but showing it as a 3D curtain can
        imply a spatial result that does not exist.  Keep the empty-state message
        visible until coordinates plus elevation/height are available.
        """
        entries = super()._visible_georef3d_entries()
        return [entry for entry in entries if self._payload_has_georeferenced_space(entry[2])]

    def _available_georef3d_entries(self) -> dict[str, tuple[str, dict]]:
        """Expose only georeferenced payloads for expanded/exportable spatial views."""
        entries = super()._available_georef3d_entries()
        return {key: value for key, value in entries.items() if self._payload_has_georeferenced_space(value[1])}

    def focus_track_view(self) -> None:
        """Switch the space results page to the track view."""
        try:
            if hasattr(self, "view_tabs"):
                self.view_tabs.setCurrentIndex(1)
        except Exception:
            pass

    def _refresh_auxiliary_space_views(self, *_args) -> None:
        """Refresh 2D track and terrain views from the currently selected 3D payload."""
        if not hasattr(self, "track_fig"):
            return
        payload = self._first_available_georef3d_payload()
        self._draw_track_view(payload)
        self._draw_terrain_profile(payload)

    def _draw_track_view(self, payload: dict | None) -> None:
        self.track_fig.clear()
        ax = self.track_fig.add_subplot(111)
        if not payload:
            ax.text(0.5, 0.5, "未加载测线轨迹", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.track_canvas.draw_idle()
            return
        preview = payload.get("preview") or {}
        x, y, distance = self._extract_track_xy(payload, preview)
        if x.size < 2:
            ax.text(0.5, 0.5, "缺少空间坐标，仅可显示距离轴预览", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.track_canvas.draw_idle()
            return
        if self.chk_layer_track.isChecked():
            ax.plot(x, y, linewidth=1.8, label="测线轨迹")
            ax.scatter([x[0], x[-1]], [y[0], y[-1]], s=34, label="起点/终点")
        ax.set_title("测线轨迹")
        ax.set_xlabel("沿测线局部 X / 经度换算 (m)")
        ax.set_ylabel("横向局部 Y / 纬度换算 (m)")
        ax.grid(True, alpha=0.25)
        if distance.size:
            ax.text(0.02, 0.02, f"长度约 {float(distance[-1]-distance[0]):.1f} m", transform=ax.transAxes)
        ax.legend(loc="best")
        self.track_fig.tight_layout()
        self.track_canvas.draw_idle()

    def _draw_terrain_profile(self, payload: dict | None) -> None:
        self.terrain_fig.clear()
        ax = self.terrain_fig.add_subplot(111)
        if not payload:
            ax.text(0.5, 0.5, "未加载地形剖面", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.terrain_canvas.draw_idle()
            return
        preview = payload.get("preview") or {}
        distance = self._array_from_payload(payload, "trace_distance_m")
        if distance.size < 2:
            distance = self._array_from_preview(preview, "curtain_x_m")
            if distance.ndim == 2:
                distance = distance[0, :]
        ground = self._extract_ground_profile(payload, preview)
        height = self._extract_height_profile(payload, preview)
        if distance.size < 2:
            distance = np.arange(max(ground.size, height.size), dtype=np.float64)
        plotted = False
        if self.chk_layer_terrain.isChecked() and ground.size:
            ax.plot(distance[: ground.size], ground, linewidth=1.8, label="地表高程")
            plotted = True
        if self.chk_layer_height.isChecked() and height.size:
            base = ground if ground.size == height.size else np.zeros_like(height)
            ax.plot(distance[: height.size], base[: height.size] + height, linewidth=1.2, linestyle="--", label="飞行轨迹高程")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "缺少地表高程 / 飞行高度", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.terrain_canvas.draw_idle()
            return
        ax.set_title("地形剖面")
        ax.set_xlabel("沿测线距离 / m")
        ax.set_ylabel("高程 / m")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        self.terrain_fig.tight_layout()
        self.terrain_canvas.draw_idle()

    def release_plot_resources(self) -> None:
        """释放本页新增的 Matplotlib 资源。"""
        super().release_plot_resources()
        for fig_name, canvas_name in [("track_fig", "track_canvas"), ("terrain_fig", "terrain_canvas")]:
            fig = getattr(self, fig_name, None)
            canvas = getattr(self, canvas_name, None)
            try:
                if fig is not None:
                    fig.clear()
            except Exception:
                pass
            try:
                if canvas is not None:
                    canvas.close()
                    canvas.deleteLater()
            except Exception:
                pass

    def _set_status_chip(self, key: str, text: str) -> None:
        chip = self._terrain_status_chips.get(key)
        if chip is not None:
            chip.setText(text)

    def _first_available_georef3d_payload(self) -> dict[str, Any] | None:
        bundle = getattr(self, "_georef3d_bundle", None) or {}
        for key in ("current", "raw", "diff"):
            payload = self._select_georef3d_payload(bundle.get(key))
            if payload:
                return payload
        return None

    @staticmethod
    def _array_from_payload(payload: dict[str, Any], key: str) -> np.ndarray:
        try:
            arr = np.asarray(payload.get(key, []), dtype=np.float64)
            return arr[np.isfinite(arr)] if arr.ndim == 1 else arr
        except Exception:
            return np.array([], dtype=np.float64)

    @staticmethod
    def _array_from_preview(preview: dict[str, Any], key: str) -> np.ndarray:
        try:
            return np.asarray(preview.get(key, []), dtype=np.float64)
        except Exception:
            return np.array([], dtype=np.float64)

    def _extract_track_xy(self, payload: dict[str, Any], preview: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self._array_from_payload(payload, "longitude")
        y = self._array_from_payload(payload, "latitude")
        if x.size < 2 or y.size < 2 or x.size != y.size:
            cx = self._array_from_preview(preview, "curtain_x_m")
            cy = self._array_from_preview(preview, "curtain_y_m")
            if cx.ndim == 2 and cy.ndim == 2 and cx.shape == cy.shape:
                x = cx[0, :]
                y = cy[0, :]
        distance = self._array_from_payload(payload, "trace_distance_m")
        return x, y, distance

    def _extract_ground_profile(self, payload: dict[str, Any], preview: dict[str, Any]) -> np.ndarray:
        for key in ("ground_elevation_m", "ground_elevation", "surface_elevation_m"):
            arr = self._array_from_payload(payload, key)
            if arr.size:
                return arr
        cz = self._array_from_preview(preview, "curtain_z_m")
        if cz.ndim == 2 and cz.size:
            return np.nanmax(cz, axis=0)
        return np.array([], dtype=np.float64)

    def _extract_height_profile(self, payload: dict[str, Any], _preview: dict[str, Any]) -> np.ndarray:
        for key in ("height_agl_m", "height_agl", "flight_height_m", "altitude_agl_m"):
            arr = self._array_from_payload(payload, key)
            if arr.size:
                return arr
        return np.array([], dtype=np.float64)

    @staticmethod
    def _track_length_from_distance(distance: Any) -> float | None:
        try:
            arr = np.asarray(distance, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size > 1:
                return float(arr[-1] - arr[0])
        except Exception:
            pass
        return None

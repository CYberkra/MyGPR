#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target interpretation page mixin for the field workbench."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from core.target_source_binding import source_label_from_target
from ui.field_panels.layout_metrics import layout_metrics_for
from ui.field_panels.target_actions import TargetActionsMixin
from ui.field_panels.widgets import Card, MetricCard, PlotCard


class InterpretationPageMixin(TargetActionsMixin):
    def _page_interpretation(self) -> QWidget:
        widget = QWidget()
        v = QVBoxLayout(widget)
        v.setContentsMargins(0, 0, 0, 0)
        lm = layout_metrics_for(self)
        v.setSpacing(lm.spacing)

        metrics = QHBoxLayout()
        metrics.setSpacing(lm.spacing)
        st = self.project_status
        for card in [
            MetricCard("◎", "已识别目标", str(st.target_count), "个"),
            MetricCard("✓", "已确认目标", str(st.confirmed_target_count), "个"),
            MetricCard("◷", "待复核目标", str(st.pending_target_count), "个"),
            MetricCard("▤", "输出定位点", str(st.spatial_point_count), "个"),
        ]:
            metrics.addWidget(card)
        v.addLayout(metrics)

        main = QHBoxLayout()
        main.setSpacing(lm.spacing)
        left = QVBoxLayout()
        left.setSpacing(lm.spacing)
        line = self._selected_line_record()
        card = Card(title=f"目标定位视图（{line.get('id', '--')} {line.get('name', '暂无测线')}）")
        card.setProperty("layoutKey", "targetBscanCard")
        card.setMaximumHeight(lm.target_bscan_h + (94 if lm.compact else 112))
        toolbar = QHBoxLayout()
        toolbar.setSpacing(lm.spacing)
        toolbar.addWidget(QLabel("标注来源"))
        self.target_source_combo = QComboBox()
        self.target_source_combo.setObjectName("filterCombo")
        self.target_source_combo.setMinimumWidth(self._compact_value(260, 210))
        self.target_source_combo.currentIndexChanged.connect(self._on_target_source_changed)
        toolbar.addWidget(self.target_source_combo)
        self._refresh_target_source_options()
        target_actions = [
            ("＋ 新建标注", self._add_preview_target),
            ("✥ 自动识别辅助", self._auto_detect_targets),
            ("▣ 保存标注", self._save_targets),
            ("⌫ 删除标注", self._delete_selected_target),
            ("⇧ 导出定位表  ▾", self._save_targets),
        ]
        for text, slot in target_actions:
            btn = QPushButton(text)
            btn.setObjectName("primaryButton" if "新建" in text or "自动" in text else "smallButton")
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)
        toolbar.addStretch(1)
        card.layout.addLayout(toolbar)

        plot = PlotCard(
            None,
            height=lm.target_bscan_h,
            expand_title="目标定位 B-scan 放大查看",
            expand_callback=self._draw_current_target_bscan,
            expand_parent=self,
        )
        plot.setProperty("layoutKey", "targetBscanPlotCard")
        plot.canvas.setObjectName("targetBscanCanvas")
        plot.canvas.setProperty("layoutKey", "targetBscanCanvas")
        plot.layout.setContentsMargins(0, 0, 0, 0)
        self.target_canvas = plot.canvas
        self._draw_current_target_bscan(plot.canvas)
        plot.canvas.mpl_connect("button_press_event", self._on_target_canvas_click)
        card.layout.addWidget(plot)
        left.addWidget(card, 0)
        left.addWidget(self._target_table_card(), 1)
        main.addLayout(left, 1)
        main.addWidget(self._target_info_panel(), 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(main, 1)
        return widget

    def _target_table_card(self) -> Card:
        line = self._selected_line_record()
        card = Card(title=f"目标标注列表（{line.get('id', '--')} {line.get('name', '暂无测线')}）")
        card.setProperty("layoutKey", "targetTableCard")
        lm = layout_metrics_for(self)
        card.setMaximumHeight(lm.target_table_max_h + 42)
        table = self._table(["□", "目标名称", "类型", "里程 (m)", "深度 (m)", "置信度", "状态", "备注"], 4)
        table.setMaximumHeight(lm.target_table_max_h)
        self.target_table = table
        self._fill_table(table, self._target_rows(), highlight_row=self.current_target_index)
        table.cellClicked.connect(self._select_target_from_table)
        card.layout.addWidget(table)
        self.target_log_label = QLabel("暂无目标标注；请导入测线并在 B-scan 上新增或识别目标。" if not self.targets else f"已加载 {len(self.targets)} 个目标标注。")
        self.target_log_label.setObjectName("activityDesc")
        card.layout.addWidget(self.target_log_label)
        return card

    def _target_info_panel(self) -> QWidget:
        lm = layout_metrics_for(self)
        wrap = QWidget()
        wrap.setProperty("layoutKey", "targetInfoSidePanel")
        wrap.setFixedWidth(lm.target_info_w)
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(lm.spacing)
        info = Card(title="目标属性")
        info.setProperty("layoutKey", "targetInfoCard")
        header = QFrame()
        header.setObjectName("targetHeroPanel")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(1)
        self.target_info_title_label = QLabel("当前目标 --")
        self.target_info_title_label.setObjectName("detailTitle")
        self.target_info_subtitle_label = QLabel("请选择或新增一个目标标注")
        self.target_info_subtitle_label.setObjectName("detailSubtitle")
        header_layout.addWidget(self.target_info_title_label)
        header_layout.addWidget(self.target_info_subtitle_label)
        info.layout.addWidget(header)

        self.target_field_labels = {}
        for k in ["类型 *", "里程 (m)", "深度 (m)", "坐标 (m)", "置信度", "状态", "备注", "来源处理结果"]:
            row = QHBoxLayout()
            row.setSpacing(lm.spacing)
            key = QLabel(k)
            key.setObjectName("keyLabel")
            key.setFixedWidth(66 if lm.compact else 76)
            lbl = QLabel("--")
            lbl.setObjectName("fieldBox")
            lbl.setWordWrap(False)
            lbl.setMaximumHeight(20 if lm.compact else 24)
            row.addWidget(key)
            row.addWidget(lbl, 1)
            info.layout.addLayout(row)
            self.target_field_labels[k] = lbl
        v.addWidget(info, 1)
        preview = PlotCard("剖面位置预览", height=lm.target_preview_h)
        preview.setProperty("layoutKey", "targetPreviewMapCard")
        preview.canvas.setObjectName("targetPreviewMapCanvas")
        preview.canvas.setProperty("layoutKey", "targetPreviewMapCanvas")
        self.target_preview_canvas = preview.canvas
        self._draw_current_line_strip(preview.canvas, marker=float(self.targets[self.current_target_index].get("mileage", 0.0)) if self.targets else None)
        v.addWidget(preview, 0)
        self._update_target_info_panel()
        return wrap


__all__ = ["InterpretationPageMixin"]

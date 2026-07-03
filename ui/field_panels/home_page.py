#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Home/project-overview page for the MyGPR field workbench."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.field_panels.widgets import Card, PlotCard


class HomePageMixin:
    """Builds the home dashboard without bloating the main window shell."""

    def _build_home_page(self) -> QWidget:
        scroll, body, layout = self._make_scroll_body()
        body_layout = layout
        top_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel("项目总览")
        title.setObjectName("pageTitle")
        sub = QLabel("欢迎使用 MyGPR 勘探定位工作台，快速掌握项目全局进展与关键指标")
        sub.setObjectName("pageSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        top_row.addLayout(title_col)
        top_row.addStretch(1)
        top_row.addWidget(QLabel(f"项目创建时间： {self.project_status.created_at}"))
        action = QPushButton("项目操作  ▾")
        action.setObjectName("smallButton")
        top_row.addWidget(action)
        top_row.addWidget(QPushButton("⌕"))
        body_layout.addLayout(top_row)
        metrics = QHBoxLayout()
        st = self.project_status
        for card in [
            self._metric_card(self.home_metric_cards, "lines", "⌘", "测线", str(st.line_count), "条", f"已导入 {st.imported_line_count} 条"),
            self._metric_card(self.home_metric_cards, "processed", "✓", "已处理", str(st.processed_line_count), "条", f"处理完成率 {st.processed_percent:.1f}%"),
            self._metric_card(self.home_metric_cards, "targets", "◎", "已确认目标", str(st.confirmed_target_count), "个", f"待复核 {st.pending_target_count} 个"),
            self._metric_card(self.home_metric_cards, "spatial", "✤", "空间定位点", str(st.spatial_point_count), "个", f"轨迹文件 {st.trajectory_file_count} 个"),
            self._metric_card(self.home_metric_cards, "reports", "▤", "报告状态", st.report_status, "", f"交付文件 {st.report_file_count} 个"),
        ]:
            metrics.addWidget(card)
        body_layout.addLayout(metrics)
        middle = QHBoxLayout()
        middle.setSpacing(8)
        # 删除“今日关注”卡片后，将横向空间完整分配给主流程和模块概览。
        # 首页优先呈现项目主线，不再在右侧堆提醒模块，避免 1080P 下布局挤压。
        middle.addWidget(self._home_flow_card(), 1)
        middle.addWidget(self._home_module_card(), 1)
        body_layout.addLayout(middle)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        # 活动卡片不再占用过大横向空间，避免挤压右侧交付与预览区域。
        bottom.addWidget(self._home_activity_card(), 4)
        bottom.addWidget(self._home_delivery_card(), 4)
        bottom.addWidget(self._home_preview_strip(), 3)
        body_layout.addLayout(bottom, 1)
        body_layout.addStretch(1)
        return scroll
    def _home_flow_card(self) -> Card:
        card = Card(title="项目流程概览")
        card.setMinimumHeight(self._compact_value(156, 132))
        row = QHBoxLayout()
        steps = [
            ("1", "项目管理", "项目创建\n数据导入\n参数配置", "已完成", "✓"),
            ("2", "测线处理", "测线处理\nB-scan 预览\n应用处理", "处理中", "◌"),
            ("3", "目标定位", "目标识别\n目标标注\n坐标定位", "部分完成", "○"),
            ("4", "空间成果", "空间映射\n高程计算\n成果生成", "已完成", "✓"),
            ("5", "成果报告", "报告编制\n成果导出\n交付管理", "已生成", "✓"),
        ]
        for idx, (num, name, body, status, mark) in enumerate(steps):
            box = QFrame()
            box.setObjectName("flowStep")
            v = QVBoxLayout(box)
            v.setContentsMargins(8, 6, 8, 6)
            badge = QLabel(num)
            badge.setObjectName("flowBadge")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setFixedSize(24, 24)
            v.addWidget(badge, 0, Qt.AlignmentFlag.AlignHCenter)
            icon = QLabel(["▤", "⌘", "◎", "✦", "▣"][idx])
            icon.setObjectName("flowIcon")
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(icon)
            title = QLabel(name)
            title.setObjectName("flowTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(title)
            desc = QLabel(body)
            desc.setObjectName("flowDesc")
            desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(desc)
            stat = QLabel(f"{status} {mark}")
            stat.setObjectName("flowStatus")
            stat.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(stat)
            row.addWidget(box, 1)
            if idx < len(steps) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("flowArrow")
                row.addWidget(arrow)
        card.layout.addLayout(row)
        return card
    def _home_module_card(self) -> Card:
        card = Card(title="模块快速概览")
        card.setMinimumHeight(self._compact_value(156, 132))
        grid = QGridLayout()
        grid.setSpacing(6)
        st = self.project_status
        items = [
            ("测线处理", f"已处理测线 {st.processed_line_count} / {st.line_count} 条\n已导入 {st.imported_line_count} 条", "前往模块  →"),
            ("目标定位", f"已确认目标 {st.confirmed_target_count} 个\n待复核目标 {st.pending_target_count} 个", "前往模块  →"),
            ("空间成果", f"定位点 {st.spatial_point_count} 个\n轨迹文件 {st.trajectory_file_count} 个", "前往模块  →"),
            ("成果报告", f"报告状态：{st.report_status}\n交付文件 {st.report_file_count} 个", "前往模块  →"),
        ]
        for i, (title, desc, link) in enumerate(items):
            tile = QFrame()
            tile.setObjectName("moduleTile")
            v = QVBoxLayout(tile)
            v.setContentsMargins(8, 6, 8, 6)
            thumb = QLabel("▨")
            thumb.setObjectName("moduleThumb")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(thumb)
            t = QLabel(title)
            t.setObjectName("moduleTitle")
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            d = QLabel(desc)
            d.setObjectName("moduleDesc")
            d.setWordWrap(True)
            v.addWidget(t)
            v.addWidget(d)
            l = QLabel(link)
            l.setObjectName("linkLabel")
            v.addWidget(l)
            grid.addWidget(tile, i // 4, i % 4)
        card.layout.addLayout(grid)
        return card
    def _home_attention_card(self) -> Card:
        card = Card(title="今日关注")
        card.setMinimumWidth(252)
        card.setMaximumWidth(280)
        card.setMinimumHeight(self._compact_value(156, 132))
        rows = self.project_status.attention_items
        for icon, title, desc, count in rows:
            row = QHBoxLayout()
            ic = QLabel(icon)
            ic.setObjectName("attentionIcon")
            row.addWidget(ic)
            col = QVBoxLayout()
            t = QLabel(title)
            t.setObjectName("attentionTitle")
            d = QLabel(desc)
            d.setObjectName("attentionDesc")
            d.setWordWrap(True)
            col.addWidget(t)
            col.addWidget(d)
            row.addLayout(col, 1)
            c = QLabel(count)
            c.setObjectName("attentionCount")
            row.addWidget(c)
            card.layout.addLayout(row)
        return card
    def _home_activity_card(self) -> Card:
        card = Card(title="最近项目活动")
        card.setMinimumHeight(self._compact_value(236, 190))
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        rows = self.project_status.activity_rows[:6]
        for idx, (icon, title, desc, time) in enumerate(rows):
            item = QFrame()
            item.setObjectName("activityTile")
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(6, 4, 6, 4)
            item_layout.setSpacing(6)
            ic = QLabel(icon)
            ic.setObjectName("activityIcon")
            ic.setFixedWidth(16)
            item_layout.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(1)
            t = QLabel(title)
            t.setObjectName("activityTitle")
            t.setWordWrap(False)
            d = QLabel(desc)
            d.setObjectName("activityDesc")
            d.setWordWrap(True)
            tm = QLabel(time)
            tm.setObjectName("timeLabel")
            col.addWidget(t)
            col.addWidget(d)
            col.addWidget(tm)
            item_layout.addLayout(col, 1)
            grid.addWidget(item, idx // 2, idx % 2)
        if not rows:
            empty = QLabel("暂无项目活动")
            empty.setObjectName("activityDesc")
            grid.addWidget(empty, 0, 0)
        card.layout.addLayout(grid, 1)
        return card
    def _home_delivery_card(self) -> Card:
        card = Card(title="交付成果概览")
        card.setMinimumHeight(self._compact_value(132, 112))
        table = self._table(["文件名称", "类型", "状态", "更新时间"], 4)
        rows = [(name, file_type, status, updated) for name, file_type, _size, updated, status, _action in self.project_status.delivery_rows]
        self._fill_table(table, rows)
        card.layout.addWidget(table, 1)
        return card
    def _home_preview_strip(self) -> QWidget:
        wrap = QWidget()
        wrap.setMinimumHeight(self._compact_value(236, 188))
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        map_card = PlotCard("项目位置概览", height=self._compact_value(90, 70))
        map_card.layout.setContentsMargins(6, 4, 6, 4)
        self._draw_current_line_strip(map_card.canvas)
        v.addWidget(map_card, 1)
        bscan = PlotCard("典型 B-scan 预览", height=self._compact_value(104, 78))
        bscan.layout.setContentsMargins(6, 4, 6, 4)
        self._draw_current_line_bscan(bscan.canvas, title="")
        v.addWidget(bscan, 1)
        return wrap


__all__ = ["HomePageMixin"]

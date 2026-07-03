#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Delivery/report page mixin for the field workbench."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from ui.field_panels.layout_metrics import layout_metrics_for
from ui.field_panels.widgets import Card, CollapsibleSidePanel, MetricCard, PlotCard
from ui.field_panels.plots import draw_bscan
from core.field_report_export import generate_project_report_package
from core.project_events import ProjectEventType


class DeliveryPageMixin:
    def _page_delivery(self) -> QWidget:
        widget = QWidget()
        v = QVBoxLayout(widget)
        v.setContentsMargins(0, 0, 0, 0)
        lm = layout_metrics_for(self)
        v.setSpacing(lm.spacing)

        top = QHBoxLayout()
        top.setSpacing(lm.spacing)
        st = self.project_status
        blockers = [item for item in st.attention_items if item[0] == "⚠"]
        for card in [
            MetricCard("▤", "报告状态", st.report_status, "", f"最后更新：{st.latest_update}"),
            MetricCard("▰", "交付文件数", str(st.report_file_count), "个", f"项目大小：{st.storage_usage_mb:.1f} MB"),
            MetricCard("☑", "已检查项", str(len(st.task_rows)), "项", f"导入：{st.imported_line_count}/{st.line_count}"),
            MetricCard("⚠", "待处理问题", str(len(blockers)), "项", "严重：0 项"),
        ]:
            top.addWidget(card)
        v.addLayout(top)

        action = QHBoxLayout()
        action.addStretch(1)
        generate_btn = QPushButton("生成报告包")
        generate_btn.setObjectName("primaryButton")
        generate_btn.clicked.connect(self._action_generate_report_package)
        action.addWidget(generate_btn)
        pdf_btn = QPushButton("生成/打开 PDF")
        pdf_btn.setObjectName("smallButton")
        pdf_btn.clicked.connect(self._action_generate_or_open_pdf_report)
        action.addWidget(pdf_btn)
        open_btn = QPushButton("打开报告目录")
        open_btn.setObjectName("smallButton")
        open_btn.clicked.connect(self._action_open_reports_dir)
        action.addWidget(open_btn)
        v.addLayout(action)
        if bool(getattr(st, "dirty_modules", {}).get("report")):
            reasons = (getattr(st, "stale_reasons", {}) or {}).get("report", [])
            reason_text = "；".join(reasons[-2:]) if reasons else "项目数据已变化"
            stale_label = QLabel(f"◷  成果报告需重新生成：{reason_text}")
            stale_label.setObjectName("staleNotice")
            v.addWidget(stale_label)

        main = QHBoxLayout()
        main.setSpacing(lm.spacing)
        main.addWidget(self._report_preview_card(), 7)
        right = QVBoxLayout()
        right.setSpacing(lm.spacing)
        right.addWidget(self._check_results_card(), 3)
        right.addWidget(self._delivery_info_card(), 2)
        right_widget = QWidget()
        right_widget.setLayout(right)
        side_panel = CollapsibleSidePanel(
            title="报告辅助",
            content=right_widget,
            expanded_width=lm.delivery_side_min_w,
            collapsed_width=34,
        )
        side_panel.setProperty("layoutKey", "deliveryAuxSidePanel")
        main.addWidget(side_panel, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(main, 3)

        v.addWidget(self._delivery_files_card(), 1)
        return widget

    def _action_generate_report_package(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "生成报告包", "请先新建或打开 MyGPR 项目。")
            return
        try:
            result = generate_project_report_package(self.project_store)
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.REPORT_GENERATED, line_id=self.selected_line, reason="成果报告包已生成", changed_paths=[result.package_dir], refresh=False)
            self._refresh_project_status_snapshot()
            self._line_status_message = f"成果报告包已生成：{result.package_dir}"
            self._refresh_project_widgets()
            QMessageBox.information(
                self,
                "生成报告包",
                "成果报告包已生成。\n"
                f"目录：{result.package_dir}\n"
                f"HTML：{result.html_path}\n"
                f"文件数：{result.file_count}",
            )
        except Exception as exc:
            self._show_operation_error("生成报告包", exc)

    def _action_open_reports_dir(self) -> None:
        if self.project_root is None:
            QMessageBox.warning(self, "打开报告目录", "请先新建或打开 MyGPR 项目。")
            return
        reports_dir = Path(self.project_root) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(reports_dir)))

    def _action_generate_or_open_pdf_report(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "PDF 报告", "请先新建或打开 MyGPR 项目。")
            return
        try:
            result = generate_project_report_package(self.project_store)
            pdf_path = Path(self.project_store.root) / result.pdf_path
            if not pdf_path.exists():
                raise FileNotFoundError(pdf_path)
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.REPORT_GENERATED, line_id=self.selected_line, reason="PDF 报告已生成", changed_paths=[pdf_path], refresh=False)
            self._refresh_project_status_snapshot()
            self._line_status_message = f"PDF 报告已生成：{pdf_path}"
            self._refresh_project_widgets()
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path)))
            QMessageBox.information(self, "PDF 报告", f"PDF 报告已生成：\n{pdf_path}")
        except Exception as exc:
            self._show_operation_error("PDF 报告", exc)

    def _report_preview_card(self) -> Card:
        card = Card(title="PDF 报告预览")
        card.setProperty("layoutKey", "deliveryReportPreviewCard")
        row = QHBoxLayout()
        lm = layout_metrics_for(self)
        cover = QFrame()
        cover.setObjectName("reportPage")
        cover.setProperty("layoutKey", "deliveryReportCover")
        cover.setFixedWidth(lm.delivery_cover_w)
        cover.setMinimumHeight(lm.delivery_cover_min_h)
        c = QVBoxLayout(cover)
        c.setContentsMargins(18, 18, 18, 18)
        c.addStretch(1)
        rt = QLabel(f"{self.project_status.project_name}\n项目成果报告")
        rt.setObjectName("reportTitle")
        rt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c.addWidget(rt)
        img = PlotCard(None, height=lm.delivery_report_thumb_h)
        img.setProperty("layoutKey", "deliveryReportThumbCard")
        img.canvas.setObjectName("deliveryReportThumbCanvas")
        img.layout.setContentsMargins(0, 0, 0, 0)
        self._draw_current_line_bscan(img.canvas, title="")
        c.addWidget(img)
        c.addWidget(QLabel(f"测区位置：{self.project_status.location}"), 0, Qt.AlignmentFlag.AlignCenter)
        c.addWidget(QLabel(f"最后更新：{self.project_status.latest_update}"), 0, Qt.AlignmentFlag.AlignCenter)
        c.addStretch(1)
        toc = QFrame()
        toc.setObjectName("reportPage")
        toc.setProperty("layoutKey", "deliveryReportToc")
        toc.setMinimumHeight(lm.delivery_toc_min_h)
        toc.setMaximumWidth(520 if lm.compact else 640)
        t = QVBoxLayout(toc)
        t.setContentsMargins(18, 18, 18, 18)
        t.setSpacing(6)
        for line in [
            "1. 项目概况",
            "   1.1 项目背景     1.2 测区概况     1.3 作业情况     1.4 数据概况",
            "2. 测线统计",
            "   2.1 测线列表     2.2 数据质量评估     2.3 数据量情况",
            "3. 目标定位成果",
            "   3.1 目标统计     3.2 典型目标剖面     3.3 目标列表",
            "4. 空间成果",
            "   4.1 平面成果     4.2 剖面成果     4.3 三维成果",
            "5. 处理记录",
            "   5.1 处理流程     5.2 参数设置     5.3 处理日志",
            "6. 质量检查结论",
            "   6.1 检查概况     6.2 问题汇总     6.3 结论与建议",
        ]:
            lbl = QLabel(line)
            lbl.setObjectName("tocLine")
            t.addWidget(lbl)
        t.addStretch(1)
        figure = QFrame()
        figure.setObjectName("reportPage")
        figure.setProperty("layoutKey", "deliveryReportFigure")
        figure.setMinimumHeight(lm.delivery_toc_min_h)
        figure.setMaximumWidth(360 if lm.compact else 460)
        f = QVBoxLayout(figure)
        f.setContentsMargins(18, 18, 18, 18)
        fig_title = QLabel("当前图件预览")
        fig_title.setObjectName("sectionTitle")
        f.addWidget(fig_title)
        figure_plot = PlotCard(None, height=max(lm.delivery_report_thumb_h + 22, 110))
        figure_plot.setProperty("layoutKey", "deliveryReportFigureThumbCard")
        figure_plot.canvas.setObjectName("deliveryReportFigureThumbCanvas")
        figure_plot.layout.setContentsMargins(0, 0, 0, 0)
        self._draw_current_line_bscan(figure_plot.canvas, title="")
        f.addWidget(figure_plot)
        for text in [
            f"测线：{self.selected_line}",
            f"目标：{self.project_status.target_count} 个",
            f"空间点：{self.project_status.spatial_point_count} 个",
            f"检查项：{len(self.project_status.task_rows)} 项",
        ]:
            lbl = QLabel(text)
            lbl.setObjectName("activityDesc")
            f.addWidget(lbl)
        f.addStretch(1)

        row.addWidget(cover, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(toc, 1, Qt.AlignmentFlag.AlignTop)
        row.addWidget(figure, 0, Qt.AlignmentFlag.AlignTop)
        card.layout.addLayout(row)

        status = QFrame()
        status.setObjectName("reportStatusStrip")
        status_row = QHBoxLayout(status)
        status_row.setContentsMargins(10, 6, 10, 6)
        status_row.setSpacing(10)
        for title, desc in [
            ("报告结构", "6 章 / 28 页，按项目交付模板组织"),
            ("成果检查", f"通过 {len(self.project_status.task_rows)} 项，待处理 {len([i for i in self.project_status.attention_items if i[0] == '⚠'])} 项"),
            ("导出格式", "HTML / PDF / CSV / JSON 可同步生成"),
        ]:
            col = QVBoxLayout()
            col.setSpacing(1)
            a = QLabel(title)
            a.setObjectName("miniStatLabel")
            b = QLabel(desc)
            b.setObjectName("detailSubtitle")
            b.setWordWrap(False)
            col.addWidget(a)
            col.addWidget(b)
            status_row.addLayout(col, 1)
        card.layout.addWidget(status)
        nav = QHBoxLayout()
        nav.addStretch(1)
        nav.addWidget(QLabel("‹     1  / 28     ›"))
        nav.addStretch(1)
        nav.addWidget(QPushButton("100%  ▾"))
        fullscreen_btn = QPushButton("全屏预览")
        fullscreen_btn.setObjectName("smallButton")
        fullscreen_btn.clicked.connect(self._action_open_report_preview_dialog)
        nav.addWidget(fullscreen_btn)
        card.layout.addLayout(nav)
        return card


    def _action_open_report_preview_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("项目报告预览 - 放大查看")
        dialog.resize(1180, 760)
        dialog.setMinimumSize(900, 600)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._report_preview_card(), 1)
        dialog.exec()

    def _check_results_card(self) -> Card:
        card = Card(title="成果检查结果")
        card.setProperty("layoutKey", "deliveryCheckResultsCard")
        lm = layout_metrics_for(self)
        card.setMinimumWidth(lm.delivery_side_min_w)
        card.setMaximumHeight(lm.delivery_side_max_h)
        run = QPushButton("运行成果检查  ▾")
        run.setObjectName("smallButton")
        card.layout.addWidget(run, alignment=Qt.AlignmentFlag.AlignRight)
        table = self._table(["检查内容", "状态", "说明", "时间"], 6)
        rows = []
        st = self.project_status
        rows.append(("测线数据导入", "● 通过" if st.imported_line_count == st.line_count and st.line_count else "⚠ 待补充", f"{st.imported_line_count}/{st.line_count} 条", st.latest_update))
        rows.append(("RTK/IMU 轨迹", "● 通过" if st.trajectory_file_count == st.line_count and st.line_count else "⚠ 待补充", f"{st.trajectory_file_count}/{st.line_count} 条", st.latest_update))
        rows.append(("处理结果", "● 通过" if st.processed_line_count else "◷ 待处理", f"{st.processed_line_count}/{st.line_count} 条", st.latest_update))
        rows.append(("目标标注", "● 通过" if st.target_count else "◷ 待标注", f"目标 {st.target_count} 个", st.latest_update))
        rows.append(("空间成果状态", "◷ 需刷新" if getattr(st, "dirty_modules", {}).get("spatial") else "● 最新", "；".join((getattr(st, "stale_reasons", {}) or {}).get("spatial", [])[-1:]) or "空间成果可用", st.latest_update))
        rows.append(("成果文件", "◷ 需重新生成" if getattr(st, "dirty_modules", {}).get("report") else ("● 通过" if st.report_file_count else "◷ 未生成"), f"交付文件 {st.report_file_count} 个", st.latest_update))
        self._fill_table(table, rows)
        card.layout.addWidget(table)
        card.layout.addWidget(QLabel(f"检查完成：{len(rows)} 项      待处理：{sum(1 for r in rows if '待' in r[1] or '未' in r[1])} 项"))
        return card

    def _delivery_files_card(self) -> Card:
        card = Card()
        card.setProperty("layoutKey", "deliveryFilesCard")
        card.setMaximumHeight(layout_metrics_for(self).delivery_files_max_h)
        tabs = QTabWidget()
        tabs.setObjectName("innerTabs")
        table = self._table(["文件名称", "类型", "大小", "更新时间", "说明", "操作"], 6)
        rows = self.project_status.delivery_rows
        if not rows:
            rows = [("暂无正式交付文件", "--", "--", "--", "请先生成成果报告或导出成果", "--")]
        self._fill_table(table, rows)
        tabs.addTab(table, "交付文件")
        task_table = self._table(["任务名称", "类型", "状态", "进度", "开始时间", "结束时间", "操作"], 5)
        self._fill_table(task_table, self.project_status.task_rows)
        tabs.addTab(task_table, "任务")
        log_table = self._table(["类型", "事件", "说明", "时间"], 5)
        self._fill_table(log_table, self.project_status.activity_rows)
        tabs.addTab(log_table, "日志")
        card.layout.addWidget(tabs)
        return card

    def _delivery_info_card(self) -> Card:
        card = Card(title="导出与交付信息")
        card.setProperty("layoutKey", "deliveryInfoCard")
        lm = layout_metrics_for(self)
        card.setMinimumWidth(lm.delivery_side_min_w)
        card.setMaximumHeight(lm.delivery_side_max_h)
        report_dir = str((self.project_root / "reports") if self.project_root is not None else "--")
        fields = [
            ("报告", f"{self.project_status.project_name}_成果报告"),
            ("路径", report_dir),
            ("格式", "待生成" if self.project_status.report_file_count == 0 else "reports 目录文件"),
            ("操作员", self.project_status.operator),
            ("时间", self.project_status.latest_update),
            ("版本", "V1.0"),
        ]
        for key, value in fields:
            row = QHBoxLayout()
            row.setSpacing(lm.spacing)
            key_label = QLabel(key)
            key_label.setObjectName("keyLabel")
            key_label.setFixedWidth(46 if lm.compact else 58)
            value_label = QLabel(str(value))
            value_label.setObjectName("fieldBox")
            value_label.setToolTip(str(value))
            value_label.setWordWrap(False)
            value_label.setMaximumHeight(24 if lm.compact else 30)
            row.addWidget(key_label)
            row.addWidget(value_label, 1)
            card.layout.addLayout(row)
        note = QLabel("可生成 CSV / JSON / HTML / PDF；CSV/JSON 保留审计源数据。")
        note.setObjectName("activityDesc")
        note.setWordWrap(True)
        card.layout.addWidget(note)
        card.layout.addStretch(1)
        return card



__all__ = ["DeliveryPageMixin"]

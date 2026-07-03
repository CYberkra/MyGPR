#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial results page mixin for the field workbench."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QEvent, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from core.project_events import ProjectEventType
from ui.field_panels.layout_metrics import layout_metrics_for
from ui.field_panels.widgets import Card, CollapsibleSidePanel, MetricCard, PlotCard
from ui.field_panels.plots import _style_axis
from ui.field_panels.preview_helpers import _set_comfort_limits
from ui.field_panels.spatial_3d_dialog import Spatial3DDialog, collect_project_spatial_scene


class SpatialPageMixin:
    def _page_spatial(self) -> QWidget:
        widget = QWidget()
        v = QVBoxLayout(widget)
        v.setContentsMargins(0, 0, 0, 0)
        lm = layout_metrics_for(self)
        v.setSpacing(lm.spacing)
        self.spatial_layer_state = getattr(
            self,
            "spatial_layer_state",
            {"trajectories": True, "targets": True, "elevation": True},
        )

        # --- Metric cards ---
        metrics = QHBoxLayout()
        metrics.setSpacing(lm.spacing)
        st = self.project_status
        completeness = 0.0 if st.line_count <= 0 else st.trajectory_file_count / st.line_count * 100.0
        metric_defs = [
            ("📐", "已定位测线", str(st.trajectory_file_count), "条"),
            ("◎", "空间定位点", str(st.spatial_point_count), "个"),
            ("✦", "目标关联数", str(st.target_count), "组"),
            ("◈", "高程完整性", f"{completeness:.1f}", "%"),
        ]
        self.spatial_metric_cards = []
        for icon, title, value, suffix in metric_defs:
            card = MetricCard(icon, title, value, suffix)
            metrics.addWidget(card)
            self.spatial_metric_cards.append(card)
        v.addLayout(metrics)

        # --- Status strip: current context ---
        selected = self._selected_line_record()
        ctx_line = selected.get("name", "") or selected.get("id", "--")
        session = getattr(self, "processing_session", None)
        step_text = f"Step {session.current_step_index:02d}" if session and session.step_count else "原始"
        status_strip = QLabel(f"当前测线：{ctx_line} | 处理：{step_text}")
        status_strip.setObjectName("statusStrip")
        v.addWidget(status_strip)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(lm.spacing)
        refresh_btn = QPushButton("↻ 刷新")
        refresh_btn.setObjectName("primaryButton")
        refresh_btn.clicked.connect(self._action_refresh_spatial_results)
        toolbar.addWidget(refresh_btn)

        export_btn = QPushButton("⇧ 导出坐标")
        export_btn.setObjectName("smallButton")
        export_btn.clicked.connect(self._action_export_spatial_coordinates)
        toolbar.addWidget(export_btn)

        more_btn = QPushButton("⋯")
        more_btn.setObjectName("smallButton")
        more_menu = QMenu(more_btn)
        more_menu.addAction("⛶ 三维视图", self._action_open_3d_view)
        more_menu.addAction("▧ 生成平面图", self._action_generate_plan_map)
        more_btn.setMenu(more_menu)
        toolbar.addWidget(more_btn)

        toolbar.addStretch(1)
        layer_btn = QPushButton("◎ 图层")
        layer_btn.setObjectName("smallButton")
        layer_btn.setMenu(self._make_spatial_layer_menu(layer_btn))
        toolbar.addWidget(layer_btn)
        v.addLayout(toolbar)

        # --- Stale notice ---
        if bool(getattr(st, "dirty_modules", {}).get("spatial")):
            reasons = (getattr(st, "stale_reasons", {}) or {}).get("spatial", [])
            reason_text = "；".join(reasons[-2:]) if reasons else "目标或轨迹数据已变化"
            stale_label = QLabel(f"◷  空间成果需刷新：{reason_text}")
            stale_label.setObjectName("staleNotice")
            v.addWidget(stale_label)

        # --- Main area ---
        main = QHBoxLayout()
        main.setSpacing(lm.spacing)

        # Left column: map + summary
        left = QVBoxLayout()
        left.setSpacing(lm.spacing)
        map_card = PlotCard("工程平面图", height=lm.spatial_map_h)
        map_card.setProperty("layoutKey", "spatialMapCard")
        map_card.canvas.setObjectName("spatialMapCanvas")
        map_card.canvas.setProperty("previewAutoAspect", True)
        map_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        map_card.layout.setContentsMargins(0, 0, 0, 0)
        self.spatial_map_canvas = map_card.canvas
        self._draw_project_spatial_map(map_card.canvas)
        left.addWidget(map_card, 1)

        summary = self._spatial_summary_table()
        summary.setMaximumHeight(lm.spatial_table_max_h)
        left.addWidget(summary, 0)
        main.addLayout(left, 7)

        # Right column: elevation and info — secondary analysis band
        right = QVBoxLayout()
        right.setSpacing(lm.spacing)
        right.setContentsMargins(0, 0, 0, 0)

        # Elevation profile — bypass Card to avoid its internal layout constraints.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as _FigureCanvas
        from matplotlib.figure import Figure as _Figure
        elevation_container = QWidget()
        elevation_layout = QVBoxLayout(elevation_container)
        elevation_layout.setContentsMargins(3, 2, 3, 2)
        elevation_layout.setSpacing(1)
        # Title
        _elev_title = QLabel("高程剖面")
        _elev_title.setObjectName("cardTitle")
        elevation_layout.addWidget(_elev_title)
        # Canvas — expanding to fill all available space
        self.spatial_elevation_canvas = _FigureCanvas(_Figure(figsize=(4, 2.4), dpi=100, facecolor="white"))
        self.spatial_elevation_canvas.setObjectName("spatialProfileCanvas")
        self.spatial_elevation_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.spatial_elevation_canvas.setMinimumHeight(0)
        self.spatial_elevation_canvas.setMaximumHeight(16777215)
        elevation_layout.addWidget(self.spatial_elevation_canvas, 1)
        # Install event filter to redraw when the canvas resizes via Qt layout.
        self.spatial_elevation_canvas.installEventFilter(self)
        self._draw_current_elevation_profile(self.spatial_elevation_canvas)
        right.addWidget(elevation_container, 1)

        info_panel = self._spatial_info_panel()
        info_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        right.addWidget(info_panel, 0)

        right_widget = QWidget()
        right_widget.setLayout(right)
        side_panel = CollapsibleSidePanel(
            title="空间辅助",
            content=right_widget,
            expanded_width=lm.spatial_side_w,
            collapsed_width=30,
        )
        side_panel.setProperty("layoutKey", "spatialAuxSidePanel")
        main.addWidget(side_panel, 1)
        v.addLayout(main, 1)

        # Redraw elevation profile after layout settles so the figure matches the
        # actual canvas size (canvas starts at its default size before the page is shown).
        QTimer.singleShot(400, self._deferred_elevation_redraw)
        return widget

    def _deferred_elevation_redraw(self) -> None:
        canvas = getattr(self, "spatial_elevation_canvas", None)
        if canvas is None:
            return
        # Query the parent container's size for figure sizing.
        parent = canvas.parentWidget()
        if parent and parent.width() > 0 and parent.height() > 0:
            dpi = canvas.figure.dpi or 100
            w = max(parent.width() - 8, 200)
            h = max(parent.height() - 20, 120)
            canvas.figure.set_size_inches(w / dpi, h / dpi)
            # Trigger a canvas resize so the layout and eventFilter update.
            canvas.resize(w, h)

    def eventFilter(self, obj, event) -> bool:
        if (event.type() == QEvent.Type.Resize
                and obj is getattr(self, "spatial_elevation_canvas", None)
                and obj.isVisible()):
            self._draw_current_elevation_profile(obj)
        return super().eventFilter(obj, event)

    def _make_spatial_layer_menu(self, parent: QWidget) -> QMenu:
        menu = QMenu(parent)
        definitions = [
            ("trajectories", "测线轨迹"),
            ("targets", "目标点"),
            ("elevation", "高程/DEM"),
        ]
        for key, label in definitions:
            action = QAction(label, menu)
            action.setCheckable(True)
            action.setChecked(bool(self.spatial_layer_state.get(key, True)))
            action.toggled.connect(lambda checked, k=key: self._set_spatial_layer(k, checked))
            menu.addAction(action)
        return menu

    def _set_spatial_layer(self, key: str, checked: bool) -> None:
        self.spatial_layer_state[key] = bool(checked)
        self._line_status_message = f"空间图层已更新：{key}={'显示' if checked else '隐藏'}"
        self._draw_project_spatial_map(self.spatial_map_canvas)

    def _action_refresh_spatial_results(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "刷新空间成果", "请先新建或打开 MyGPR 项目。")
            return
        exported = []
        try:
            for line in self.project_store.list_lines():
                try:
                    if self.project_store.load_targets(line.line_id):
                        exported.append(self.project_store.export_spatial_targets_xy(line.line_id))
                except Exception:
                    continue
            self._refresh_project_status_snapshot()
            self._line_status_message = f"空间成果已刷新：{len(exported)} 个目标坐标文件。"
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.SPATIAL_RESULTS_REFRESHED, line_id=self.selected_line, reason=f"空间成果已刷新：{len(exported)} 个目标坐标文件", refresh=False)
            self._post_project_operation_refresh(switch_to="spatial")
        except Exception as exc:
            self._show_operation_error("刷新空间成果", exc)

    def _action_export_spatial_coordinates(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "导出坐标成果", "请先新建或打开 MyGPR 项目。")
            return
        try:
            path = self.project_store.export_project_spatial_coordinates()
            self._refresh_project_status_snapshot()
            self._line_status_message = f"坐标成果已导出：{path}"
            if getattr(self, "linkage_controller", None) is not None:
                self.linkage_controller.emit(ProjectEventType.SPATIAL_EXPORT_GENERATED, line_id=self.selected_line, reason="项目空间坐标成果已导出", changed_paths=[path], refresh=False)
            self._post_project_operation_refresh(switch_to="spatial")
            QMessageBox.information(self, "导出坐标成果", f"已导出项目空间坐标成果：\n{path}")
        except Exception as exc:
            self._show_operation_error("导出坐标成果", exc)

    def _action_open_3d_view(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "打开三维视图", "请先新建或打开 MyGPR 项目。")
            return
        dialog = Spatial3DDialog(self, project_store=self.project_store, selected_line=self.selected_line)
        dialog.exec()

    def _action_generate_plan_map(self) -> None:
        if self.project_store is None:
            QMessageBox.warning(self, "生成平面图", "请先新建或打开 MyGPR 项目。")
            return
        default = self.project_store.root / "spatial" / f"project_plan_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "生成平面图", str(default), "PNG 图像 (*.png);;PDF 文件 (*.pdf);;SVG 文件 (*.svg)")
        if not path:
            return
        out = Path(path)
        suffix = out.suffix.lower()
        if suffix not in {".png", ".pdf", ".svg"}:
            out = out.with_suffix(".png")
        try:
            if not hasattr(self, "spatial_map_canvas") or self.spatial_map_canvas is None:
                raise RuntimeError("空间平面图画布未初始化")
            out.parent.mkdir(parents=True, exist_ok=True)
            self.spatial_map_canvas.figure.savefig(out, dpi=180, bbox_inches="tight")
            self._line_status_message = f"平面图已生成：{out}"
            self._refresh_project_widgets()
            QMessageBox.information(self, "生成平面图", f"已生成平面成果图：\n{out}")
        except Exception as exc:
            self._show_operation_error("生成平面图", exc)

    def _spatial_summary_table(self) -> Card:
        card = Card(title="测线汇总")
        card.setMaximumHeight(layout_metrics_for(self).spatial_table_max_h)
        table = self._table(["测线", "长度 (m)", "RTK", "定位点", "目标", "状态"], 4)
        table.setObjectName("spatialSummaryTable")
        rows = []
        for line in self.line_records:
            line_id = str(line.get("id", "--"))
            target_count = str(line.get("targets", 0) or 0)
            start_xy = end_xy = "--"
            point_count = "0"
            status = line.get("status", "--")
            if self.project_store is not None:
                try:
                    line_record = self.project_store.get_line(line_id)
                    if not line_record.trajectory_path:
                        raise FileNotFoundError("trajectory not attached")
                    traj = self.project_store.load_trajectory(line_id)
                    point_count = str(len(traj.points))
                    if traj.points:
                        start_xy = f"({traj.points[0].x:.3f}, {traj.points[0].y:.3f})"
                        end_xy = f"({traj.points[-1].x:.3f}, {traj.points[-1].y:.3f})"
                except Exception:
                    pass
            length_str = f"{float(line.get('length', 0.0)):.2f}"
            rtk = line.get("rtk", "--")
            rows.append((line_id, length_str, rtk, point_count, target_count, status))
        if not rows:
            rows = [("暂无测线", "--", "--", "0", "0", "--")]
        self._fill_table(table, rows, highlight_row=self._selected_line_row() if self.line_records else -1)
        card.layout.addWidget(table)
        return card

    def _spatial_info_panel(self) -> Card:
        card = Card(title="空间信息")
        card.setProperty("layoutKey", "spatialInfoPanel")
        line = self._selected_line_record()
        coord_sys = self.project_status.coordinate_system or '--'

        # Group 1: coordinate system
        lbl = QLabel(f"坐标系：{coord_sys}")
        lbl.setObjectName("smallInfo")
        lbl.setWordWrap(True)
        card.layout.addWidget(lbl)

        # Group 2: selected line details
        lbl = QLabel(f"当前测线：{line.get('id', '--')} {line.get('name', '暂无测线')}")
        lbl.setObjectName("smallInfo")
        lbl.setWordWrap(True)
        card.layout.addWidget(lbl)

        lbl = QLabel(f"长度：{float(line.get('length', 0.0)):.2f} m　状态：{line.get('status', '--')}")
        lbl.setObjectName("smallInfo")
        lbl.setWordWrap(True)
        card.layout.addWidget(lbl)

        # Separator
        sep = QLabel("─" * 30)
        sep.setObjectName("activityDesc")
        card.layout.addWidget(sep)

        # Group 3: layer summary
        lbl = QLabel(self._spatial_layer_summary())
        lbl.setObjectName("smallInfo")
        lbl.setWordWrap(True)
        card.layout.addWidget(lbl)

        return card

    def _spatial_layer_summary(self) -> str:
        state = getattr(self, "spatial_layer_state", {"trajectories": True, "targets": True, "elevation": True})
        labels = []
        if state.get("trajectories", True):
            labels.append("测线轨迹")
        if state.get("targets", True):
            labels.append("目标点")
        if state.get("elevation", True):
            labels.append("高程剖面/DEM")
        return "图层：" + (" / ".join(labels) if labels else "全部隐藏")

    def _draw_project_spatial_map(self, canvas) -> None:
        if self.project_store is None or not self.line_records:
            self._draw_empty_plot(canvas, "暂无空间成果\n请先导入含定位信息的测线")
            return
        trajectories, targets = collect_project_spatial_scene(self.project_store)
        has_trajectory = bool(trajectories) and self.spatial_layer_state.get("trajectories", True)
        has_targets = bool(targets) and self.spatial_layer_state.get("targets", True)
        if not has_trajectory and not has_targets:
            self._draw_empty_plot(canvas, "当前图层无可显示数据\n请在图层控制中启用轨迹或目标点")
            return
        x_values = []
        y_values = []
        if has_trajectory:
            for item in trajectories:
                x_values.extend(list(item["x"]))
                y_values.extend(list(item["y"]))
        if has_targets:
            for target in targets:
                if target.get("x") is not None and target.get("y") is not None:
                    x_values.append(float(target["x"]))
                    y_values.append(float(target["y"]))
        if not x_values or not y_values:
            self._draw_empty_plot(canvas, "暂无空间坐标\n导入 RTK/IMU 或目标坐标后显示")
            return
        x_min = float(np.nanmin(x_values))
        y_min = float(np.nanmin(y_values))
        fig = canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if has_trajectory:
            for item in trajectories:
                x = item["x"] - x_min
                y = item["y"] - y_min
                line_width = 2.0 if item["line_id"] == self.selected_line else 1.3
                ax.plot(x, y, linewidth=line_width, marker="o", markersize=2, markevery=max(1, len(x)//20), label=item["line_id"])
        if has_targets:
            target_points = [target for target in targets if target.get("x") is not None and target.get("y") is not None]
            if target_points:
                tx = np.asarray([float(t["x"]) - x_min for t in target_points], dtype=float)
                ty = np.asarray([float(t["y"]) - y_min for t in target_points], dtype=float)
                ax.scatter(tx, ty, marker="^", s=24, label="目标点")
                for t in target_points[:24]:
                    ax.text(float(t["x"]) - x_min, float(t["y"]) - y_min, str(t.get("target_id", "T")), fontsize=6)
        ax.legend(loc="upper right", fontsize=7)
        # Use a comfortable viewport for sparse or short survey lines.  Pure
        # autoscale can make one point or a short line float in a huge empty
        # plot, which passes geometry checks but looks visually unfinished.
        rel_x_values = [float(x) - x_min for x in x_values]
        rel_y_values = [float(y) - y_min for y in y_values]
        # Keep engineering aspect ratio, but avoid the old over-wide 36 m
        # fallback that made short imported lines look tiny in a sea of blank
        # space.  A 12 m minimum still prevents single-point over-zooming while
        # making 5–10 m demo/short survey lines readable.
        _set_comfort_limits(ax, rel_x_values, rel_y_values, min_span=12.0, pad_ratio=0.18)
        if bool(canvas.property("previewAutoAspect")):
            # In-page preview prioritizes using the available card area so the
            # page does not look empty.  The enlarged viewer keeps the true
            # equal-aspect engineering view.
            ax.set_aspect("auto", adjustable="box")
        else:
            ax.set_aspect("equal", adjustable="box")
        ax.set_title("项目测线轨迹与目标点（工程坐标，相对显示）", fontsize=9, loc="left", fontweight="bold")
        ax.set_xlabel("相对 X (m)", fontsize=8)
        ax.set_ylabel("相对 Y (m)", fontsize=8)
        _style_axis(ax)
        fig.tight_layout(pad=0.55)
        canvas.draw_idle()

    def _draw_current_elevation_profile(self, canvas) -> None:
        if not getattr(self, "spatial_layer_state", {"elevation": True}).get("elevation", True):
            self._draw_empty_plot(canvas, "高程图层已隐藏")
            return
        if self.trajectory_model is None or not getattr(self.trajectory_model, "points", None):
            self._draw_empty_plot(canvas, "暂无高程数据\n当前测线缺少轨迹信息")
            return
        fig = canvas.figure
        fig.clear()
        # Size figure to match the canvas's natural size (not the parent).
        # The canvas's sizeHint equals the figure's size, so the layout gives
        # the canvas exactly the space the figure requests.  We query the
        # canvas's current size which Qt has already allocated.
        dpi = fig.dpi or 100
        w = max(canvas.width(), 200)
        h = max(canvas.height(), 120)
        fig.set_size_inches(w / dpi, h / dpi)
        # NOTE: we do NOT call canvas.resize() here — that would make the
        # canvas bigger than its parent and cause overflow.  Instead, the
        # eventFilter catches future resize events and redraws the figure.
        d = self.trajectory_model.distance
        z = self.trajectory_model.z
        # --- Top subplot: elevation profile line ---
        ax1 = fig.add_subplot(211)
        ax1.plot(d, z, color="#0D91B2", linewidth=1.5)
        ax1.fill_between(d, z, np.nanmin(z), color="#0D91B2", alpha=0.12)
        ax1.set_title(f"高程剖面（{self.selected_line}）", fontsize=9, loc="left", fontweight="bold")
        ax1.set_ylabel("高程", fontsize=8)
        ax1.tick_params(labelsize=7)
        _style_axis(ax1)
        # --- Bottom subplot: DEM color strip ---
        ax2 = fig.add_subplot(212)
        image = np.tile(z, (8, 1))
        im = ax2.imshow(image, aspect="auto", cmap="terrain", extent=[float(d[0]), float(d[-1]), 0, 1])
        ax2.set_yticks([])
        ax2.set_xlabel("距离 (m)", fontsize=8)
        ax2.set_title("高程色带", fontsize=9, loc="left", fontweight="bold")
        ax2.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        _style_axis(ax2)
        fig.tight_layout(pad=0.55)
        canvas.draw_idle()


__all__ = ["SpatialPageMixin"]

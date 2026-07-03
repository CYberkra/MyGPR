#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D spatial deliverable dialog for the MyGPR field workbench."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.field_panels.plots import _style_axis


class Spatial3DDialog(QDialog):
    """Interactive 3D spatial deliverable view.

    The dialog is intentionally self-contained so the main spatial page can keep
    a compact 1080P layout while a larger, exportable 3D result is available on
    demand.
    """

    def __init__(self, parent: QWidget | None, *, project_store: Any, selected_line: str = "") -> None:
        super().__init__(parent)
        self.project_store = project_store
        self.selected_line = selected_line
        self.setWindowTitle("MyGPR 三维空间成果")
        self.resize(1120, 720)
        self.setMinimumSize(900, 600)
        self.trajectories, self.targets = collect_project_spatial_scene(project_store)
        self._setup_ui()
        self._draw_3d_scene()
        self._draw_plan_scene()
        self._fill_summary_table()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("三维空间成果：测线轨迹 / 高程 / 目标点")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        for text, slot in [
            ("导出三维场景 PNG", self.export_scene_png),
            ("导出三维点云 CSV", self.export_point_cloud_csv),
            ("打开 spatial 目录", self.open_spatial_dir),
        ]:
            btn = QPushButton(text)
            btn.setObjectName("smallButton")
            btn.clicked.connect(slot)
            header.addWidget(btn)
        root.addLayout(header)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("innerTabs")
        self.figure_3d = Figure(figsize=(7.2, 4.6), dpi=100, facecolor="white")
        self.canvas_3d = FigureCanvas(self.figure_3d)
        page3d = QWidget()
        page3d_layout = QVBoxLayout(page3d)
        page3d_layout.setContentsMargins(0, 0, 0, 0)
        page3d_layout.addWidget(self.canvas_3d)
        self.tabs.addTab(page3d, "三维视图")
        self.figure_plan = Figure(figsize=(7.2, 4.6), dpi=100, facecolor="white")
        self.canvas_plan = FigureCanvas(self.figure_plan)
        page_plan = QWidget()
        page_plan_layout = QVBoxLayout(page_plan)
        page_plan_layout.setContentsMargins(0, 0, 0, 0)
        page_plan_layout.addWidget(self.canvas_plan)
        self.tabs.addTab(page_plan, "平面视图")
        self.summary_table = QTableWidget()
        self.summary_table.setObjectName("dataTable")
        self.summary_table.setAlternatingRowColors(True)
        self.tabs.addTab(self.summary_table, "数据汇总")
        root.addWidget(self.tabs, 1)
        note = QLabel("说明：目标点的 Z 值按“轨迹高程 - 目标深度”估算；无 RTK/IMU 轨迹的测线不会生成三维轨迹。")
        note.setObjectName("smallInfo")
        note.setWordWrap(True)
        root.addWidget(note)

    def _draw_3d_scene(self) -> None:
        fig = self.figure_3d
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        if not self.trajectories:
            ax.text2D(
                0.35,
                0.5,
                "需要轨迹数据才能显示三维视图\n\n"
                "请在测线处理页面完成空间定位后刷新",
                transform=ax.transAxes,
                fontsize=12,
                ha="center",
                va="center",
                color="#666",
            )
            ax.set_axis_off()
            fig.tight_layout(pad=1.0)
            self.canvas_3d.draw_idle()
            return

        # Collect all trajectory points for the scene.
        all_x = np.concatenate([item["x"] for item in self.trajectories])
        all_y = np.concatenate([item["y"] for item in self.trajectories])
        all_z = np.concatenate([item["z"] for item in self.trajectories])

        # Use absolute coordinates when the range is small enough (< 1 000 000)
        # to avoid meaningless large offsets; otherwise fall back to relative.
        use_absolute = (
            np.ptp(all_x) < 1_000_000
            and np.ptp(all_y) < 1_000_000
            and np.ptp(all_z) < 1_000_000
        )
        if use_absolute:
            x_offset, y_offset = 0.0, 0.0
            x_label, y_label, z_label = "东坐标 Easting (m)", "北坐标 Northing (m)", "高程 Elevation (m)"
        else:
            x_offset = float(np.nanmin(all_x))
            y_offset = float(np.nanmin(all_y))
            x_label, y_label, z_label = "相对 X (m)", "相对 Y (m)", "高程 / 目标 Z (m)"

        # --- Terrain mesh via plot_trisurf ---
        # Build a surface when there are enough distinct points and elevation
        # variation is non-trivial (more than 1 cm across the site).
        valid_mask = np.isfinite(all_x) & np.isfinite(all_y) & np.isfinite(all_z)
        px, py, pz = all_x[valid_mask] - x_offset, all_y[valid_mask] - y_offset, all_z[valid_mask]
        has_terrain = (
            len(px) >= 10
            and np.ptp(pz) > 0.01  # elevation varies > 1 cm
        )
        if has_terrain:
            # Down-sample when the point cloud is large to keep the mesh
            # lightweight and interactive.
            max_vertices = 4000
            if len(px) > max_vertices:
                step = max(1, len(px) // max_vertices)
                px_s, py_s, pz_s = px[::step], py[::step], pz[::step]
            else:
                px_s, py_s, pz_s = px, py, pz

            try:
                tri = Triangulation(px_s, py_s)
                ax.plot_trisurf(
                    tri,
                    pz_s,
                    cmap="terrain",
                    alpha=0.55,
                    edgecolor="none",
                    antialiased=True,
                    zorder=1,
                )
            except Exception:
                # Degenerate triangulation (e.g. collinear points) -- skip
                # the surface; trajectory lines will still render.
                pass

        # --- Trajectory lines ---
        for item in self.trajectories:
            x = item["x"] - x_offset
            y = item["y"] - y_offset
            z = item["z"]
            lw = 2.2 if item["line_id"] == self.selected_line else 1.4
            ax.plot(
                x,
                y,
                z,
                linewidth=lw,
                marker="o",
                markersize=2,
                markevery=max(1, len(x) // 30),
                label=item["line_id"],
                zorder=3,
            )

        # --- Target scatter ---
        target_points = [
            t for t in self.targets if t.get("x") is not None and t.get("y") is not None
        ]
        if target_points:
            tx = np.asarray([float(t["x"]) - x_offset for t in target_points], dtype=float)
            ty = np.asarray([float(t["y"]) - y_offset for t in target_points], dtype=float)
            tz = np.asarray([float(t.get("z", 0.0)) for t in target_points], dtype=float)
            ax.scatter(tx, ty, tz, marker="^", s=28, label="目标点", zorder=4)

        ax.set_title(
            "三维空间成果" + ("（工程坐标）" if use_absolute else "（相对工程坐标显示）"),
            fontsize=11,
            fontweight="bold",
            loc="left",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.legend(loc="upper left", fontsize=8)
        try:
            ax.view_init(elev=28, azim=-52)
        except Exception:
            pass
        fig.tight_layout(pad=1.0)
        self.canvas_3d.draw_idle()

    def _draw_plan_scene(self) -> None:
        fig = self.figure_plan
        fig.clear()
        ax = fig.add_subplot(111)
        if not self.trajectories:
            ax.text(
                0.5,
                0.5,
                "需要轨迹数据才能显示平面视图",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#666",
            )
            ax.set_axis_off()
            self.canvas_plan.draw_idle()
            return
        x0 = min(float(np.nanmin(item["x"])) for item in self.trajectories)
        y0 = min(float(np.nanmin(item["y"])) for item in self.trajectories)
        for item in self.trajectories:
            x = item["x"] - x0
            y = item["y"] - y0
            line_width = 2.0 if item["line_id"] == self.selected_line else 1.4
            ax.plot(x, y, linewidth=line_width, marker="o", markersize=2, markevery=max(1, len(x) // 30), label=item["line_id"])
        target_points = [target for target in self.targets if target.get("x") is not None and target.get("y") is not None]
        if target_points:
            tx = np.asarray([float(t["x"]) - x0 for t in target_points], dtype=float)
            ty = np.asarray([float(t["y"]) - y0 for t in target_points], dtype=float)
            ax.scatter(tx, ty, marker="^", s=26, label="目标点")
            for t in target_points[:30]:
                ax.text(float(t["x"]) - x0, float(t["y"]) - y0, str(t.get("target_id", "T")), fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("平面成果（相对工程坐标显示）", fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("相对 X (m)")
        ax.set_ylabel("相对 Y (m)")
        ax.legend(loc="upper right", fontsize=8)
        _style_axis(ax)
        fig.tight_layout(pad=1.0)
        self.canvas_plan.draw_idle()

    def _fill_summary_table(self) -> None:
        headers = ["测线", "轨迹点", "目标点", "长度(m)", "X范围", "Y范围", "Z范围"]
        rows = []
        target_count_by_line: dict[str, int] = {}
        for target in self.targets:
            target_count_by_line[target.get("line_id", "")] = target_count_by_line.get(target.get("line_id", ""), 0) + 1
        for item in self.trajectories:
            rows.append(
                [
                    item["line_id"],
                    str(len(item["distance"])),
                    str(target_count_by_line.get(item["line_id"], 0)),
                    f"{float(item['distance'][-1] - item['distance'][0]):.2f}" if len(item["distance"]) else "0.00",
                    f"{float(np.nanmin(item['x'])):.2f} ~ {float(np.nanmax(item['x'])):.2f}",
                    f"{float(np.nanmin(item['y'])):.2f} ~ {float(np.nanmax(item['y'])):.2f}",
                    f"{float(np.nanmin(item['z'])):.2f} ~ {float(np.nanmax(item['z'])):.2f}",
                ]
            )
        if not rows:
            rows = [["暂无轨迹", "0", "0", "0.00", "--", "--", "--"]]
        table = self.summary_table
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def closeEvent(self, event) -> None:
        self.canvas_3d.close()
        plt.close(self.figure_3d)
        self.canvas_plan.close()
        plt.close(self.figure_plan)
        super().closeEvent(event)

    def export_scene_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出三维场景 PNG", str(self._default_spatial_path("spatial_3d_scene.png")), "PNG 图像 (*.png)")
        if not path:
            return
        out = Path(path).with_suffix(".png")
        self.figure_3d.savefig(out, dpi=180, bbox_inches="tight")
        QMessageBox.information(self, "导出三维场景", f"已导出：\n{out}")

    def export_point_cloud_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出三维点云 CSV", str(self._default_spatial_path("spatial_3d_point_cloud.csv")), "CSV 文件 (*.csv)")
        if not path:
            return
        out = Path(path).with_suffix(".csv")
        write_spatial_scene_csv(out, self.trajectories, self.targets)
        QMessageBox.information(self, "导出三维点云", f"已导出：\n{out}")

    def open_spatial_dir(self) -> None:
        directory = self._default_spatial_path("").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _default_spatial_path(self, name: str) -> Path:
        root = Path(getattr(self.project_store, "root", Path.cwd()))
        directory = root / "spatial"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name if name else directory


def collect_project_spatial_scene(project_store: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    if project_store is None:
        return trajectories, targets
    for line in project_store.list_lines():
        line_id = line.line_id
        trajectory = None
        try:
            trajectory = project_store.load_trajectory(line_id)
        except Exception:
            trajectory = None
        if trajectory is not None and getattr(trajectory, "points", None):
            trajectories.append(
                {
                    "line_id": line_id,
                    "distance": np.asarray(trajectory.distance, dtype=float),
                    "x": np.asarray(trajectory.x, dtype=float),
                    "y": np.asarray(trajectory.y, dtype=float),
                    "z": np.asarray(trajectory.z, dtype=float),
                }
            )
        try:
            line_targets = project_store.load_targets(line_id)
        except Exception:
            line_targets = []
        for idx, target in enumerate(line_targets, start=1):
            distance_m = _to_float(target.get("distance_m", target.get("mileage", 0.0)), 0.0) or 0.0
            depth_m = _to_float(target.get("depth_m", target.get("depth", 0.0)), 0.0) or 0.0
            x_val = _to_float(target.get("x"), None)
            y_val = _to_float(target.get("y"), None)
            elevation = None
            if trajectory is not None:
                point = trajectory.interpolate(distance_m)
                # Prefer measured/interpolated trajectory coordinates over
                # stale target x/y fields so plan/3D views stay spatially
                # coherent with the selected line.
                x_val = point.x
                y_val = point.y
                elevation = point.z
            targets.append(
                {
                    "line_id": line_id,
                    "target_id": target.get("target_id") or target.get("name") or f"T-{idx:04d}",
                    "distance_m": distance_m,
                    "depth_m": depth_m,
                    "x": x_val,
                    "y": y_val,
                    "z": (float(elevation) - depth_m) if elevation is not None else 0.0,
                    "type": target.get("type", ""),
                    "status": target.get("status", ""),
                    "confidence": target.get("confidence", ""),
                }
            )
    return trajectories, targets


def write_spatial_scene_csv(path: Path, trajectories: list[dict[str, Any]], targets: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["record_type", "line_id", "id", "distance_m", "x", "y", "z", "depth_m", "target_type", "status", "confidence"])
        writer.writeheader()
        for item in trajectories:
            for idx, (d, x, y, z) in enumerate(zip(item["distance"], item["x"], item["y"], item["z"]), start=1):
                writer.writerow({"record_type": "trajectory", "line_id": item["line_id"], "id": f"P{idx:05d}", "distance_m": f"{float(d):.3f}", "x": f"{float(x):.3f}", "y": f"{float(y):.3f}", "z": f"{float(z):.3f}", "depth_m": "", "target_type": "", "status": "", "confidence": ""})
        for target in targets:
            writer.writerow({"record_type": "target", "line_id": target.get("line_id", ""), "id": target.get("target_id", ""), "distance_m": f"{float(target.get('distance_m', 0.0)):.3f}", "x": f"{float(target.get('x', 0.0)):.3f}" if target.get("x") is not None else "", "y": f"{float(target.get('y', 0.0)):.3f}" if target.get("y") is not None else "", "z": f"{float(target.get('z', 0.0)):.3f}", "depth_m": f"{float(target.get('depth_m', 0.0)):.3f}", "target_type": target.get("type", ""), "status": target.get("status", ""), "confidence": target.get("confidence", "")})
    tmp.replace(path)
    return path


def _to_float(value: Any, default: float | None = 0.0) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["Spatial3DDialog", "collect_project_spatial_scene", "write_spatial_scene_csv"]

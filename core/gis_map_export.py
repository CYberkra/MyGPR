#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless, transactional engineering plan-map export.

The Qt canvas is only a preview.  Formal PNG/PDF/SVG exports are rebuilt from
project data in a worker thread so large raster reprojection and figure writing
do not block the GUI or race with interactive redraws.
"""
from __future__ import annotations

import csv
import math
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.figure import Figure

from core.gis_layers import GISLayerStore
from core.plot_font_policy import configure_matplotlib_cjk_fonts

CancelCheck = Callable[[], bool] | None
ProgressCallback = Callable[[int, int, str], None] | None


def _check_cancel(cancel_requested: CancelCheck) -> None:
    if cancel_requested is not None and cancel_requested():
        from core.job_manager import JobCancelled
        raise JobCancelled("GIS 平面图导出已取消")


def _finite_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def _apply_limits(ax, xs: list[float], ys: list[float]) -> None:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        raise ValueError("项目中没有可绘制的真实空间坐标。")
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    span_x = max(x1 - x0, 10.0)
    span_y = max(y1 - y0, 10.0)
    ax.set_xlim((x0 + x1) / 2 - span_x * 0.56, (x0 + x1) / 2 + span_x * 0.56)
    ax.set_ylim((y0 + y1) / 2 - span_y * 0.56, (y0 + y1) / 2 + span_y * 0.56)


def _load_interface_xy(store: Any, line_id: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    try:
        annotation = store.load_basal_interface_annotation(line_id)
    except Exception:
        annotation = None
    if annotation is None:
        return None
    path = store.root / "spatial" / f"{line_id}_basal_interface_xyz.csv"
    if not path.exists():
        path = store.export_spatial_interface_curve(line_id, annotation)
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                xs.append(float(row.get("x", "")))
                ys.append(float(row.get("y", "")))
            except (TypeError, ValueError):
                continue
    x, y = _finite_xy(xs, ys)
    if x.size == 0:
        return None
    return x, y, str(getattr(annotation, "status", "draft"))


def export_project_plan_map(
    store: Any,
    destination: str | Path,
    *,
    selected_line: str = "",
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
    dpi: int = 220,
) -> Path:
    """Rebuild and atomically export a georeferenced project plan map."""
    configure_matplotlib_cjk_fonts()
    out = Path(destination)
    suffix = out.suffix.lower()
    if suffix not in {".png", ".pdf", ".svg"}:
        out = out.with_suffix(".png")
        suffix = ".png"
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f".{out.stem}.{uuid.uuid4().hex}.tmp{suffix}")

    gis_store = GISLayerStore(store.root)
    layers = [layer for layer in gis_store.list_layers() if layer.visible]
    lines = list(store.list_lines())
    total_steps = max(len(layers) + len(lines) * 2 + 1, 1)
    completed = 0
    project_crs = str(getattr(store.manifest, "coordinate_system", "") or "")
    warnings: list[str] = []
    bounds_x: list[float] = []
    bounds_y: list[float] = []
    committed = False

    fig = Figure(figsize=(11.69, 8.27), dpi=max(int(dpi), 96), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#F8FAFC")
    try:
        for layer in layers:
            _check_cancel(cancel_requested)
            if progress_callback is not None:
                progress_callback(completed, total_steps, f"绘制 GIS 图层：{layer.name}")
            try:
                if layer.kind == "raster":
                    preview = gis_store.load_raster_preview(layer, max_size=2400, target_crs=project_crs)
                    ax.imshow(
                        preview.array,
                        extent=preview.extent,
                        origin="upper",
                        cmap="terrain" if preview.is_dem else None,
                        alpha=float(layer.opacity),
                        zorder=0,
                    )
                    bounds_x.extend([preview.extent[0], preview.extent[1]])
                    bounds_y.extend([preview.extent[2], preview.extent[3]])
                else:
                    features = gis_store.load_vector(layer, target_crs=project_crs)
                    for feature_index, feature in enumerate(features):
                        if feature_index % 128 == 0:
                            _check_cancel(cancel_requested)
                        for coords in feature.coordinates:
                            if coords.size == 0:
                                continue
                            x, y = _finite_xy(coords[:, 0], coords[:, 1])
                            if x.size == 0:
                                continue
                            bounds_x.extend(x.tolist())
                            bounds_y.extend(y.tolist())
                            label = layer.name if feature_index == 0 else "_nolegend_"
                            if feature.geometry_type == "Point":
                                ax.scatter(x, y, s=16, alpha=float(layer.opacity), label=label, zorder=2)
                            elif feature.geometry_type == "Polygon":
                                ax.fill(x, y, alpha=min(float(layer.opacity) * 0.22, 0.35), zorder=1)
                                ax.plot(x, y, linewidth=0.9, alpha=float(layer.opacity), label=label, zorder=2)
                            else:
                                ax.plot(x, y, linewidth=1.0, alpha=float(layer.opacity), label=label, zorder=2)
            except Exception as exc:
                warnings.append(f"{layer.name}: {exc}")
            completed += 1

        for line in lines:
            _check_cancel(cancel_requested)
            line_id = str(line.line_id)
            if progress_callback is not None:
                progress_callback(completed, total_steps, f"绘制测线轨迹：{line_id}")
            try:
                trajectory = store.load_trajectory(line_id)
            except Exception:
                trajectory = None
            if trajectory is not None and getattr(trajectory, "points", None):
                x, y = _finite_xy(trajectory.x, trajectory.y)
                if x.size:
                    bounds_x.extend(x.tolist())
                    bounds_y.extend(y.tolist())
                    width = 2.2 if line_id == selected_line else 1.2
                    ax.plot(x, y, linewidth=width, color="#475569", alpha=0.9, label=f"{line_id} 轨迹", zorder=5)
                    middle = len(x) // 2
                    ax.text(float(x[middle]), float(y[middle]), line_id, fontsize=8, fontweight="bold", zorder=8)
            completed += 1

            _check_cancel(cancel_requested)
            if progress_callback is not None:
                progress_callback(completed, total_steps, f"绘制基覆界面：{line_id}")
            try:
                interface = _load_interface_xy(store, line_id)
                if interface is not None:
                    x, y, status = interface
                    bounds_x.extend(x.tolist())
                    bounds_y.extend(y.tolist())
                    color = "#F59E0B" if status == "confirmed" else "#0EA5E9"
                    width = 3.0 if line_id == selected_line else 2.0
                    ax.plot(x, y, linewidth=width, color=color, alpha=0.96, label=f"{line_id} 基覆界面", zorder=7)
            except Exception as exc:
                warnings.append(f"{line_id} 界面: {exc}")
            completed += 1

        _check_cancel(cancel_requested)
        _apply_limits(ax, bounds_x, bounds_y)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        span_x = max(abs(x1 - x0), 1.0)
        span_y = max(abs(y1 - y0), 1.0)
        ax.annotate(
            "N",
            xy=(0.055, 0.94),
            xytext=(0.055, 0.83),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#101828", lw=1.4),
            zorder=10,
        )
        raw_scale = span_x / 5.0
        exponent = 10 ** math.floor(math.log10(max(raw_scale, 1e-9)))
        scale_len = max(1.0, round(raw_scale / exponent) * exponent)
        sx0 = x0 + span_x * 0.04
        sy0 = y0 + span_y * 0.06
        ax.plot([sx0, sx0 + scale_len], [sy0, sy0], color="#101828", linewidth=2.5, zorder=10)
        ax.text(sx0 + scale_len / 2, sy0 + span_y * 0.018, f"{scale_len:g} m", fontsize=8, ha="center", va="bottom", zorder=10)
        ax.set_aspect("equal", adjustable="box")
        ax.ticklabel_format(style="plain", useOffset=False, axis="both")
        ax.grid(True, linewidth=0.35, alpha=0.28)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique: dict[str, Any] = {}
            for handle, label in zip(handles, labels):
                if label and label != "_nolegend_":
                    unique.setdefault(label, handle)
            ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=7, ncol=2)
        ax.set_title(f"{store.manifest.name} · 项目空间成果图\n{project_crs or 'CRS 未设置'}", fontsize=12, loc="left", fontweight="bold")
        ax.set_xlabel("X / Easting")
        ax.set_ylabel("Y / Northing")
        if warnings:
            ax.text(0.01, 0.01, "；".join(warnings[:3]), transform=ax.transAxes, fontsize=7, color="#B45309", va="bottom")
        fig.tight_layout(pad=1.0)
        if progress_callback is not None:
            progress_callback(completed, total_steps, "写出工程平面图")
        _check_cancel(cancel_requested)
        fig.savefig(temp, dpi=max(int(dpi), 96), bbox_inches="tight", format=suffix.lstrip("."))
        _check_cancel(cancel_requested)
        temp.replace(out)
        committed = True
        if progress_callback is not None:
            progress_callback(total_steps, total_steps, "工程平面图已生成")
        return out
    finally:
        fig.clear()
        if not committed:
            temp.unlink(missing_ok=True)


__all__ = ["export_project_plan_map"]

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


def _apply_limits(ax, xs: list[float], ys: list[float], *, minimum_span: float = 10.0) -> None:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        raise ValueError("项目中没有可绘制的真实空间坐标。")
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    span_x = max(x1 - x0, minimum_span)
    span_y = max(y1 - y0, minimum_span)
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


def _plot_trajectory(ax, trajectory, line_id, project_crs, ranges, highlighted):
    from pyproj import CRS, Transformer
    from pyproj.exceptions import ProjError

    points = trajectory.points
    xy = np.column_stack((trajectory.x, trajectory.y))
    declared = {point.coordinate_system for point in points if point.coordinate_system}
    if len(declared) > 1:
        return None
    source = next(iter(declared), project_crs)
    if source and project_crs:
        try:
            src, dst = CRS.from_user_input(source), CRS.from_user_input(project_crs)
            if src != dst:
                x, y = Transformer.from_crs(src, dst, always_xy=True).transform(*xy.T)
                xy = np.column_stack((x, y))
        except (ProjError, ValueError):
            if source != project_crs:
                return None
    indices = np.array([point.trace_index for point in points], dtype=int)
    valid = np.isfinite(xy).all(axis=1) & np.array([
        point.alignment_status not in ('gap', 'missing', 'unmatched', 'outside', 'rtk_out_of_range', 'rtk_time_residual')
        for point in points
    ])
    positions = np.flatnonzero(valid)
    if not positions.size:
        return None
    known_indices = (trajectory.trace_indices_explicit
                     and np.all(indices >= 0) and np.all(np.diff(indices) > 0))
    breaks = np.diff(positions) > 1
    if known_indices:
        breaks |= np.diff(indices[positions]) > 1
    parts = np.split(positions, np.flatnonzero(breaks) + 1)
    for part_index, part in enumerate(parts):
        ax.plot(*xy[part].T, color="#475569", linewidth=2.2 if highlighted else 1.3,
                marker='.' if len(part) == 1 else None,
                label=f"{line_id} 轨迹" if part_index == 0 else None, zorder=5)
    start, end = xy[positions[0]], xy[positions[-1]]
    ax.scatter(*start, s=25, marker='o', color='#166534', zorder=7)
    ax.annotate(f'{line_id} 有效起', start, xytext=(4, 4), textcoords='offset points', fontsize=8)
    if np.any(end != start):
        ax.scatter(*end, s=25, marker='s', color='#1e3a8a', zorder=7)
        ax.annotate(f'{line_id} 有效终', end, xytext=(4, 4), textcoords='offset points', fontsize=8)
    longest = max(parts, key=len)
    if len(longest) >= 2:
        middle = len(longest) // 2
        first, second = xy[longest[middle - 1]], xy[longest[middle]]
        if np.any(first != second):
            ax.annotate('', second, xytext=first,
                        arrowprops={'arrowstyle': '->', 'color': '#1e3a8a', 'lw': 1.6}, zorder=8)
    if known_indices:
        label_indices = []
        for code, left, right in ranges:
            visible = []
            for part in parts:
                segment = part[(indices[part] >= left) & (indices[part] < right)]
                if segment.size:
                    ax.plot(*xy[segment].T, color='#b45309', linewidth=2.7,
                            marker='.' if segment.size == 1 else None, zorder=7)
                    visible.append(segment)
            if visible and (code.endswith('-F01') or len(label_indices) < 8):
                segment = max(visible, key=len)
                centre = segment[len(segment) // 2]
                distance = max(1, int(len(points) * .055))
                neighbours = sum(abs(centre - prior) < distance for prior in label_indices)
                if neighbours < 2:
                    ax.annotate(code, xy[centre], xytext=(4, -11 - 10 * neighbours),
                                textcoords='offset points', fontsize=8, color='#92400e', zorder=9)
                    label_indices.append(centre)
    return xy[positions]


def _draw_gis_layer(ax, gis_store, layer, project_crs, bounds_x, bounds_y, cancel_requested):
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


def _style_plan_map(ax, project_name, project_crs, crs, selected_line, line_ids, warnings):
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
    metres = crs is not None and crs.is_projected and np.isclose(crs.axis_info[0].unit_conversion_factor, 1)
    raw_scale = span_x / 5.0
    exponent = 10 ** math.floor(math.log10(max(raw_scale, 1e-9)))
    scale_len = max(1.0, round(raw_scale / exponent) * exponent)
    sx0 = x0 + span_x * 0.04
    sy0 = y0 + span_y * 0.06
    if metres:
        ax.plot([sx0, sx0 + scale_len], [sy0, sy0], color="#101828", linewidth=2.5, zorder=10)
        ax.text(sx0 + scale_len / 2, sy0 + span_y * 0.018, f"{scale_len:g} m", fontsize=8, ha="center", va="bottom", zorder=10)
    if crs and crs.is_geographic:
        mid_latitude = np.clip((y0 + y1) / 2, -89.9, 89.9)
        ax.set_aspect(1 / np.cos(np.deg2rad(mid_latitude)), adjustable='box')
    else:
        ax.set_aspect('equal', adjustable='box')
    ax.ticklabel_format(style="plain", useOffset=False, axis="both")
    ax.grid(True, linewidth=0.35, alpha=0.28)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if label and label != "_nolegend_":
                unique.setdefault(label, handle)
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=7, ncol=2)
    map_title = (f'{selected_line} 路线' if selected_line and line_ids == {selected_line}
                 else '测区路线总图' if line_ids is not None else '项目空间成果图')
    ax.set_title(f"{project_name} · {map_title}\n{project_crs or 'CRS 未设置'}", fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel('经度 (°)' if crs and crs.is_geographic else 'X / Easting')
    ax.set_ylabel('纬度 (°)' if crs and crs.is_geographic else 'Y / Northing')
    if warnings:
        ax.text(0.01, 0.01, "；".join(warnings[:3]), transform=ax.transAxes, fontsize=7, color="#B45309", va="bottom")


def _plan_map_trajectories(store, lines, project_crs, line_ids, cancel_requested):
    from pyproj import CRS
    from pyproj.exceptions import ProjError

    def load_trajectory(line):
        _check_cancel(cancel_requested)
        try:
            return store.load_trajectory(line.line_id)
        except (OSError, ValueError, KeyError):
            return None

    trajectories = {line.line_id: load_trajectory(line) for line in lines}
    if line_ids is not None and not project_crs:
        declared = {point.coordinate_system for trajectory in trajectories.values() if trajectory is not None
                    for point in trajectory.points if point.coordinate_system}
        if len(declared) == 1:
            project_crs = declared.pop()
    try:
        crs = CRS.from_user_input(project_crs)
    except (ProjError, ValueError):
        crs = None
    if line_ids is not None and crs is None:
        raise ValueError('请为报告路线图设置有效的项目坐标参考。')
    return trajectories, project_crs, crs


def _fit_plan_bounds(ax, line_ids, crs, route_x, route_y, bounds_x, bounds_y):
    minimum_span = .0001 if crs and crs.is_geographic else 10.0
    if line_ids is not None:
        _apply_limits(ax, route_x, route_y, minimum_span=minimum_span)
    else:
        _apply_limits(ax, bounds_x, bounds_y, minimum_span=minimum_span)


def export_project_plan_map(
    store: Any,
    destination: str | Path,
    *,
    selected_line: str = "",
    line_ids: set[str] | None = None,
    line_ranges: dict | None = None,
    include_interpretation: bool = True,
    include_layers: bool = True,
    require_trajectory: bool = False,
    located_lines: set[str] | None = None,
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
    layers = [layer for layer in gis_store.list_layers() if layer.visible] if include_layers else []
    lines = [line for line in store.list_lines() if line_ids is None or line.line_id in line_ids]
    total_steps = max(len(layers) + len(lines) * 2 + 1, 1)
    completed = 0
    project_crs = str(getattr(store.manifest, "coordinate_system", "") or "")
    trajectories, project_crs, crs = _plan_map_trajectories(store, lines, project_crs, line_ids, cancel_requested)
    warnings: list[str] = []
    bounds_x: list[float] = []
    bounds_y: list[float] = []
    route_x: list[float] = []
    route_y: list[float] = []
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
                _draw_gis_layer(ax, gis_store, layer, project_crs, bounds_x, bounds_y, cancel_requested)
            except Exception as exc:
                warnings.append(f"{layer.name}: {exc}")
            completed += 1

        for line in lines:
            _check_cancel(cancel_requested)
            line_id = str(line.line_id)
            if progress_callback is not None:
                progress_callback(completed, total_steps, f"绘制测线轨迹：{line_id}")
            trajectory = trajectories[line_id]
            if trajectory is not None and getattr(trajectory, "points", None):
                xy = _plot_trajectory(ax, trajectory, line_id, project_crs,
                                      (line_ranges or {}).get(line_id, ()), line_id == selected_line)
                if xy is not None:
                    x, y = xy.T
                    bounds_x.extend(x.tolist())
                    bounds_y.extend(y.tolist())
                    route_x.extend(x.tolist())
                    route_y.extend(y.tolist())
                    if located_lines is not None:
                        located_lines.add(line_id)
            completed += 1

            _check_cancel(cancel_requested)
            if progress_callback is not None:
                progress_callback(completed, total_steps, f"绘制基覆界面：{line_id}")
            try:
                interface = _load_interface_xy(store, line_id) if include_interpretation else None
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
        if require_trajectory and not route_x:
            raise ValueError('测线缺少定位轨迹')
        _fit_plan_bounds(ax, line_ids, crs, route_x, route_y, bounds_x, bounds_y)
        _style_plan_map(ax, store.manifest.name, project_crs, crs, selected_line, line_ids, warnings)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网格化服务：轨迹分组 → 属性网格 → GIS 图层 GeoJSON。

application 层只依赖 domain 与 ProjectService 委托；core 投影、
持久化、GIS 入库全部经 ProjectSessionPort 的 infrastructure 混入完成。
属性数据源参数化：v1 内置 interface_depth（界面深度切片），
Phase 3.1/4 换数据源即可复用整条管线。
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from mygpr.application.project.service import ProjectService
from mygpr.domain.grid.clustering import group_tracks
from mygpr.domain.grid.errors import GridAnalysisError
from mygpr.domain.grid.models import (
    LINE_GROUPS_SCHEMA,
    AttributeGrid,
    AttributeGridRequest,
    TrackGrouping,
)

DEFAULT_GROUP_TOLERANCE_M = 20.0
INTERFACE_DEPTH_ATTRIBUTE = "interface_depth_m"


def group_project_tracks(
    projects: ProjectService, project_id: str, *, tolerance_m: float = DEFAULT_GROUP_TOLERANCE_M
) -> TrackGrouping:
    """读取空间轨迹并聚类分组（不落盘；落盘走 save_line_groups）。"""
    tracks = projects.load_spatial_tracks(project_id)
    return group_tracks(list(tracks), tolerance_m=tolerance_m)


def _grid_geojson_payload(grid: AttributeGrid) -> dict[str, Any]:
    """把规则网格序列化为 GeoJSON FeatureCollection（Polygon cell 中心点）。

    用 Point + 值属性承载规则网格，GIS 侧可直接做点插值或格网渲染；
    crs 由调用方在 import 时声明。
    """
    features: list[dict[str, Any]] = []
    for row_idx, row in enumerate(grid.values):
        y = grid.y_origin_m - row_idx * grid.cell_size_m
        for col_idx, value in enumerate(row):
            if value is None:
                continue
            x = grid.x_origin_m + col_idx * grid.cell_size_m
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(x, 4), round(y, 4)]},
                "properties": {grid.attribute_name: round(float(value), 4)},
            })
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "attribute": grid.attribute_name,
            "cell_size_m": grid.cell_size_m,
            "ncols": grid.ncols,
            "nrows": grid.nrows,
            "x_origin_m": grid.x_origin_m,
            "y_origin_m": grid.y_origin_m,
            "valid_count": grid.valid_count,
        },
    }


def gridded_interface_depth(
    projects: ProjectService,
    project_id: str,
    line_ids: Sequence[str],
    *,
    cell_size_m: float = 1.0,
) -> AttributeGrid:
    """从界面标注采集 (x, y, 深度) 点并网格化。

    每条测线：load_interface_annotation 的界面点 → sample→depth 轴换算 →
    轨迹按 trace_index 插值取 x/y。无标注或无轨迹的测线跳过。
    """
    if not line_ids:
        raise GridAnalysisError(
            "网格化至少需要一条测线。", hint="先选择有界面标注的测线。")
    xs: list[float] = []
    ys: list[float] = []
    values: list[float] = []
    for line_id in line_ids:
        annotation = projects.load_interface_annotation(project_id, line_id, create=False)
        if annotation is None or not annotation.points:
            continue
        tracks = {t.line_id: t for t in projects.load_spatial_tracks(project_id)}
        track = tracks.get(line_id)
        if track is None or not track.points:
            continue
        samples = [float(p.sample_index) for p in annotation.points]
        depths = projects.depth_at_samples(project_id, line_id, samples)
        trace_to_xy = {int(p.trace_index): (float(p.x), float(p.y)) for p in track.points}
        for point, depth in zip(annotation.points, depths):
            xy = trace_to_xy.get(int(round(float(point.trace_index))))
            if xy is None or not math.isfinite(depth):
                continue
            xs.append(xy[0])
            ys.append(xy[1])
            values.append(float(depth))
    if not values:
        raise GridAnalysisError(
            "没有可用的界面深度点（需测线同时具备界面标注与投影轨迹）。",
            hint="先在解释页拾取界面并完成测线轨迹投影。",
        )
    request = AttributeGridRequest(
        x_m=tuple(xs), y_m=tuple(ys), values=tuple(values),
        attribute_name=INTERFACE_DEPTH_ATTRIBUTE, cell_size_m=cell_size_m,
    )
    return grid_attribute(request)


def grid_attribute(request: AttributeGridRequest) -> AttributeGrid:
    """把散乱属性点放进规则网格（cell 内取均值，空 cell 为 None）。

    行序 = y 降序（GIS 惯例，北在上），列序 = x 升序。
    """
    points = list(zip(request.x_m, request.y_m, request.values))
    if request.value_missing_ok:
        points = [(x, y, v) for x, y, v in points if math.isfinite(v)]
    else:
        for x, y, v in points:
            if not math.isfinite(v):
                raise GridAnalysisError(
                    f"属性点含 NaN/Inf（{request.attribute_name}）。",
                    hint="检查属性来源数据。",
                )
    if not points:
        raise GridAnalysisError(
            "网格化没有有效属性点。", hint="检查输入点是否全部为 NaN。")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cell = float(request.cell_size_m)
    ncols = max(int(math.floor((x_max - x_min) / cell)) + 1, 1)
    nrows = max(int(math.floor((y_max - y_min) / cell)) + 1, 1)
    if ncols * nrows > 4_000_000:
        raise GridAnalysisError(
            f"网格过大（{ncols}x{nrows}），请增大 cell_size_m。",
            hint="cell_size_m 至少应约为测区边长的 1/500。",
        )
    buckets: dict[tuple[int, int], list[float]] = {}
    for x, y, v in points:
        col = min(int((x - x_min) / cell), ncols - 1)
        row = min(int((y_max - y) / cell), nrows - 1)
        buckets.setdefault((row, col), []).append(v)
    rows: list[tuple[float | None, ...]] = []
    for row_idx in range(nrows):
        row: list[float | None] = []
        for col_idx in range(ncols):
            vals = buckets.get((row_idx, col_idx))
            row.append(sum(vals) / len(vals) if vals else None)
        rows.append(tuple(row))
    return AttributeGrid(
        attribute_name=request.attribute_name,
        x_origin_m=x_min + cell / 2.0,
        y_origin_m=y_max - cell / 2.0,
        cell_size_m=cell,
        ncols=ncols,
        nrows=nrows,
        values=tuple(rows),
    )


def write_grid_geojson(
    grid: AttributeGrid, destination: Path, *, crs_name: str
) -> Path:
    """把网格写成 GeoJSON（投影坐标），crs 写入 payload。原子性由
    临时文件 + replace 保证。"""
    payload = _grid_geojson_payload(grid)
    if crs_name:
        payload["crs"] = {"type": "name", "properties": {"name": crs_name}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        tmp_path.replace(destination)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination


def persist_grouping(
    projects: ProjectService, project_id: str, grouping: TrackGrouping,
    *, crs_name: str,
) -> dict[str, Any]:
    """把分组结果（含 schema、容差、CRS）交给持久化层写旁车。"""
    payload: dict[str, Any] = {
        "schema": LINE_GROUPS_SCHEMA,
        "tolerance_m": grouping.tolerance_m,
        "crs": crs_name,
        "groups": [
            {
                "group_id": g.group_id,
                "line_ids": list(g.line_ids),
                "representative_xy_m": [g.representative_x_m, g.representative_y_m],
                "track_count": g.track_count,
                "max_pair_distance_m": g.max_pair_distance_m,
            }
            for g in grouping.groups
        ],
    }
    projects.save_line_groups(project_id, payload)
    return payload


__all__ = [
    "DEFAULT_GROUP_TOLERANCE_M",
    "INTERFACE_DEPTH_ATTRIBUTE",
    "AttributeGrid",
    "AttributeGridRequest",
    "TrackGrouping",
    "grid_attribute",
    "gridded_interface_depth",
    "group_project_tracks",
    "persist_grouping",
    "write_grid_geojson",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轨迹聚类成组（纯 numpy/scipy，无 Qt/core 依赖）。

每条轨迹取全部顶点的算术平均作为代表点，对代表点做层次聚类
（scipy hierarchy average-linkage + fcluster 距离判据）：代表点间距
≤ tolerance_m 的测线归入同组。组间按代表点坐标字典序排序，保证
同输入永远得到同输出。
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from mygpr.domain.grid.errors import GridAnalysisError
from mygpr.domain.grid.models import LineGroup, TrackGrouping
from mygpr.domain.spatial.models import SpatialTrack

_MIN_TRACKS = 1


def _representative_points(tracks: list[SpatialTrack]) -> np.ndarray:
    pts = np.empty((len(tracks), 2), dtype=np.float64)
    for idx, track in enumerate(tracks):
        xs = [float(p.x) for p in track.points]
        ys = [float(p.y) for p in track.points]
        pts[idx, 0] = float(np.mean(xs)) if xs else float("nan")
        pts[idx, 1] = float(np.mean(ys)) if ys else float("nan")
    return pts


def group_tracks(tracks: list[SpatialTrack], *, tolerance_m: float) -> TrackGrouping:
    """按轨迹代表点把测线分成若干组。

    Args:
        tracks: 已投影的空间轨迹（x/y 为米制投影坐标）。
        tolerance_m: 组内代表点最大间距（米）。

    Raises:
        GridAnalysisError: 轨迹为空、代表点含 NaN/Inf、tolerance 非正，
            或 scipy 距离矩阵退化（不应发生，防御性）。
    """
    tolerance = float(tolerance_m)
    if not tolerance > 0.0:
        raise GridAnalysisError(
            f"tolerance_m 必须为正数，实际 {tolerance}。",
            hint="按测线间距经验设置，例如 5~50 米。",
        )
    if len(tracks) < _MIN_TRACKS:
        raise GridAnalysisError(
            "至少需要一条轨迹才能分组。",
            hint="先完成测线导入与轨迹投影。",
        )

    # 组内一致排序：line_id 字典序（ SpatialTrack.line_id 保证非空唯一）
    ordered = sorted(tracks, key=lambda t: str(t.line_id))
    pts = _representative_points(ordered)
    if not np.all(np.isfinite(pts)):
        bad = [str(ordered[i].line_id) for i in range(len(ordered))
               if not np.isfinite(pts[i]).all()]
        raise GridAnalysisError(
            f"轨迹代表点坐标含 NaN/Inf: {', '.join(bad)}。",
            hint="确认测线轨迹已完成投影（rtk_status=已投影）。",
        )

    distances = pdist(pts)
    labels = (
        fcluster(linkage(distances, method="average"), t=tolerance, criterion="distance")
        if len(ordered) > 1 else np.array([1])
    )

    buckets: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        buckets.setdefault(int(label), []).append(idx)

    groups: list[LineGroup] = []
    for members in buckets.values():
        line_ids = tuple(str(ordered[i].line_id) for i in members)
        center = pts[members].mean(axis=0)
        if len(members) == 1:
            max_pair = 0.0
        else:
            pair = pdist(pts[members])
            max_pair = float(np.max(pair))
        groups.append(LineGroup(
            group_id="G%02d" % (len(groups) + 1),
            line_ids=line_ids,
            representative_x_m=float(center[0]),
            representative_y_m=float(center[1]),
            track_count=len(members),
            max_pair_distance_m=max_pair,
        ))

    groups.sort(key=lambda g: (g.representative_x_m, g.representative_y_m))
    renumbered = tuple(
        LineGroup(
            group_id="G%02d" % (idx + 1),
            line_ids=g.line_ids,
            representative_x_m=g.representative_x_m,
            representative_y_m=g.representative_y_m,
            track_count=g.track_count,
            max_pair_distance_m=g.max_pair_distance_m,
        )
        for idx, g in enumerate(groups)
    )
    return TrackGrouping(
        tolerance_m=tolerance,
        groups=renumbered,
        ungrouped_line_ids=(),
    )


__all__ = ["group_tracks"]

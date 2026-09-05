#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.2 无界面验收：轨迹聚类 → 分组持久化 → 界面深度网格 → GIS 图层。

domain 层纯测试（无项目依赖）+ 经 ``MyGPRBackend`` 公共 API 的 job 链路验收，
不 import 任何 Qt 模块。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mygpr.application.grid.facade_service import GridService
from mygpr.application.grid.service import grid_attribute
from mygpr.application.jobs.models import JobStatus
from mygpr.domain.grid.errors import GridAnalysisError
from mygpr.domain.grid.models import (
    LINE_GROUPS_SCHEMA,
    AttributeGridRequest,
)
from mygpr.domain.grid.clustering import group_tracks
from mygpr.domain.interpretation.models import InterfaceAnnotation, InterpretationPoint
from mygpr.domain.spatial.models import SpatialTrack, SpatialTrackPoint
from mygpr.interfaces.backend import MyGPRBackend

pytestmark = [
    pytest.mark.integration,
]


def _track(line_id: str, x_m: float, y_m: float, *, count: int = 3) -> SpatialTrack:
    """以 (x, y) 为中心造一条东西向短轨迹，代表点即 (x, y)。"""
    half = (count - 1) / 2.0
    points = tuple(
        SpatialTrackPoint(
            trace_index=idx,
            x=x_m + (idx - half) * 0.5,
            y=y_m,
        )
        for idx in range(count)
    )
    return SpatialTrack(line_id, line_id, points, "EPSG:4527", "synthetic")


# ---------------------------------------------------------------------------
# domain：聚类纯测试
# ---------------------------------------------------------------------------

def test_group_tracks_two_clusters() -> None:
    """两组相距 50 m 的轨迹 → 恰分两组，组内 line_id 有序。"""
    tracks = [
        _track("L03", 0.0, 0.0),
        _track("L01", 1.5, 0.5),
        _track("L02", 50.0, 30.0),
        _track("L04", 51.5, 30.5),
    ]
    grouping = group_tracks(tracks, tolerance_m=20.0)
    assert grouping.tolerance_m == pytest.approx(20.0)
    assert grouping.ungrouped_line_ids == ()
    assert len(grouping.groups) == 2
    # 组间按代表点字典序：近原点组在前
    first, second = grouping.groups
    assert first.group_id == "G01" and second.group_id == "G02"
    assert first.line_ids == ("L01", "L03")
    assert second.line_ids == ("L02", "L04")
    assert first.max_pair_distance_m <= 20.0
    assert first.representative_x_m == pytest.approx((0.0 + 1.5) / 2.0)
    assert first.track_count == 2
    # group_of 反查
    assert grouping.group_of("L02") is second
    assert grouping.group_of("L99") is None


def test_group_tracks_deterministic_under_input_order() -> None:
    """同输入逆序重排 → 输出逐字段一致（确定性契约）。"""
    tracks = [
        _track("L01", 0.0, 0.0),
        _track("L02", 0.8, 0.2),
        _track("L03", 40.0, 25.0),
    ]
    forward = group_tracks(tracks, tolerance_m=20.0)
    backward = group_tracks(list(reversed(tracks)), tolerance_m=20.0)
    assert forward == backward


def test_group_tracks_tolerance_merges_clusters() -> None:
    """容差从 5 m 提到 50 m：两组并作一组。"""
    tracks = [_track("L01", 0.0, 0.0), _track("L02", 30.0, 0.0)]
    tight = group_tracks(tracks, tolerance_m=5.0)
    loose = group_tracks(tracks, tolerance_m=50.0)
    assert len(tight.groups) == 2
    assert len(loose.groups) == 1
    assert loose.groups[0].line_ids == ("L01", "L02")
    assert loose.groups[0].max_pair_distance_m == pytest.approx(30.0)


def test_group_tracks_single_track_bypass() -> None:
    """单轨迹 → 单组、max_pair=0，不触发 scipy 退化。"""
    grouping = group_tracks([_track("L01", 10.0, 20.0)], tolerance_m=5.0)
    assert len(grouping.groups) == 1
    only = grouping.groups[0]
    assert only.group_id == "G01"
    assert only.line_ids == ("L01",)
    assert only.max_pair_distance_m == 0.0
    assert only.track_count == 1


def test_group_tracks_rejects_bad_input() -> None:
    """tolerance≤0 / 空轨迹 / NaN 代表点 → GridAnalysisError。"""
    good = _track("L01", 0.0, 0.0)
    with pytest.raises(GridAnalysisError):
        group_tracks([good], tolerance_m=0.0)
    with pytest.raises(GridAnalysisError):
        group_tracks([good], tolerance_m=-1.0)
    with pytest.raises(GridAnalysisError):
        group_tracks([], tolerance_m=5.0)
    nan_track = SpatialTrack(
        "L02", "L02",
        (SpatialTrackPoint(0, float("nan"), 0.0),),
        "EPSG:4527", "synthetic")
    with pytest.raises(GridAnalysisError, match="L02"):
        group_tracks([good, nan_track], tolerance_m=5.0)


# ---------------------------------------------------------------------------
# domain：网格化纯测试
# ---------------------------------------------------------------------------

def test_grid_attribute_bucket_mean() -> None:
    """同 cell 多点取均值；y 降序成行、x 升序成列；空 cell 为 None。

    网格锚定在数据包围盒：origin 为首 cell 中心（x_min+cell/2, y_max−cell/2）。
    """
    request = AttributeGridRequest(
        x_m=(0.0, 0.5, 1.2, 1.2),
        y_m=(0.1, 0.2, 2.2, 2.25),
        values=(1.0, 3.0, 7.0, 9.0),
        attribute_name="test_attr",
        cell_size_m=1.0,
    )
    grid = grid_attribute(request)
    assert grid.ncols == 2 and grid.nrows == 3
    # x=0 与 x=0.5 同落列 0（floor((x−x_min)/cell)）；4 点聚成 2 个有效 cell
    assert grid.valid_count == 2
    # origin = 首 cell 中心
    assert grid.x_origin_m == pytest.approx(0.5)
    assert grid.y_origin_m == pytest.approx(1.75)
    # 行 0 = 最高 y 带 (1.25, 2.25]：x=1.2 落列 1，两点均值 (7+9)/2
    assert grid.values[0] == (None, pytest.approx(8.0))
    # 行 1 = 中带 (0.25, 1.25]：空
    assert grid.values[1] == (None, None)
    # 行 2 = 最低带：x=0/0.5 同 cell 取均值 (1+3)/2
    assert grid.values[2] == (pytest.approx(2.0), None)

def _write_simple_csv(path: Path, *, traces: int, samples: int,
                      lon0: float, lat0: float, lon_step: float) -> None:
    """合成无结构 B-scan CSV（MyGPR 侧车格式），lon 偏移可控分组。"""
    lines = [
        f"Number of Samples = {samples},,,\n",
        "Time windows (ns) = 512,,\n",
        f"Number of Traces = {traces},,\n",
        "Trace interval (m) = 0.5,,\n",
    ]
    for tr in range(traces):
        lon = lon0 + tr * lon_step
        lat = lat0
        for s in range(samples):
            value = 0.0 if s != samples // 2 else 1.0
            lines.append(f"{lon:.8f},{lat:.8f},441.7,{value:.6f},1.2000\n")
    path.write_text("".join(lines), encoding="utf-8")


def _create_grouped_project(backend: MyGPRBackend, tmp_path: Path,
                            name: str) -> tuple[str, list[str]]:
    """两条相距 >20 m 的测线（CSV 经纬度错开）→ 创建 + 导入。"""
    project = backend.projects.create_project(
        tmp_path / name, name=name,
        coordinate_system="CGCS2000 / 3-degree GK zone 36",
    )
    project_id = project.project_id
    # 3° 带 zone36 中央经线 108°E；两条测线 lon 相差 0.0002° ≈ 19 m < tolerance
    specs = [
        ("L01", 107.9950, 31.2000),
        ("L02", 107.9952, 31.2000),
    ]
    for line_id, lon0, lat0 in specs:
        csv_path = tmp_path / f"{name}-{line_id}.csv"
        _write_simple_csv(csv_path, traces=11, samples=64,
                          lon0=lon0, lat0=lat0, lon_step=1e-5)
        backend.projects.import_line_source(project_id, line_id, csv_path)
    return project_id, [line_id for line_id, _, _ in specs]


def test_line_grouping_job_persists_sidecar(tmp_path: Path) -> None:
    """job：分组 → 旁车 line_groups.json → 回读一致（schema/容差/成员）。"""
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id, line_ids = _create_grouped_project(
            backend, tmp_path, "grid-group")
        # 前置确认：两条轨迹确实存在
        tracks = backend.projects.load_spatial_tracks(project_id)
        assert {t.line_id for t in tracks} == set(line_ids)

        job_id = backend.submit_line_grouping(project_id, tolerance_m=20.0)
        snapshot = backend.jobs.wait(job_id, timeout=60)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        payload = snapshot.result
        assert isinstance(payload, dict)
        assert payload["schema"] == LINE_GROUPS_SCHEMA
        assert payload["tolerance_m"] == pytest.approx(20.0)
        assert len(payload["groups"]) == 1
        assert tuple(payload["groups"][0]["line_ids"]) == tuple(sorted(line_ids))

        # 旁车落盘 + ProjectService 回读
        stored = backend.projects.load_line_groups(project_id)
        assert stored == payload
    finally:
        backend.shutdown()


def test_grid_layer_job_imports_gis(tmp_path: Path) -> None:
    """job：界面标注 → 深度网格 → GeoJSON → GIS 图层注册。"""
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id, line_ids = _create_grouped_project(
            backend, tmp_path, "grid-gis")
        # 每条测线给 3 个界面点（trace 2/5/8，样本 20 附近）
        for line_id in line_ids:
            points = tuple(
                InterpretationPoint(trace_index=float(tr), sample_index=20.0)
                for tr in (2, 5, 8)
            )
            annotation = InterfaceAnnotation(
                annotation_id=f"B-{line_id}", line_id=line_id,
                name=f"{line_id} 基覆界面", version=1, status="confirmed",
                points=points,
            )
            backend.projects.save_interface_annotation(project_id, annotation)

        job_id = backend.submit_grid_layer(project_id, line_ids, cell_size_m=2.0)
        snapshot = backend.jobs.wait(job_id, timeout=90)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        result = snapshot.result
        assert isinstance(result, dict)
        grid = result["grid"]
        assert grid["attribute"] == "interface_depth_m"
        assert grid["cell_size_m"] == pytest.approx(2.0)
        assert grid["valid_count"] >= 6  # 2 线 × 3 点至少各落 1 cell
        assert grid["ncols"] >= 1 and grid["nrows"] >= 1

        # GIS 图层注册表回读
        layers = backend.projects.list_gis_layers(project_id)
        names = [str(item.get("name")) for item in layers]
        assert result["layer"]["name"] in names
        matched = next(item for item in layers
                       if item.get("name") == result["layer"]["name"])
        assert matched.get("kind") in ("geojson", "vector", "unknown", "")
        assert matched.get("role") == "analysis"

        # 几何抽查：GeoJSON 中每有效 cell 一个 Point feature
        layer_path = Path(str(matched.get("source_path", "")))
        if layer_path.is_file():
            payload = json.loads(layer_path.read_text(encoding="utf-8"))
            features = payload.get("features", [])
            assert features, "GeoJSON 应含 Point features"
            assert all(
                feature["geometry"]["type"] == "Point" for feature in features
            )
    finally:
        backend.shutdown()


def test_grid_service_rejects_empty_lines(tmp_path: Path) -> None:
    """无界面标注 → gridded_interface_depth 报 GridAnalysisError（job failed）。"""
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project_id, line_ids = _create_grouped_project(
            backend, tmp_path, "grid-empty")
        service = GridService(backend.projects)
        with pytest.raises(GridAnalysisError):
            service.export_interface_depth_grid(project_id, line_ids)
    finally:
        backend.shutdown()

# -*- coding: utf-8 -*-
"""ui.geo_utils 纯函数的针对性测试（无 Qt 依赖，后端 CI 可跑）。

覆盖空间信息页"项目覆盖统计"卡的关键数学：投影米坐标欧氏距离、
经纬度 haversine 距离、非法点跳过、CRS 缺失时的幅值兜底启发。
"""
from __future__ import annotations

from types import SimpleNamespace

from ui.geo_utils import coverage_statistics, format_distance


def _point(x, y):
    return SimpleNamespace(x=x, y=y)


def _track(points, coordinate_system='EPSG:32648'):
    return SimpleNamespace(
        coordinate_system=coordinate_system,
        points=tuple(_point(x, y) for x, y in points),
    )


def test_empty_tracks_yield_zero_statistics() -> None:
    assert coverage_statistics([]) == {
        'track_count': 0,
        'point_count': 0,
        'segment_count': 0,
        'length_m': 0.0,
    }
    # 无有效点的轨迹不计入测线数
    assert coverage_statistics([_track(((float('nan'), 1.0),))])['track_count'] == 0


def test_missing_crs_uses_magnitude_fallback() -> None:
    """CRS 缺失且坐标都在经纬度幅值内时按地理距离（遗留兜底启发）。"""
    statistics = coverage_statistics([
        _track(((120.0, 30.0), (120.0, 30.01)), ''),
    ])
    # 0.01 度纬度约 1.1 km（空 CRS 的幅值兜底把坐标当经纬度）
    assert 1_100.0 < statistics['length_m'] < 1_120.0
    # 空 CRS + 小幅值坐标与显式 EPSG:4326 结果一致；
    # 局部米坐标若恰在经纬度幅值内会被误判（原实现既有行为）
    implicit = coverage_statistics([_track(((0.0, 0.0), (3.0, 4.0)), '')])
    explicit = coverage_statistics([_track(((0.0, 0.0), (3.0, 4.0)), 'EPSG:4326')])
    assert implicit['length_m'] == explicit['length_m']
    assert implicit['length_m'] > 100_000.0


def test_geographic_crs_keywords_are_detected() -> None:
    """EPSG:4490 / wgs84 等关键字一律按地理坐标测距。"""
    statistics = coverage_statistics([
        _track(((120.0, 30.0), (120.0, 30.01)), 'WGS84'),
        _track(((120.0, 30.0), (120.0, 30.01)), 'EPSG:4490'),
    ])
    assert 2_200.0 < statistics['length_m'] < 2_250.0


def test_single_point_track_has_no_segments() -> None:
    statistics = coverage_statistics([_track(((1.0, 2.0),))])
    assert statistics['track_count'] == 1
    assert statistics['point_count'] == 1
    assert statistics['segment_count'] == 0
    assert statistics['length_m'] == 0.0


def test_mixed_valid_invalid_points_keep_valid_segments() -> None:
    """NaN 点被跳过：不参与计数，也不与相邻点构成段。"""
    track = SimpleNamespace(
        coordinate_system='EPSG:32648',
        points=(_point(0.0, 0.0), _point(float('nan'), 1.0), _point(1.0, 0.0)),
    )
    statistics = coverage_statistics([track])
    assert statistics['point_count'] == 2
    assert statistics['segment_count'] == 1
    assert statistics['length_m'] == 1.0


def test_non_numeric_coordinates_are_skipped() -> None:
    track = SimpleNamespace(
        coordinate_system='EPSG:32648',
        points=(_point('abc', 0.0), _point(0.0, 0.0), _point(0.0, 0.0)),
    )
    statistics = coverage_statistics([track])
    assert statistics['track_count'] == 1
    assert statistics['point_count'] == 2


def test_format_distance_boundary() -> None:
    assert format_distance(0.0) == '0 m'
    assert format_distance(999.9) == '1000 m'
    assert format_distance(1000.0) == '1.00 km'

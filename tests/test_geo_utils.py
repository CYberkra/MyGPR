# -*- coding: utf-8 -*-
"""ui.geo_utils 纯函数的针对性测试（无 Qt 依赖，后端 CI 可跑）。

覆盖空间信息页"项目覆盖统计"卡的关键数学：投影米坐标欧氏距离、
经纬度 haversine 距离、非法点跳过、CRS 缺失时的幅值兜底启发。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

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


# ---------------------------------------------------------------- 向量化回归
class TestVectorizedEquivalence:
    """``coverage_statistics`` 的向量化实现必须与逐点实现在数值上一致。

    背景：该函数曾是逐点 Python 循环 + 逐段 ``np.hypot``，实测 6 线 /
    10,416 点耗 25 ms（投影）、64 ms（经纬 haversine）。它挂在
    ``spatial._refresh_coverage_statistics`` 上，一次「打开项目」会被调
    2–3 次，足以让主线程超出一个重绘周期（就是"打开项目后整窗闪一下"
    的贡献项之一）。向量化后降到 4 ms / 6 ms。

    本测试用**逐点参考实现**做黄金对照，确保优化没改变任何数字。
    """

    @staticmethod
    def _reference(tracks) -> dict:
        """逐点参考实现（修复前的原始逻辑，作为黄金标准）。"""
        import numpy as np

        from ui.geo_utils import _is_geographic, _haversine_m

        track_count = point_count = segment_count = 0
        length_m = 0.0
        for track in tracks or []:
            points = []
            for point in getattr(track, 'points', ()) or ():
                try:
                    x = float(getattr(point, 'x', float('nan')))
                    y = float(getattr(point, 'y', float('nan')))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    points.append((x, y))
            if not points:
                continue
            track_count += 1
            point_count += len(points)
            geographic = _is_geographic(
                str(getattr(track, 'coordinate_system', '') or ''), points)
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                segment_count += 1
                if geographic:
                    length_m += _haversine_m(x0, y0, x1, y1)
                else:
                    length_m += float(np.hypot(x1 - x0, y1 - y0))
        return {
            'track_count': track_count,
            'point_count': point_count,
            'segment_count': segment_count,
            'length_m': length_m,
        }

    @staticmethod
    def _big_tracks(n_tracks=6, per=1736, *, geographic=False):
        """构造与真实项目同量级的轨迹（~10k 点）。"""
        tracks = []
        for t in range(n_tracks):
            x0 = 500_000.0 + t * 1000.0
            y0 = 3_200_000.0 + t * 500.0
            if geographic:
                pts = tuple((113.5 + i * 1e-5, 30.2 + i * 1e-5)
                            for i in range(per))
            else:
                pts = tuple((x0 + i * 0.5, y0 + i * 0.3) for i in range(per))
            tracks.append(_track(pts, 'EPSG:4326' if geographic else 'EPSG:4547'))
        return tracks

    def test_projected_matches_reference_exactly(self) -> None:
        tracks = self._big_tracks()
        got = coverage_statistics(tracks)
        want = self._reference(tracks)
        assert got['track_count'] == want['track_count'] == 6
        assert got['point_count'] == want['point_count'] == 6 * 1736
        assert got['segment_count'] == want['segment_count']
        # 浮点求和顺序变了（逐段累加 → np.sum），allow 1e-6 相对误差
        assert got['length_m'] == pytest.approx(want['length_m'], rel=1e-9)

    def test_geographic_matches_reference_exactly(self) -> None:
        tracks = self._big_tracks(geographic=True)
        got = coverage_statistics(tracks)
        want = self._reference(tracks)
        assert got['track_count'] == want['track_count'] == 6
        assert got['segment_count'] == want['segment_count']
        assert got['length_m'] == pytest.approx(want['length_m'], rel=1e-9)

    def test_per_point_skip_contract_preserved(self) -> None:
        """非法点只剔除自己，不得让整条测线消失（对外契约）。"""
        mixed = _track(((1.0, 1.0), (float('nan'), 2.0), ('abc', 3.0),
                        (2.0, 2.0), (3.0, 3.0)), 'EPSG:32648')
        got = coverage_statistics([mixed])
        want = self._reference([mixed])
        assert got == pytest.approx(want)
        assert got['track_count'] == 1
        assert got['point_count'] == 3
        assert got['segment_count'] == 2

    def test_all_invalid_track_dropped(self) -> None:
        bad = _track((('abc', 'def'), (float('inf'), 1.0)), 'EPSG:32648')
        assert coverage_statistics([bad])['track_count'] == 0

    def test_empty_and_single_point_unchanged(self) -> None:
        assert coverage_statistics([])['length_m'] == 0.0
        one = coverage_statistics([_track(((5.0, 6.0),))])
        assert one == {'track_count': 1, 'point_count': 1,
                       'segment_count': 0, 'length_m': 0.0}

    def test_is_speedup_not_regression(self) -> None:
        """向量化实现应显著快于逐点实现（守 3× 下限，避免退化回循环）。"""
        import time

        tracks = self._big_tracks()
        # 预热 numpy（首次调用含数组分配，不计入）
        coverage_statistics(tracks)
        self._reference(tracks)

        t0 = time.perf_counter()
        for _ in range(5):
            coverage_statistics(tracks)
        fast = (time.perf_counter() - t0) / 5

        t0 = time.perf_counter()
        for _ in range(5):
            self._reference(tracks)
        slow = (time.perf_counter() - t0) / 5

        assert fast * 3 < slow, (
            f'向量化未体现优势：fast={fast * 1000:.2f}ms '
            f'slow={slow * 1000:.2f}ms（应至少 3×）')

"""Focused tests for the frontend-only SpatialPage coverage summary."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from ui.pages.spatial_page import _coverage_statistics, _format_distance  # noqa: E402


def _track(points, coordinate_system='EPSG:32648'):
    return SimpleNamespace(
        coordinate_system=coordinate_system,
        points=tuple(SimpleNamespace(x=x, y=y) for x, y in points),
    )


def test_coverage_statistics_sums_projected_track_lengths() -> None:
    statistics = _coverage_statistics([
        _track(((0, 0), (3, 4), (3, 8))),
        _track(((10, 10),)),
    ])

    assert statistics == {
        'track_count': 2,
        'point_count': 4,
        'segment_count': 2,
        'length_m': 9.0,
    }


def test_coverage_statistics_uses_geographic_distance_and_ignores_invalid_points() -> None:
    statistics = _coverage_statistics([
        _track(((120.0, 30.0), (120.0, 30.01)), 'EPSG:4326'),
        _track(((float('nan'), 0.0),)),
    ])

    assert statistics['track_count'] == 1
    assert statistics['point_count'] == 2
    assert statistics['segment_count'] == 1
    assert 1_100.0 < statistics['length_m'] < 1_120.0


def test_distance_format_switches_to_kilometres() -> None:
    assert _format_distance(999.4) == '999 m'
    assert _format_distance(1_250.0) == '1.25 km'

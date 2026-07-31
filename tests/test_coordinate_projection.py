#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for core.coordinate_projection — CGCS2000 / 3-degree GK projection."""

from __future__ import annotations

import numpy as np
import pytest

from core.coordinate_projection import (
    ProjectionError,
    ProjectionSpec,
    _zone_to_cgcs2000_3deg_epsg,
    infer_3deg_zone_from_longitude,
    project_lonlat_to_xy,
    resolve_projection_spec,
)


# ── _zone_to_cgcs2000_3deg_epsg ────────────────────────────────────────────

class TestZoneToEPSG:
    def test_zone_39_returns_4527(self) -> None:
        assert _zone_to_cgcs2000_3deg_epsg(39) == 4527

    def test_zone_25_lower_bound(self) -> None:
        assert _zone_to_cgcs2000_3deg_epsg(25) == 4513

    def test_zone_45_upper_bound(self) -> None:
        assert _zone_to_cgcs2000_3deg_epsg(45) == 4533

    def test_zone_below_25_raises(self) -> None:
        with pytest.raises(ProjectionError, match="out of expected range"):
            _zone_to_cgcs2000_3deg_epsg(24)

    def test_zone_above_45_raises(self) -> None:
        with pytest.raises(ProjectionError, match="out of expected range"):
            _zone_to_cgcs2000_3deg_epsg(46)

    def test_zone_zero_raises(self) -> None:
        with pytest.raises(ProjectionError, match="out of expected range"):
            _zone_to_cgcs2000_3deg_epsg(0)


# ── infer_3deg_zone_from_longitude ──────────────────────────────────────────

class TestInferZone:
    def test_117e_returns_zone_39(self) -> None:
        assert infer_3deg_zone_from_longitude(117.0) == 39

    def test_108e_returns_zone_36(self) -> None:
        assert infer_3deg_zone_from_longitude(108.0) == 36

    def test_105e_returns_zone_35(self) -> None:
        assert infer_3deg_zone_from_longitude(105.0) == 35

    def test_120e_returns_zone_40(self) -> None:
        assert infer_3deg_zone_from_longitude(120.0) == 40

    def test_very_west_longitude_clamped_to_25(self) -> None:
        assert infer_3deg_zone_from_longitude(0.0) == 25

    def test_very_east_longitude_clamped_to_45(self) -> None:
        assert infer_3deg_zone_from_longitude(180.0) == 45

    def test_negative_longitude_clamped(self) -> None:
        assert infer_3deg_zone_from_longitude(-10.0) == 25

    def test_nan_longitude_raises(self) -> None:
        with pytest.raises(ProjectionError, match="non-finite"):
            infer_3deg_zone_from_longitude(float("nan"))

    def test_inf_longitude_raises(self) -> None:
        with pytest.raises(ProjectionError, match="non-finite"):
            infer_3deg_zone_from_longitude(float("inf"))

    def test_negative_inf_raises(self) -> None:
        with pytest.raises(ProjectionError, match="non-finite"):
            infer_3deg_zone_from_longitude(float("-inf"))


# ── resolve_projection_spec ─────────────────────────────────────────────────

class TestResolveProjectionSpec:
    # EPSG format
    def test_explicit_epsg_4544(self) -> None:
        spec = resolve_projection_spec("EPSG:4544")
        assert spec.epsg == 4544
        assert spec.zone is None
        assert spec.is_auto is False

    def test_epsg_case_insensitive(self) -> None:
        spec = resolve_projection_spec("epsg:4527")
        assert spec.epsg == 4527

    def test_epsg_with_chinese_colon(self) -> None:
        spec = resolve_projection_spec("EPSG：4544")
        assert spec.epsg == 4544

    def test_epsg_extracts_from_noisy_text(self) -> None:
        spec = resolve_projection_spec("使用 EPSG:4544 坐标系统")
        assert spec.epsg == 4544

    # CGCS2000 3-degree GK with explicit zone
    def test_cgcs2000_3deg_gk_zone_39(self) -> None:
        spec = resolve_projection_spec("CGCS2000 / 3-degree GK Zone 39")
        assert spec.epsg == 4527
        assert spec.zone == 39
        assert spec.is_auto is False

    def test_cgcs2000_chinese_text_with_zone(self) -> None:
        spec = resolve_projection_spec("CGCS2000 3-degree Gauss-Kruger zone 36")
        assert spec.epsg == 4524
        assert spec.zone == 36

    def test_cgcs2000_with_zone_number_only(self) -> None:
        spec = resolve_projection_spec("CGCS2000 3度带 39 带")
        assert spec.epsg == 4527
        assert spec.zone == 39

    # CGCS2000 with auto zone from longitude
    def test_cgcs2000_auto_zone_from_longitude(self) -> None:
        spec = resolve_projection_spec("CGCS2000 / 3-degree GK", mean_longitude=117.0)
        assert spec.epsg == 4527
        assert spec.zone == 39
        assert spec.is_auto is True

    # Auto infer from longitude when no system specified
    def test_auto_infer_from_longitude(self) -> None:
        spec = resolve_projection_spec(None, mean_longitude=108.0)
        assert spec.zone == 36
        assert spec.is_auto is True

    def test_empty_string_with_longitude(self) -> None:
        spec = resolve_projection_spec("", mean_longitude=117.0)
        assert spec.zone == 39

    def test_empty_string_without_longitude_raises(self) -> None:
        with pytest.raises(ProjectionError, match="未填写"):
            resolve_projection_spec("")

    def test_unknown_system_without_longitude_raises(self) -> None:
        with pytest.raises(ProjectionError, match="暂不支持"):
            resolve_projection_spec("Mars 坐标系统")

    def test_none_without_longitude_raises(self) -> None:
        with pytest.raises(ProjectionError, match="未填写"):
            resolve_projection_spec(None)

    # CGCS2000 with zone missing and no longitude raises
    def test_cgcs2000_without_zone_or_longitude_raises(self) -> None:
        with pytest.raises(ProjectionError, match="缺少"):
            resolve_projection_spec("CGCS2000 / 3-degree GK")


# ── ProjectionSpec ──────────────────────────────────────────────────────────

class TestProjectionSpec:
    def test_description_with_zone(self) -> None:
        spec = ProjectionSpec(name="CGCS2000", epsg=4527, zone=39)
        assert "zone 39" in spec.description.lower()

    def test_description_without_zone(self) -> None:
        spec = ProjectionSpec(name="EPSG:4544", epsg=4544)
        assert "zone" not in spec.description.lower()

    def test_default_source_epsg_is_wgs84(self) -> None:
        spec = ProjectionSpec(name="t", epsg=4527)
        assert spec.source_epsg == 4326

    def test_auto_flag_defaults_false(self) -> None:
        spec = ProjectionSpec(name="t", epsg=4527)
        assert spec.is_auto is False


# ── project_lonlat_to_xy ────────────────────────────────────────────────────

class TestProjectLonLatToXY:
    def test_projects_single_point(self) -> None:
        x, y, spec = project_lonlat_to_xy(
            [117.0], [30.0], coordinate_system="EPSG:4544"
        )
        assert x.shape == (1,)
        assert y.shape == (1,)
        assert x.dtype == np.float64
        assert y.dtype == np.float64
        assert spec.epsg == 4544

    def test_projects_multiple_points(self) -> None:
        lon = [117.0, 117.1, 117.2]
        lat = [30.0, 30.1, 30.2]
        x, y, spec = project_lonlat_to_xy(lon, lat, coordinate_system="EPSG:4544")
        assert x.shape == (3,)
        assert y.shape == (3,)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ProjectionError, match="shape mismatch"):
            project_lonlat_to_xy([117.0, 117.1], [30.0], coordinate_system="EPSG:4544")

    def test_empty_arrays_raise(self) -> None:
        with pytest.raises(ProjectionError, match="empty"):
            project_lonlat_to_xy([], [], coordinate_system="EPSG:4544")

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ProjectionError, match="no finite"):
            project_lonlat_to_xy(
                [float("nan")], [float("nan")], coordinate_system="EPSG:4544"
            )

    def test_mixed_finite_and_nan_still_projects(self) -> None:
        """Finite values should project even when some are NaN."""
        lon = [117.0, float("nan")]
        lat = [30.0, 30.1]
        x, y, spec = project_lonlat_to_xy(lon, lat, coordinate_system="EPSG:4544")
        assert x.shape == (2,)
        assert np.isfinite(x[0])

    def test_auto_zone_from_mean_longitude(self) -> None:
        x, y, spec = project_lonlat_to_xy(
            [117.0, 117.05], [30.0, 30.05], coordinate_system=None
        )
        assert spec.is_auto is True
        assert 25 <= spec.zone <= 45

    def test_result_is_float64(self) -> None:
        x, y, _ = project_lonlat_to_xy([117.0], [30.0], coordinate_system="EPSG:4544")
        assert x.dtype == np.float64
        assert y.dtype == np.float64

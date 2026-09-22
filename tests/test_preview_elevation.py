# -*- coding: utf-8 -*-
"""逐道地面高程（海拔纵轴的数据层）测试。

链路：core.gui_rendering.PreviewBundle.trace_elevation_m
      ← ui.desktop_backend_facade.build_preview_bundle
      ← ui.controllers.project_controller._bundle_from_window
      ← mygpr.projects.line_trace_elevation

关键语义（实测确认，勿改）：MyGPR 航空堆叠 CSV 第 3 列是**地面高程**，
不是天线海拔；存储层 TrajectoryPoint.z = 地面 + 离地高度，所以从轨迹文件
回读时地面 = z − flight_height_m。详见
``output/probes_archive/_probe_column3_semantics.py``。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.gui_rendering import align_trace_vector, bundle_from_dataset, make_preview_bundle
from mygpr.infrastructure.persistence.spatial_adapter import (
    _fill_gaps,
    _metadata_ground_elevation,
)


def _dataset(rows: list[dict[str, Any]], *, shape: tuple[int, int] = (12, 40),
             extra: dict[str, Any] | None = None) -> Any:
    matrix = np.zeros(shape, dtype=np.float32)

    class _DS:
        line_id = "L01"
        sample_count = shape[0]
        trace_count = shape[1]
        metadata = {"trajectory_rows": rows}

        def preview_matrix(self, max_samples: int, max_traces: int) -> np.ndarray:
            step_s = max(1, int(np.ceil(shape[0] / max_samples)))
            step_t = max(1, int(np.ceil(shape[1] / max_traces)))
            return matrix[::step_s, ::step_t]

    ds = _DS()
    ds.matrix = matrix
    for key, value in (extra or {}).items():
        setattr(ds, key, np.asarray(value, dtype=np.float64))
    return ds


class TestPreviewBundleElevation:
    def test_default_has_no_elevation(self) -> None:
        bundle = make_preview_bundle(np.zeros((8, 10), dtype=np.float32))
        assert bundle.trace_elevation_m is None

    def test_downsamples_in_sync_with_trace_stride(self) -> None:
        traces = 3600  # > _MAX_PREVIEW_TRACES(1800) → 跨步 2
        elevation = np.linspace(400.0, 500.0, traces)
        matrix = np.zeros((20, traces), dtype=np.float32)
        bundle = make_preview_bundle(matrix, trace_elevation_m=elevation)
        got = bundle.trace_elevation_m
        assert got is not None
        assert got.size == bundle.matrix.shape[1]
        # 每列的高程必须与降采样后该列代表的那一道严格对应
        assert np.allclose(got, elevation[::2][: got.size], atol=1e-6)

    def test_from_dataset_uses_trajectory_rows_elevation(self) -> None:
        rows = [{"elevation": 442.7 + 0.1 * i, "height_m": 9.5} for i in range(40)]
        bundle = bundle_from_dataset(_dataset(rows))
        got = bundle.trace_elevation_m
        expected = np.array([r["elevation"] for r in rows])
        assert got is not None
        assert got.size == 40
        assert np.allclose(got, expected, atol=1e-6)

    def test_from_dataset_prefers_explicit_ground_elevation(self) -> None:
        rows = [{"elevation": 100.0 + i} for i in range(40)]
        explicit = np.linspace(500.0, 520.0, 40)
        bundle = bundle_from_dataset(_dataset(rows, extra={"ground_elevation_m": explicit}))
        assert bundle.trace_elevation_m is not None
        assert np.allclose(bundle.trace_elevation_m, explicit, atol=1e-6)

    def test_no_elevation_data_is_none(self) -> None:
        bundle = bundle_from_dataset(_dataset([]))
        assert bundle.trace_elevation_m is None


class TestAlignTraceVector:
    def test_matching_size_passthrough(self) -> None:
        series = np.array([1.0, 2.0, 3.0])
        assert np.allclose(align_trace_vector(series, 3), series)

    def test_resamples_to_trace_grid(self) -> None:
        series = np.array([0.0, 10.0])  # 2 点 → 5 道
        got = align_trace_vector(series, 5)
        assert got is not None
        assert np.allclose(got, [0.0, 2.5, 5.0, 7.5, 10.0], atol=1e-9)

    def test_unusable_inputs_are_none(self) -> None:
        assert align_trace_vector(None, 10) is None
        assert align_trace_vector([], 10) is None
        assert align_trace_vector([5.0], 10) is None  # 单点无法构成插值区间


class TestWindowElevation:
    """UI 侧：窗口抽样必须与矩阵同索引（PyQt 可选环境）。"""

    @pytest.fixture(autouse=True)
    def _require_qt(self, qapp) -> None:  # noqa: ANN001
        pass

    def test_elevation_follows_trace_indices(self, qapp) -> None:
        from ui.controllers.project_controller import ProjectController

        total = 48
        elevation = np.linspace(430.0, 460.0, total)
        trace_idx = np.arange(0, total, 2, dtype=np.int64)  # 每两道抽一道
        bundle = ProjectController._bundle_from_window(
            np.zeros((72, len(trace_idx)), dtype=np.float32), "L01", title="t",
            time_window_ns=500.0, length_m=24.0,
            sample_indices=np.arange(0, 72, dtype=np.int64), trace_indices=trace_idx,
            total_samples=72, total_traces=total,
            trace_elevation_m=elevation,
        )
        got = bundle.trace_elevation_m
        assert got is not None
        assert got.size == len(trace_idx)
        assert np.allclose(got, elevation[trace_idx], atol=1e-6)

    def test_missing_elevation_stays_none(self, qapp) -> None:
        from ui.controllers.project_controller import ProjectController

        total = 8
        bundle = ProjectController._bundle_from_window(
            np.zeros((10, total), dtype=np.float32), "L01", title="t",
            trace_indices=np.arange(total, dtype=np.int64), total_traces=total,
        )
        assert bundle.trace_elevation_m is None


class TestMetadataGroundElevation:
    def test_prefers_explicit_ground(self) -> None:
        metadata = {
            "ground_elevation_m": np.array([10.0, 11.0]),
            "local_z_m": np.array([99.0, 99.0]),
            "flight_height_m": np.array([5.0, 6.0]),
        }
        assert np.allclose(_metadata_ground_elevation(metadata), [10.0, 11.0])

    def test_altitude_minus_flight_height(self) -> None:
        metadata = {
            "local_z_m": np.array([15.0, 17.0]),
            "flight_height_m": np.array([5.0, 6.0]),
        }
        assert np.allclose(_metadata_ground_elevation(metadata), [10.0, 11.0])

    def test_missing_flight_height_yields_none(self) -> None:
        # 只有轨迹海拔时不能伪造地面高程，否则把天线海拔当地面
        assert _metadata_ground_elevation({"local_z_m": np.array([15.0, 16.0])}) is None

    def test_fill_gaps_interpolates(self) -> None:
        series = np.array([1.0, np.nan, 3.0])
        assert np.allclose(_fill_gaps(series), [1.0, 2.0, 3.0])

    def test_fill_gaps_all_invalid_is_none(self) -> None:
        assert _fill_gaps(np.array([np.nan, np.nan])) is None

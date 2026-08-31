#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless tests for ``core/gui_rendering.py`` (no Qt required)."""
from __future__ import annotations

import numpy as np
import pytest

from core.gpr_data_model import GPRDataSet
from core.gui_rendering import (
    COLORMAPS,
    PreviewBundle,
    bundle_from_dataset,
    colormap_names,
    compute_levels,
    downsample_matrix,
    make_preview_bundle,
)


@pytest.mark.unit
def test_colormap_names_matches_contract() -> None:
    names = colormap_names()
    assert names == ["seismic", "hot", "jet", "gray", "viridis", "plasma", "inferno", "magma", "cividis"]
    assert names is not COLORMAPS  # defensive copy
    assert names[0] == "seismic"


@pytest.mark.unit
def test_downsample_matrix_bounds_large_input() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((2000, 4000))
    preview = downsample_matrix(matrix, max_samples=900, max_traces=1800)
    assert preview.shape[0] <= 900
    assert preview.shape[1] <= 1800
    assert preview.dtype == np.float32
    # strided sampling: values must come from the source matrix
    assert np.allclose(preview, matrix[::3, ::3].astype(np.float32))


@pytest.mark.unit
def test_downsample_matrix_small_input_passthrough() -> None:
    matrix = np.arange(50 * 30, dtype=np.float64).reshape(50, 30)
    preview = downsample_matrix(matrix)
    assert preview.shape == (50, 30)
    assert preview.dtype == np.float32
    assert np.allclose(preview, matrix)


@pytest.mark.unit
def test_compute_levels_random_matrix() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((200, 100)) * 10.0
    vmin, vmax = compute_levels(matrix)
    assert np.isfinite(vmin) and np.isfinite(vmax)
    assert vmin < vmax
    assert matrix.min() <= vmin < vmax <= matrix.max()


@pytest.mark.unit
def test_compute_levels_nan_inf_safe() -> None:
    matrix = np.linspace(-5.0, 5.0, 400, dtype=np.float64).reshape(20, 20)
    matrix[0, 0] = np.nan
    matrix[1, 1] = np.inf
    matrix[2, 2] = -np.inf
    vmin, vmax = compute_levels(matrix)
    assert np.isfinite(vmin) and np.isfinite(vmax)
    assert vmin < vmax


@pytest.mark.unit
def test_compute_levels_all_nan_fallback() -> None:
    matrix = np.full((10, 10), np.nan)
    assert compute_levels(matrix) == (-1.0, 1.0)


@pytest.mark.unit
def test_compute_levels_constant_matrix_degenerates_to_pm_max() -> None:
    matrix = np.full((8, 8), 3.5)
    vmin, vmax = compute_levels(matrix)
    assert (vmin, vmax) == (-3.5, 3.5)
    zero = np.zeros((8, 8))
    assert compute_levels(zero) == (-1.0, 1.0)


@pytest.mark.unit
def test_compute_levels_respects_percentile_args() -> None:
    matrix = np.linspace(0.0, 99.0, 100).reshape(10, 10)
    vmin, vmax = compute_levels(matrix, p_low=10.0, p_high=90.0)
    assert vmin == pytest.approx(9.9, rel=1e-6)
    assert vmax == pytest.approx(89.1, rel=1e-6)


@pytest.mark.unit
def test_make_preview_bundle_counts_and_axes() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((1500, 2400))
    trace_axis = np.linspace(0.0, 120.0, 2400)
    sample_axis = np.linspace(0.0, 250.0, 1500)
    bundle = make_preview_bundle(
        matrix,
        title="合成测线",
        trace_axis_m=trace_axis,
        sample_axis=sample_axis,
        sample_axis_label="时间 (ns)",
    )
    assert isinstance(bundle, PreviewBundle)
    assert bundle.sample_count == 1500
    assert bundle.trace_count == 2400
    assert bundle.matrix.shape[0] <= 900
    assert bundle.matrix.shape[1] <= 1800
    assert bundle.matrix.dtype == np.float32
    assert bundle.title == "合成测线"
    assert bundle.x_label == "道数"
    assert bundle.y_label == "采样点"
    assert bundle.trace_axis_m is not None and len(bundle.trace_axis_m) == bundle.matrix.shape[1]
    assert bundle.sample_axis is not None and len(bundle.sample_axis) == bundle.matrix.shape[0]
    assert bundle.sample_axis_label == "时间 (ns)"
    assert bundle.vmin < bundle.vmax


@pytest.mark.unit
def test_make_preview_bundle_with_nan() -> None:
    matrix = np.random.default_rng(3).standard_normal((120, 80))
    matrix[::7, ::5] = np.nan
    bundle = make_preview_bundle(matrix)
    assert np.isfinite(bundle.vmin) and np.isfinite(bundle.vmax)
    assert np.isnan(bundle.matrix).any()  # nan passes through for display masking


@pytest.mark.unit
def test_make_preview_bundle_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        make_preview_bundle(np.zeros((3, 3, 3)))


@pytest.mark.unit
def test_bundle_from_dataset_uses_bounded_preview() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((1200, 2000)).astype(np.float32)
    matrix[100:110, 200:210] = np.nan
    dataset = GPRDataSet(
        line_id="L01",
        matrix=matrix,
        distance_axis_m=np.linspace(0.0, 50.0, 2000, dtype=np.float32),
        time_axis_ns=np.linspace(0.0, 250.0, 1200, dtype=np.float32),
        depth_axis_m=np.linspace(0.0, 12.5, 1200, dtype=np.float32),
    )
    bundle = bundle_from_dataset(dataset)
    assert bundle.sample_count == 1200
    assert bundle.trace_count == 2000
    assert bundle.matrix.shape[0] <= 900
    assert bundle.matrix.shape[1] <= 1800
    assert bundle.title == "L01"
    assert bundle.sample_axis_label == "时间 (ns)"
    assert bundle.trace_axis_m is not None and len(bundle.trace_axis_m) == bundle.matrix.shape[1]
    assert bundle.sample_axis is not None and len(bundle.sample_axis) == bundle.matrix.shape[0]
    assert np.isfinite(bundle.vmin) and np.isfinite(bundle.vmax)
    assert bundle.vmin < bundle.vmax


@pytest.mark.unit
def test_bundle_from_dataset_honours_overrides() -> None:
    matrix = np.linspace(-1.0, 1.0, 100 * 60, dtype=np.float32).reshape(100, 60)
    dataset = GPRDataSet(
        line_id="L02",
        matrix=matrix,
        distance_axis_m=np.zeros(60, dtype=np.float32),
        time_axis_ns=np.zeros(100, dtype=np.float32),
        depth_axis_m=np.zeros(100, dtype=np.float32),
    )
    bundle = bundle_from_dataset(dataset, title="自定义", p_low=0.0, p_high=100.0)
    assert bundle.title == "自定义"
    assert bundle.vmin == pytest.approx(-1.0, rel=1e-5)
    assert bundle.vmax == pytest.approx(1.0, rel=1e-5)

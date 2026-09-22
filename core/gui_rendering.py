#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qt-free rendering helpers for the MyGPR Qt frontend.

This module owns the numpy-side of B-scan preview rendering: bounded
downsampling, nan/inf safe percentile levels and the immutable
:class:`PreviewBundle` consumed by ``ui/widgets/bscan_view.py``.

No Qt imports are allowed here so the module stays usable from headless
tests, CLI tools and worker threads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

COLORMAPS: list[str] = [
    "seismic",
    "hot",
    "jet",
    "gray",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
]

DEFAULT_COLORMAP = "seismic"

_MAX_PREVIEW_SAMPLES = 900
_MAX_PREVIEW_TRACES = 1800


@dataclass(frozen=True)
class PreviewBundle:
    """Immutable, display-ready preview of one B-scan matrix."""

    matrix: np.ndarray            # float32 (samples, traces), downsampled
    vmin: float
    vmax: float
    sample_count: int             # original sample count
    trace_count: int              # original trace count
    title: str = ""
    x_label: str = "道数"
    y_label: str = "采样点"
    trace_axis_m: np.ndarray | None = None   # optional distance axis (downsampled)
    sample_axis: np.ndarray | None = None    # optional time/depth axis (downsampled)
    sample_axis_label: str = ""
    trace_elevation_m: np.ndarray | None = None  # 逐道地面高程 (m, downsampled)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"preview matrix must be 2D, got shape={matrix.shape!r}")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "vmin", float(self.vmin))
        object.__setattr__(self, "vmax", float(self.vmax))
        object.__setattr__(self, "sample_count", max(0, int(self.sample_count)))
        object.__setattr__(self, "trace_count", max(0, int(self.trace_count)))
        for name in ("trace_axis_m", "sample_axis"):
            axis = getattr(self, name)
            if axis is not None:
                object.__setattr__(self, name, np.asarray(axis, dtype=np.float64))
        if self.trace_elevation_m is not None:
            aligned = align_trace_vector(self.trace_elevation_m, matrix.shape[1])
            object.__setattr__(self, "trace_elevation_m", aligned)


def align_trace_vector(series: Any, trace_count: int) -> np.ndarray | None:
    """Coerce a per-trace series to a float64 vector of ``trace_count`` entries.

    轨迹点数可能与道数不等（RTK 对齐后会抽/补点），此函数把逐道序列线性
    重采样到道数网格上；点数为 0 或无法构成插值区间时返回 ``None``（视为
    「无高程数据」，UI 据此禁用海拔纵轴）。
    """
    if series is None:
        return None
    array = np.asarray(series, dtype=np.float64).ravel()
    target = max(0, int(trace_count))
    if array.size == 0 or target == 0:
        return None
    if array.size == target:
        return np.ascontiguousarray(array)
    if array.size < 2:
        return None
    source = np.linspace(0.0, 1.0, array.size)
    return np.ascontiguousarray(np.interp(np.linspace(0.0, 1.0, target), source, array))


def colormap_names() -> list[str]:
    """Return the supported colormap names (default first)."""
    return list(COLORMAPS)


def _strides(rows: int, cols: int, max_samples: int, max_traces: int) -> tuple[int, int]:
    row_step = max(1, int(np.ceil(rows / max(max_samples, 1)))) if rows else 1
    col_step = max(1, int(np.ceil(cols / max(max_traces, 1)))) if cols else 1
    return row_step, col_step


def downsample_matrix(
    matrix: Any,
    max_samples: int = _MAX_PREVIEW_SAMPLES,
    max_traces: int = _MAX_PREVIEW_TRACES,
) -> np.ndarray:
    """Return a bounded float32 strided view/copy of ``matrix``.

    The strided slice never materialises the full-resolution input, so
    mmap/lazy-backed matrices stay cheap.  NaN/Inf values pass through
    untouched; level computation is handled separately.
    """
    array = np.asanyarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={array.shape!r}")
    row_step, col_step = _strides(array.shape[0], array.shape[1], max_samples, max_traces)
    view = array[::row_step, ::col_step]
    if view.dtype != np.float32:
        view = np.asarray(view, dtype=np.float32)
    return view


def _finite_values(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return np.empty(0, dtype=np.float64)
    return array[finite]


def compute_levels(
    matrix: Any,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> tuple[float, float]:
    """Compute nan/inf safe percentile display levels.

    Degenerate inputs (empty, all non-finite, or constant matrices) fall
    back to ``±max(abs(values))`` (or ``±1`` when everything is zero).
    """
    values = _finite_values(np.asanyarray(matrix))
    if values.size == 0:
        return -1.0, 1.0
    low = float(np.clip(float(p_low), 0.0, 100.0))
    high = float(np.clip(float(p_high), 0.0, 100.0))
    if low >= high:
        low, high = 0.0, 100.0
    vmin = float(np.percentile(values, low))
    vmax = float(np.percentile(values, high))
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin >= vmax:
        scale = float(np.max(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return -scale, scale
    return vmin, vmax


def _downsample_axis(axis: Any, step: int, expected: int) -> np.ndarray | None:
    if axis is None:
        return None
    array = np.asanyarray(axis, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return None
    sliced = array[:: max(1, int(step))]
    return np.ascontiguousarray(sliced[:expected])


def make_preview_bundle(
    matrix: Any,
    *,
    title: str = "",
    p_low: float = 2.0,
    p_high: float = 98.0,
    trace_axis_m: Any = None,
    sample_axis: Any = None,
    sample_axis_label: str = "",
    trace_elevation_m: Any = None,
    max_samples: int = _MAX_PREVIEW_SAMPLES,
    max_traces: int = _MAX_PREVIEW_TRACES,
) -> PreviewBundle:
    """Build a :class:`PreviewBundle` from an in-memory 2D matrix."""
    array = np.asanyarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={array.shape!r}")
    sample_count, trace_count = int(array.shape[0]), int(array.shape[1])
    row_step, col_step = _strides(sample_count, trace_count, max_samples, max_traces)
    preview = downsample_matrix(array, max_samples=max_samples, max_traces=max_traces)
    vmin, vmax = compute_levels(preview, p_low=p_low, p_high=p_high)
    return PreviewBundle(
        matrix=preview,
        vmin=vmin,
        vmax=vmax,
        sample_count=sample_count,
        trace_count=trace_count,
        title=str(title or ""),
        trace_axis_m=_downsample_axis(trace_axis_m, col_step, preview.shape[1]),
        sample_axis=_downsample_axis(sample_axis, row_step, preview.shape[0]),
        sample_axis_label=str(sample_axis_label or ""),
        trace_elevation_m=_downsample_axis(
            align_trace_vector(trace_elevation_m, trace_count), col_step, preview.shape[1]
        ),
    )


def _dataset_axis(dataset: Any, *names: str) -> np.ndarray | None:
    for name in names:
        axis = getattr(dataset, name, None)
        if axis is None:
            continue
        array = np.asanyarray(axis)
        if array.ndim == 1 and array.size:
            return array
    return None


def _metadata_mapping(dataset: Any) -> dict[str, Any]:
    metadata = getattr(dataset, "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _dataset_elevation(dataset: Any, trace_count: int) -> np.ndarray | None:
    """Resolve per-trace ground elevation (m) from a ``GPRDataSet``-like object.

    取值顺序（均为逐道「地面」高程，非天线海拔）：

    1. 数据集现有属性/字典里的 ``ground_elevation_m``；
    2. 航空 metadata 的 ``trajectory_rows``：``elevation`` 字段本身就是地面
       高程（与该 CSV 第 3 列的语义一致——**不是**天线海拔，见下）。

    ⚠️ 语义已实测确认（`output/probes_archive/_probe_column3_semantics.py`）：
    MyGPR 航空堆叠 CSV 第 3 列是**地面高程**，第 5 列是离地飞行高度。判据是
    交叉测线：Line9 与 Line3 在 43 个交叉点（间距 ≤2 m）上第 3 列仅差
    **0.10 m**，而两条航次的离地高度相差 **3.18 m**——若第 3 列是天线海拔，
    同一地面点上的两条航次必然相差 3.18 m 而不是 0.10 m。
    因此这里**不要**再做 ``elevation - height_m``，那会把全线高程系统性
    压低约 9.5 m（营山 Line9：真值 ~442.7 m vs 误算 ~433.2 m）。

    取不到返回 ``None``，由 UI 据此禁用海拔纵轴。
    """
    direct = _dataset_axis(dataset, "ground_elevation_m", "trace_elevation_m")
    if direct is not None:
        return align_trace_vector(direct, trace_count)
    metadata = _metadata_mapping(dataset)
    series = metadata.get("ground_elevation_m")
    if series is not None:
        return align_trace_vector(series, trace_count)
    rows = metadata.get("trajectory_rows")
    if not rows:
        return None
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("elevation", row.get("elevation_m"))
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        values.append(value)
    if not values:
        return None
    return align_trace_vector(values, trace_count)


def bundle_from_dataset(dataset: Any, **kw: Any) -> PreviewBundle:
    """Build a :class:`PreviewBundle` from a ``GPRDataSet``-like object.

    Uses ``dataset.preview_matrix(max_samples, max_traces)`` so lazy or
    mmap-backed stores only materialise the bounded preview.  Extra
    keyword arguments (``title``/``p_low``/``p_high``/...) are forwarded
    to :func:`make_preview_bundle`.
    """
    max_samples = int(kw.pop("max_samples", _MAX_PREVIEW_SAMPLES))
    max_traces = int(kw.pop("max_traces", _MAX_PREVIEW_TRACES))
    matrix = getattr(dataset, "matrix")
    sample_count = int(getattr(dataset, "sample_count", 0) or np.asanyarray(matrix).shape[0])
    trace_count = int(getattr(dataset, "trace_count", 0) or np.asanyarray(matrix).shape[1])
    row_step, col_step = _strides(sample_count, trace_count, max_samples, max_traces)
    preview_fn = getattr(dataset, "preview_matrix", None)
    if callable(preview_fn):
        preview = np.asarray(preview_fn(max_samples=max_samples, max_traces=max_traces), dtype=np.float32)
    else:
        preview = downsample_matrix(matrix, max_samples=max_samples, max_traces=max_traces)

    trace_axis = _dataset_axis(dataset, "distance_axis_m")
    sample_axis = _dataset_axis(dataset, "time_axis_ns")
    sample_axis_label = "时间 (ns)"
    if sample_axis is None:
        sample_axis = _dataset_axis(dataset, "depth_axis_m")
        sample_axis_label = "深度 (m)" if sample_axis is not None else ""

    title = kw.pop("title", "") or str(getattr(dataset, "line_id", "") or "")
    elevation = kw.pop("trace_elevation_m", None)
    if elevation is None:
        elevation = _dataset_elevation(dataset, trace_count)
    else:
        elevation = align_trace_vector(elevation, trace_count)
    vmin, vmax = compute_levels(
        preview,
        p_low=float(kw.pop("p_low", 2.0)),
        p_high=float(kw.pop("p_high", 98.0)),
    )
    if kw:
        unexpected = ", ".join(sorted(kw))
        raise TypeError(f"unexpected keyword arguments: {unexpected}")
    return PreviewBundle(
        matrix=preview,
        vmin=vmin,
        vmax=vmax,
        sample_count=sample_count,
        trace_count=trace_count,
        title=title,
        trace_axis_m=_downsample_axis(trace_axis, col_step, preview.shape[1]),
        sample_axis=_downsample_axis(sample_axis, row_step, preview.shape[0]),
        sample_axis_label=sample_axis_label if sample_axis is not None else "",
        trace_elevation_m=_downsample_axis(elevation, col_step, preview.shape[1]),
    )


__all__ = [
    "COLORMAPS",
    "DEFAULT_COLORMAP",
    "PreviewBundle",
    "align_trace_vector",
    "bundle_from_dataset",
    "colormap_names",
    "compute_levels",
    "downsample_matrix",
    "make_preview_bundle",
]

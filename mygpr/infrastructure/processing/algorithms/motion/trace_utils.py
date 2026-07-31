"""Trace-aligned geometry helpers used by native motion compensation."""
from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.domain.common.scalars import to_float

EARTH_RADIUS_M = 6378137.0

def _as_1d_array(values: Any, dtype: np.dtype | type) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError("trace/sidecar metadata fields must be 1D arrays")
    return arr


def _trace_count(metadata: dict[str, np.ndarray]) -> int:
    if not metadata:
        raise ValueError("trace_metadata must not be empty")
    first_key = next(iter(metadata))
    count = int(np.asarray(metadata[first_key]).size)
    if count <= 1:
        for values in metadata.values():
            candidate = int(np.asarray(values).size)
            if candidate > 1:
                count = candidate
                break
    if count <= 0:
        raise ValueError("trace_metadata must contain at least one trace")
    for key, values in metadata.items():
        size = int(np.asarray(values).size)
        if size not in {1, count}:
            raise ValueError(f"trace_metadata field '{key}' length mismatch")
    return count


def derive_local_xy_m(
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    origin_longitude: float | None = None,
    origin_latitude: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to a simple local tangent-plane XY in meters."""
    lon = _as_1d_array(longitude, np.float64)
    lat = _as_1d_array(latitude, np.float64)
    if lon.size != lat.size:
        raise ValueError("longitude and latitude must have the same length")
    if lon.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    lon0 = float(lon[0] if origin_longitude is None else origin_longitude)
    lat0 = float(lat[0] if origin_latitude is None else origin_latitude)
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    lon0_rad = np.radians(lon0)
    lat0_rad = np.radians(lat0)

    x_m = (lon_rad - lon0_rad) * np.cos(lat0_rad) * EARTH_RADIUS_M
    y_m = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x_m.astype(np.float32), y_m.astype(np.float32)


def build_uniform_trace_distance_m(
    trace_distance_m: np.ndarray,
    *,
    spacing_m: float | None = None,
) -> np.ndarray:
    """Build an equal-distance trace axis over the current distance span."""
    distance = _as_1d_array(trace_distance_m, np.float64)
    if distance.size == 0:
        raise ValueError("trace_distance_m must contain at least one trace")
    if distance.size == 1:
        return distance.astype(np.float32, copy=True)

    if np.any(np.diff(distance) < 0):
        raise ValueError("trace_distance_m must be monotonically non-decreasing")

    if spacing_m is None:
        deltas = np.diff(distance)
        positive = deltas[deltas > 0]
        spacing = float(np.median(positive)) if positive.size else 1.0
    else:
        spacing = to_float(spacing_m, default=0.0)
    if spacing <= 0:
        raise ValueError("spacing_m must be positive")

    start = float(distance[0])
    end = float(distance[-1])
    steps = max(1, int(round((end - start) / spacing)))
    uniform = start + np.arange(steps + 1, dtype=np.float64) * spacing
    uniform[-1] = end
    return uniform.astype(np.float32, copy=False)


def resample_bscan_columns_linear(
    data: np.ndarray,
    source_distance_m: np.ndarray,
    target_distance_m: np.ndarray,
) -> np.ndarray:
    """Resample B-scan columns along a trace-distance axis with linear interpolation."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D B-scan matrix")

    source_distance = _as_1d_array(source_distance_m, np.float64)
    target_distance = _as_1d_array(target_distance_m, np.float64)
    if source_distance.size != arr.shape[1]:
        raise ValueError("source_distance_m length must match B-scan trace count")
    if source_distance.size == 0:
        raise ValueError("source_distance_m must contain at least one trace")
    source_deltas = np.diff(source_distance)
    if np.any(source_deltas < 0):
        raise ValueError("source_distance_m must be monotonically non-decreasing")
    if target_distance.size == 0:
        return np.empty((arr.shape[0], 0), dtype=np.float32)

    if np.all(source_deltas > 0):
        unique_distance = source_distance
        source_values = arr
    else:
        unique_distance, reverse_index = np.unique(
            source_distance[::-1],
            return_index=True,
        )
        source_indices = (source_distance.size - 1 - reverse_index).astype(np.int64)
        source_values = arr[:, source_indices]
    if unique_distance.size == 1:
        return np.repeat(source_values[:, :1], target_distance.size, axis=1)

    right_idx = np.searchsorted(unique_distance, target_distance, side="left")
    right_idx = np.clip(right_idx, 1, unique_distance.size - 1)
    left_idx = right_idx - 1

    left_x = unique_distance[left_idx]
    right_x = unique_distance[right_idx]
    denom = np.where(right_x > left_x, right_x - left_x, 1.0)
    weight = ((target_distance - left_x) / denom).astype(np.float32)
    weight[target_distance <= unique_distance[0]] = 0.0
    weight[target_distance >= unique_distance[-1]] = 1.0

    result = np.empty((arr.shape[0], target_distance.size), dtype=np.float32)
    np.multiply(
        source_values[:, left_idx],
        (1.0 - weight)[None, :],
        out=result,
        casting="unsafe",
    )
    result += source_values[:, right_idx] * weight[None, :]
    return result


def resample_trace_metadata(
    trace_metadata: dict[str, np.ndarray],
    *,
    target_trace_distance_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Resample per-trace metadata onto a new equal-distance trace axis."""
    trace_count = _trace_count(trace_metadata)
    source_distance = _as_1d_array(trace_metadata.get("trace_distance_m"), np.float64)
    if source_distance.size != trace_count:
        raise ValueError("trace_distance_m length must match metadata trace count")
    if np.any(np.diff(source_distance) < 0):
        raise ValueError("trace_distance_m must be monotonically non-decreasing")

    target_distance = _as_1d_array(target_trace_distance_m, np.float64)
    if target_distance.size == 0:
        raise ValueError("target_trace_distance_m must contain at least one trace")
    if np.any(np.diff(target_distance) < 0):
        raise ValueError("target_trace_distance_m must be monotonically non-decreasing")

    resampled: dict[str, np.ndarray] = {}
    insert_idx = np.searchsorted(source_distance, target_distance, side="left")
    right_idx = np.clip(insert_idx, 0, trace_count - 1)
    left_idx = np.clip(insert_idx - 1, 0, trace_count - 1)
    choose_left = (
        np.abs(target_distance - source_distance[left_idx])
        <= np.abs(source_distance[right_idx] - target_distance)
    )
    nearest_idx = np.where(choose_left, left_idx, right_idx)

    for key, values in trace_metadata.items():
        arr = np.asarray(values)
        if arr.size == 1:
            resampled[key] = arr.copy()
            continue
        if arr.ndim != 1 or arr.size != trace_count:
            raise ValueError(f"trace_metadata field '{key}' must be 1D and length-consistent")

        if key == "trace_index":
            resampled[key] = np.arange(target_distance.size, dtype=np.int32)
            continue
        if key == "trace_distance_m":
            resampled[key] = target_distance.astype(np.float32)
            continue
        if key == "alignment_status":
            resampled[key] = np.full(target_distance.size, "resampled", dtype="<U16")
            continue

        if np.issubdtype(arr.dtype, np.number):
            interp = np.interp(target_distance, source_distance, arr.astype(np.float64))
            if np.issubdtype(arr.dtype, np.integer):
                resampled[key] = np.rint(interp).astype(arr.dtype)
            else:
                resampled[key] = interp.astype(arr.dtype)
        else:
            resampled[key] = arr[nearest_idx].astype(arr.dtype, copy=True)

    if "alignment_status" not in resampled:
        resampled["alignment_status"] = np.full(target_distance.size, "resampled", dtype="<U16")
    return resampled

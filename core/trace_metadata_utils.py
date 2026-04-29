#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for motion-ready per-trace metadata enrichment."""

from __future__ import annotations

from typing import Any

import numpy as np

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


def _normalize_sidecar_records(
    sidecar_records: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "timestamp_s" not in sidecar_records:
        raise ValueError("sidecar_records must include 'timestamp_s'")
    if "longitude" not in sidecar_records or "latitude" not in sidecar_records:
        raise ValueError("sidecar_records must include longitude and latitude")

    timestamp_s = _as_1d_array(sidecar_records["timestamp_s"], np.float64)
    longitude = _as_1d_array(sidecar_records["longitude"], np.float64)
    latitude = _as_1d_array(sidecar_records["latitude"], np.float64)

    if timestamp_s.size == 0:
        raise ValueError("sidecar_records must contain at least one timestamp")
    if longitude.size != timestamp_s.size or latitude.size != timestamp_s.size:
        raise ValueError("sidecar longitude/latitude must match timestamp length")

    order = np.argsort(timestamp_s, kind="stable")
    return timestamp_s[order], longitude[order], latitude[order]


def _normalize_timestamped_payload(
    payload: dict[str, np.ndarray],
    *,
    required_fields: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if "timestamp_s" not in payload:
        raise ValueError("sidecar payload must include 'timestamp_s'")
    timestamp_s = _as_1d_array(payload["timestamp_s"], np.float64)
    if timestamp_s.size == 0:
        raise ValueError("sidecar payload must contain at least one timestamp")

    normalized: dict[str, np.ndarray] = {}
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"sidecar payload must include '{field}'")
        values = _as_1d_array(payload[field], np.float64)
        if values.size != timestamp_s.size:
            raise ValueError(f"sidecar field '{field}' length mismatch")
        normalized[field] = values

    order = np.argsort(timestamp_s, kind="stable")
    return timestamp_s[order], {key: value[order] for key, value in normalized.items()}


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
        spacing = float(spacing_m)
    if spacing <= 0:
        raise ValueError("spacing_m must be positive")

    start = float(distance[0])
    end = float(distance[-1])
    steps = max(1, int(round((end - start) / spacing)))
    uniform = start + np.arange(steps + 1, dtype=np.float64) * spacing
    uniform[-1] = end
    return uniform.astype(np.float32)


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
    nearest_idx = np.searchsorted(source_distance, target_distance, side="left")
    nearest_idx = np.clip(nearest_idx, 0, trace_count - 1)

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


def align_sidecar_records(
    trace_metadata: dict[str, np.ndarray],
    sidecar_records: dict[str, np.ndarray],
    *,
    trace_timestamps_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """Align normalized sidecar records onto per-trace timestamps.

    This is the smallest Phase-1 helper: preserve legacy per-trace metadata,
    add `trace_timestamp_s`, derive local XY from aligned lon/lat, and emit
    a per-trace alignment status array without touching GUI/runtime wiring.
    """
    trace_count = _trace_count(trace_metadata)
    timestamps = _as_1d_array(trace_timestamps_s, np.float64)
    if timestamps.size != trace_count:
        raise ValueError("trace_timestamps_s length must match trace_metadata")

    sidecar_t, sidecar_lon, sidecar_lat = _normalize_sidecar_records(sidecar_records)
    aligned_lon = np.interp(timestamps, sidecar_t, sidecar_lon)
    aligned_lat = np.interp(timestamps, sidecar_t, sidecar_lat)
    local_x_m, local_y_m = derive_local_xy_m(aligned_lon, aligned_lat)

    enriched = {
        key: np.asarray(values).copy() for key, values in trace_metadata.items()
    }
    enriched["trace_timestamp_s"] = timestamps.copy()
    enriched["local_x_m"] = local_x_m
    enriched["local_y_m"] = local_y_m
    enriched["alignment_status"] = np.full(trace_count, "aligned", dtype="<U16")
    return enriched


def integrate_optional_sidecars(
    trace_metadata: dict[str, np.ndarray],
    *,
    trace_timestamps_s: np.ndarray | None = None,
    rtk_payload: dict[str, np.ndarray] | None = None,
    imu_payload: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Merge normalized RTK/IMU payloads into per-trace metadata when provided."""
    _trace_count(trace_metadata)

    if rtk_payload is None and imu_payload is None:
        return {key: np.asarray(values).copy() for key, values in trace_metadata.items()}

    if trace_timestamps_s is None:
        raise ValueError("trace_timestamps_s is required when integrating sidecars")

    timestamps = _as_1d_array(trace_timestamps_s, np.float64)
    enriched = {key: np.asarray(values).copy() for key, values in trace_metadata.items()}

    if rtk_payload is not None:
        enriched = align_sidecar_records(
            enriched,
            rtk_payload,
            trace_timestamps_s=timestamps,
        )
    else:
        enriched["trace_timestamp_s"] = timestamps.copy()

    if imu_payload is not None:
        imu_timestamps, imu_fields = _normalize_timestamped_payload(
            imu_payload,
            required_fields=("roll_deg", "pitch_deg", "yaw_deg"),
        )
        for field, values in imu_fields.items():
            enriched[field] = np.interp(timestamps, imu_timestamps, values).astype(np.float32)

    return enriched

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


def _nearest_indices(source_timestamps_s: np.ndarray, target_timestamps_s: np.ndarray) -> np.ndarray:
    """Return nearest source indices for each target timestamp."""
    insert = np.searchsorted(source_timestamps_s, target_timestamps_s, side="left")
    right = np.clip(insert, 0, source_timestamps_s.size - 1)
    left = np.clip(insert - 1, 0, source_timestamps_s.size - 1)
    choose_left = (
        np.abs(target_timestamps_s - source_timestamps_s[left])
        <= np.abs(source_timestamps_s[right] - target_timestamps_s)
    )
    return np.where(choose_left, left, right)


def _timestamp_coverage_mask(
    source_timestamps_s: np.ndarray,
    target_timestamps_s: np.ndarray,
    *,
    tolerance_s: float = 1.0e-9,
) -> np.ndarray:
    """Return True where target timestamps stay within source coverage."""
    source = _as_1d_array(source_timestamps_s, np.float64)
    target = _as_1d_array(target_timestamps_s, np.float64)
    if source.size == 0:
        raise ValueError("source_timestamps_s must contain at least one timestamp")
    start = float(np.min(source))
    end = float(np.max(source))
    tol = max(float(tolerance_s), 0.0)
    return (target >= start - tol) & (target <= end + tol)


def _ordered_optional_field(
    payload: dict[str, np.ndarray],
    key: str,
    order: np.ndarray,
    expected_size: int,
    dtype: np.dtype | type,
) -> np.ndarray | None:
    if key not in payload:
        return None
    values = _as_1d_array(payload[key], dtype)
    if values.size != expected_size:
        raise ValueError(f"sidecar field '{key}' length mismatch")
    return values[order]


def _xy_trace_distance_m(local_x_m: np.ndarray, local_y_m: np.ndarray) -> np.ndarray:
    x = _as_1d_array(local_x_m, np.float64)
    y = _as_1d_array(local_y_m, np.float64)
    if x.size != y.size:
        raise ValueError("local_x_m and local_y_m must have the same length")
    if x.size == 0:
        return np.array([], dtype=np.float32)
    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate(([0.0], np.cumsum(step))).astype(np.float32)


def _normalize_altimeter_payload(
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    timestamp_s, fields = _normalize_timestamped_payload(
        payload,
        required_fields=("height_agl_m",),
    )
    for key in ("snr", "target_count", "valid"):
        if key not in payload:
            continue
        values = _as_1d_array(payload[key], np.float64)
        if values.size != timestamp_s.size:
            raise ValueError(f"sidecar field '{key}' length mismatch")
        order = np.argsort(_as_1d_array(payload["timestamp_s"], np.float64), kind="stable")
        fields[key] = values[order]
    if "height_source" in payload:
        values = _as_1d_array(payload["height_source"], str)
        if values.size != timestamp_s.size:
            raise ValueError("sidecar field 'height_source' length mismatch")
        order = np.argsort(_as_1d_array(payload["timestamp_s"], np.float64), kind="stable")
        fields["height_source"] = values[order]
    return timestamp_s, fields


def _build_altimeter_confidence(fields: dict[str, np.ndarray], size: int) -> np.ndarray:
    confidence = np.ones(size, dtype=np.float64)
    if "snr" in fields:
        snr = np.asarray(fields["snr"], dtype=np.float64)
        confidence *= np.clip(snr / 20.0, 0.05, 1.0)
    if "target_count" in fields:
        target_count = np.asarray(fields["target_count"], dtype=np.float64)
        confidence *= np.where(target_count > 0, 1.0, 0.25)
    if "valid" in fields:
        valid = np.asarray(fields["valid"], dtype=np.float64)
        confidence *= np.where(valid > 0, 1.0, 0.0)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


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
    if np.any(np.diff(source_distance) < 0):
        raise ValueError("source_distance_m must be monotonically non-decreasing")
    if target_distance.size == 0:
        return np.empty((arr.shape[0], 0), dtype=np.float32)

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
    weight = np.where(target_distance <= unique_distance[0], 0.0, weight)
    weight = np.where(target_distance >= unique_distance[-1], 1.0, weight)

    return (
        source_values[:, left_idx] * (1.0 - weight)[None, :]
        + source_values[:, right_idx] * weight[None, :]
    ).astype(np.float32, copy=False)


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
    raw_sidecar_t = _as_1d_array(sidecar_records["timestamp_s"], np.float64)
    sidecar_order = np.argsort(raw_sidecar_t, kind="stable")
    aligned_lon = np.interp(timestamps, sidecar_t, sidecar_lon)
    aligned_lat = np.interp(timestamps, sidecar_t, sidecar_lat)
    sidecar_local_x = _ordered_optional_field(
        sidecar_records, "local_x_m", sidecar_order, sidecar_t.size, np.float64
    )
    sidecar_local_y = _ordered_optional_field(
        sidecar_records, "local_y_m", sidecar_order, sidecar_t.size, np.float64
    )
    explicit_local_xy = sidecar_local_x is not None and sidecar_local_y is not None
    if explicit_local_xy:
        local_x_m = np.interp(timestamps, sidecar_t, sidecar_local_x).astype(np.float32)
        local_y_m = np.interp(timestamps, sidecar_t, sidecar_local_y).astype(np.float32)
    else:
        local_x_m, local_y_m = derive_local_xy_m(aligned_lon, aligned_lat)
    coverage_mask = _timestamp_coverage_mask(sidecar_t, timestamps)

    enriched = {
        key: np.asarray(values).copy() for key, values in trace_metadata.items()
    }
    enriched["trace_timestamp_s"] = timestamps.copy()
    enriched["local_x_m"] = local_x_m
    enriched["local_y_m"] = local_y_m
    if explicit_local_xy or "trace_distance_m" not in enriched:
        enriched["trace_distance_m"] = _xy_trace_distance_m(local_x_m, local_y_m)
    sidecar_local_z = _ordered_optional_field(
        sidecar_records, "local_z_m", sidecar_order, sidecar_t.size, np.float64
    )
    if sidecar_local_z is not None:
        enriched["local_z_m"] = np.interp(
            timestamps, sidecar_t, sidecar_local_z
        ).astype(np.float32)
    enriched["alignment_status"] = np.where(
        coverage_mask, "aligned", "extrapolated"
    ).astype("<U16")
    return enriched


def integrate_optional_sidecars(
    trace_metadata: dict[str, np.ndarray],
    *,
    trace_timestamps_s: np.ndarray | None = None,
    rtk_payload: dict[str, np.ndarray] | None = None,
    imu_payload: dict[str, np.ndarray] | None = None,
    altimeter_payload: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Merge normalized RTK/IMU payloads into per-trace metadata when provided."""
    _trace_count(trace_metadata)

    if rtk_payload is None and imu_payload is None and altimeter_payload is None:
        return {key: np.asarray(values).copy() for key, values in trace_metadata.items()}

    if trace_timestamps_s is None:
        raise ValueError("trace_timestamps_s is required when integrating sidecars")

    timestamps = _as_1d_array(trace_timestamps_s, np.float64)
    enriched = {key: np.asarray(values).copy() for key, values in trace_metadata.items()}
    alignment_status = np.full(timestamps.size, "aligned", dtype="<U16")

    if rtk_payload is not None:
        enriched = align_sidecar_records(
            enriched,
            rtk_payload,
            trace_timestamps_s=timestamps,
        )
        rtk_timestamps, _, _ = _normalize_sidecar_records(rtk_payload)
        alignment_status = np.where(
            _timestamp_coverage_mask(rtk_timestamps, timestamps),
            alignment_status,
            "extrapolated",
        )
    else:
        enriched["trace_timestamp_s"] = timestamps.copy()

    if imu_payload is not None:
        imu_timestamps, imu_fields = _normalize_timestamped_payload(
            imu_payload,
            required_fields=("roll_deg", "pitch_deg", "yaw_deg"),
        )
        alignment_status = np.where(
            _timestamp_coverage_mask(imu_timestamps, timestamps),
            alignment_status,
            "extrapolated",
        )
        for field, values in imu_fields.items():
            enriched[field] = np.interp(timestamps, imu_timestamps, values).astype(np.float32)

    if altimeter_payload is not None:
        altimeter_timestamps, altimeter_fields = _normalize_altimeter_payload(
            altimeter_payload
        )
        alignment_status = np.where(
            _timestamp_coverage_mask(altimeter_timestamps, timestamps),
            alignment_status,
            "extrapolated",
        )
        height_agl_m = np.interp(
            timestamps,
            altimeter_timestamps,
            altimeter_fields["height_agl_m"],
        ).astype(np.float32)
        enriched["height_agl_m"] = height_agl_m
        enriched["height_confidence"] = _build_altimeter_confidence(
            {
                key: np.interp(timestamps, altimeter_timestamps, value)
                for key, value in altimeter_fields.items()
                if key in {"snr", "target_count", "valid"}
            },
            timestamps.size,
        )
        if "height_source" in altimeter_fields:
            nearest = _nearest_indices(altimeter_timestamps, timestamps)
            source = np.asarray(altimeter_fields["height_source"])[nearest]
            source = np.where(source == "", "altimeter", source)
            enriched["height_source"] = source.astype("<U32")
        else:
            enriched["height_source"] = np.full(timestamps.size, "altimeter", dtype="<U32")

    if rtk_payload is not None or imu_payload is not None or altimeter_payload is not None:
        enriched["alignment_status"] = alignment_status

    return enriched

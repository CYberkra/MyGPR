#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR 三维地理参考预览与导出工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.trace_metadata_utils import derive_local_xy_m

AIR_LIGHT_SPEED_M_PER_NS = 0.299792458
AIR_TWO_WAY_DEPTH_SCALE_M_PER_NS = AIR_LIGHT_SPEED_M_PER_NS / 2.0
DEFAULT_MAX_PREVIEW_TRACES = 240
DEFAULT_MAX_PREVIEW_SAMPLES = 160


def _as_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("trace metadata fields must be one-dimensional")
    return arr


def _as_int_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("trace metadata fields must be one-dimensional")
    return arr


def _first_existing_1d(metadata: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in metadata:
            arr = np.asarray(metadata[key])
            if arr.ndim == 1 and arr.size > 0:
                return arr
    return None


def _coerce_length(values: np.ndarray | None, target_len: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0:
        return None
    if arr.size == target_len:
        return arr.copy()
    if arr.size == 1:
        return np.repeat(arr, target_len)
    if arr.size > target_len:
        return arr[:target_len].copy()

    # Best-effort interpolation for partial metadata.
    source_axis = np.linspace(0.0, 1.0, arr.size, dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    if np.issubdtype(arr.dtype, np.number):
        interp = np.interp(target_axis, source_axis, arr.astype(np.float64))
        if np.issubdtype(arr.dtype, np.integer):
            return np.rint(interp).astype(arr.dtype)
        return interp.astype(arr.dtype)
    index = np.linspace(0, arr.size - 1, target_len).round().astype(np.int64)
    return arr[index]


def _safe_numeric(values: np.ndarray | None, target_len: int, fill_value: float = 0.0) -> np.ndarray:
    arr = _coerce_length(values, target_len)
    if arr is None:
        return np.full(target_len, fill_value, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size != target_len:
        arr = np.resize(arr, target_len).astype(np.float64)
    return np.where(np.isfinite(arr), arr, fill_value).astype(np.float64)


def _dominant_trace_axis(
    trace_metadata: dict[str, Any] | None,
    trace_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return local X/Y and along-track distance with best-effort fallbacks."""
    quality_flags: list[str] = []
    metadata = trace_metadata or {}

    trace_distance = _first_existing_1d(metadata, ("trace_distance_m",))
    trace_index = _first_existing_1d(metadata, ("trace_index",))
    longitude = _first_existing_1d(metadata, ("longitude",))
    latitude = _first_existing_1d(metadata, ("latitude",))
    local_x = _first_existing_1d(metadata, ("local_x_m",))
    local_y = _first_existing_1d(metadata, ("local_y_m",))

    if local_x is not None and local_y is not None:
        x = _safe_numeric(local_x, trace_count)
        y = _safe_numeric(local_y, trace_count)
    elif longitude is not None and latitude is not None:
        n = min(longitude.size, latitude.size)
        lon = np.asarray(longitude[:n], dtype=np.float64)
        lat = np.asarray(latitude[:n], dtype=np.float64)
        x, y = derive_local_xy_m(lon, lat)
        x = _safe_numeric(x, trace_count)
        y = _safe_numeric(y, trace_count)
        quality_flags.append("derived_local_xy_from_lon_lat")
    elif trace_distance is not None:
        distance = _safe_numeric(trace_distance, trace_count)
        x = distance
        y = np.zeros(trace_count, dtype=np.float64)
        quality_flags.append("fallback_trace_distance_axis")
    else:
        x = np.arange(trace_count, dtype=np.float64)
        y = np.zeros(trace_count, dtype=np.float64)
        quality_flags.append("fallback_trace_index_axis")

    if trace_distance is not None:
        distance = _safe_numeric(trace_distance, trace_count)
    else:
        step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        distance = np.concatenate(([0.0], np.cumsum(step))).astype(np.float64)
        quality_flags.append("derived_trace_distance_from_xy")

    if trace_index is None:
        quality_flags.append("missing_trace_index")

    return x, y, distance, quality_flags


def _resolve_height_profiles(
    metadata: dict[str, Any] | None,
    trace_count: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]:
    """Return flight/AGL/ground profiles with fallbacks."""
    quality_flags: list[str] = []
    meta = metadata or {}
    ground = _first_existing_1d(meta, ("ground_elevation_m",))
    flight = _first_existing_1d(meta, ("height_agl_m", "flight_height_m"))
    agl = _first_existing_1d(meta, ("height_agl_m",))

    ground_arr = _safe_numeric(ground, trace_count, fill_value=np.nan) if ground is not None else None
    flight_arr = _safe_numeric(flight, trace_count, fill_value=np.nan) if flight is not None else None
    agl_arr = _safe_numeric(agl, trace_count, fill_value=np.nan) if agl is not None else None

    if ground_arr is None:
        quality_flags.append("missing_ground_elevation")
    if flight_arr is None:
        quality_flags.append("missing_flight_height")
    if agl_arr is None:
        quality_flags.append("missing_height_agl")

    if agl_arr is None and flight_arr is not None:
        agl_arr = flight_arr.copy()
        quality_flags.append("using_flight_height_as_agl")

    if agl_arr is None:
        agl_arr = np.zeros(trace_count, dtype=np.float64)

    if ground_arr is not None:
        finite_ground = ground_arr[np.isfinite(ground_arr)]
        ground_fill = float(np.nanmedian(finite_ground)) if finite_ground.size else 0.0
        ground_arr = np.where(np.isfinite(ground_arr), ground_arr, ground_fill)

    if agl_arr is not None:
        finite_agl = agl_arr[np.isfinite(agl_arr)]
        agl_fill = float(np.nanmedian(finite_agl)) if finite_agl.size else 0.0
        agl_arr = np.where(np.isfinite(agl_arr), agl_arr, agl_fill)

    if ground_arr is None:
        airborne_z = agl_arr.copy()
        quality_flags.append("no_ground_reference_for_absolute_z")
    else:
        airborne_z = np.where(
            np.isfinite(ground_arr) & np.isfinite(agl_arr),
            ground_arr + agl_arr,
            np.where(np.isfinite(ground_arr), ground_arr, agl_arr),
        ).astype(np.float64)

    return airborne_z, ground_arr, agl_arr, quality_flags


def _resolve_time_axis(
    data: np.ndarray,
    header_info: dict[str, Any] | None,
    trace_metadata: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    meta = trace_metadata or {}
    quality_flags: list[str] = []

    sample_count = int(data.shape[0]) if data.ndim == 2 else int(np.asarray(data).shape[0])
    total_time_ns = None
    if header_info and header_info.get("total_time_ns") is not None:
        try:
            total_time_ns = float(header_info.get("total_time_ns"))
        except Exception:
            total_time_ns = None
    if (total_time_ns is None or total_time_ns <= 0.0) and "time_window_ns" in meta:
        try:
            total_time_ns = float(np.asarray(meta["time_window_ns"]).reshape(-1)[0])
        except Exception:
            total_time_ns = None
    if total_time_ns is None or total_time_ns <= 0.0:
        total_time_ns = float(max(sample_count - 1, 1))
        quality_flags.append("missing_total_time_ns")
        time_axis_ns = np.arange(sample_count, dtype=np.float64)
        depth_axis_m = np.arange(sample_count, dtype=np.float64)
    else:
        time_axis_ns = np.linspace(0.0, total_time_ns, sample_count, dtype=np.float64)
        depth_axis_m = time_axis_ns * AIR_TWO_WAY_DEPTH_SCALE_M_PER_NS
    return time_axis_ns, depth_axis_m, float(total_time_ns), quality_flags


def _preview_indices(length: int, limit: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.int32)
    if length <= limit:
        return np.arange(length, dtype=np.int32)
    step = int(np.ceil(length / limit))
    idx = np.arange(0, length, step, dtype=np.int32)
    if idx.size == 0 or idx[-1] != length - 1:
        idx = np.unique(np.append(idx, length - 1)).astype(np.int32)
    return idx


def build_airborne_georeference_3d_payload(
    data: np.ndarray,
    header_info: dict[str, Any] | None,
    trace_metadata: dict[str, Any] | None,
    *,
    selected_trace_index: int | None = None,
    max_preview_traces: int = DEFAULT_MAX_PREVIEW_TRACES,
    max_preview_samples: int = DEFAULT_MAX_PREVIEW_SAMPLES,
) -> dict[str, Any] | None:
    """Build a 3D georeference payload for preview and export."""
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.size == 0:
        return None

    trace_count = int(arr.shape[1])
    trace_count = max(trace_count, 1)

    x_m, y_m, trace_distance_m, axis_flags = _dominant_trace_axis(trace_metadata, trace_count)
    airborne_z_m, ground_elevation_m, height_agl_m, height_flags = _resolve_height_profiles(
        trace_metadata, trace_count
    )
    time_axis_ns, depth_axis_m, total_time_ns, time_flags = _resolve_time_axis(
        arr, header_info, trace_metadata
    )

    trace_index = _first_existing_1d(trace_metadata or {}, ("trace_index",))
    if trace_index is None or trace_index.size == 0:
        trace_index = np.arange(trace_count, dtype=np.int32)
    else:
        trace_index = _coerce_length(trace_index, trace_count)
        trace_index = np.asarray(trace_index, dtype=np.int32)

    longitude = _coerce_length(_first_existing_1d(trace_metadata or {}, ("longitude",)), trace_count)
    latitude = _coerce_length(_first_existing_1d(trace_metadata or {}, ("latitude",)), trace_count)

    trace_count = min(trace_count, arr.shape[1], x_m.size, y_m.size, trace_distance_m.size, airborne_z_m.size, trace_index.size)
    x_m = np.asarray(x_m[:trace_count], dtype=np.float64)
    y_m = np.asarray(y_m[:trace_count], dtype=np.float64)
    trace_distance_m = np.asarray(trace_distance_m[:trace_count], dtype=np.float64)
    airborne_z_m = np.asarray(airborne_z_m[:trace_count], dtype=np.float64)
    trace_index = np.asarray(trace_index[:trace_count], dtype=np.int32)
    if ground_elevation_m is not None:
        ground_elevation_m = np.asarray(ground_elevation_m[:trace_count], dtype=np.float64)
    if height_agl_m is not None:
        height_agl_m = np.asarray(height_agl_m[:trace_count], dtype=np.float64)
    if longitude is not None:
        longitude = np.asarray(longitude[:trace_count], dtype=np.float64)
    if latitude is not None:
        latitude = np.asarray(latitude[:trace_count], dtype=np.float64)

    sample_count = int(arr.shape[0])
    preview_trace_idx = _preview_indices(trace_count, max_preview_traces)
    preview_sample_idx = _preview_indices(sample_count, max_preview_samples)
    preview_data = np.asarray(arr[np.ix_(preview_sample_idx, preview_trace_idx)], dtype=np.float64)

    preview_x = x_m[preview_trace_idx]
    preview_y = y_m[preview_trace_idx]
    preview_z_top = airborne_z_m[preview_trace_idx]
    preview_depth = depth_axis_m[preview_sample_idx]
    curtain_x = np.repeat(preview_x[np.newaxis, :], preview_sample_idx.size, axis=0)
    curtain_y = np.repeat(preview_y[np.newaxis, :], preview_sample_idx.size, axis=0)
    curtain_z = preview_z_top[np.newaxis, :] - preview_depth[:, np.newaxis]

    finite_preview = np.isfinite(preview_data)
    if np.any(finite_preview):
        amp_min = float(np.nanmin(preview_data[finite_preview]))
        amp_max = float(np.nanmax(preview_data[finite_preview]))
    else:
        amp_min = amp_max = 0.0

    quality_flags = sorted(
        set(axis_flags + height_flags + time_flags + (["downsampled_preview"] if (preview_trace_idx.size < trace_count or preview_sample_idx.size < sample_count) else []))
    )

    payload = {
        "schema_version": 1,
        "source_kind": "uav_gpr_georeference_3d",
        "trace_count": int(trace_count),
        "sample_count": int(sample_count),
        "selected_trace_index": int(selected_trace_index) if selected_trace_index is not None else None,
        "trace_index": trace_index,
        "trace_distance_m": trace_distance_m,
        "longitude": longitude,
        "latitude": latitude,
        "local_x_m": x_m,
        "local_y_m": y_m,
        "ground_elevation_m": ground_elevation_m,
        "height_agl_m": height_agl_m,
        "airborne_z_m": airborne_z_m,
        "time_axis_ns": time_axis_ns,
        "depth_axis_m": depth_axis_m,
        "total_time_ns": float(total_time_ns),
        "preview": {
            "trace_indices": preview_trace_idx,
            "sample_indices": preview_sample_idx,
            "x_m": preview_x,
            "y_m": preview_y,
            "z_top_m": preview_z_top,
            "curtain_x_m": curtain_x,
            "curtain_y_m": curtain_y,
            "curtain_z_m": curtain_z,
            "amplitude": preview_data,
            "amplitude_min": amp_min,
            "amplitude_max": amp_max,
            "depth_axis_m": preview_depth,
            "trace_stride": int(preview_trace_idx[1] - preview_trace_idx[0]) if preview_trace_idx.size > 1 else 1,
            "sample_stride": int(preview_sample_idx[1] - preview_sample_idx[0]) if preview_sample_idx.size > 1 else 1,
        },
        "quality_flags": quality_flags,
        "has_ground_elevation": ground_elevation_m is not None,
        "has_height_agl": height_agl_m is not None,
        "has_longitude_latitude": longitude is not None and latitude is not None,
        "x_axis_label": "局部 X (m)" if longitude is not None else "距离 (m)",
        "y_axis_label": "局部 Y (m)" if longitude is not None else "纬度方向 (m)",
        "z_axis_label": "等效高度/深度 (m)",
        "depth_model": "time_ns_to_equivalent_air_distance_m",
        "depth_scale_m_per_ns": AIR_TWO_WAY_DEPTH_SCALE_M_PER_NS,
        "amp_min": amp_min,
        "amp_max": amp_max,
    }
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_vtk_structured_grid(
    payload: dict[str, Any],
    vtk_path: Path,
) -> Path:
    preview = payload["preview"]
    curtain_x = np.asarray(preview["curtain_x_m"], dtype=np.float64)
    curtain_y = np.asarray(preview["curtain_y_m"], dtype=np.float64)
    curtain_z = np.asarray(preview["curtain_z_m"], dtype=np.float64)
    amplitude = np.asarray(preview["amplitude"], dtype=np.float64)

    sample_count, trace_count = curtain_x.shape
    vtk_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vtk_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MyGPR UAV-GPR 3D georeference preview\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {trace_count} {sample_count} 1\n")
        f.write(f"POINTS {trace_count * sample_count} float\n")
        points = np.column_stack(
            (
                curtain_x.reshape(-1),
                curtain_y.reshape(-1),
                curtain_z.reshape(-1),
            )
        )
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")
        f.write(f"POINT_DATA {trace_count * sample_count}\n")
        f.write("SCALARS amplitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, amplitude.reshape(-1), fmt="%.6f")
    return vtk_path


def _write_trace_summary_csv(payload: dict[str, Any], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    trace_index = np.asarray(payload["trace_index"], dtype=np.int32)
    trace_distance = np.asarray(payload["trace_distance_m"], dtype=np.float64)
    longitude = payload.get("longitude")
    latitude = payload.get("latitude")
    x_m = np.asarray(payload["local_x_m"], dtype=np.float64)
    y_m = np.asarray(payload["local_y_m"], dtype=np.float64)
    ground = payload.get("ground_elevation_m")
    agl = payload.get("height_agl_m")
    airborne_z = np.asarray(payload["airborne_z_m"], dtype=np.float64)
    selected_trace = payload.get("selected_trace_index")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_index",
                "trace_distance_m",
                "longitude",
                "latitude",
                "local_x_m",
                "local_y_m",
                "ground_elevation_m",
                "height_agl_m",
                "airborne_z_m",
                "selected_trace_index",
            ],
        )
        writer.writeheader()
        for idx in range(trace_index.size):
            writer.writerow(
                {
                    "trace_index": int(trace_index[idx]),
                    "trace_distance_m": float(trace_distance[idx]),
                    "longitude": float(longitude[idx]) if isinstance(longitude, np.ndarray) else "",
                    "latitude": float(latitude[idx]) if isinstance(latitude, np.ndarray) else "",
                    "local_x_m": float(x_m[idx]),
                    "local_y_m": float(y_m[idx]),
                    "ground_elevation_m": float(ground[idx]) if isinstance(ground, np.ndarray) else "",
                    "height_agl_m": float(agl[idx]) if isinstance(agl, np.ndarray) else "",
                    "airborne_z_m": float(airborne_z[idx]),
                    "selected_trace_index": selected_trace if selected_trace is not None else "",
                }
            )
    return csv_path


def export_airborne_georeference_3d_bundle(
    payload: dict[str, Any],
    out_base_path: str | Path,
) -> dict[str, Any]:
    """Export a 3D georeference preview as VTK + CSV + JSON."""
    base = Path(out_base_path)
    if base.suffix.lower() == ".vtk":
        stem = base.with_suffix("")
        vtk_path = base
    else:
        stem = base
        vtk_path = base.with_suffix(".vtk")
    base.parent.mkdir(parents=True, exist_ok=True)

    csv_path = stem.with_suffix(".csv")
    json_path = stem.with_suffix(".json")

    _write_vtk_structured_grid(payload, vtk_path)
    _write_trace_summary_csv(payload, csv_path)

    summary = {
        "schema_version": int(payload.get("schema_version", 1)),
        "source_kind": payload.get("source_kind", "uav_gpr_georeference_3d"),
        "trace_count": int(payload.get("trace_count", 0)),
        "sample_count": int(payload.get("sample_count", 0)),
        "selected_trace_index": payload.get("selected_trace_index"),
        "total_time_ns": float(payload.get("total_time_ns", 0.0)),
        "depth_model": payload.get("depth_model"),
        "depth_scale_m_per_ns": float(payload.get("depth_scale_m_per_ns", AIR_TWO_WAY_DEPTH_SCALE_M_PER_NS)),
        "quality_flags": list(payload.get("quality_flags") or []),
        "axis_labels": {
            "x": payload.get("x_axis_label"),
            "y": payload.get("y_axis_label"),
            "z": payload.get("z_axis_label"),
        },
        "preview": {
            "trace_stride": int(payload.get("preview", {}).get("trace_stride", 1)),
            "sample_stride": int(payload.get("preview", {}).get("sample_stride", 1)),
            "shape": [
                int(np.asarray(payload.get("preview", {}).get("amplitude", np.empty((0, 0)))).shape[0]),
                int(np.asarray(payload.get("preview", {}).get("amplitude", np.empty((0, 0)))).shape[1]),
            ],
            "amplitude_min": float(payload.get("preview", {}).get("amplitude_min", 0.0)),
            "amplitude_max": float(payload.get("preview", {}).get("amplitude_max", 0.0)),
        },
        "trace_summary_csv": str(csv_path),
        "vtk_path": str(vtk_path),
    }
    json_path.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "vtk_path": str(vtk_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "json_path": str(json_path.resolve()),
        "summary": summary,
    }

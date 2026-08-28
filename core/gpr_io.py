#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enhanced GPR I/O module.

Supports:
- Standard CSV/TXT B-scan files (with/without header)
- Folder of A-scan CSV files
- Lightweight native subsets: MALA RD3/RD7, ImpulseRadar IPRB, fixed SEG-Y, ENVI BSQ, NPY/NPZ
- Recognized vendor formats with explicit conversion guidance: GSSI DZT, Sensors & Software DT1/HD, OKO GPR/GPR2

Author: MyGPR Team
Date: 2026-03-31
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np


logger = logging.getLogger(__name__)

# 直接导入 read_file_data 模块
from PythonModule.read_file_data import readcsv, savecsv, save_image, show_image
from mygpr.domain.autotune.data_context import (
    DATA_CONTEXT_GPRMAX,
    DATA_CONTEXT_GPRMAX_IMPULSE,
    DATA_CONTEXT_UAV_GPR_SFCW_FIELD,
    apply_data_context_defaults,
)
from mygpr.domain.common.scalars import to_float, to_float_or_none, to_int

from core.gpr_format_registry import get_format_spec
from core.gpr_vendor_readers import (
    GPRFormatReadError,
    read_envi_bsq,
    read_impulseradar_iprb,
    read_mala_rd,
    read_numpy_profile,
    read_segy_fixed,
    unsupported_known_format_message,
)


# ============ Auto-detect and Load ============


def auto_load_data(path: str, **kwargs) -> Dict[str, Any]:
    """Auto-detect file type and load GPR data.

    The return payload is normalized to include ``data`` for profile-like files.
    Known but not natively decoded vendor formats raise a clear error instead
    of being mis-read as generic binary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_dir():
        return read_ascans_folder(str(path), **kwargs)

    suffix = path.suffix.lower()
    spec = get_format_spec(path)

    if suffix in {".csv", ".txt"}:
        data = readcsv(str(path))
        return {
            "data": data,
            "header_info": {
                "a_scan_length": int(data.shape[0]),
                "num_traces": int(data.shape[1]) if data.ndim >= 2 else 1,
                "total_time_ns": 0.0,
                "trace_interval_m": 0.0,
                "source": "matrix_text",
                "path": str(path),
            },
            "type": "bscan_text",
            "source": str(path),
            "path": str(path),
        }
    if suffix in {".npy", ".npz"}:
        return read_numpy_profile(str(path))
    if suffix in {".rd3", ".rd7", ".rad"}:
        return read_mala_rd(str(path))
    if suffix in {".iprb", ".iprh"}:
        return read_impulseradar_iprb(str(path))
    if suffix in {".sgy", ".segy"}:
        return read_segy_fixed(str(path))
    if suffix in {".dat", ".hdr"}:
        return read_envi_bsq(str(path))
    if spec is not None and spec.support == "recognized":
        raise GPRFormatReadError(
            unsupported_known_format_message(str(path), spec.display_name, spec.notes)
        )
    raise ValueError(f"Unsupported file type: {suffix}")


# ============ 文件夹 A-scan 数据加载 ============


def extract_airborne_csv_payload(
    raw_data: np.ndarray,
    header_info: dict[str, Any] | None,
    *,
    trace_timestamps_s: np.ndarray | None = None,
    rtk_path: str | Path | None = None,
    imu_path: str | Path | None = None,
    altimeter_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None, dict[str, Any] | None]:
    """Extract amplitude matrix and per-trace airborne metadata from imported CSV.

    Supported primary airborne format:
    - first 4 lines header
    - then rows of: longitude, latitude, ground elevation, amplitude, flight height
    - optional sixth column: explicit trace timestamp in seconds
    - rows are stacked trace-by-trace, each trace containing `samples` rows
    """
    arr = np.asarray(raw_data)
    if arr.size == 0:
        raise ValueError("CSV 未读取到有效数据")

    metadata = None
    updated_header = dict(header_info or {}) if header_info else None

    if header_info:
        samples = int(header_info["a_scan_length"])
        traces = int(header_info["num_traces"])
        required_rows = samples * traces

        if arr.shape[1] >= 4 and arr.shape[0] >= required_rows:
            use_rows = arr[:required_rows, :]

            # Typical airborne stacked format: [lon, lat, ground_z, amplitude, flight_h]
            amp_col = 3
            signal_1d = use_rows[:, amp_col].astype(np.float32, copy=False)
            data = signal_1d.reshape((traces, samples)).T

            if use_rows.shape[1] >= 5:
                metadata = _extract_trace_metadata_from_stacked_rows(
                    use_rows, samples, traces
                )
                if trace_timestamps_s is None and "trace_timestamp_s" in metadata:
                    trace_timestamps_s = metadata["trace_timestamp_s"]
                if header_info and "total_time_ns" in header_info:
                    metadata["time_window_ns"] = to_float(
                        header_info["total_time_ns"],
                        default=0.0,
                    )
                metadata = _integrate_optional_airborne_sidecars(
                    metadata,
                    trace_timestamps_s=trace_timestamps_s,
                    rtk_path=rtk_path,
                    imu_path=imu_path,
                    altimeter_path=altimeter_path,
                )
                if updated_header is None:
                    updated_header = {}
                updated_header.update(_build_airborne_header_summary(metadata))
                updated_header = apply_data_context_defaults(
                    updated_header,
                    trace_metadata=metadata,
                    context=DATA_CONTEXT_UAV_GPR_SFCW_FIELD,
                )
                return data, metadata, updated_header

            return data, metadata, updated_header

        if (
            arr.ndim == 2
            and arr.shape[0] > 0
            and arr.shape[1] > 0
            and (rtk_path is not None or imu_path is not None or altimeter_path is not None)
        ):
            data = arr.astype(np.float32, copy=False)
            trace_count = int(data.shape[1])
            if trace_timestamps_s is None:
                raise ValueError(
                    "trace_timestamps_s is required when integrating sidecars with matrix CSV data"
                )
            timestamps = np.asarray(trace_timestamps_s, dtype=np.float64).reshape(-1)
            if timestamps.size != trace_count:
                raise ValueError(
                    "trace_timestamps_s length must match matrix CSV trace count "
                    f"({timestamps.size} != {trace_count})"
                )
            if not np.isfinite(timestamps).all():
                raise ValueError("trace_timestamps_s must contain only finite values")
            metadata = {
                "trace_index": np.arange(trace_count, dtype=np.int32),
                "trace_timestamp_s": timestamps.copy(),
                "trace_distance_m": np.arange(trace_count, dtype=np.float32),
            }
            metadata = _integrate_optional_airborne_sidecars(
                metadata,
                trace_timestamps_s=timestamps,
                rtk_path=rtk_path,
                imu_path=imu_path,
                altimeter_path=altimeter_path,
            )
            if updated_header is None:
                updated_header = {}
            updated_header.setdefault("a_scan_length", int(data.shape[0]))
            updated_header.setdefault("num_traces", int(trace_count))
            updated_header.update(_build_airborne_header_summary(metadata))
            updated_header["source"] = "matrix_csv_with_sidecars"
            updated_header = apply_data_context_defaults(
                updated_header,
                trace_metadata=metadata,
                context=DATA_CONTEXT_UAV_GPR_SFCW_FIELD,
            )
            return data, metadata, updated_header

        if arr.shape[0] == traces and arr.shape[1] >= samples:
            data = arr[:, :samples].T.astype(np.float32, copy=False)
            return data, metadata, updated_header

        if arr.shape[0] >= samples and arr.shape[1] >= traces:
            data = arr[:samples, :traces].astype(np.float32, copy=False)
            return data, metadata, updated_header

    data = arr.astype(np.float32, copy=False)
    return data, metadata, updated_header


def _integrate_optional_airborne_sidecars(
    metadata: dict[str, np.ndarray] | None,
    *,
    trace_timestamps_s: np.ndarray | None,
    rtk_path: str | Path | None,
    imu_path: str | Path | None,
    altimeter_path: str | Path | None,
) -> dict[str, np.ndarray] | None:
    """Optionally merge parsed RTK/IMU/altimeter sidecars into airborne trace metadata."""
    if rtk_path is None and imu_path is None and altimeter_path is None:
        return metadata
    if metadata is None:
        raise ValueError("optional sidecar integration requires airborne trace metadata")

    integration_module = importlib.import_module("core.sidecar_integration")
    return integration_module.load_and_integrate_optional_sidecars(
        metadata,
        trace_timestamps_s=trace_timestamps_s,
        rtk_path=rtk_path,
        imu_path=imu_path,
        altimeter_path=altimeter_path,
    )


def subset_trace_metadata(
    metadata: dict[str, np.ndarray] | None, trace_indices: np.ndarray | slice | None
) -> dict[str, np.ndarray] | None:
    """Subset per-trace metadata using trace indices."""
    if metadata is None or trace_indices is None:
        return metadata
    subset = {}
    for key, values in metadata.items():
        arr = np.asarray(values)
        if arr.ndim == 0:
            subset[key] = np.array(arr, copy=True)
            continue
        try:
            subset[key] = np.asarray(arr[trace_indices]).copy()
        except (IndexError, TypeError, ValueError):
            subset[key] = np.array(arr, copy=True)
    return subset


def compute_trace_distance_m(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Compute cumulative along-track distance from lon/lat in meters."""
    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    n = min(lon.size, lat.size)
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([0.0], dtype=np.float32)

    lon_rad = np.radians(lon[:n])
    lat_rad = np.radians(lat[:n])
    dlon = np.diff(lon_rad)
    dlat = np.diff(lat_rad)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    distances = 6371000.0 * c
    cumulative = np.empty(n, dtype=np.float32)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(distances, dtype=np.float64).astype(np.float32, copy=False)
    return cumulative


def _extract_trace_metadata_from_stacked_rows(
    rows: np.ndarray, samples: int, traces: int
) -> dict[str, np.ndarray]:
    trace_rows = rows.reshape((traces, samples, rows.shape[1]))[:, 0, :]
    longitude = trace_rows[:, 0].astype(np.float64, copy=False)
    latitude = trace_rows[:, 1].astype(np.float64, copy=False)
    ground_elevation_m = trace_rows[:, 2].astype(np.float32, copy=False)
    flight_height_m = trace_rows[:, 4].astype(np.float32, copy=False)
    distance_m = compute_trace_distance_m(longitude, latitude)
    metadata = {
        "trace_index": np.arange(traces, dtype=np.int32),
        "longitude": longitude.astype(np.float64, copy=False),
        "latitude": latitude.astype(np.float64, copy=False),
        "ground_elevation_m": ground_elevation_m,
        "flight_height_m": flight_height_m,
        "trace_distance_m": distance_m,
    }
    if rows.shape[1] >= 6:
        metadata["trace_timestamp_s"] = trace_rows[:, 5].astype(np.float64, copy=False)
    return metadata


def _build_airborne_header_summary(metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    distance = np.asarray(metadata.get("trace_distance_m", []), dtype=np.float64)
    ground = np.asarray(metadata.get("ground_elevation_m", []), dtype=np.float64)
    flight = np.asarray(metadata.get("flight_height_m", []), dtype=np.float64)
    height_agl = np.asarray(metadata.get("height_agl_m", []), dtype=np.float64)
    timestamps = np.asarray(metadata.get("trace_timestamp_s", []), dtype=np.float64)
    height_confidence = np.asarray(metadata.get("height_confidence", []), dtype=np.float64)
    alignment_status = np.asarray(metadata.get("alignment_status", []), dtype="<U16")

    if distance.size > 1:
        trace_steps = np.diff(distance)
        mean_interval = float(np.mean(trace_steps)) if trace_steps.size else 0.0
        min_interval = float(np.min(trace_steps)) if trace_steps.size else 0.0
        max_interval = float(np.max(trace_steps)) if trace_steps.size else 0.0
    else:
        mean_interval = min_interval = max_interval = 0.0

    if timestamps.size:
        timestamp_min = float(np.min(timestamps))
        timestamp_max = float(np.max(timestamps))
    else:
        timestamp_min = timestamp_max = None

    confidence_valid = height_confidence[np.isfinite(height_confidence)]
    if confidence_valid.size:
        confidence_min = float(np.min(confidence_valid))
        confidence_mean = float(np.mean(confidence_valid))
        confidence_max = float(np.max(confidence_valid))
        confidence_low_count = int(np.count_nonzero(confidence_valid < 0.5))
    else:
        confidence_min = confidence_mean = confidence_max = None
        confidence_low_count = None

    if alignment_status.size:
        alignment_extrapolated_count = int(np.count_nonzero(alignment_status == "extrapolated"))
        alignment_resampled_count = int(np.count_nonzero(alignment_status == "resampled"))
        alignment_extrapolated_fraction = (
            float(alignment_extrapolated_count) / float(alignment_status.size)
        )
    else:
        alignment_extrapolated_count = None
        alignment_resampled_count = None
        alignment_extrapolated_fraction = None

    return {
        "source": "airborne_csv",
        "trace_interval_m": mean_interval,
        "track_length_m": float(distance[-1]) if distance.size else 0.0,
        "trace_interval_min_m": min_interval,
        "trace_interval_max_m": max_interval,
        "ground_elevation_min_m": float(np.min(ground)) if ground.size else 0.0,
        "ground_elevation_max_m": float(np.max(ground)) if ground.size else 0.0,
        "flight_height_min_m": float(np.min(flight)) if flight.size else 0.0,
        "flight_height_max_m": float(np.max(flight)) if flight.size else 0.0,
        "height_agl_min_m": float(np.min(height_agl)) if height_agl.size else 0.0,
        "height_agl_max_m": float(np.max(height_agl)) if height_agl.size else 0.0,
        "trace_timestamp_min_s": timestamp_min,
        "trace_timestamp_max_s": timestamp_max,
        "height_confidence_min": confidence_min,
        "height_confidence_mean": confidence_mean,
        "height_confidence_max": confidence_max,
        "height_confidence_low_count": confidence_low_count,
        "alignment_extrapolated_trace_count": alignment_extrapolated_count,
        "alignment_extrapolated_fraction": alignment_extrapolated_fraction,
        "alignment_resampled_trace_count": alignment_resampled_count,
        "has_airborne_metadata": True,
    }


_ASCAN_NUM_RE = re.compile(r"(\d+)(?=\.csv$)", re.IGNORECASE)


def _ascan_sort_key(filename: str) -> int:
    """从文件名中提取排序编号（如 lineData_0000001.csv -> 1）"""
    m = _ASCAN_NUM_RE.search(filename)
    return int(m.group(1)) if m else 0


def _find_data_start(lines: list[str]) -> int:
    """查找 CSV 文件中数值数据的起始行号"""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            float(stripped.split(",")[0])
            return i
        except (ValueError, IndexError):
            continue
    return len(lines)


def _parse_ascan_amplitudes(
    lines: list[str],
    data_start: int,
    *,
    max_samples: int | None = None,
) -> list[float]:
    """Parse one A-scan amplitude column, skipping malformed rows."""
    values: list[float] = []
    for line in lines[data_start:]:
        if max_samples is not None and len(values) >= max_samples:
            break
        parts = line.strip().split(",")
        if len(parts) >= 2:
            value_text = parts[1]
        elif len(parts) == 1:
            value_text = parts[0]
        else:
            continue
        try:
            values.append(float(value_text))
        except ValueError:
            continue
    return values


def read_ascans_folder(folder_path: str, max_files: int = 0, progress_cb=None) -> dict:
    """从文件夹加载多条 A-scan CSV，组装为 B-scan 矩阵

    每个 CSV 文件包含一条 A-scan（第二列幅值），按文件名数字排序后
    拼接为 samples x traces 矩阵。

    Args:
        folder_path: 包含 A-scan CSV 的文件夹路径
        max_files: 最大加载文件数（0=不限制）
        progress_cb: 进度回调 (current, total, message)

    Returns:
        dict: {
            "data": np.ndarray (samples x traces, float32),
            "num_traces": int,
            "samples_per_trace": int,
            "time_step_s": float or None,
        }
    """
    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")],
        key=_ascan_sort_key,
    )
    if not csv_files:
        raise ValueError(f"文件夹中没有 CSV 文件: {folder_path}")

    if max_files > 0:
        csv_files = csv_files[:max_files]

    total = len(csv_files)

    # 用第一个文件确定 header 行数和采样点数
    first_path = os.path.join(folder_path, csv_files[0])
    with open(first_path, "r", encoding="utf-8", errors="ignore") as f:
        first_lines = f.readlines()

    data_start = _find_data_start(first_lines)
    if data_start >= len(first_lines):
        raise ValueError(f"无法在 {csv_files[0]} 中找到数值数据")

    # 读取第一个文件的幅值（第二列；单列时取第一列）
    first_data = _parse_ascan_amplitudes(first_lines, data_start)

    samples = len(first_data)
    if samples == 0:
        raise ValueError(f"第一列 A-scan 无有效数据: {csv_files[0]}")

    # 计算时间步长
    time_step_s = None
    if data_start + 1 < len(first_lines):
        try:
            t0 = float(first_lines[data_start].strip().split(",")[0])
            t1 = float(first_lines[data_start + 1].strip().split(",")[0])
            time_step_s = t1 - t0
        except (ValueError, IndexError):
            pass

    # 预分配矩阵
    matrix = np.zeros((samples, total), dtype=np.float32)
    matrix[:, 0] = first_data

    if progress_cb:
        progress_cb(1, total, f"读取 {csv_files[0]} ({samples} 采样点)")

    # 读取剩余文件（复用 data_start 偏移量，无需重新检测）
    for idx in range(1, total):
        fpath = os.path.join(folder_path, csv_files[idx])
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            values = _parse_ascan_amplitudes(
                lines,
                data_start,
                max_samples=samples,
            )
            if values:
                matrix[: len(values), idx] = values
        except OSError as exc:
            logger.warning("跳过 A-scan 文件 %s: %s", csv_files[idx], exc)

        if progress_cb and (idx % 200 == 0 or idx == total - 1):
            progress_cb(idx + 1, total, f"读取 {csv_files[idx]} ({idx + 1}/{total})")

    return {
        "data": matrix,
        "num_traces": total,
        "samples_per_trace": samples,
        "time_step_s": time_step_s,
    }


# ============ Backward Compatibility ============

# Keep old function names for compatibility
load_bscan_csv = readcsv
load_ascans_folder = read_ascans_folder


__all__ = [
    "readcsv",
    "savecsv",
    "save_image",
    "show_image",
    "extract_airborne_csv_payload",
    "subset_trace_metadata",
    "compute_trace_distance_m",
    "auto_load_data",
    "read_mala_rd",
    "read_impulseradar_iprb",
    "read_segy_fixed",
    "read_envi_bsq",
    "read_numpy_profile",
    "read_ascans_folder",
    "load_bscan_csv",
    "load_ascans_folder",
]

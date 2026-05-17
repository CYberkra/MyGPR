#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enhanced GPR I/O module.

Supports:
- Standard CSV B-scan files (with/without header)
- Folder of A-scan CSV files
- gprMax .out files
- gprMax .in configuration files

Author: GPR_GUI Team
Date: 2026-03-31
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

# Try to import h5py for gprMax .out support
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available. gprMax .out loading disabled.")

# 直接导入 read_file_data 模块
from read_file_data import readcsv, savecsv, save_image, show_image
from core.data_context import (
    DATA_CONTEXT_GPRMAX,
    DATA_CONTEXT_GPRMAX_IMPULSE,
    DATA_CONTEXT_UAV_GPR_SFCW_FIELD,
    apply_data_context_defaults,
)
from core.scalar_utils import to_float, to_float_or_none, to_int


def read_gprmax_in(in_path: str) -> Dict[str, Any]:
    """Parse gprMax .in configuration file.

    Extracts key parameters like domain size, dx, time window, etc.

    Args:
        in_path: Path to .in file

    Returns:
        dict: Configuration parameters
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f".in file not found: {in_path}")

    config = {
        "title": "",
        "domain": None,
        "dx_dy_dz": None,
        "time_window": None,
        "materials": [],
        "geometry_files": [],
        "waveform": None,
        "src_position": None,
        "rx_position": None,
        "src_steps": None,
        "rx_steps": None,
    }

    with open(in_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("#title:"):
                config["title"] = line.replace("#title:", "").strip()
            elif line.startswith("#domain:"):
                parts = line.replace("#domain:", "").strip().split()
                config["domain"] = [float(p) for p in parts]
            elif line.startswith("#dx_dy_dz:"):
                parts = line.replace("#dx_dy_dz:", "").strip().split()
                config["dx_dy_dz"] = [float(p) for p in parts]
            elif line.startswith("#time_window:"):
                config["time_window"] = float(line.replace("#time_window:", "").strip())
            elif line.startswith("#material:"):
                config["materials"].append(line)
            elif line.startswith("#geometry_objects_read:"):
                parts = line.replace("#geometry_objects_read:", "").strip().split()
                if len(parts) >= 5:
                    config["geometry_files"].append(parts[3])  # h5 file
                    config["geometry_files"].append(parts[4])  # materials file
            elif line.startswith("#waveform:"):
                config["waveform"] = line
            elif line.startswith("#hertzian_dipole:"):
                parts = line.replace("#hertzian_dipole:", "").strip().split()
                if len(parts) >= 5:
                    config["src_position"] = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
            elif line.startswith("#rx:"):
                parts = line.replace("#rx:", "").strip().split()
                if len(parts) >= 3:
                    config["rx_position"] = [
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                    ]
            elif line.startswith("#src_steps:"):
                parts = line.replace("#src_steps:", "").strip().split()
                config["src_steps"] = [float(p) for p in parts]
            elif line.startswith("#rx_steps:"):
                parts = line.replace("#rx_steps:", "").strip().split()
                config["rx_steps"] = [float(p) for p in parts]

    return config


# ============ Auto-detect and Load ============


def auto_load_data(path: str, **kwargs) -> Dict[str, Any]:
    """Auto-detect file type and load GPR data.

    Supports:
    - .out: gprMax simulation output
    - .in: gprMax configuration file
    - .csv: B-scan CSV file
    - folder: Folder of A-scan CSV files

    Args:
        path: File or folder path
        **kwargs: Additional arguments passed to specific loaders

    Returns:
        dict: Loaded data with metadata
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_dir():
        # Folder of A-scan CSV files
        return read_ascans_folder(str(path), **kwargs)

    suffix = path.suffix.lower()

    if suffix == ".out":
        # gprMax simulation output (.out HDF5)
        return read_gprmax_out(str(path))

    elif suffix == ".in":
        # gprMax configuration file
        return read_gprmax_in(str(path))

    elif suffix == ".csv":
        # B-scan CSV file
        data = readcsv(str(path))
        return {
            "data": data,
            "type": "bscan_csv",
            "source": str(path),
        }

    else:
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
            amp_col = 3 if use_rows.shape[1] >= 4 else _select_amp_column(use_rows)
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
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    return cumulative.astype(np.float32)


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


def _find_gprmax_input_for_output(out_path: Path) -> Path | None:
    """Find the most likely .in file that produced a gprMax .out file."""
    trace_prefix = _gprmax_output_prefix(out_path)
    candidates = [
        out_path.with_suffix(".in"),
        out_path.with_name(out_path.stem.replace("_merged", "") + ".in"),
        out_path.with_name(trace_prefix + ".in"),
    ]
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            return candidate
    in_files = sorted(out_path.parent.glob("*.in"))
    return in_files[0] if in_files else None


def _gprmax_output_prefix(out_path: Path) -> str:
    """Return the shared filename prefix for gprMax per-trace .out files."""
    stem = out_path.stem
    if stem.endswith("_merged"):
        stem = stem[: -len("_merged")]
    match = re.match(r"^(.*?)(\d+)$", stem)
    return match.group(1) if match else stem


def _gprmax_trace_index(path: Path, prefix: str) -> int | None:
    """Return the per-trace numeric suffix for a related gprMax .out file."""
    stem = path.stem
    if "merged" in stem.lower() or not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _related_gprmax_out_files(out_path: Path) -> list[Path]:
    """List .out files that belong to the same gprMax B-scan run."""
    prefix = _gprmax_output_prefix(out_path)
    related: list[tuple[int, str, Path]] = []
    for candidate in out_path.parent.glob("*.out"):
        trace_index = _gprmax_trace_index(candidate, prefix)
        if trace_index is not None:
            related.append((trace_index, candidate.name, candidate))
    if related:
        return [item[2] for item in sorted(related, key=lambda item: (item[0], item[1]))]
    if "merged" not in out_path.stem.lower():
        return [out_path]
    return []


def _find_gprmax_manifest_for_output(out_path: Path) -> Path | None:
    """Find a manifest JSON near a gprMax output file."""
    candidates: list[Path] = []
    for pattern in ("*_manifest.json", "manifest.json", "dataset_manifest.json"):
        candidates.extend(sorted(out_path.parent.glob(pattern)))
    unique = list(dict.fromkeys(candidates))
    if not unique:
        return None

    matching: list[Path] = []
    for candidate in unique:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        primary = _manifest_path_value(
            payload,
            "primary_out_file",
            "primary_out_path",
            "out_file",
            "merged_out_file",
        )
        if primary and Path(str(primary)).name == out_path.name:
            matching.append(candidate)
    if len(matching) == 1:
        return matching[0]
    return unique[0] if len(unique) == 1 else None


def _manifest_path_value(payload: dict[str, Any], *keys: str) -> str | None:
    groups = [
        payload,
        payload.get("paths_relative_to_output_dir"),
        payload.get("paths"),
        payload.get("files"),
    ]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for key in keys:
            value = group.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _attach_gprmax_ground_truth(
    header_info: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """Attach converted gprMax ground truth from a nearby manifest when available."""
    manifest_path = _find_gprmax_manifest_for_output(out_path)
    if manifest_path is None:
        return header_info
    header = dict(header_info)
    try:
        from core.gprmax_ground_truth import load_ground_truth_from_manifest

        samples = int(header.get("a_scan_length", 0) or 0)
        traces = int(header.get("num_traces", 0) or 0)
        data_shape = (samples, traces) if samples > 0 and traces > 0 else None
        ground_truth = load_ground_truth_from_manifest(
            str(manifest_path),
            data_shape=data_shape,
        )
    except Exception as exc:
        header["ground_truth_load_error"] = str(exc)
        header["ground_truth_manifest_path"] = str(manifest_path)
        return header
    if ground_truth:
        header["ground_truth"] = ground_truth
        header["ground_truth_manifest_path"] = str(manifest_path)
    return header


def _safe_attr_list(value: Any) -> list[float] | None:
    parsed: list[float] = []
    try:
        iterator = list(value)
    except Exception:
        return None
    for item in iterator:
        number = to_float_or_none(item)
        if number is None:
            return None
        parsed.append(number)
    return parsed


def _build_gprmax_trace_metadata(
    traces: int,
    gprmax_config: dict[str, Any] | None,
) -> dict[str, np.ndarray] | None:
    """Create deterministic per-trace distance metadata from gprMax step commands."""
    if traces <= 0:
        return None
    step = None
    if gprmax_config:
        src_steps = gprmax_config.get("src_steps")
        rx_steps = gprmax_config.get("rx_steps")
        for steps in (rx_steps, src_steps):
            if steps and len(steps) >= 1:
                candidate = to_float(steps[0], default=0.0)
                if candidate > 0.0:
                    step = candidate
                    break
    if step is None:
        return None
    return {
        "trace_index": np.arange(traces, dtype=np.int32),
        "trace_distance_m": (np.arange(traces, dtype=np.float32) * np.float32(step)),
    }


def _build_gprmax_header_info(
    *,
    out_path: Path,
    samples: int,
    traces: int,
    time_step_s: float | None,
    total_time_ns: float | None,
    attrs: dict[str, Any],
    gprmax_config: dict[str, Any] | None,
) -> dict[str, Any]:
    trace_interval_m = 0.0
    if gprmax_config:
        for steps in (gprmax_config.get("rx_steps"), gprmax_config.get("src_steps")):
            if steps and len(steps) >= 1:
                trace_interval_m = to_float(steps[0], default=0.0)
                if trace_interval_m > 0.0:
                    break

    header = {
        "a_scan_length": int(samples),
        "num_traces": int(traces),
        "total_time_ns": to_float(total_time_ns, default=0.0),
        "time_step_s": time_step_s,
        "trace_interval_m": trace_interval_m,
        "source": "gprmax_out",
        "source_format": "gprmax_out",
        "out_path": str(out_path),
    }
    if "Iterations" in attrs:
        header["gprmax_iterations"] = to_int(attrs["Iterations"], default=0)
    if "dt" in attrs:
        header["gprmax_dt_s"] = to_float(attrs["dt"], default=0.0)
    if "nx_ny_nz" in attrs:
        nx_ny_nz = _safe_attr_list(attrs.get("nx_ny_nz"))
        if nx_ny_nz is not None:
            header["gprmax_nx_ny_nz"] = nx_ny_nz

    context = DATA_CONTEXT_GPRMAX
    if gprmax_config and "impulse" in str(gprmax_config.get("waveform") or "").lower():
        context = DATA_CONTEXT_GPRMAX_IMPULSE
    return apply_data_context_defaults(
        header,
        gprmax_config=gprmax_config,
        source_path=out_path,
        context=context,
    )


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
        except OSError:
            pass

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


def read_gprmax_out(out_path: str) -> dict:
    """读取 gprMax .out HDF5 文件，提取电场数据并组装为 B-scan 矩阵

    Args:
        out_path: gprMax .out 文件路径

    Returns:
        dict: {
            "data": np.ndarray (samples x traces, float32),
            "num_traces": int,
            "samples_per_trace": int,
            "time_step_s": float or None,
            "total_time_ns": float or None,
        }
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required to read gprMax .out files. Install with: pip install h5py"
        )

    out_path = Path(out_path)
    if not out_path.exists():
        raise FileNotFoundError(f"gprMax .out file not found: {out_path}")

    gprmax_config = None
    in_path = _find_gprmax_input_for_output(out_path)
    if in_path is not None:
        try:
            gprmax_config = read_gprmax_in(str(in_path))
        except Exception:
            gprmax_config = None

    with h5py.File(out_path, "r") as f:
        # 读取属性
        attrs = dict(f.attrs)
        iterations = attrs.get("Iterations", 0)
        dt = attrs.get("dt", 0)
        nx_ny_nz = attrs.get("nx_ny_nz", [1, 1, 1])

        # 读取电场数据
        if "rxs" in f and "rx1" in f["rxs"] and "Ez" in f["rxs"]["rx1"]:
            data = f["rxs"]["rx1"]["Ez"][:]
        else:
            # 文件可能为空（如合并失败的 merged.out）
            # 尝试降级到读取同目录的单独 .out 文件
            out_files = _related_gprmax_out_files(out_path)
            if not out_files:
                raise ValueError(
                    f"Cannot find 'rxs/rx1/Ez' in {out_path} and no other .out files found"
                )

            # 读取第一个文件获取参数
            with h5py.File(out_files[0], "r") as f0:
                first_attrs = dict(f0.attrs)
                iterations = first_attrs.get("Iterations", iterations)
                dt = first_attrs.get("dt", dt)
                attrs = first_attrs
                data0 = f0["rxs"]["rx1"]["Ez"][:]

            # 合并所有文件
            samples = to_int(iterations, default=int(np.asarray(data0).shape[0]))
            n_traces = len(out_files)
            matrix = np.zeros((samples, n_traces), dtype=np.float32)
            matrix[:, 0] = data0

            for i, out_file in enumerate(out_files[1:], 1):
                with h5py.File(out_file, "r") as fi:
                    matrix[:, i] = fi["rxs"]["rx1"]["Ez"][:]
            data = matrix

    # 处理数据形状
    # gprMax 输出: (iterations,) - 单道数据
    # 需要根据文件数量重塑为矩阵

    samples = to_int(iterations, default=int(np.asarray(data).shape[0]))

    # 如果数据是二维的（已合并的 merged.out），直接返回
    if data.ndim == 2 and data.shape[1] > 1:
        time_step_value = to_float(dt, default=0.0)
        time_step_s = time_step_value if time_step_value > 0.0 else None
        total_time_ns = time_step_s * samples * 1e9 if time_step_s else None
        header_info = _build_gprmax_header_info(
            out_path=out_path,
            samples=data.shape[0],
            traces=data.shape[1],
            time_step_s=time_step_s,
            total_time_ns=total_time_ns,
            attrs=attrs,
            gprmax_config=gprmax_config,
        )
        header_info = _attach_gprmax_ground_truth(header_info, out_path)
        trace_metadata = _build_gprmax_trace_metadata(data.shape[1], gprmax_config)
        return {
            "data": data.astype(np.float32, copy=False),
            "num_traces": data.shape[1],
            "samples_per_trace": data.shape[0],
            "time_step_s": time_step_s,
            "total_time_ns": total_time_ns,
            "header_info": header_info,
            "trace_metadata": trace_metadata,
            "gprmax_config": gprmax_config,
            "in_path": str(in_path) if in_path else None,
        }

    # 尝试读取道数信息
    n_traces = 1
    if "rxsteps" in attrs:
        rxsteps = attrs["rxsteps"]
        # 计算步进次数
        if len(rxsteps) >= 1 and rxsteps[0] > 0:
            # 估算道数
            n_traces = 1

    # 如果数据是一维的，尝试查找同目录的其他 .out 文件
    if data.ndim == 1:
        out_files = _related_gprmax_out_files(out_path)

        if len(out_files) > 1:
            # 多个文件，需要合并
            n_traces = len(out_files)
            matrix = np.zeros((samples, n_traces), dtype=np.float32)

            for i, out_file in enumerate(out_files):
                with h5py.File(out_file, "r") as f:
                    matrix[:, i] = f["rxs"]["rx1"]["Ez"][:]
            data = matrix
        else:
            # 单道数据，重塑为列向量
            data = data.reshape(-1, 1)

    # 计算时间参数
    time_step_value = to_float(dt, default=0.0)
    time_step_s = time_step_value if time_step_value > 0.0 else None
    total_time_ns = time_step_s * samples * 1e9 if time_step_s else None
    traces = data.shape[1] if data.ndim == 2 else 1
    header_info = _build_gprmax_header_info(
        out_path=out_path,
        samples=data.shape[0],
        traces=traces,
        time_step_s=time_step_s,
        total_time_ns=total_time_ns,
        attrs=attrs,
        gprmax_config=gprmax_config,
    )
    header_info = _attach_gprmax_ground_truth(header_info, out_path)
    trace_metadata = _build_gprmax_trace_metadata(traces, gprmax_config)

    return {
        "data": data.astype(np.float32, copy=False),
        "num_traces": traces,
        "samples_per_trace": data.shape[0],
        "time_step_s": time_step_s,
        "total_time_ns": total_time_ns,
        "header_info": header_info,
        "trace_metadata": trace_metadata,
        "gprmax_config": gprmax_config,
        "in_path": str(in_path) if in_path else None,
    }


def save_gprmax_out_as_csv(out_path: str, csv_path: str = None) -> str:
    """将 gprMax .out 文件转换为 CSV 格式

    Args:
        out_path: gprMax .out 文件路径
        csv_path: 输出 CSV 路径（可选，默认同目录同名）

    Returns:
        str: 生成的 CSV 文件路径
    """
    result = read_gprmax_out(out_path)
    data = result["data"]

    if csv_path is None:
        csv_path = str(Path(out_path).with_suffix(".csv"))

    # 保存为 CSV（无 header）
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6f")

    return csv_path


__all__ = [
    "readcsv",
    "savecsv",
    "save_image",
    "show_image",
    "read_gprmax_in",
    "read_gprmax_out",
    "save_gprmax_out_as_csv",
    "extract_airborne_csv_payload",
    "subset_trace_metadata",
    "compute_trace_distance_m",
    "auto_load_data",
    "read_ascans_folder",
    "load_bscan_csv",
    "load_ascans_folder",
]

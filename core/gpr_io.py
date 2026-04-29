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
                    metadata["time_window_ns"] = float(header_info["total_time_ns"])
                metadata = _integrate_optional_airborne_sidecars(
                    metadata,
                    trace_timestamps_s=trace_timestamps_s,
                    rtk_path=rtk_path,
                    imu_path=imu_path,
                )
                if updated_header is None:
                    updated_header = {}
                updated_header.update(_build_airborne_header_summary(metadata))
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
) -> dict[str, np.ndarray] | None:
    """Optionally merge parsed RTK/IMU sidecars into airborne trace metadata."""
    if rtk_path is None and imu_path is None:
        return metadata
    if metadata is None:
        raise ValueError("optional sidecar integration requires airborne trace metadata")

    integration_module = importlib.import_module("core.sidecar_integration")
    return integration_module.load_and_integrate_optional_sidecars(
        metadata,
        trace_timestamps_s=trace_timestamps_s,
        rtk_path=rtk_path,
        imu_path=imu_path,
    )


def subset_trace_metadata(
    metadata: dict[str, np.ndarray] | None, trace_indices: np.ndarray | slice | None
) -> dict[str, np.ndarray] | None:
    """Subset per-trace metadata using trace indices."""
    if metadata is None or trace_indices is None:
        return metadata
    subset = {}
    for key, values in metadata.items():
        subset[key] = np.asarray(values)[trace_indices].copy()
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

    if distance.size > 1:
        trace_steps = np.diff(distance)
        mean_interval = float(np.mean(trace_steps)) if trace_steps.size else 0.0
        min_interval = float(np.min(trace_steps)) if trace_steps.size else 0.0
        max_interval = float(np.max(trace_steps)) if trace_steps.size else 0.0
    else:
        mean_interval = min_interval = max_interval = 0.0

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
    first_data = []
    for line in first_lines[data_start:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                first_data.append(float(parts[1]))
            except ValueError:
                continue
        elif len(parts) == 1:
            try:
                first_data.append(float(parts[0]))
            except ValueError:
                continue

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
            row_idx = 0
            for line in lines[data_start:]:
                parts = line.strip().split(",")
                if len(parts) >= 2 and row_idx < samples:
                    try:
                        matrix[row_idx, idx] = float(parts[1])
                    except ValueError:
                        pass
                    row_idx += 1
                elif len(parts) == 1 and row_idx < samples:
                    try:
                        matrix[row_idx, idx] = float(parts[0])
                    except ValueError:
                        pass
                    row_idx += 1
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
            folder = str(out_path.parent)
            out_files = sorted(
                [
                    fn
                    for fn in os.listdir(folder)
                    if fn.endswith(".out") and "merged" not in fn
                ],
                key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
            )
            if not out_files:
                raise ValueError(
                    f"Cannot find 'rxs/rx1/Ez' in {out_path} and no other .out files found"
                )

            # 读取第一个文件获取参数
            first_path = os.path.join(folder, out_files[0])
            with h5py.File(first_path, "r") as f0:
                first_attrs = dict(f0.attrs)
                iterations = first_attrs.get("Iterations", iterations)
                dt = first_attrs.get("dt", dt)
                data0 = f0["rxs"]["rx1"]["Ez"][:]

            # 合并所有文件
            samples = int(iterations)
            n_traces = len(out_files)
            matrix = np.zeros((samples, n_traces), dtype=np.float32)
            matrix[:, 0] = data0

            for i, fname in enumerate(out_files[1:], 1):
                fpath = os.path.join(folder, fname)
                with h5py.File(fpath, "r") as fi:
                    matrix[:, i] = fi["rxs"]["rx1"]["Ez"][:]
            data = matrix

    # 处理数据形状
    # gprMax 输出: (iterations,) - 单道数据
    # 需要根据文件数量重塑为矩阵

    samples = int(iterations)

    # 如果数据是二维的（已合并的 merged.out），直接返回
    if data.ndim == 2 and data.shape[1] > 1:
        time_step_s = float(dt) if dt else None
        total_time_ns = time_step_s * samples * 1e9 if time_step_s else None
        return {
            "data": data.astype(np.float32),
            "num_traces": data.shape[1],
            "samples_per_trace": data.shape[0],
            "time_step_s": time_step_s,
            "total_time_ns": total_time_ns,
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
        folder = out_path.parent
        # 按文件名中的数字排序，而不是字符串排序
        out_files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".out") and "merged" not in f],
            key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
        )

        if len(out_files) > 1:
            # 多个文件，需要合并
            n_traces = len(out_files)
            matrix = np.zeros((samples, n_traces), dtype=np.float32)

            for i, fname in enumerate(out_files):
                fpath = os.path.join(folder, fname)
                with h5py.File(fpath, "r") as f:
                    matrix[:, i] = f["rxs"]["rx1"]["Ez"][:]
            data = matrix
        else:
            # 单道数据，重塑为列向量
            data = data.reshape(-1, 1)

    # 计算时间参数
    time_step_s = float(dt) if dt else None
    total_time_ns = time_step_s * samples * 1e9 if time_step_s else None

    return {
        "data": data.astype(np.float32),
        "num_traces": data.shape[1] if data.ndim == 2 else 1,
        "samples_per_trace": data.shape[0],
        "time_step_s": time_step_s,
        "total_time_ns": total_time_ns,
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

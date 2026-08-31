#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight native readers for selected common GPR profile formats.

These readers intentionally cover conservative, documented subsets.  They return
MyGPR's normalized payload: ``data`` as samples x traces plus ``header_info``.
"""

from __future__ import annotations

import os
import re
import struct
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from mygpr.domain.common.errors import MyGPRError


class GPRFormatReadError(MyGPRError):
    """Raised when a known GPR format cannot be safely decoded."""


class GprReaderFormat(str, Enum):
    """GPR reader format type identifiers returned by native readers."""

    NUMPY_ARRAY = "numpy_array"
    MALA_RD = "mala_rd"
    IMPULSERADAR_IPRB = "impulseradar_iprb"
    SEGY_FIXED = "segy_fixed"
    ENVI_BSQ = "envi_bsq"
    SENSORS_SOFTWARE_DT1 = "sensors_software_dt1"
    GSSI_DZT = "gssi_dzt"


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except (OSError, UnicodeError):
            continue
    return path.read_bytes().decode("latin-1", errors="ignore")


def _parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().strip("\ufeff")
        if not line or line.startswith(("#", "//")):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
        elif "=" in line:
            key, val = line.split("=", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts
        norm = re.sub(r"[^A-Z0-9]+", "_", key.strip().upper()).strip("_")
        values[norm] = val.strip()
    return values


def _num(raw: Any, default: float | None = None) -> float | None:
    if raw is None:
        return default
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))
    if not m:
        return default
    try:
        return float(m.group(0))
    except ValueError:
        return default


def _int(raw: Any, default: int | None = None) -> int | None:
    val = _num(raw, None)
    return int(round(val)) if val is not None else default


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_suffix(suffix)


def _ensure_2d_matrix(data: np.ndarray, *, source: str) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise GPRFormatReadError(f"{source} 不是二维 B-scan 矩阵: shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise GPRFormatReadError(f"{source} 不是数值矩阵: dtype={arr.dtype}")
    arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        finite = np.isfinite(arr)
        fill = float(np.mean(arr[finite])) if finite.any() else 0.0
        arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr


def read_numpy_profile(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as z:
            key = "data" if "data" in z.files else z.files[0]
            data = z[key]
            header = {"npz_key": key, "npz_keys": list(z.files)}
    else:
        data = np.load(p, mmap_mode="r", allow_pickle=False)
        header = {}
    arr = _ensure_2d_matrix(data, source=str(p))
    header.update(
        {
            "a_scan_length": int(arr.shape[0]),
            "num_traces": int(arr.shape[1]),
            "total_time_ns": 0.0,
            "trace_interval_m": 0.0,
            "source": GprReaderFormat.NUMPY_ARRAY,
            "path": str(p),
        }
    )
    return {"data": arr, "header_info": header, "path": str(p), "format": GprReaderFormat.NUMPY_ARRAY}


def read_mala_rd(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".rad":
        rad = p
        rd3 = p.with_suffix(".rd3")
        rd7 = p.with_suffix(".rd7")
        data_path = rd3 if rd3.exists() else rd7
    else:
        data_path = p
        rad = _sidecar(p, ".rad")
    if not rad.exists():
        raise GPRFormatReadError(f"MALÅ RD3/RD7 需要同名 .rad 头文件: {rad}")
    if not data_path.exists():
        raise GPRFormatReadError(f"MALÅ 数据文件不存在: {data_path}")
    kv = _parse_key_values(_read_text(rad))
    samples = _int(kv.get("SAMPLES"))
    if not samples or samples <= 0:
        raise GPRFormatReadError("MALÅ .rad 缺少有效 SAMPLES")
    dtype = np.dtype("<i2") if data_path.suffix.lower() == ".rd3" else np.dtype("<i4")
    raw = np.fromfile(data_path, dtype=dtype)
    traces = raw.size // samples
    if traces <= 0:
        raise GPRFormatReadError("MALÅ 数据长度不足一个 trace")
    raw = raw[: traces * samples]
    data = raw.reshape((traces, samples)).T.astype(np.float32, copy=False)
    time_window = _num(kv.get("TIME_WINDOW") or kv.get("TIMEWINDOW"), 0.0) or 0.0
    trace_interval = _num(kv.get("DISTANCE_INTERVAL"), 0.0) or 0.0
    last_trace = _int(kv.get("LAST_TRACE"), traces) or traces
    header = {
        "a_scan_length": int(samples),
        "num_traces": int(traces),
        "declared_last_trace": int(last_trace),
        "total_time_ns": float(time_window),
        "trace_interval_m": float(trace_interval),
        "source": GprReaderFormat.MALA_RD,
        "rad_path": str(rad),
        "data_path": str(data_path),
    }
    return {"data": data, "header_info": header, "path": str(data_path), "format": GprReaderFormat.MALA_RD}


def read_impulseradar_iprb(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".iprh":
        header_path = p
        data_path = p.with_suffix(".iprb")
    else:
        data_path = p
        header_path = p.with_suffix(".iprh")
    if not header_path.exists():
        raise GPRFormatReadError(f"ImpulseRadar IPRB 需要同名 .iprh 头文件: {header_path}")
    if not data_path.exists():
        raise GPRFormatReadError(f"ImpulseRadar 数据文件不存在: {data_path}")
    kv = _parse_key_values(_read_text(header_path))
    samples = _int(kv.get("SAMPLES"))
    data_version = _int(kv.get("DATA_VERSION"), 16) or 16
    if not samples or samples <= 0:
        raise GPRFormatReadError("ImpulseRadar .iprh 缺少有效 SAMPLES")
    if data_version not in {16, 32}:
        raise GPRFormatReadError(f"暂不支持 DATA VERSION={data_version}")
    dtype = np.dtype("<i2") if data_version == 16 else np.dtype("<i4")
    raw = np.fromfile(data_path, dtype=dtype)
    traces = raw.size // samples
    if traces <= 0:
        raise GPRFormatReadError("ImpulseRadar 数据长度不足一个 trace")
    raw = raw[: traces * samples]
    data = raw.reshape((traces, samples)).T.astype(np.float32, copy=False)
    freq_khz = _num(kv.get("FREQUENCY"), None)
    total_time_ns = 0.0
    if freq_khz and freq_khz > 0:
        # Header frequency is typically sampling frequency in kHz; period in ns.
        total_time_ns = float(samples) * (1e6 / float(freq_khz))
    header = {
        "a_scan_length": int(samples),
        "num_traces": int(traces),
        "total_time_ns": total_time_ns,
        "trace_interval_m": _num(kv.get("DISTANCE_INTERVAL"), 0.0) or 0.0,
        "source": GprReaderFormat.IMPULSERADAR_IPRB,
        "iprh_path": str(header_path),
        "data_path": str(data_path),
        "data_version": int(data_version),
    }
    return {"data": data, "header_info": header, "path": str(data_path), "format": GprReaderFormat.IMPULSERADAR_IPRB}


def _segy_format_dtype(format_code: int) -> np.dtype | None:
    # Conservative subset used in many fixed-length profile exports.
    if format_code == 2:
        return np.dtype(">i4")
    if format_code == 3:
        return np.dtype(">i2")
    if format_code == 5:
        return np.dtype(">f4")
    if format_code == 8:
        return np.dtype(">i1")
    return None


def read_segy_fixed(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    blob = p.read_bytes()
    if len(blob) < 3600 + 240:
        raise GPRFormatReadError("SEG-Y 文件过短")
    bin_header = blob[3200:3600]
    sample_interval_us = struct.unpack(">H", bin_header[16:18])[0]
    samples = struct.unpack(">H", bin_header[20:22])[0]
    fmt = struct.unpack(">H", bin_header[24:26])[0]
    if samples <= 0:
        # Try trace header fallback bytes 115-116 inside first 240-byte trace header.
        samples = struct.unpack(">H", blob[3600 + 114 : 3600 + 116])[0]
    dtype = _segy_format_dtype(fmt)
    if dtype is None or samples <= 0:
        raise GPRFormatReadError(f"暂不支持该 SEG-Y 样本格式或采样数: format={fmt}, samples={samples}")
    bytes_per_trace = 240 + samples * dtype.itemsize
    ntr = (len(blob) - 3600) // bytes_per_trace
    if ntr <= 0:
        raise GPRFormatReadError("SEG-Y 未检测到完整 trace")
    data = np.empty((samples, ntr), dtype=np.float32)
    offset = 3600
    for idx in range(ntr):
        start = offset + 240
        end = start + samples * dtype.itemsize
        data[:, idx] = np.frombuffer(blob[start:end], dtype=dtype, count=samples).astype(np.float32, copy=False)
        offset += bytes_per_trace
    header = {
        "a_scan_length": int(samples),
        "num_traces": int(ntr),
        "sample_interval_us": int(sample_interval_us),
        "total_time_ns": float(sample_interval_us) * float(samples) * 1000.0,
        "trace_interval_m": 0.0,
        "source": GprReaderFormat.SEGY_FIXED,
        "path": str(p),
        "sample_format_code": int(fmt),
    }
    return {"data": data, "header_info": header, "path": str(p), "format": GprReaderFormat.SEGY_FIXED}


def read_envi_bsq(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".hdr":
        hdr = p
        data_path = p.with_suffix(".dat")
    else:
        data_path = p
        hdr = p.with_suffix(".hdr")
    if not hdr.exists():
        raise GPRFormatReadError(f"ENVI BSQ 需要同名 .hdr 头文件: {hdr}")
    kv = _parse_key_values(_read_text(hdr))
    samples = _int(kv.get("SAMPLES"))
    lines = _int(kv.get("LINES"))
    bands = _int(kv.get("BANDS"), 1) or 1
    data_type = _int(kv.get("DATA_TYPE"), None)
    byte_order = _int(kv.get("BYTE_ORDER"), 0) or 0
    dtype_map = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4"}
    if not samples or not lines or data_type not in dtype_map:
        raise GPRFormatReadError("ENVI .hdr 缺少 samples/lines/data type 或 data type 不支持")
    endian = ">" if byte_order == 1 and dtype_map[data_type] != "u1" else "<"
    dtype = np.dtype((endian + dtype_map[data_type]) if dtype_map[data_type] != "u1" else dtype_map[data_type])
    raw = np.fromfile(data_path, dtype=dtype)
    expected = samples * lines * bands
    if raw.size < expected:
        raise GPRFormatReadError(f"ENVI 数据长度不足: {raw.size} < {expected}")
    raw = raw[:expected]
    arr = raw.reshape((bands, lines, samples))
    data = arr[0].T.astype(np.float32, copy=False)
    header = {
        "a_scan_length": int(samples),
        "num_traces": int(lines),
        "total_time_ns": 0.0,
        "trace_interval_m": 0.0,
        "source": GprReaderFormat.ENVI_BSQ,
        "hdr_path": str(hdr),
        "data_path": str(data_path),
        "bands": int(bands),
        "data_type": int(data_type),
    }
    return {"data": data, "header_info": header, "path": str(data_path), "format": GprReaderFormat.ENVI_BSQ}


def read_sensors_software_dt1(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Decode Sensors & Software (PulseEKKO) .DT1 traces with the .HD text header.

    二进制布局（经 GPRPy XLINE00 样例实证）：
    - 无文件头；``N × (128B 道头 + samples×int16)`` 连续排列
    - 32-float 道头：``[0]`` 道号(1-based)、``[1]`` 位置、``[2]`` 每道采样数、``[7]`` 叠加次数
    - ``.HD`` 为纯文本键值头，位置单位可为 ft（换算为 m）
    """
    p = Path(path)
    if p.suffix.lower() == ".hd":
        header_path = p
        data_path = p.with_suffix(".dt1")
    else:
        data_path = p
        header_path = p.with_suffix(".hd")
    if not data_path.exists():
        raise GPRFormatReadError(f"Sensors & Software .DT1 数据文件不存在: {data_path}")
    if not header_path.exists():
        raise GPRFormatReadError(f"Sensors & Software .DT1 需要同名 .hd 文本头文件: {header_path}")

    kv = _parse_key_values(_read_text(header_path))
    file_size = data_path.stat().st_size
    with data_path.open("rb") as stream:
        first_head = struct.unpack("<32f", stream.read(128))
        samples = int(first_head[2]) if first_head[2] > 0 else _int(kv.get("NUMBER_OF_PTS_TRC"), 0) or 0
        if samples <= 0:
            raise GPRFormatReadError("DT1 道头缺少有效每道采样数")
        bytes_per_trace = 128 + samples * 2
        traces = file_size // bytes_per_trace
        if traces <= 0 or file_size % bytes_per_trace != 0:
            raise GPRFormatReadError(
                f"DT1 文件长度与道头声明的采样数不符: size={file_size}, samples={samples}"
            )
        data = np.empty((samples, traces), dtype=np.float32)
        positions = np.empty(traces, dtype=np.float64)
        for index in range(traces):
            stream.seek(index * bytes_per_trace)
            trace_head = struct.unpack("<32f", stream.read(128))
            positions[index] = float(trace_head[1])
            data[:, index] = np.frombuffer(
                stream.read(samples * 2), dtype="<i2", count=samples
            ).astype(np.float32)

    total_time_ns = _num(kv.get("TOTAL_TIME_WINDOW"), 0.0) or 0.0
    step_size = _num(kv.get("STEP_SIZE_USED"), 0.0) or 0.0
    pos_units = str(kv.get("POSITION_UNITS", "m")).lower()
    if pos_units == "ft":
        step_size *= 0.3048
    header = {
        "a_scan_length": int(samples),
        "num_traces": int(traces),
        "total_time_ns": float(total_time_ns),
        "trace_interval_m": float(step_size),
        "nominal_frequency_mhz": _num(kv.get("NOMINAL_FREQUENCY"), 0.0) or 0.0,
        "stacks": _int(kv.get("NUMBER_OF_STACKS"), 0) or 0,
        "source": GprReaderFormat.SENSORS_SOFTWARE_DT1,
        "hd_path": str(header_path),
        "data_path": str(data_path),
        "trace_positions": positions,
    }
    return {
        "data": data,
        "header_info": header,
        "path": str(data_path),
        "format": GprReaderFormat.SENSORS_SOFTWARE_DT1,
    }


def read_gssi_dzt(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Decode GSSI .DZT profiles (conservative single-channel subset).

    布局（DZT.File.Format 文档 + GPRPy/readgssi 实现共识）：
    - 1024B 固定头（little-endian）；``rh_data`` 决定头块数（每块 1024B）
    - 样本位宽 ``rh_bits`` ∈ {8, 16, 32}；uint8/uint16 需减 ``2^(bits-1)`` 转有符号
    - 数据按道连续存储，reshape 为 (traces, samples) 后转置
    """
    p = Path(path)
    file_size = p.stat().st_size
    if file_size < 1024:
        raise GPRFormatReadError(f"DZT 文件过短: {file_size}")
    with p.open("rb") as stream:
        head = stream.read(1024)
    _rh_tag, rh_data, samples, rh_bits, _rh_zero = struct.unpack("<5h", head[:10])
    sps, spm, _mpm, _position, time_range_ns = struct.unpack("<5f", head[10:30])
    rh_npass, = struct.unpack("<h", head[30:32])
    rh_nchan, = struct.unpack("<h", head[52:54])
    # rh_data<1024 时为 1024B 头块数；≥1024 时为直接字节数（1024 处两种解读一致）
    header_bytes = rh_data if rh_data >= 1024 else 1024 * rh_data
    if samples <= 0 or rh_bits not in (8, 16, 32):
        raise GPRFormatReadError(f"DZT 头字段无效: samples={samples}, bits={rh_bits}")
    if file_size <= header_bytes:
        raise GPRFormatReadError(f"DZT 无数据体: size={file_size}, header={header_bytes}")
    dtype = {8: np.dtype("u1"), 16: np.dtype("<u2"), 32: np.dtype("<i4")}[rh_bits]
    element_size = dtype.itemsize
    available = file_size - header_bytes
    bytes_per_trace = samples * element_size
    traces = available // bytes_per_trace
    if traces <= 0:
        raise GPRFormatReadError("DZT 数据体不足一个完整 trace")
    with p.open("rb") as stream:
        stream.seek(header_bytes)
        raw = np.frombuffer(
            stream.read(traces * bytes_per_trace), dtype=dtype, count=traces * samples
        )
    if rh_bits in (8, 16):
        data = (raw.astype(np.float64) - float(2 ** (rh_bits - 1))).astype(np.float32)
    else:
        data = raw.astype(np.float32)
    data = data.reshape((traces, samples)).T
    header = {
        "a_scan_length": int(samples),
        "num_traces": int(traces),
        "total_time_ns": float(time_range_ns),
        "trace_interval_m": float(spm) if spm and spm > 0 else 0.0,
        "scans_per_second": float(sps) if sps and sps > 0 else 0.0,
        "bits_per_sample": int(rh_bits),
        "channels": int(rh_nchan),
        "passes": int(rh_npass),
        "source": GprReaderFormat.GSSI_DZT,
        "path": str(p),
    }
    return {"data": data, "header_info": header, "path": str(p), "format": GprReaderFormat.GSSI_DZT}


def unsupported_known_format_message(path: str | os.PathLike[str], display_name: str, notes: str = "") -> str:
    return (
        f"{display_name} 已被识别为常见 GPR 数据格式，但 V0.8.40 尚未内置可靠解码器。"
        f"建议先用设备软件/RGPR/GPRPy 等转换为 CSV、SEG-Y、ENVI 或 MyGPR-readable 数据后导入。"
        + (f"\n说明：{notes}" if notes else "")
    )


__all__ = [
    "GPRFormatReadError",
    "GprReaderFormat",
    "read_numpy_profile",
    "read_mala_rd",
    "read_impulseradar_iprb",
    "read_sensors_software_dt1",
    "read_gssi_dzt",
    "read_segy_fixed",
    "read_envi_bsq",
    "unsupported_known_format_message",
]

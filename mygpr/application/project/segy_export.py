#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEG-Y 2.1 fixed-length IEEE float32 export.

The extended binary interval is in microseconds; trace delay uses the standard
millisecond time scalar. Along-line distances use group X in metres at 1 mm
precision. Axes that cannot be represented by this profile are rejected.
Reference: SEG Technical Standards Committee, SEG-Y rev 2.1 (October 2023),
https://seg.org/wp-content/uploads/2025/11/seg_y_rev2_1-oct2023.pdf,
binary bytes 3273–3280 and trace bytes 71–72, 81–84, 109–110, 215–216.
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_FORMAT_FLOAT32 = 5


def _text_header(result_name: str, line_id: str) -> bytes:
    lines = [
        f"C 1 MYGPR EXPORTED PROFILE {result_name[:20]}",
        f"C 2 LINE {line_id[:24]}",
        "C 3 SEG-Y REVISION 2.1 IEEE FLOAT32 BIG ENDIAN",
        "C 4 GROUP X IS ALONG-LINE DISTANCE METRES WHEN PRESENT",
        "C 5 TRACE DELAY AND TIME SCALAR MAY BE NONZERO",
    ]
    rows = [line.encode("ascii", errors="replace").ljust(80) for line in lines]
    body = b"".join(rows)
    return body + b" " * (3200 - len(body))


def write_segy(
    destination: str | Path,
    matrix: np.ndarray,
    *,
    time_axis_ns: np.ndarray,
    line_id: str = "",
    result_name: str = "",
    trace_positions: np.ndarray | None = None,
) -> Path:
    """把 samples×traces float32 矩阵写成 SEG-Y 固定长度 profile。"""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"SEG-Y 导出需要非空二维矩阵，当前 shape={arr.shape}")
    samples, ntr = arr.shape
    source_axis = np.asarray(time_axis_ns)
    axis = np.asarray(source_axis, dtype=np.float64)
    if axis.shape != (samples,) or samples < 2 or not np.all(np.isfinite(axis)):
        raise ValueError("SEG-Y 需要完整且有效的时间轴")
    step_ns = float((axis[-1] - axis[0]) / (samples - 1))
    ideal = axis[0] + np.arange(samples) * step_ns
    if source_axis.dtype.kind == "f":
        tolerance = np.abs(np.spacing(source_axis)).astype(np.float64) * 0.51
    else:
        tolerance = np.zeros(samples)
    tolerance += np.finfo(np.float64).eps * np.maximum(np.abs(ideal), 1) * 4
    if step_ns <= 0 or np.any(np.abs(axis - ideal) > tolerance):
        raise ValueError("SEG-Y 不支持非均匀时间轴")
    # SEG rev 2.1 standard trace bytes 109-110 and 215-216: delay ms with
    # decimal scalar.  Reject starts that cannot be represented in this profile.
    delay = None
    for divisor in (10000, 1000, 100, 10, 1):
        scaled = float(axis[0]) * divisor / 1e6
        integer = round(scaled)
        if -32768 <= integer <= 32767 and abs(scaled - integer) < 1e-9:
            delay = (integer, -divisor if divisor != 1 else 1)
            break
    if delay is None:
        raise ValueError("SEG-Y 无法精确表示时间轴起点")
    if samples > 65535:
        raise ValueError("SEG-Y 当前格式最多支持 65535 个采样点")
    original_positions = None if trace_positions is None else np.asarray(trace_positions)
    positions = None if original_positions is None else np.asarray(original_positions, dtype=np.float64)
    if positions is not None:
        scaled = positions * 1000
        rounded = np.rint(scaled) / 1000
        if (positions.shape != (ntr,) or not np.all(np.isfinite(positions))
                or np.any(rounded.astype(original_positions.dtype) != original_positions)
                or np.any(np.abs(scaled) > 2147483647)):
            raise ValueError("SEG-Y 坐标无法以毫米精度表示")
    out = Path(destination).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = _text_header(result_name or out.stem, line_id)
    bin_bytes = bytearray(400)
    interval_us = step_ns / 1000.0
    legacy_interval = int(interval_us) if interval_us.is_integer() and interval_us <= 65535 else 0
    struct.pack_into(">H", bin_bytes, 16, legacy_interval)
    struct.pack_into(">H", bin_bytes, 20, samples)
    struct.pack_into(">H", bin_bytes, 24, SAMPLE_FORMAT_FLOAT32)
    struct.pack_into(">d", bin_bytes, 72, interval_us)  # SEG rev 2.1 bytes 3273-3280, microseconds
    struct.pack_into(">H", bin_bytes, 54, 1)  # metres
    bin_bytes[300:302] = bytes((2, 1))
    struct.pack_into(">H", bin_bytes, 302, 1)  # fixed length
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=out.parent, prefix='.' + out.name, suffix='.tmp', delete=False) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.write(bin_bytes)
            for tr in range(ntr):
                trace_header = bytearray(240)
                struct.pack_into(">I", trace_header, 0, tr + 1)
                struct.pack_into(">I", trace_header, 4, tr + 1)
                if positions is not None:
                    struct.pack_into(">h", trace_header, 70, -1000)
                    struct.pack_into(">i", trace_header, 80, int(round(positions[tr] * 1000)))
                    struct.pack_into(">h", trace_header, 88, 1)
                struct.pack_into(">h", trace_header, 108, delay[0])
                struct.pack_into(">h", trace_header, 214, delay[1])
                struct.pack_into(">H", trace_header, 114, samples)
                struct.pack_into(">H", trace_header, 116, legacy_interval)
                handle.write(trace_header)
                handle.write(arr[:, tr].astype(">f4").tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, out)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return out


__all__ = ["write_segy"]

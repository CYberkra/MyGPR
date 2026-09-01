#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEG-Y 导出（任务 F 候选 4 / 需求 C1）：处理成果 → SEG-Y 行业交付格式。

写固定长度 profile：3200B 文本头 + 400B 二进制头（float32 大端，格式码 5）
+ N×(240B 道头 + 样本)。道号/坐标留空由调用方按需补齐。
纯函数、无 Qt 依赖。
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SAMPLE_FORMAT_FLOAT32 = 5


def _text_header(result_name: str, line_id: str) -> bytes:
    lines = [
        f"C 1 MYGPR EXPORTED PROFILE {result_name[:20]}",
        f"C 2 LINE {line_id[:24]}",
        "C 3 SAMPLE FORMAT FLOAT32 BIG ENDIAN",
    ]
    rows = [line.encode("ascii", errors="replace").ljust(80) for line in lines]
    body = b"".join(rows)
    return body + b" " * (3200 - len(body))


def write_segy(
    destination: str | Path,
    matrix: np.ndarray,
    *,
    sample_interval_us: int = 500,
    line_id: str = "",
    result_name: str = "",
    trace_positions: np.ndarray | None = None,
) -> Path:
    """把 samples×traces float32 矩阵写成 SEG-Y 固定长度 profile。"""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"SEG-Y 导出需要非空二维矩阵，当前 shape={arr.shape}")
    samples, ntr = arr.shape
    out = Path(destination).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    text = _text_header(result_name or out.stem, line_id)
    bin_bytes = bytearray(400)
    struct.pack_into(">H", bin_bytes, 16, sample_interval_us)   # sample interval (us)
    struct.pack_into(">H", bin_bytes, 20, samples)              # samples per trace
    struct.pack_into(">H", bin_bytes, 24, SAMPLE_FORMAT_FLOAT32)

    with out.open("wb") as handle:
        handle.write(text)
        handle.write(bytes(bin_bytes))
        for tr in range(ntr):
            trace_header = bytearray(240)
            struct.pack_into(">I", trace_header, 0, tr + 1)   # trace seq
            struct.pack_into(">I", trace_header, 4, tr + 1)   # REEL trace seq
            if trace_positions is not None and tr < len(trace_positions):
                struct.pack_into(">i", trace_header, 84, int(trace_positions[tr] * 1000))
            struct.pack_into(">H", trace_header, 114, samples)
            struct.pack_into(">H", trace_header, 116, sample_interval_us)
            handle.write(bytes(trace_header))
            samples_le = arr[:, tr].astype("<f4")
            handle.write(samples_le.astype(">f4").tobytes())
    return out


__all__ = ["write_segy"]

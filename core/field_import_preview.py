#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight, cancellable import preflight for field measurement files.

Preflight must never materialise a multi-gigabyte matrix.  It inspects headers,
array metadata or a bounded CSV sample and leaves the full two-pass scan to the
transactional background import job.
"""
from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from core.gpr_data_model import detect_mygpr_airborne_sidecar_csv
from core.gpr_format_registry import get_format_spec

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None

DIRECT_IMPORT_EXTENSIONS = {".csv", ".txt", ".npy", ".npz", ".h5", ".hdf5"}
CancelCheck = Callable[[], bool] | None
ProgressCallback = Callable[[str, int, int], None] | None


@dataclass(frozen=True)
class ImportPreflightResult:
    """Result shown to users before importing one measurement file."""

    path: str
    exists: bool
    is_file: bool
    extension: str
    format_name: str
    support: str
    can_import: bool
    message: str
    suggestions: tuple[str, ...] = ()
    sample_count: int = 0
    trace_count: int = 0
    length_m: float = 0.0
    time_window_ns: float = 0.0
    dielectric_constant: float = 0.0
    data_min: float = 0.0
    data_max: float = 0.0
    source_kind: str = ""
    has_trajectory: bool = False
    column_summary: str = ""

    @property
    def shape_text(self) -> str:
        return f"{self.sample_count} × {self.trace_count}" if self.sample_count and self.trace_count else "导入后确定"

    def to_log_lines(self) -> list[str]:
        lines = [
            f"文件：{Path(self.path).name}",
            f"格式：{self.format_name or self.extension or '--'}",
            f"支持状态：{self.support}",
            f"矩阵尺寸：{self.shape_text}",
        ]
        if self.source_kind:
            lines.append(f"数据类型：{self.source_kind}")
        if self.length_m:
            lines.append(f"测线长度：{self.length_m:.2f} m")
        if self.time_window_ns:
            lines.append(f"时间窗：{self.time_window_ns:.2f} ns")
        if self.column_summary:
            lines.append(f"列识别：{self.column_summary}")
        lines.append(f"定位信息：{'已识别' if self.has_trajectory else '未识别'}")
        lines.append(f"结论：{self.message}")
        if self.suggestions:
            lines.append("建议：" + "；".join(self.suggestions))
        return lines


def _check_cancel(cancel_requested: CancelCheck) -> None:
    if cancel_requested is not None and cancel_requested():
        from core.job_manager import JobCancelled
        raise JobCancelled("导入预检已取消")


def _emit(callback: ProgressCallback, stage: str, current: int, total: int) -> None:
    if callback is not None:
        callback(stage, int(current), int(max(total, 1)))


def _sample_minmax(array: np.ndarray, *, max_values: int = 65536) -> tuple[float, float]:
    if array.size <= max_values:
        sample = np.asarray(array, dtype=np.float32).reshape(-1)
    else:
        row_step = max(int(np.ceil(array.shape[0] / 128)), 1)
        col_step = max(int(np.ceil(array.shape[1] / 512)), 1)
        sample = np.asarray(array[::row_step, ::col_step], dtype=np.float32).reshape(-1)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.min(finite)), float(np.max(finite))


def _sample_numeric_csv(
    path: Path,
    *,
    cancel_requested: CancelCheck,
    progress_callback: ProgressCallback,
    max_numeric_rows: int = 4096,
) -> tuple[int, int, float, float, int]:
    """Inspect a bounded CSV sample instead of scanning the entire source."""
    total = max(path.stat().st_size, 1)
    numeric_rows = 0
    min_cols: int | None = None
    data_min = float("inf")
    data_max = float("-inf")
    consumed = 0
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        for row_index, row in enumerate(reader, start=1):
            if row_index % 256 == 0:
                _check_cancel(cancel_requested)
                try:
                    consumed = int(fh.buffer.tell())
                except (AttributeError, OSError, ValueError):
                    consumed = min(total, row_index)
                _emit(progress_callback, "sample_csv", consumed, total)
            values: list[float] = []
            for item in row:
                try:
                    values.append(float(str(item).strip()))
                except (TypeError, ValueError):
                    continue
            if not values:
                continue
            numeric_rows += 1
            min_cols = len(values) if min_cols is None else min(min_cols, len(values))
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                data_min = min(data_min, float(np.min(finite)))
                data_max = max(data_max, float(np.max(finite)))
            if numeric_rows >= max_numeric_rows:
                break
        try:
            consumed = int(fh.buffer.tell())
        except (AttributeError, OSError, ValueError):
            consumed = total if numeric_rows < max_numeric_rows else consumed
    _check_cancel(cancel_requested)
    _emit(progress_callback, "sample_csv", min(consumed or total, total), total)
    return (
        numeric_rows,
        int(min_cols or 0),
        0.0 if data_min == float("inf") else data_min,
        0.0 if data_max == float("-inf") else data_max,
        consumed,
    )


def _npz_matrix_shape(path: Path) -> tuple[tuple[int, ...], str]:
    """Read an embedded NPY header without decompressing the matrix payload."""
    with zipfile.ZipFile(path, "r") as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".npy")]
        if not names:
            raise ValueError("NPZ 内未找到 NPY 数组。")
        preferred = ["matrix.npy", "data.npy", "bscan.npy", "radargram.npy", "arr_0.npy"]
        member = next((name for wanted in preferred for name in names if Path(name).name.lower() == wanted), names[0])
        with archive.open(member, "r") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    return tuple(int(v) for v in shape), str(dtype)


def _hdf5_matrix_shape(path: Path) -> tuple[tuple[int, ...], str, str]:
    if h5py is None:
        raise RuntimeError("h5py 不可用，无法预检 HDF5。")
    found: tuple[tuple[int, ...], str, str] | None = None
    with h5py.File(path, "r") as handle:
        for key in ("matrix", "data", "bscan", "radargram"):
            if key in handle and getattr(handle[key], "ndim", 0) == 2:
                dataset = handle[key]
                return tuple(int(v) for v in dataset.shape), str(dataset.dtype), key

        def visitor(name, obj) -> None:
            nonlocal found
            if found is None and getattr(obj, "ndim", 0) == 2:
                found = (tuple(int(v) for v in obj.shape), str(obj.dtype), str(name))

        handle.visititems(visitor)
    if found is None:
        raise ValueError("HDF5 内未找到二维矩阵数据集。")
    return found


def build_import_preflight(
    source: str | Path,
    *,
    line_id: str = "L01",
    dielectric_constant: float = 9.0,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
) -> ImportPreflightResult:
    del line_id  # identity is validated by the project operation layer
    src = Path(source).expanduser().resolve()
    suffix = src.suffix.lower()
    spec = get_format_spec(src)
    display_name = spec.display_name if spec else (suffix or "未知格式")

    if not src.exists():
        return ImportPreflightResult(str(src), False, False, suffix, display_name, "missing", False, "数据文件不存在。", ("检查文件路径是否正确",))
    if not src.is_file():
        return ImportPreflightResult(str(src), True, False, suffix, display_name, "directory", False, "当前导入入口只支持单个文件，不支持目录。", ("请选择一个 CSV / NPY / NPZ / H5 文件",))

    if suffix in DIRECT_IMPORT_EXTENSIONS:
        try:
            _check_cancel(cancel_requested)
            size = max(src.stat().st_size, 1)
            _emit(progress_callback, "inspect_header", 0, size)
            if suffix in {".csv", ".txt"}:
                sidecar = detect_mygpr_airborne_sidecar_csv(src)
                if sidecar is not None:
                    sample_count = int(sidecar["sample_count"])
                    trace_count = int(sidecar["trace_count"])
                    interval = float(sidecar["trace_interval_m"])
                    _emit(progress_callback, "inspect_header", size, size)
                    return ImportPreflightResult(
                        path=str(src), exists=True, is_file=True, extension=suffix,
                        format_name="mygpr-airborne-sidecar-csv", support="direct", can_import=True,
                        message="可直接导入；已由头信息识别为 MyGPR 航空 GPR 主数据。",
                        suggestions=("完整矩阵解析、轨迹提取和质检将在后台导入任务中执行",),
                        sample_count=sample_count, trace_count=trace_count,
                        length_m=interval * max(trace_count - 1, 1),
                        time_window_ns=float(sidecar["time_window_ns"]), dielectric_constant=float(dielectric_constant),
                        source_kind="MyGPR 航空 GPR 主数据 CSV", has_trajectory=True,
                        column_summary="longitude, latitude, elevation_m, amplitude, height_m, timestamp_s",
                    )
                rows, cols, low, high, consumed = _sample_numeric_csv(
                    src, cancel_requested=cancel_requested, progress_callback=progress_callback,
                )
                if rows < 32 or cols < 16:
                    raise ValueError(f"数值抽样不足以构成 B-scan：rows={rows}, cols={cols}")
                sampled_all = consumed >= size
                return ImportPreflightResult(
                    path=str(src), exists=True, is_file=True, extension=suffix,
                    format_name="csv-matrix", support="direct", can_import=True,
                    message="可直接导入；完整行列数将在后台两遍扫描后确定。" if not sampled_all else "可直接导入；数值矩阵抽样通过。",
                    suggestions=("导入任务支持进度、取消和事务回滚",),
                    sample_count=rows if sampled_all else 0, trace_count=cols if sampled_all else 0,
                    dielectric_constant=float(dielectric_constant), data_min=low, data_max=high,
                    source_kind="二维数值矩阵", has_trajectory=False,
                    column_summary=f"已抽样 {rows} 个数值行，至少 {cols} 列" + ("（完整文件）" if sampled_all else "（有限抽样）"),
                )

            if suffix == ".npy":
                array = np.load(src, mmap_mode="r", allow_pickle=False)
                if array.ndim != 2:
                    raise ValueError(f"GPR 矩阵必须为二维，当前 shape={array.shape}")
                low, high = _sample_minmax(array)
                shape = tuple(int(v) for v in array.shape)
                dtype = str(array.dtype)
                del array
            elif suffix == ".npz":
                shape, dtype = _npz_matrix_shape(src)
                low = high = 0.0
            else:
                shape, dtype, dataset_name = _hdf5_matrix_shape(src)
                low = high = 0.0
                display_name = f"HDF5:{dataset_name}"
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError(f"GPR 矩阵必须为非空二维数组，当前 shape={shape}")
            _emit(progress_callback, "inspect_header", size, size)
            return ImportPreflightResult(
                path=str(src), exists=True, is_file=True, extension=suffix,
                format_name=display_name, support="direct", can_import=True,
                message="可直接导入；预检仅读取数组元数据，没有展开完整矩阵。",
                suggestions=("完整复制、转换和质检将在统一后台任务中执行",),
                sample_count=int(shape[0]), trace_count=int(shape[1]),
                dielectric_constant=float(dielectric_constant), data_min=low, data_max=high,
                source_kind="二维矩阵数据", column_summary=f"dtype={dtype}",
            )
        except Exception as exc:
            from core.job_manager import JobCancelled
            if isinstance(exc, JobCancelled):
                raise
            return ImportPreflightResult(
                path=str(src), exists=True, is_file=True, extension=suffix,
                format_name=display_name, support="direct-failed", can_import=False,
                message=f"文件格式已支持，但轻量预检未通过：{exc}",
                suggestions=("确认文件包含非空二维数值矩阵", "必要时先导出为标准 CSV / NPY / NPZ / H5"),
            )

    if spec is not None:
        return ImportPreflightResult(
            path=str(src), exists=True, is_file=True, extension=suffix,
            format_name=spec.display_name, support=spec.support, can_import=False,
            message=f"已识别 {spec.display_name}，但当前项目向导尚未直接解码该格式。",
            suggestions=(spec.notes or "建议先转换为 CSV / NPY / NPZ / H5 后导入", "后续可作为厂商格式专项适配"),
        )
    return ImportPreflightResult(
        path=str(src), exists=True, is_file=True, extension=suffix,
        format_name=display_name, support="unsupported", can_import=False,
        message=f"暂不支持该数据格式：{suffix or '无扩展名'}。",
        suggestions=("当前正式导入入口支持 CSV / TXT / NPY / NPZ / H5 / HDF5",),
    )


__all__ = ["DIRECT_IMPORT_EXTENSIONS", "ImportPreflightResult", "build_import_preflight"]

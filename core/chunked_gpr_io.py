#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chunked/cancellable matrix IO primitives for large GPR files.

The functions here deliberately avoid Qt so they can be reused by the GUI,
CLI and tests.  Cancellation is cooperative and checked while copying,
scanning, parsing and writing chunks.  A cancelled operation raises
:class:`ImportCancelled`; callers are expected to roll back their project
transaction and remove the staging directory.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Any

import numpy as np

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

CancelCheck = Callable[[], bool] | None
ProgressCallback = Callable[[str, int, int], None] | None

DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_MATRIX_CHUNK_ROWS = 256
LARGE_DATASET_THRESHOLD_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 128
MAX_AXIS_BYTES = 256 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 64 * 1024
MAX_IMPORT_EXPANDED_BYTES = 512 * 1024 * 1024 * 1024
MAX_HDF5_OBJECTS = 100_000


class ImportCancelled(RuntimeError):
    """Raised when the user cancels a running file import."""


@dataclass(frozen=True)
class MatrixLoadResult:
    matrix: np.ndarray
    format_name: str
    staging_files: tuple[Path, ...] = ()
    metadata: dict[str, Any] | None = None


def check_cancel(cancel_requested: CancelCheck) -> None:
    if cancel_requested is not None and bool(cancel_requested()):
        raise ImportCancelled("用户取消了当前文件导入。")


def emit_progress(callback: ProgressCallback, stage: str, current: int, total: int) -> None:
    if callback is not None:
        callback(stage, int(current), int(max(total, 0)))


def copy_file_chunked(
    source: str | Path,
    destination: str | Path,
    *,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Path:
    """Copy a file without loading it into memory, checking cancellation per chunk."""
    src = Path(source)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = src.stat().st_size
    copied = 0
    try:
        with src.open("rb") as rf, dst.open("wb") as wf:
            while True:
                check_cancel(cancel_requested)
                block = rf.read(max(int(chunk_bytes), 64 * 1024))
                if not block:
                    break
                wf.write(block)
                copied += len(block)
                emit_progress(progress_callback, "copy_source", copied, total)
            wf.flush()
            os.fsync(wf.fileno())
        shutil.copystat(src, dst)
        return dst
    except Exception:
        # A cancelled copy must never leave a truncated file that later looks
        # like a valid source artifact.  Higher-level project transactions also
        # roll back their directory, but this local cleanup protects direct API
        # use and abrupt cancellation between transaction checkpoints.
        dst.unlink(missing_ok=True)
        raise


def _numeric_values(row: Iterable[str]) -> list[float]:
    values: list[float] = []
    for item in row:
        try:
            values.append(float(str(item).strip()))
        except (TypeError, ValueError):
            continue
    return values


def scan_numeric_csv(
    path: str | Path,
    *,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
) -> tuple[int, int]:
    """Return numeric row count and common numeric column count using one pass."""
    src = Path(path)
    total = src.stat().st_size
    row_count = 0
    min_cols: int | None = None
    with src.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        for index, row in enumerate(reader, start=1):
            if index % 256 == 0:
                check_cancel(cancel_requested)
                emit_progress(progress_callback, "scan_csv", fh.buffer.tell(), total)
            values = _numeric_values(row)
            if not values:
                continue
            row_count += 1
            min_cols = len(values) if min_cols is None else min(min_cols, len(values))
    check_cancel(cancel_requested)
    emit_progress(progress_callback, "scan_csv", total, total)
    return row_count, int(min_cols or 0)


def load_numeric_csv_to_memmap(
    path: str | Path,
    output_npy: str | Path,
    *,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
) -> np.memmap:
    rows, cols = scan_numeric_csv(
        path,
        cancel_requested=cancel_requested,
        progress_callback=progress_callback,
    )
    if cols < 16 or rows < 32:
        raise ValueError(
            f"CSV numeric content is too small for a B-scan matrix: rows={rows}, cols={cols}, path={path}"
        )
    out_path = Path(output_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(rows, cols))
    total = Path(path).stat().st_size
    written = 0
    try:
        with Path(path).open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                values = _numeric_values(row)
                if not values:
                    continue
                if written % 128 == 0:
                    check_cancel(cancel_requested)
                    emit_progress(progress_callback, "parse_csv", fh.buffer.tell(), total)
                matrix[written, :] = np.asarray(values[:cols], dtype=np.float32)
                written += 1
        if written != rows:
            raise ValueError(f"CSV changed while importing: expected={rows}, actual={written}")
        matrix.flush()
        check_cancel(cancel_requested)
        emit_progress(progress_callback, "parse_csv", total, total)
        return matrix
    except Exception:
        del matrix
        out_path.unlink(missing_ok=True)
        raise


def _copy_array_rows_to_npy(
    source: Any,
    output_npy: str | Path,
    *,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
    stage: str = "copy_matrix",
    chunk_rows: int = DEFAULT_MATRIX_CHUNK_ROWS,
) -> np.memmap:
    if len(source.shape) != 2:
        raise ValueError(f"GPR matrix must be 2D, got shape={source.shape!r}")
    rows, cols = (int(source.shape[0]), int(source.shape[1]))
    out_path = Path(output_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(rows, cols))
    try:
        for start in range(0, rows, max(int(chunk_rows), 1)):
            check_cancel(cancel_requested)
            end = min(rows, start + max(int(chunk_rows), 1))
            output[start:end] = np.asarray(source[start:end], dtype=np.float32)
            emit_progress(progress_callback, stage, end, rows)
        output.flush()
        return output
    except Exception:
        del output
        out_path.unlink(missing_ok=True)
        raise


def load_npy_for_import(
    path: str | Path,
    output_npy: str | Path,
    *,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
) -> np.ndarray:
    """Load/copy an NPY matrix with a file-copy fast path for float32 C arrays."""
    src = Path(path)
    array = np.load(src, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"GPR matrix must be 2D, got shape={array.shape!r}")
    out = Path(output_npy)
    # NPY files already contain the required header.  A byte-for-byte copy is
    # both faster and lower-memory than touching every mmap page.
    if array.dtype == np.float32 and bool(array.flags.c_contiguous):
        del array
        copy_file_chunked(
            src,
            out,
            cancel_requested=cancel_requested,
            progress_callback=progress_callback,
        )
        return np.load(out, mmap_mode="r+", allow_pickle=False)
    return _copy_array_rows_to_npy(
        array,
        out,
        cancel_requested=cancel_requested,
        progress_callback=progress_callback,
        stage="convert_npy",
    )


def _extract_zip_member_chunked(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
    *,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
) -> Path:
    info = archive.getinfo(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = 0
    try:
        with archive.open(member, "r") as rf, destination.open("wb") as wf:
            while True:
                check_cancel(cancel_requested)
                block = rf.read(DEFAULT_CHUNK_BYTES)
                if not block:
                    break
                wf.write(block)
                current += len(block)
                emit_progress(progress_callback, "extract_npz", current, int(info.file_size))
        return destination
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _require_output_space(output_path: str | Path, required_bytes: int) -> None:
    if required_bytes < 0 or required_bytes > MAX_IMPORT_EXPANDED_BYTES:
        raise ValueError(f"导入数据展开大小异常：{required_bytes} bytes")
    parent = Path(output_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    reserve = max(64 * 1024 * 1024, required_bytes // 20)
    if shutil.disk_usage(parent).free < required_bytes + reserve:
        raise OSError(f"磁盘空间不足，至少需要 {required_bytes + reserve} bytes")


def _validated_npz_members(archive: zipfile.ZipFile) -> list[str]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"NPZ 数组成员过多：{len(infos)}")
    names: list[str] = []
    folded: set[str] = set()
    total = 0
    for info in infos:
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts or len(parts) != 1 or not name.lower().endswith(".npy"):
            raise ValueError(f"NPZ 包含非法成员：{info.filename}")
        if info.flag_bits & 0x1:
            raise ValueError(f"NPZ 不支持加密成员：{info.filename}")
        if name in names or name.casefold() in folded:
            raise ValueError(f"NPZ 包含重复或大小写冲突成员：{name}")
        if info.file_size < 0 or info.file_size > MAX_IMPORT_EXPANDED_BYTES:
            raise ValueError(f"NPZ 成员展开大小异常：{name}")
        if Path(name).stem in {"distance_axis_m", "time_axis_ns", "depth_axis_m"} and info.file_size > MAX_AXIS_BYTES + MAX_NPY_HEADER_BYTES:
            raise ValueError(f"NPZ 坐标轴成员过大：{name}")
        names.append(name)
        folded.add(name.casefold())
        total += int(info.file_size)
    if total > MAX_IMPORT_EXPANDED_BYTES:
        raise ValueError(f"NPZ 总展开大小异常：{total} bytes")
    return names


def load_npz_for_import(
    path: str | Path,
    output_npy: str | Path,
    *,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stream the matrix NPY member out of an NPZ, then mmap it.

    Axis arrays are normally tiny relative to the matrix and are loaded through
    NumPy after the matrix has been extracted.
    """
    src = Path(path)
    out_path = Path(output_npy)
    completed = False
    try:
        with zipfile.ZipFile(src, "r") as archive:
            names = _validated_npz_members(archive)
            if not names:
                raise ValueError(f"NPZ does not contain NPY arrays: {src}")
            matrix_member = "matrix.npy" if "matrix.npy" in names else names[0]
            matrix_info = archive.getinfo(matrix_member)
            _require_output_space(out_path, int(matrix_info.file_size))
            _extract_zip_member_chunked(
                archive,
                matrix_member,
                out_path,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
        matrix = np.load(out_path, mmap_mode="r+", allow_pickle=False)
        if matrix.ndim != 2 or matrix.dtype.kind not in "biufc":
            raise ValueError(f"GPR matrix must be a numeric 2D array, got shape={matrix.shape!r}, dtype={matrix.dtype}")
        axes: dict[str, np.ndarray] = {}
        with np.load(src, allow_pickle=False) as npz:
            for key in ("distance_axis_m", "time_axis_ns", "depth_axis_m"):
                if key in npz:
                    axis = np.asarray(npz[key], dtype=np.float32)
                    if axis.ndim != 1 or axis.nbytes > MAX_AXIS_BYTES:
                        raise ValueError(f"NPZ axis is invalid or too large: {key}, shape={axis.shape}")
                    axes[key] = axis
        completed = True
        return matrix, axes
    finally:
        if not completed:
            out_path.unlink(missing_ok=True)


def _first_2d_hdf5_dataset(h5: Any) -> Any:
    objects: list[tuple[str, Any]] = []
    stack: list[tuple[str, Any]] = [("", h5)]
    seen = 0
    while stack:
        prefix, group = stack.pop()
        for key in group.keys():
            seen += 1
            if seen > MAX_HDF5_OBJECTS:
                raise ValueError(f"HDF5 object count exceeds limit: {seen}")
            name = f"{prefix}/{key}".strip("/")
            link = group.get(key, getlink=True)
            if h5py is not None and isinstance(link, (h5py.ExternalLink, h5py.SoftLink)):
                raise ValueError(f"HDF5 external/soft links are not allowed: {name}")
            obj = group.get(key)
            if h5py is not None and isinstance(obj, h5py.Group):
                stack.append((name, obj))
            elif hasattr(obj, "shape"):
                objects.append((name, obj))
    preferred_names = ("matrix", "data", "bscan", "radargram")
    objects.sort(key=lambda item: (item[0].split("/")[-1] not in preferred_names, item[0]))
    for name, obj in objects:
        if len(obj.shape) != 2:
            continue
        dtype = np.dtype(obj.dtype)
        if dtype.kind not in "biufc" or (h5py is not None and h5py.check_dtype(vlen=dtype) is not None):
            raise ValueError(f"HDF5 matrix must use a fixed numeric dtype: {dtype}")
        rows, cols = int(obj.shape[0]), int(obj.shape[1])
        if rows <= 0 or cols <= 0:
            raise ValueError(f"HDF5 matrix shape is invalid: {obj.shape}")
        return obj
    raise ValueError("No 2D matrix dataset found in HDF5 file")


def load_hdf5_for_import(
    path: str | Path,
    output_npy: str | Path,
    *,
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is not available; cannot load HDF5 GPR data")
    out_path = Path(output_npy)
    completed = False
    try:
        with h5py.File(Path(path), "r") as h5:
            dataset = _first_2d_hdf5_dataset(h5)
            required_bytes = int(dataset.shape[0]) * int(dataset.shape[1]) * np.dtype(np.float32).itemsize
            _require_output_space(out_path, required_bytes)
            result = _copy_array_rows_to_npy(
                dataset,
                out_path,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
                stage="copy_hdf5",
            )
            completed = True
            return result
    finally:
        if not completed:
            out_path.unlink(missing_ok=True)


def save_dataset_directory(
    destination: str | Path,
    *,
    matrix: np.ndarray,
    distance_axis_m: np.ndarray,
    time_axis_ns: np.ndarray,
    depth_axis_m: np.ndarray,
    metadata: dict[str, Any],
    cancel_requested: CancelCheck = None,
    progress_callback: ProgressCallback = None,
    source_npy: str | Path | None = None,
) -> Path:
    """Atomically write a directory-backed, mmap-friendly GPR dataset."""
    final_dir = Path(destination)
    staging = final_dir.with_name(f".{final_dir.name}.staging_{os.getpid()}_{id(matrix):x}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        matrix_path = staging / "matrix.npy"
        if source_npy is not None:
            source_path = Path(source_npy)
            source_array = np.load(source_path, mmap_mode="r", allow_pickle=False)
            same_layout = (
                source_array.ndim == 2
                and source_array.dtype == np.float32
                and tuple(source_array.shape) == tuple(matrix.shape)
                and bool(source_array.flags.c_contiguous)
            )
            del source_array
            if same_layout:
                copy_file_chunked(
                    source_path,
                    matrix_path,
                    cancel_requested=cancel_requested,
                    progress_callback=progress_callback,
                )
            else:
                _copy_array_rows_to_npy(
                    matrix,
                    matrix_path,
                    cancel_requested=cancel_requested,
                    progress_callback=progress_callback,
                )
        else:
            _copy_array_rows_to_npy(
                matrix,
                matrix_path,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
        np.save(staging / "distance_axis_m.npy", np.asarray(distance_axis_m, dtype=np.float32), allow_pickle=False)
        np.save(staging / "time_axis_ns.npy", np.asarray(time_axis_ns, dtype=np.float32), allow_pickle=False)
        np.save(staging / "depth_axis_m.npy", np.asarray(depth_axis_m, dtype=np.float32), allow_pickle=False)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        check_cancel(cancel_requested)
        # 先把旧目录改名到备份位再换入 staging，避免"先删后换"在崩溃时
        # 同时失去新旧两份数据；同卷 rename 是原子操作。
        backup_dir: Path | None = None
        if final_dir.exists():
            if final_dir.is_dir():
                backup_dir = final_dir.with_name(
                    f".{final_dir.name}.backup_{os.getpid()}_{id(matrix):x}"
                )
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                final_dir.replace(backup_dir)
            else:
                final_dir.unlink()
        try:
            staging.replace(final_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
                backup_dir.replace(final_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
        return final_dir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_dataset_directory(path: str | Path, *, mmap_mode: str = "r") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    root = Path(path)
    matrix = np.load(root / "matrix.npy", mmap_mode=mmap_mode, allow_pickle=False)
    distance = np.load(root / "distance_axis_m.npy", mmap_mode=mmap_mode, allow_pickle=False)
    time_axis = np.load(root / "time_axis_ns.npy", mmap_mode=mmap_mode, allow_pickle=False)
    depth = np.load(root / "depth_axis_m.npy", mmap_mode=mmap_mode, allow_pickle=False)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return matrix, distance, time_axis, depth, metadata


__all__ = [
    "CancelCheck",
    "ProgressCallback",
    "ImportCancelled",
    "LARGE_DATASET_THRESHOLD_BYTES",
    "check_cancel",
    "copy_file_chunked",
    "load_numeric_csv_to_memmap",
    "load_npy_for_import",
    "load_npz_for_import",
    "load_hdf5_for_import",
    "save_dataset_directory",
    "load_dataset_directory",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-line HDF5 container used by the MyGPR hybrid project store."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from core.gpr_data_model import GPRDataSet, time_to_depth_axis
from core.hdf5_array_proxy import HDF5ArrayProxy
from core.storage_primitives import atomic_output_path, fsync_directory, fsync_file, utc_now

LINE_CONTAINER_SCHEMA = "mygpr.line_container.v1"
RAW_MATRIX_PATH = "/raw/bscan"
RAW_DISTANCE_PATH = "/raw/distance_m"
RAW_TIME_PATH = "/raw/time_ns"
RAW_DEPTH_PATH = "/raw/depth_m"
PROCESSING_ROOT = "/processing/artifacts"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _coerce_float(value: Any) -> float | None:
    """宽容浮点转换（None/非法值返回 None）。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def choose_chunk_shape(shape: tuple[int, int], dtype: np.dtype, *, target_bytes: int = 1024 * 1024) -> tuple[int, int]:
    rows, cols = (max(int(shape[0]), 1), max(int(shape[1]), 1))
    itemsize = max(int(np.dtype(dtype).itemsize), 1)
    target_values = max(target_bytes // itemsize, 1)
    # Preserve trace locality while keeping chunks close to 1 MiB.
    chunk_cols = min(cols, 256)
    chunk_rows = min(rows, max(16, target_values // max(chunk_cols, 1)))
    chunk_rows = max(1, min(rows, int(2 ** math.floor(math.log2(max(chunk_rows, 1))))))
    return chunk_rows, chunk_cols


def _iter_row_blocks(shape: tuple[int, int], block_rows: int):
    rows = int(shape[0])
    for start in range(0, rows, max(int(block_rows), 1)):
        yield start, min(start + max(int(block_rows), 1), rows)


def _write_matrix(
    group: h5py.Group,
    name: str,
    matrix: Any,
    *,
    compression: str = "gzip",
    compression_opts: int = 2,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[h5py.Dataset, str]:
    shape = tuple(int(v) for v in matrix.shape)
    if len(shape) != 2 or not all(v > 0 for v in shape):
        raise ValueError(f"HDF5 matrix must be a non-empty 2D array, got {shape!r}")
    dtype = np.dtype(matrix.dtype)
    if not np.issubdtype(dtype, np.floating) or dtype.itemsize > 4:
        dtype = np.dtype(np.float32)
    chunks = choose_chunk_shape(shape, dtype)
    dataset = group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
        fletcher32=True,
    )
    digest = hashlib.sha256()
    digest.update(str(dtype).encode("utf-8"))
    digest.update(str(shape).encode("utf-8"))
    total = int(math.ceil(shape[0] / chunks[0]))
    for index, (start, end) in enumerate(_iter_row_blocks(shape, chunks[0]), start=1):
        if cancel_requested is not None and cancel_requested():
            from core.job_manager import JobCancelled
            raise JobCancelled("HDF5 数据写入已取消")
        block = np.asarray(matrix[start:end, :], dtype=dtype)
        dataset[start:end, :] = block
        digest.update(np.ascontiguousarray(block).tobytes())
        if progress_callback is not None:
            progress_callback(index, total, f"写入 HDF5 数据块 {index}/{total}")
    return dataset, digest.hexdigest()


def initialize_line_container(path: str | Path, *, project_id: str, line_id: str) -> Path:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with h5py.File(out, "r+", libver="latest") as handle:
            handle.attrs.setdefault("schema", LINE_CONTAINER_SCHEMA)
            handle.attrs.setdefault("project_id", str(project_id))
            handle.attrs.setdefault("line_id", str(line_id))
            handle.attrs["updated_at"] = utc_now()
            for group_name in ("raw", "navigation", "processing", "interpretation", "qc", "provenance", "_staging"):
                handle.require_group(group_name)
            handle.require_group("processing/artifacts")
            handle.require_group("processing/recipes")
            handle.require_group("processing/branches")
            handle.flush()
        fsync_file(out)
        return out
    with atomic_output_path(out, suffix=".h5.tmp") as temporary:
        with h5py.File(temporary, "w", libver="latest") as handle:
            handle.attrs.update(
                schema=LINE_CONTAINER_SCHEMA,
                project_id=str(project_id),
                line_id=str(line_id),
                created_at=utc_now(),
                updated_at=utc_now(),
            )
            for group_name in ("raw", "navigation", "processing", "interpretation", "qc", "provenance", "_staging"):
                handle.require_group(group_name)
            handle.require_group("processing/artifacts")
            handle.require_group("processing/recipes")
            handle.require_group("processing/branches")
            handle.flush()
    return out

def write_raw_dataset(
    path: str | Path,
    dataset: GPRDataSet,
    *,
    project_id: str,
    line_id: str,
    cancel_requested=None,
    progress_callback=None,
) -> tuple[Path, str]:
    """Replace the raw group through fsynced temporary output and atomic rename."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = ""
    with atomic_output_path(out, suffix=".h5.tmp") as temporary:
        with h5py.File(temporary, "w", libver="latest") as handle:
            handle.attrs.update(
                schema=LINE_CONTAINER_SCHEMA,
                project_id=str(project_id),
                line_id=str(line_id),
                created_at=utc_now(),
                updated_at=utc_now(),
            )
            raw = handle.require_group("raw")
            matrix_ds, digest = _write_matrix(
                raw,
                "bscan",
                dataset.matrix,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
            raw.create_dataset(
                "distance_m", data=np.asarray(dataset.distance_axis_m, dtype=np.float32),
                compression="gzip", compression_opts=1, shuffle=True,
            )
            raw.create_dataset(
                "time_ns", data=np.asarray(dataset.time_axis_ns, dtype=np.float32),
                compression="gzip", compression_opts=1, shuffle=True,
            )
            raw.create_dataset(
                "depth_m", data=np.asarray(dataset.depth_axis_m, dtype=np.float32),
                compression="gzip", compression_opts=1, shuffle=True,
            )
            metadata = dataset.to_metadata()
            metadata.update(
                storage_mode="hdf5_line_container",
                matrix_dataset=RAW_MATRIX_PATH,
                data_sha256=digest,
                dtype=str(matrix_ds.dtype),
                chunks=list(matrix_ds.chunks or ()),
            )
            raw.attrs["metadata_json"] = _json(metadata)
            raw.attrs["immutable"] = False
            raw.attrs["source_files_immutable"] = True
            raw.attrs["write_policy"] = "controlled_replace_with_backup"
            raw.attrs["committed_at"] = utc_now()
            preserved_groups = ("navigation", "processing", "interpretation", "qc", "provenance")
            if out.exists():
                with h5py.File(out, "r", libver="latest", swmr=True) as previous:
                    for group_name in preserved_groups:
                        if group_name in previous:
                            previous.copy(group_name, handle, name=group_name)
            for group_name in preserved_groups + ("_staging",):
                handle.require_group(group_name)
            handle.require_group("processing/artifacts")
            handle.require_group("processing/recipes")
            handle.require_group("processing/branches")
            handle.flush()
    return out, digest

def load_raw_dataset(path: str | Path, *, line_id: str) -> GPRDataSet:
    source = Path(path).resolve()
    # 只读消费者且写路径走原子替换（读到的要么旧要么新），不开 SWMR：
    # SWMR reader 对截断损坏文件会无限期等待（复现实证），预览路径必须快失败
    with h5py.File(source, "r", libver="latest") as handle:
        raw = handle["raw"]
        metadata_raw = raw.attrs.get("metadata_json", "{}")
        if isinstance(metadata_raw, bytes):
            metadata_raw = metadata_raw.decode("utf-8")
        try:
            metadata = json.loads(str(metadata_raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            metadata = {}
        distance = np.asarray(raw["distance_m"][...], dtype=np.float32)
        time_axis = np.asarray(raw["time_ns"][...], dtype=np.float32)
        depth = np.asarray(raw["depth_m"][...], dtype=np.float32)
    proxy = HDF5ArrayProxy(source, RAW_MATRIX_PATH)
    return GPRDataSet(
        line_id=line_id,
        matrix=proxy,
        distance_axis_m=distance,
        time_axis_ns=time_axis,
        depth_axis_m=depth,
        source_path=str(source),
        time_window_ns=float(metadata.get("time_window_ns") or (time_axis[-1] if time_axis.size else 250.0)),
        dielectric_constant=float(metadata.get("dielectric_constant") or 9.0),
        format_name=str(metadata.get("format_name") or "mygpr-hdf5"),
        metadata=metadata,
    )


def _verify_written_artifact(path: Path, artifact_id: str, digest: str,
                             shape: tuple[int, ...], dtype: np.dtype) -> None:
    """Re-open a freshly written container and validate the committed artifact.

    fletcher32 chunks are checksum-verified by HDF5 on read, so touching a
    sample block detects torn metadata as well as silent data corruption
    (the "bad version number for layout message" failure mode seen when an
    in-place ``r+`` write is interrupted).  Raises RuntimeError on any
    mismatch so the caller can discard the temporary file.
    """
    final_path = f"{PROCESSING_ROOT}/{artifact_id}"
    with h5py.File(path, "r", libver="latest") as handle:
        if final_path not in handle:
            raise RuntimeError(f"写回验证失败：{final_path} 不存在")
        group = handle[final_path]
        dataset = group["bscan"]
        if tuple(dataset.shape) != tuple(shape):
            raise RuntimeError(
                f"写回验证失败：形状 {dataset.shape} != 预期 {shape}")
        if dataset.dtype != dtype:
            raise RuntimeError(
                f"写回验证失败：dtype {dataset.dtype} != 预期 {dtype}")
        dataset[shape[0] // 2, 0:1]  # fletcher32-verified read of one row
        if str(group.attrs.get("status", "")) != "committed":
            raise RuntimeError("写回验证失败：artifact 状态不是 committed")
        written = _load_json_attr(group, "manifest_json").get("output_data_sha256", "")
        if digest and written != digest:
            raise RuntimeError("写回验证失败：sha256 摘要不匹配")


ARTIFACT_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,160}")


def _validate_artifact_id(artifact_id: str) -> str:
    """Enforce the artifact-id charset shared with the field project adapter.

    artifact_id 直接成为 sidecar 文件名与 HDF5 组名；拒绝空串、路径分隔符
    与 ``..`` 等路径逃逸形态（规则与 field_project_adapter 现行校验一致）。
    """
    text = str(artifact_id or "")
    if ARTIFACT_ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"非法 artifact_id：{text!r}")
    return text


def artifacts_dir_path(path: str | Path) -> Path:
    """Return the per-line sidecar directory sibling to a line container.

    ``data/lines/L01.h5`` → ``data/lines/L01.artifacts/``。与 backend 的
    ``line_artifacts_dir`` 必须解析到同一目录（写路径与 delete/move 共享约定）。
    """
    container = Path(path).resolve()
    return container.parent / f"{container.stem}.artifacts"


def locate_processing_artifact(path: str | Path, artifact_id: str) -> tuple[Path, str]:
    """Locate an artifact's backing file (dual-read: sidecar first, legacy next).

    过渡期兼容两种布局：sidecar ``<stem>.artifacts/<artifact_id>.h5`` 优先，
    其次旧容器内嵌组 ``/processing/artifacts/<artifact_id>``。两处均做可读
    探测（能打开且含 ``bscan``）：sidecar 损坏时回退容器，容器打不开时不
    影响有效 sidecar；两处均不可读才抛 FileNotFoundError。
    返回 ``(file_path, dataset_path)``。
    """
    _validate_artifact_id(artifact_id)
    source = Path(path).resolve()
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    sidecar = artifacts_dir_path(source) / f"{artifact_id}.h5"
    if sidecar.is_file():
        try:
            with h5py.File(sidecar, "r", libver="latest", swmr=True) as handle:
                if f"{group_path}/bscan" in handle:
                    return sidecar, f"{group_path}/bscan"
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            pass
    if source.is_file():
        try:
            with h5py.File(source, "r", libver="latest", swmr=True) as handle:
                if f"{group_path}/bscan" in handle:
                    return source, f"{group_path}/bscan"
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            pass
    raise FileNotFoundError(f"Processing artifact not found: {artifact_id}")
def write_processing_artifact(
    path: str | Path,
    *,
    artifact_id: str,
    matrix: Any,
    manifest: dict[str, Any],
    params: dict[str, Any],
    cancel_requested=None,
    progress_callback=None,
) -> dict[str, Any]:
    """Commit a processing artifact as a per-artifact sidecar file.

    成品矩阵不再写回测线容器（旧实现 copy2 整容器 + ``r+`` 原地改写，保存
    耗时 O(N×容器大小)，且中断曾损坏容器元数据导致整条测线不可读）。现改
    为写入容器同目录 sidecar ``<stem>.artifacts/<artifact_id>.h5``：文件内
    直接建立最终组名 ``/processing/artifacts/<artifact_id>/bscan``，经
    ``atomic_output_path`` 整文件原子发布。容器本体一个字节不动——任何时刻
    中断都绝不影响已提交数据；读取方经 ``locate_processing_artifact`` 双读，
    旧容器内嵌组继续可用，不迁移不重写。

    Sidecar 内部布局与旧容器内嵌布局逐字段一致，``dataset_path`` 字符串
    （URI 的 dataset 部分）完全不变；result 新增 ``h5_file`` 记录实际落盘
    文件（绝对路径），供 backend 把 catalog/URI 的 file 部分指向 sidecar。
    """
    _validate_artifact_id(artifact_id)
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    final_path = f"{PROCESSING_ROOT}/{artifact_id}"
    if processing_artifact_exists(source, artifact_id):
        raise FileExistsError(f"Processing artifact already exists: {artifact_id}")
    sidecar = artifacts_dir_path(source) / f"{artifact_id}.h5"
    result: dict[str, Any]
    with atomic_output_path(sidecar, suffix=".h5.artifact.tmp") as temporary:
        temporary = Path(temporary)
        with h5py.File(temporary, "w", libver="latest") as handle:
            group = handle.require_group(PROCESSING_ROOT).require_group(artifact_id)
            dataset, digest = _write_matrix(
                group,
                "bscan",
                matrix,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
            committed = dict(manifest)
            committed.update(
                artifact_id=artifact_id,
                dataset_path=f"{final_path}/bscan",
                output_data_sha256=digest,
                output_shape=list(dataset.shape),
                output_dtype=str(dataset.dtype),
                committed_at=utc_now(),
            )
            group.attrs["manifest_json"] = _json(committed)
            group.attrs["params_json"] = _json(params)
            group.attrs["status"] = "committed"
            handle.flush()
            result = {
                "artifact_id": artifact_id,
                "dataset_path": f"{final_path}/bscan",
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "sha256": digest,
                "manifest": committed,
                "h5_file": str(sidecar),
            }
        _verify_written_artifact(
            temporary, artifact_id, result["sha256"],
            tuple(result["shape"]), np.dtype(result["dtype"]))
    return result

def delete_processing_artifact(path: str | Path, artifact_id: str) -> bool:
    """Remove an artifact from the legacy container group and/or its sidecar.

    容器仅在确认存在内嵌组时才以 ``r+`` 打开（sidecar-only 场景容器零接触）；
    sidecar 直接 unlink 后 fsync 所在目录。二者任一删除成功即返回 True。
    """
    _validate_artifact_id(artifact_id)
    source = Path(path).resolve()
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    deleted = False
    if source.is_file():
        present = False
        try:
            with h5py.File(source, "r", libver="latest", swmr=True) as handle:
                present = group_path in handle
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            present = False
        if present:
            with h5py.File(source, "r+", libver="latest") as handle:
                if group_path in handle:
                    del handle[group_path]
                    handle.attrs["updated_at"] = utc_now()
                    handle.flush()
                    deleted = True
            fsync_file(source)
    sidecar = artifacts_dir_path(source) / f"{artifact_id}.h5"
    if sidecar.is_file():
        sidecar.unlink()
        fsync_directory(sidecar.parent)
        deleted = True
    return deleted

def processing_artifact_exists(path: str | Path, artifact_id: str) -> bool:
    _validate_artifact_id(artifact_id)
    source = Path(path).resolve()
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    sidecar = artifacts_dir_path(source) / f"{artifact_id}.h5"
    if sidecar.is_file():
        try:
            with h5py.File(sidecar, "r", libver="latest", swmr=True) as handle:
                if f"{group_path}/bscan" in handle:
                    return True
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            pass
    if not source.is_file():
        return False
    try:
        with h5py.File(source, "r", libver="latest", swmr=True) as handle:
            return group_path in handle and f"{group_path}/bscan" in handle
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


def _load_json_attr(group: h5py.Group, name: str) -> dict[str, Any]:
    raw = group.attrs.get(name, "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        value = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def read_raw_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with h5py.File(source, "r", libver="latest", swmr=True) as handle:
        return _load_json_attr(handle["raw"], "metadata_json")


def read_processing_artifact_record(path: str | Path, artifact_id: str) -> dict[str, Any]:
    source = Path(path).resolve()
    artifact_file, _dataset_path = locate_processing_artifact(source, artifact_id)
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    with h5py.File(artifact_file, "r", libver="latest", swmr=True) as handle:
        group = handle[group_path]
        dataset = group["bscan"]
        manifest = _load_json_attr(group, "manifest_json")
        params = _load_json_attr(group, "params_json")
        return {
            "artifact_id": artifact_id,
            "dataset_path": f"{group_path}/bscan",
            "shape": [int(value) for value in dataset.shape],
            "dtype": str(dataset.dtype),
            "sha256": str(manifest.get("output_data_sha256") or ""),
            "manifest": manifest,
            "params": params,
            "status": str(group.attrs.get("status") or ""),
            "h5_file": str(artifact_file),
        }


def compute_dataset_sha256(
    path: str | Path,
    dataset_path: str,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> str:
    """Recompute the canonical array digest without materialising the matrix."""
    source = Path(path).resolve()
    with h5py.File(source, "r", libver="latest", swmr=True) as handle:
        dataset = handle[dataset_path]
        if dataset.ndim != 2:
            raise ValueError(f"HDF5 digest requires a 2D dataset: {dataset_path}")
        shape = tuple(int(value) for value in dataset.shape)
        dtype = np.dtype(dataset.dtype)
        block_rows = int(dataset.chunks[0]) if dataset.chunks else min(shape[0], 1024)
        digest = hashlib.sha256()
        digest.update(str(dtype).encode("utf-8"))
        digest.update(str(shape).encode("utf-8"))
        total = int(math.ceil(shape[0] / max(block_rows, 1)))
        for index, (start, end) in enumerate(_iter_row_blocks(shape, block_rows), start=1):
            if cancel_requested is not None and cancel_requested():
                from core.job_manager import JobCancelled
                raise JobCancelled("HDF5 深度校验已取消")
            block = np.asarray(dataset[start:end, :], dtype=dtype)
            digest.update(np.ascontiguousarray(block).tobytes())
            if progress_callback is not None:
                progress_callback(index, total, f"校验 HDF5 数据块 {index}/{total}")
        return digest.hexdigest()


def list_processing_artifact_ids(path: str | Path) -> list[str]:
    source = Path(path).resolve()
    found: set[str] = set()
    if source.is_file():
        try:
            with h5py.File(source, "r", libver="latest", swmr=True) as handle:
                group = handle.get(PROCESSING_ROOT)
                if group is not None:
                    found.update(str(name) for name in group.keys())
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            pass
    sidecar_dir = artifacts_dir_path(source)
    if sidecar_dir.is_dir():
        found.update(item.stem for item in sidecar_dir.glob("*.h5") if item.is_file())
    return sorted(found)


def read_processing_manifest(path: str | Path, artifact_id: str) -> dict[str, Any]:
    source = Path(path).resolve()
    artifact_file, _dataset_path = locate_processing_artifact(source, artifact_id)
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    with h5py.File(artifact_file, "r", libver="latest", swmr=True) as handle:
        group = handle[group_path]
        raw = group.attrs.get("manifest_json", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def load_processing_dataset(path: str | Path, *, artifact_id: str, raw_dataset: GPRDataSet | None = None) -> GPRDataSet:
    source = Path(path).resolve()
    artifact_file, dataset_path = locate_processing_artifact(source, artifact_id)
    manifest = read_processing_manifest(source, artifact_id)
    proxy = HDF5ArrayProxy(artifact_file, dataset_path)
    input_meta = manifest.get("input_dataset") if isinstance(manifest.get("input_dataset"), dict) else {}
    if raw_dataset is not None:
        distance = raw_dataset.distance_axis_m
        time_axis = raw_dataset.time_axis_ns
        depth = raw_dataset.depth_axis_m
        dielectric = raw_dataset.dielectric_constant
        time_window = raw_dataset.time_window_ns
    else:
        # 轴永远来自容器 raw 组（sidecar 不携带轴）；容器缺失或无 raw 组时
        # 退回 manifest input_meta 默认值，distance 由下方按输出形状重建。
        distance = time_axis = depth = None
        try:
            with h5py.File(source, "r", libver="latest", swmr=True) as handle:
                if "raw" in handle:
                    raw = handle["raw"]
                    if "distance_m" in raw:
                        distance = np.asarray(raw["distance_m"][...], dtype=np.float32)
                    if "time_ns" in raw:
                        time_axis = np.asarray(raw["time_ns"][...], dtype=np.float32)
                    if "depth_m" in raw:
                        depth = np.asarray(raw["depth_m"][...], dtype=np.float32)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            pass
        dielectric = float(input_meta.get("dielectric_constant") or 9.0)
        time_window = float(input_meta.get("time_window_ns")
                            or (time_axis[-1] if time_axis is not None and time_axis.size else 250.0))
        if distance is None:
            distance = np.zeros(0, dtype=np.float32)
        if time_axis is None:
            time_axis = np.zeros(0, dtype=np.float32)
        if depth is None:
            depth = np.zeros(0, dtype=np.float32)
    # P1-1：优先用 manifest 持久化的输出 header 重建物理轴（形状变更方法如
    # time_cut/set_zero_time 会改变时间窗与零点，否则二次处理/成像按原始时窗算错）。
    output_header = manifest.get("output_header") if isinstance(manifest.get("output_header"), dict) else {}
    out_time_window = _coerce_float(output_header.get("total_time_ns") or output_header.get("time_window_ns"))
    out_offset = _coerce_float(output_header.get("time_cut_offset_ns")) or 0.0
    if out_time_window and out_time_window > 0:
        time_window = out_time_window
    sample_count = int(getattr(proxy, "shape", (0, 0))[0])
    if output_header and sample_count > 0:
        time_axis = out_offset + np.linspace(0.0, time_window, sample_count, dtype=np.float32)
        depth = time_to_depth_axis(time_axis, dielectric)
        if len(distance) != int(getattr(proxy, "shape", (0, 0))[1]):
            distance = np.linspace(0.0, max(float(int(getattr(proxy, "shape", (0, 0))[1]) - 1), 1.0),
                                   int(getattr(proxy, "shape", (0, 0))[1]), dtype=np.float32)
    # Shape-changing processing nodes may define explicit axes in future.  Until
    # then GPRDataSet rebuilds a mismatched axis rather than returning invalid lengths.
    return GPRDataSet(
        line_id=str(manifest.get("line_id") or ""),
        matrix=proxy,
        distance_axis_m=np.asarray(distance, dtype=np.float32),
        time_axis_ns=np.asarray(time_axis, dtype=np.float32),
        depth_axis_m=np.asarray(depth, dtype=np.float32),
        source_path=str(source),
        time_window_ns=time_window,
        dielectric_constant=dielectric,
        format_name=f"processed-hdf5:{manifest.get('method_id') or artifact_id}",
        metadata={"processing_manifest": manifest},
    )


def validate_line_container(path: str | Path, *, project_id: str | None = None, line_id: str | None = None) -> list[str]:
    issues: list[str] = []
    source = Path(path)
    if not source.exists():
        return [f"missing container: {source}"]
    try:
        with h5py.File(source, "r", libver="latest", swmr=True) as handle:
            if str(handle.attrs.get("schema") or "") != LINE_CONTAINER_SCHEMA:
                issues.append("invalid line container schema")
            if project_id and str(handle.attrs.get("project_id") or "") != str(project_id):
                issues.append("project_id mismatch")
            if line_id and str(handle.attrs.get("line_id") or "") != str(line_id):
                issues.append("line_id mismatch")
            for dataset_path in (RAW_MATRIX_PATH, RAW_DISTANCE_PATH, RAW_TIME_PATH, RAW_DEPTH_PATH):
                if dataset_path not in handle:
                    issues.append(f"missing dataset {dataset_path}")
            if RAW_MATRIX_PATH in handle:
                matrix = handle[RAW_MATRIX_PATH]
                if matrix.ndim != 2 or matrix.dtype.kind not in "f":
                    issues.append("raw bscan must be a 2D floating dataset")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        issues.append(f"cannot open container: {exc}")
    return issues


__all__ = [
    "LINE_CONTAINER_SCHEMA",
    "PROCESSING_ROOT",
    "RAW_MATRIX_PATH",
    "artifacts_dir_path",
    "choose_chunk_shape",
    "compute_dataset_sha256",
    "delete_processing_artifact",
    "initialize_line_container",
    "list_processing_artifact_ids",
    "load_processing_dataset",
    "load_raw_dataset",
    "locate_processing_artifact",
    "processing_artifact_exists",
    "read_processing_artifact_record",
    "read_processing_manifest",
    "read_raw_metadata",
    "validate_line_container",
    "write_processing_artifact",
    "write_raw_dataset",
]

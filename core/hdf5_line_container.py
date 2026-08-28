#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-line HDF5 container used by the MyGPR hybrid project store."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from core.gpr_data_model import GPRDataSet, time_to_depth_axis
from core.hdf5_array_proxy import HDF5ArrayProxy
from core.storage_primitives import atomic_output_path, fsync_file, utc_now

LINE_CONTAINER_SCHEMA = "mygpr.line_container.v1"
RAW_MATRIX_PATH = "/raw/bscan"
RAW_DISTANCE_PATH = "/raw/distance_m"
RAW_TIME_PATH = "/raw/time_ns"
RAW_DEPTH_PATH = "/raw/depth_m"
PROCESSING_ROOT = "/processing/artifacts"
STAGING_ROOT = "/_staging"


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
    with h5py.File(source, "r", libver="latest", swmr=True) as handle:
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
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    staging_id = uuid.uuid4().hex
    staging_path = f"{STAGING_ROOT}/processing_{staging_id}"
    final_path = f"{PROCESSING_ROOT}/{artifact_id}"
    result: dict[str, Any]
    with h5py.File(source, "r+", libver="latest") as handle:
        if final_path in handle:
            raise FileExistsError(f"Processing artifact already exists: {artifact_id}")
        staging = handle.require_group(staging_path)
        try:
            dataset, digest = _write_matrix(
                staging,
                "bscan",
                matrix,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
            staged_manifest = dict(manifest)
            staged_manifest.update(
                artifact_id=artifact_id,
                dataset_path=f"{final_path}/bscan",
                output_data_sha256=digest,
                output_shape=list(dataset.shape),
                output_dtype=str(dataset.dtype),
                committed_at=utc_now(),
            )
            staging.attrs["manifest_json"] = _json(staged_manifest)
            staging.attrs["params_json"] = _json(params)
            staging.attrs["status"] = "staged"
            handle.flush()
            handle.move(staging_path, final_path)
            final = handle[final_path]
            final.attrs["status"] = "committed"
            handle.attrs["updated_at"] = utc_now()
            handle.flush()
            result = {
                "artifact_id": artifact_id,
                "dataset_path": f"{final_path}/bscan",
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "sha256": digest,
                "manifest": staged_manifest,
            }
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            if staging_path in handle:
                del handle[staging_path]
                handle.flush()
            raise
    fsync_file(source)
    return result

def delete_processing_artifact(path: str | Path, artifact_id: str) -> bool:
    """Remove an uncommitted/orphan processing group after catalog failure."""
    source = Path(path).resolve()
    if not source.exists():
        return False
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    deleted = False
    with h5py.File(source, "r+", libver="latest") as handle:
        if group_path in handle:
            del handle[group_path]
            handle.attrs["updated_at"] = utc_now()
            handle.flush()
            deleted = True
    if deleted:
        fsync_file(source)
    return deleted

def processing_artifact_exists(path: str | Path, artifact_id: str) -> bool:
    source = Path(path).resolve()
    if not source.exists():
        return False
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
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
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    with h5py.File(source, "r", libver="latest", swmr=True) as handle:
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
    if not source.exists():
        return []
    with h5py.File(source, "r", libver="latest", swmr=True) as handle:
        group = handle.get(PROCESSING_ROOT)
        return sorted(str(name) for name in group.keys()) if group is not None else []


def read_processing_manifest(path: str | Path, artifact_id: str) -> dict[str, Any]:
    source = Path(path).resolve()
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    with h5py.File(source, "r", libver="latest", swmr=True) as handle:
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
    dataset_path = f"{PROCESSING_ROOT}/{artifact_id}/bscan"
    manifest = read_processing_manifest(source, artifact_id)
    proxy = HDF5ArrayProxy(source, dataset_path)
    input_meta = manifest.get("input_dataset") if isinstance(manifest.get("input_dataset"), dict) else {}
    if raw_dataset is not None:
        distance = raw_dataset.distance_axis_m
        time_axis = raw_dataset.time_axis_ns
        depth = raw_dataset.depth_axis_m
        dielectric = raw_dataset.dielectric_constant
        time_window = raw_dataset.time_window_ns
    else:
        with h5py.File(source, "r", libver="latest", swmr=True) as handle:
            raw = handle["raw"]
            distance = np.asarray(raw["distance_m"][...], dtype=np.float32)
            time_axis = np.asarray(raw["time_ns"][...], dtype=np.float32)
            depth = np.asarray(raw["depth_m"][...], dtype=np.float32)
        dielectric = float(input_meta.get("dielectric_constant") or 9.0)
        time_window = float(input_meta.get("time_window_ns") or (time_axis[-1] if time_axis.size else 250.0))
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
    "choose_chunk_shape",
    "compute_dataset_sha256",
    "initialize_line_container",
    "load_processing_dataset",
    "load_raw_dataset",
    "processing_artifact_exists",
    "read_processing_artifact_record",
    "read_processing_manifest",
    "read_raw_metadata",
    "validate_line_container",
    "write_processing_artifact",
    "write_raw_dataset",
    "delete_processing_artifact",
    "list_processing_artifact_ids",
]

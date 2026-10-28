"""Integrity-protected HDF5 trace snapshots."""
from __future__ import annotations

import hashlib
import h5py
import numpy as np


def read_trace_snapshot(group: h5py.Group) -> dict[str, np.ndarray] | None:
    if "trace_metadata" not in group:
        return None
    result = {}
    snapshot = group["trace_metadata"]
    for key, dataset in snapshot.items():
        if not isinstance(snapshot.get(key, getlink=True), h5py.HardLink) or not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"无效逐道元数据: {key}")
        values = dataset[...]
        if "unicode_dtype" in dataset.attrs:
            values = np.char.decode(values, "utf-8").astype(str(dataset.attrs["unicode_dtype"]))
        if hashlib.sha256(values.tobytes()).hexdigest() != dataset.attrs.get("sha256", ""):
            raise ValueError(f"成果逐道元数据校验失败: {key}")
        result[key] = values
    return result


def write_trace_snapshot(group: h5py.Group, trace_metadata: dict, trace_count: int) -> None:
    snapshot = group.create_group("trace_metadata")
    for key, value in trace_metadata.items():
        values = np.asarray(value)
        if not key or "/" in key or values.ndim != 1 or len(values) != trace_count:
            raise ValueError(f"无效逐道元数据: {key}")
        stored = np.char.encode(values, "utf-8") if values.dtype.kind == "U" else values
        entry = snapshot.create_dataset(key, data=stored, fletcher32=True, chunks=True)
        if values.dtype.kind == "U":
            entry.attrs["unicode_dtype"] = values.dtype.str
        entry.attrs["sha256"] = hashlib.sha256(values.tobytes()).hexdigest()

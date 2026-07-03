#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight GPR B-scan data contract used by the field workbench.

Round 4 deliberately keeps this layer small and dependency-light: GUI code asks
for a :class:`GPRDataSet`, while file-format adapters can be swapped or extended
later for vendor formats such as DZT/RD3/DT1.  CSV, NPY and H5/HDF5 are supported
now so the product workflow can move beyond synthetic screenshots.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # optional, only needed for .h5/.hdf5
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    h5py = None


GPR_DATA_SCHEMA = "mygpr.gpr_dataset.v1"


@dataclass
class GPRDataSet:
    """Normalized in-memory representation of one B-scan line."""

    line_id: str
    matrix: np.ndarray
    distance_axis_m: np.ndarray
    time_axis_ns: np.ndarray
    depth_axis_m: np.ndarray
    source_path: str = ""
    sample_count: int = 0
    trace_count: int = 0
    time_window_ns: float = 250.0
    dielectric_constant: float = 9.0
    format_name: str = "memory"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"GPR matrix must be 2D, got shape={matrix.shape!r}")
        # Convention: rows = samples/time/depth, cols = traces/distance.
        self.matrix = matrix
        self.sample_count, self.trace_count = matrix.shape
        if len(self.distance_axis_m) != self.trace_count:
            self.distance_axis_m = np.linspace(0.0, max(float(self.trace_count - 1), 1.0), self.trace_count, dtype=np.float32)
        if len(self.time_axis_ns) != self.sample_count:
            self.time_axis_ns = np.linspace(0.0, float(self.time_window_ns), self.sample_count, dtype=np.float32)
        if len(self.depth_axis_m) != self.sample_count:
            self.depth_axis_m = time_to_depth_axis(self.time_axis_ns, self.dielectric_constant)

    @property
    def length_m(self) -> float:
        if self.distance_axis_m.size == 0:
            return 0.0
        return float(self.distance_axis_m[-1] - self.distance_axis_m[0])

    @property
    def normalized_matrix(self) -> np.ndarray:
        matrix = self.matrix.astype(np.float32, copy=False)
        scale = float(np.nanpercentile(np.abs(matrix), 99.5)) if matrix.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        return np.clip((matrix - float(np.nanmean(matrix))) / scale, -1.0, 1.0)

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("matrix", None)
        payload.pop("distance_axis_m", None)
        payload.pop("time_axis_ns", None)
        payload.pop("depth_axis_m", None)
        payload["schema"] = GPR_DATA_SCHEMA
        payload["length_m"] = self.length_m
        return payload

    def save_npz(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            matrix=self.matrix,
            distance_axis_m=self.distance_axis_m,
            time_axis_ns=self.time_axis_ns,
            depth_axis_m=self.depth_axis_m,
            metadata=np.array(json.dumps(self.to_metadata(), ensure_ascii=False)),
        )
        return out

    @classmethod
    def from_matrix(
        cls,
        line_id: str,
        matrix: np.ndarray,
        *,
        length_m: float | None = None,
        time_window_ns: float = 250.0,
        dielectric_constant: float = 9.0,
        source_path: str = "",
        format_name: str = "memory",
        metadata: dict[str, Any] | None = None,
    ) -> "GPRDataSet":
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"GPR matrix must be 2D, got shape={arr.shape!r}")
        rows, cols = arr.shape
        length = float(length_m if length_m is not None else max(cols - 1, 1))
        distance_axis_m = np.linspace(0.0, length, cols, dtype=np.float32)
        time_axis_ns = np.linspace(0.0, float(time_window_ns), rows, dtype=np.float32)
        depth_axis_m = time_to_depth_axis(time_axis_ns, dielectric_constant)
        return cls(
            line_id=line_id,
            matrix=arr,
            distance_axis_m=distance_axis_m,
            time_axis_ns=time_axis_ns,
            depth_axis_m=depth_axis_m,
            source_path=source_path,
            time_window_ns=float(time_window_ns),
            dielectric_constant=float(dielectric_constant),
            format_name=format_name,
            metadata=metadata or {},
        )

    @classmethod
    def synthetic(cls, line_id: str = "L03", *, rows: int = 240, cols: int = 420, length_m: float = 212.35, seed: int = 14) -> "GPRDataSet":
        """Create a deterministic radargram-like dataset for demos/tests."""
        rng = np.random.default_rng(seed)
        y = np.arange(rows, dtype=np.float32)[:, None]
        x = np.arange(cols, dtype=np.float32)[None, :]
        yy = y / max(rows - 1, 1)
        data = 0.12 * rng.normal(size=(rows, cols)).astype(np.float32)
        data += 0.24 * np.sin(0.26 * y + 0.015 * x) * np.exp(-yy * 1.3)
        data += 0.13 * np.cos(0.55 * y + 0.025 * np.sin(x / 14.0)) * np.exp(-yy * 2.0)
        for depth, amp, freq in [(18, 0.78, 1.15), (32, 0.42, 0.9), (47, 0.25, 0.8)]:
            data += amp * np.exp(-((y - depth) ** 2) / (2 * 5.0**2)) * np.cos(freq * (y - depth) + 0.025 * x)
        for cx, cy, sigma, amp in [(42, 64, 8, 1.2), (122, 78, 8, 1.1), (203, 67, 10, 1.4), (302, 88, 8, 1.0), (370, 97, 7, 0.8)]:
            curve = cy + 0.010 * (x - cx) ** 2
            aperture = np.exp(-((x - cx) ** 2) / (2 * (cols * 0.15) ** 2))
            data += amp * np.exp(-((y - curve) ** 2) / (2 * sigma**2)) * np.cos((y - curve) * 1.25) * aperture
        data -= np.mean(data)
        return cls.from_matrix(
            line_id,
            data,
            length_m=length_m,
            time_window_ns=250.0,
            dielectric_constant=9.0,
            format_name="synthetic-demo",
            metadata={"generator": "GPRDataSet.synthetic", "seed": seed},
        )


def time_to_depth_axis(time_axis_ns: np.ndarray, dielectric_constant: float) -> np.ndarray:
    """Convert two-way travel time in ns to depth in meters."""
    c_m_per_ns = 0.299792458
    eps = max(float(dielectric_constant), 1.0)
    return (np.asarray(time_axis_ns, dtype=np.float32) * c_m_per_ns / (2.0 * np.sqrt(eps))).astype(np.float32)



# Normalized header key mapping for legacy YingShan airborne CSV files.
SIDE_CAR_HEADER_KEYS = {
    "number of samples": "sample_count",
    "time windows (ns)": "time_window_ns",
    "number of traces": "trace_count",
    "trace interval (m)": "trace_interval_m",
}


def _first_float_from_text(value: str) -> float:
    """Extract the first numeric token from a CSV header value.

    Real YingShan measurement files export header rows like
    ``Number of Samples = 501,,,,``.  ``float("501,,,,")`` fails,
    but the row is still a valid legacy MyGPR sidecar header.
    """
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    if not match:
        raise ValueError(value)
    return float(match.group(0))


def _parse_mygpr_sidecar_header(path: Path) -> tuple[dict[str, float], int]:
    """Parse the legacy MyGPR airborne sidecar CSV header.

    Expected first lines are for example:
      Number of Samples = 160
      Time windows (ns) = 95.000000
      Number of Traces = 180
      Trace interval (m) = 0.343213
    """
    header: dict[str, float] = {}
    header_lines = 0
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                header_lines += 1
                continue
            if "=" not in line:
                break
            key, value = [part.strip() for part in line.split("=", 1)]
            norm = key.lower()
            mapped = SIDE_CAR_HEADER_KEYS.get(norm)
            if mapped is None:
                break
            try:
                header[mapped] = _first_float_from_text(value)
            except ValueError:
                break
            header_lines += 1
            if {"sample_count", "time_window_ns", "trace_count", "trace_interval_m"}.issubset(header):
                break
    return header, header_lines


def detect_mygpr_airborne_sidecar_csv(path: str | Path) -> dict[str, float] | None:
    """Return parsed header if *path* is the legacy airborne MyGPR main CSV."""
    src = Path(path)
    header, _header_lines = _parse_mygpr_sidecar_header(src)
    required = {"sample_count", "time_window_ns", "trace_count", "trace_interval_m"}
    if required.issubset(header):
        sample_count = int(header["sample_count"])
        trace_count = int(header["trace_count"])
        if sample_count > 0 and trace_count > 0:
            return header
    return None


def _load_mygpr_airborne_sidecar_csv(path: Path, *, line_id: str, dielectric_constant: float = 9.0) -> GPRDataSet:
    header, header_lines = _parse_mygpr_sidecar_header(path)
    if not {"sample_count", "time_window_ns", "trace_count", "trace_interval_m"}.issubset(header):
        raise ValueError(f"Not a MyGPR airborne sidecar CSV: {path}")
    sample_count = int(header["sample_count"])
    trace_count = int(header["trace_count"])
    time_window_ns = float(header["time_window_ns"])
    trace_interval_m = float(header["trace_interval_m"])
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        for _ in range(header_lines):
            next(reader, None)
        for row in reader:
            vals: list[float] = []
            for item in row:
                try:
                    vals.append(float(str(item).strip()))
                except (TypeError, ValueError):
                    vals.append(float("nan"))
            if len(vals) >= 4:
                rows.append(vals[:6])
    expected = sample_count * trace_count
    if len(rows) < expected:
        raise ValueError(
            f"MyGPR sidecar CSV has insufficient data rows: expected={expected}, actual={len(rows)}, path={path}"
        )
    if len(rows) > expected:
        rows = rows[:expected]
    data = np.asarray(rows, dtype=np.float32)
    # Raw order in legacy files is trace-major: each trace contributes
    # sample_count consecutive rows. Convert to rows=samples, cols=traces.
    amplitude = data[:, 3].reshape(trace_count, sample_count).T
    length_m = trace_interval_m * max(trace_count - 1, 1)
    distance_axis_m = np.linspace(0.0, length_m, trace_count, dtype=np.float32)
    time_axis_ns = np.linspace(0.0, time_window_ns, sample_count, dtype=np.float32)
    depth_axis_m = time_to_depth_axis(time_axis_ns, dielectric_constant)
    # One trajectory point per trace, sampled from the first sample in each trace.
    trace_rows = data.reshape(trace_count, sample_count, data.shape[1])[:, 0, :]
    trajectory_rows = []
    for idx, row in enumerate(trace_rows):
        trajectory_rows.append(
            {
                "distance_m": float(distance_axis_m[idx]),
                "longitude": float(row[0]),
                "latitude": float(row[1]),
                "elevation": float(row[2]),
                "height_m": float(row[4]) if data.shape[1] > 4 else 0.0,
                "timestamp_s": float(row[5]) if data.shape[1] > 5 else float(idx),
                "quality": "未知",
            }
        )
    columns = ["longitude", "latitude", "elevation_m", "amplitude"]
    if data.shape[1] > 4:
        columns.append("height_m")
    if data.shape[1] > 5:
        columns.append("timestamp_s")
    metadata = {
        "source_name": path.name,
        "source_format": "mygpr_airborne_sidecar_csv",
        "header_lines": header_lines,
        "trace_interval_m": trace_interval_m,
        "columns": columns,
        "trajectory_rows": trajectory_rows,
        "raw_row_count": len(rows),
        "data_column_count": int(data.shape[1]),
    }
    ds = GPRDataSet(
        line_id=line_id,
        matrix=amplitude,
        distance_axis_m=distance_axis_m,
        time_axis_ns=time_axis_ns,
        depth_axis_m=depth_axis_m,
        source_path=str(path),
        time_window_ns=time_window_ns,
        dielectric_constant=dielectric_constant,
        format_name="mygpr-airborne-sidecar-csv",
        metadata=metadata,
    )
    return ds


def _load_numeric_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            vals: list[float] = []
            for item in row:
                try:
                    vals.append(float(str(item).strip()))
                except (TypeError, ValueError):
                    pass
            if vals:
                rows.append(vals)
    if not rows:
        raise ValueError(f"CSV does not contain numeric matrix data: {path}")
    min_len = min(len(row) for row in rows)
    if min_len < 16 or len(rows) < 32:
        raise ValueError(f"CSV numeric content is too small for a B-scan matrix: rows={len(rows)}, cols={min_len}, path={path}")
    return np.asarray([row[:min_len] for row in rows], dtype=np.float32)


def load_gpr_dataset(path: str | Path, *, line_id: str = "L03", length_m: float | None = None, dielectric_constant: float = 9.0) -> GPRDataSet:
    """Load a normalized dataset from CSV, NPY, NPZ or H5/HDF5."""
    src = Path(path).resolve()
    suffix = src.suffix.lower()
    if suffix in {".csv", ".txt"}:
        sidecar_header = detect_mygpr_airborne_sidecar_csv(src)
        if sidecar_header is not None:
            return _load_mygpr_airborne_sidecar_csv(src, line_id=line_id, dielectric_constant=dielectric_constant)
        matrix = _load_numeric_csv(src)
        fmt = "csv-matrix"
    elif suffix == ".npy":
        matrix = np.load(src)
        fmt = "npy-matrix"
    elif suffix == ".npz":
        npz = np.load(src, allow_pickle=False)
        matrix = npz["matrix"] if "matrix" in npz else npz[npz.files[0]]
        length_axis = npz["distance_axis_m"] if "distance_axis_m" in npz else None
        time_axis = npz["time_axis_ns"] if "time_axis_ns" in npz else None
        depth_axis = npz["depth_axis_m"] if "depth_axis_m" in npz else None
        ds = GPRDataSet.from_matrix(
            line_id,
            matrix,
            length_m=float(length_axis[-1]) if length_axis is not None and len(length_axis) else length_m,
            dielectric_constant=dielectric_constant,
            source_path=str(src),
            format_name="npz-dataset",
        )
        if length_axis is not None:
            ds.distance_axis_m = np.asarray(length_axis, dtype=np.float32)
        if time_axis is not None:
            ds.time_axis_ns = np.asarray(time_axis, dtype=np.float32)
        if depth_axis is not None:
            ds.depth_axis_m = np.asarray(depth_axis, dtype=np.float32)
        return ds
    elif suffix in {".h5", ".hdf5"}:
        if h5py is None:
            raise RuntimeError("h5py is not available; cannot load HDF5 GPR data")
        with h5py.File(src, "r") as h5:
            for key in ("matrix", "data", "bscan", "radargram"):
                if key in h5:
                    matrix = h5[key][()]
                    break
            else:
                # first 2D dataset in the file
                matrix = None
                def visitor(_name: str, obj: Any) -> None:
                    nonlocal matrix
                    if matrix is None and hasattr(obj, "shape") and len(obj.shape) == 2:
                        matrix = obj[()]
                h5.visititems(visitor)
                if matrix is None:
                    raise ValueError(f"No 2D matrix dataset found in {src}")
        fmt = "hdf5-matrix"
    else:
        raise ValueError(f"Unsupported GPR data format: {src.suffix}")
    return GPRDataSet.from_matrix(
        line_id,
        np.asarray(matrix, dtype=np.float32),
        length_m=length_m,
        dielectric_constant=dielectric_constant,
        source_path=str(src),
        format_name=fmt,
        metadata={"source_name": src.name},
    )


__all__ = ["GPR_DATA_SCHEMA", "GPRDataSet", "detect_mygpr_airborne_sidecar_csv", "load_gpr_dataset", "time_to_depth_axis"]

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
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

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
    matrix: Any
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
        # Preserve float32 ndarrays and ``np.memmap`` objects.  The former
        # implementation coerced every dataset to float64, doubling memory and
        # materialising mmap-backed files during import.
        raw_matrix = self.matrix
        is_lazy_array = (
            not isinstance(raw_matrix, np.ndarray)
            and hasattr(raw_matrix, "shape")
            and hasattr(raw_matrix, "dtype")
            and hasattr(raw_matrix, "__getitem__")
        )
        matrix = raw_matrix if is_lazy_array else np.asanyarray(raw_matrix)
        if int(getattr(matrix, "ndim", len(getattr(matrix, "shape", ())))) != 2:
            raise ValueError(f"GPR matrix must be 2D, got shape={getattr(matrix, 'shape', None)!r}")
        dtype = np.dtype(matrix.dtype)
        if not np.issubdtype(dtype, np.floating) or dtype.itemsize > 4:
            # Lazy stores are expected to expose float32 data.  Unsupported lazy
            # dtypes are converted only when explicitly materialised by callers.
            matrix = np.asarray(matrix, dtype=np.float32)
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

    @staticmethod
    def _normalize_array(matrix: np.ndarray) -> np.ndarray:
        array = np.asarray(matrix, dtype=np.float32)
        scale = float(np.nanpercentile(np.abs(array), 99.5)) if array.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        center = float(np.nanmean(array)) if array.size else 0.0
        return np.clip((array - center) / scale, -1.0, 1.0)

    @property
    def normalized_matrix(self) -> np.ndarray:
        """Return the complete normalized matrix.

        UI code should prefer :meth:`normalized_preview` for large files.
        This property is retained for algorithm compatibility.
        """
        return self._normalize_array(self.matrix)

    def preview_matrix(self, *, max_samples: int = 900, max_traces: int = 1800) -> np.ndarray:
        """Return a bounded strided view/copy suitable for interactive plots."""
        rows, cols = self.matrix.shape
        row_step = max(1, int(np.ceil(rows / max(max_samples, 1))))
        col_step = max(1, int(np.ceil(cols / max(max_traces, 1))))
        return np.asarray(self.matrix[::row_step, ::col_step], dtype=np.float32)

    def normalized_preview(self, *, max_samples: int = 900, max_traces: int = 1800) -> np.ndarray:
        return self._normalize_array(self.preview_matrix(max_samples=max_samples, max_traces=max_traces))

    def preview_window(
        self,
        *,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
        max_samples: int = 900,
        max_traces: int = 1800,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a bounded preview for a requested viewport.

        The function slices mmap-backed data before downsampling, so zoomed views
        retain full local resolution without materialising the complete B-scan.
        """
        s0 = int(np.clip(sample_start, 0, max(self.sample_count - 1, 0)))
        s1 = self.sample_count if sample_end is None else int(np.clip(sample_end, s0 + 1, self.sample_count))
        t0 = int(np.clip(trace_start, 0, max(self.trace_count - 1, 0)))
        t1 = self.trace_count if trace_end is None else int(np.clip(trace_end, t0 + 1, self.trace_count))
        row_step = max(1, int(np.ceil((s1 - s0) / max(max_samples, 1))))
        col_step = max(1, int(np.ceil((t1 - t0) / max(max_traces, 1))))
        matrix = np.asarray(self.matrix[s0:s1:row_step, t0:t1:col_step], dtype=np.float32)
        if normalize:
            matrix = self._normalize_array(matrix)
        sample_indices = np.arange(s0, s1, row_step, dtype=np.int64)[: matrix.shape[0]]
        trace_indices = np.arange(t0, t1, col_step, dtype=np.int64)[: matrix.shape[1]]
        return matrix, sample_indices, trace_indices

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("matrix", None)
        payload.pop("distance_axis_m", None)
        payload.pop("time_axis_ns", None)
        payload.pop("depth_axis_m", None)
        payload["schema"] = GPR_DATA_SCHEMA
        payload["length_m"] = self.length_m
        return payload

    def save_npz(
        self,
        path: str | Path,
        *,
        cancel_checker: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        """Atomically save a compact dataset without replacing a valid file early."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.{next(tempfile._get_candidate_names())}.tmp.npz")

        def check_cancel() -> None:
            if cancel_checker is not None and cancel_checker():
                from core.job_manager import JobCancelled
                raise JobCancelled("数据保存已取消")

        try:
            check_cancel()
            if progress_callback is not None:
                progress_callback(0, 1, "压缩数据集")
            np.savez_compressed(
                tmp,
                matrix=self.matrix,
                distance_axis_m=self.distance_axis_m,
                time_axis_ns=self.time_axis_ns,
                depth_axis_m=self.depth_axis_m,
                metadata=np.array(json.dumps(self.to_metadata(), ensure_ascii=False)),
            )
            check_cancel()
            tmp.replace(out)
            if progress_callback is not None:
                progress_callback(1, 1, "数据集已保存")
            return out
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

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
        arr = np.asanyarray(matrix)
        if not np.issubdtype(arr.dtype, np.floating) or arr.dtype.itemsize > 4:
            arr = arr.astype(np.float32, copy=False)
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


    @classmethod
    def synthetic_basal_interface(
        cls,
        line_id: str = "L03",
        *,
        rows: int = 240,
        cols: int = 420,
        length_m: float = 212.35,
        seed: int = 25,
    ) -> "GPRDataSet":
        """Create a deterministic landslide-style basal-interface radargram.

        The preview deliberately avoids pipe-like diffraction hyperbolas.  It
        contains shallow ringing, layered cover, an undulating basal/bedrock
        boundary, a weak-visibility interval and a texturally distinct lower
        unit so the annotation workstation demonstrates its real task.
        """
        rng = np.random.default_rng(seed)
        y = np.arange(rows, dtype=np.float32)[:, None]
        x = np.arange(cols, dtype=np.float32)[None, :]
        xn = x / max(cols - 1, 1)
        yn = y / max(rows - 1, 1)

        interface = (
            rows * 0.63
            + rows * 0.018 * np.sin(2.2 * np.pi * xn + 0.35)
            + rows * 0.010 * np.sin(5.0 * np.pi * xn)
            + rows * 0.014 * (xn - 0.5)
        )
        transition = np.clip((y - interface) / 9.0, -1.0, 1.0)
        cover_weight = 0.5 * (1.0 - np.tanh(transition))
        bedrock_weight = 1.0 - cover_weight

        data = 0.075 * rng.normal(size=(rows, cols)).astype(np.float32)
        # Early direct-wave and ground-surface ringing.
        data += 0.38 * np.sin(0.60 * y + 0.018 * x) * np.exp(-yn * 3.2)
        for depth, amp, freq, width in [(17, 0.90, 1.12, 4.0), (31, 0.48, 0.92, 5.5), (48, 0.28, 0.78, 7.0)]:
            band = np.exp(-((y - depth) ** 2) / (2.0 * width**2))
            data += amp * band * np.cos(freq * (y - depth) + 0.020 * x)

        # Fine, gently warped cover layering.
        warped_y = y + 2.6 * np.sin(2.8 * np.pi * xn) + 1.4 * np.sin(7.0 * np.pi * xn)
        cover = (
            0.24 * np.sin(0.31 * warped_y + 0.012 * x)
            + 0.15 * np.cos(0.53 * warped_y - 0.010 * x)
            + 0.08 * np.sin(0.82 * warped_y + 0.025 * np.sin(x / 20.0))
        )
        data += cover * cover_weight * np.exp(-yn * 0.75)

        # Coarser, dipping bedrock texture below the boundary.
        bedrock = (
            0.20 * np.sin(0.19 * y + 0.050 * x)
            + 0.13 * np.cos(0.27 * y - 0.034 * x)
            + 0.07 * np.sin(0.43 * y + 0.012 * x)
        )
        data += bedrock * bedrock_weight

        # Continuous basal reflection; attenuate the designated weak segment.
        weak = 1.0 - 0.74 * np.exp(-((xn - 0.485) ** 2) / (2.0 * 0.060**2))
        boundary_wave = np.exp(-((y - interface) ** 2) / (2.0 * 5.0**2)) * np.cos((y - interface) * 1.05)
        data += 1.05 * boundary_wave * weak

        # Mild trace-to-trace coupling variation and depth attenuation.
        trace_gain = 1.0 + 0.055 * np.sin(4.0 * np.pi * xn) + 0.025 * rng.normal(size=(1, cols))
        data *= trace_gain.astype(np.float32)
        data *= (1.0 - 0.18 * yn).astype(np.float32)
        data -= np.mean(data, dtype=np.float64)
        scale = max(float(np.percentile(np.abs(data), 99.5)), 1e-6)
        data = np.clip(data / scale, -1.0, 1.0).astype(np.float32)
        return cls.from_matrix(
            line_id,
            data,
            length_m=length_m,
            time_window_ns=250.0,
            dielectric_constant=9.0,
            format_name="synthetic-basal-interface-demo",
            metadata={
                "generator": "GPRDataSet.synthetic_basal_interface",
                "seed": seed,
                "interface_fraction": 0.63,
                "weak_interval_fraction": [0.42, 0.55],
            },
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


def load_gpr_dataset(
    path: str | Path,
    *,
    line_id: str = "L03",
    length_m: float | None = None,
    dielectric_constant: float = 9.0,
    mmap_mode: str | None = "r",
) -> GPRDataSet:
    """Load a normalized dataset from a dataset directory, CSV, NPY, NPZ or HDF5.

    Directory-backed datasets keep the matrix as ``np.memmap`` and therefore
    do not need to fit in RAM.  Source-file imports use
    :func:`load_gpr_dataset_for_import` to obtain cancellation and chunked IO.
    """
    src = Path(path).resolve()
    if src.is_dir() and (src / "matrix.npy").exists():
        from core.chunked_gpr_io import load_dataset_directory

        matrix, distance_axis, time_axis, depth_axis, stored_meta = load_dataset_directory(
            src, mmap_mode=mmap_mode or "r"
        )
        metadata = dict(stored_meta.get("metadata") or stored_meta)
        return GPRDataSet(
            line_id=line_id,
            matrix=matrix,
            distance_axis_m=np.asarray(distance_axis, dtype=np.float32),
            time_axis_ns=np.asarray(time_axis, dtype=np.float32),
            depth_axis_m=np.asarray(depth_axis, dtype=np.float32),
            source_path=str(stored_meta.get("source_path", src)),
            time_window_ns=float(stored_meta.get("time_window_ns", float(time_axis[-1]) if len(time_axis) else 250.0)),
            dielectric_constant=float(stored_meta.get("dielectric_constant", dielectric_constant)),
            format_name=str(stored_meta.get("format_name", "chunked-dataset")),
            metadata=metadata,
        )
    suffix = src.suffix.lower()
    if suffix in {".csv", ".txt"}:
        sidecar_header = detect_mygpr_airborne_sidecar_csv(src)
        if sidecar_header is not None:
            return _load_mygpr_airborne_sidecar_csv(src, line_id=line_id, dielectric_constant=dielectric_constant)
        matrix = _load_numeric_csv(src)
        fmt = "csv-matrix"
    elif suffix == ".npy":
        matrix = np.load(src, mmap_mode=mmap_mode, allow_pickle=False)
        fmt = "npy-matrix"
    elif suffix == ".npz":
        with np.load(src, allow_pickle=False) as npz:
            matrix = np.asarray(npz["matrix"] if "matrix" in npz else npz[npz.files[0]], dtype=np.float32)
            length_axis = np.asarray(npz["distance_axis_m"], dtype=np.float32) if "distance_axis_m" in npz else None
            time_axis = np.asarray(npz["time_axis_ns"], dtype=np.float32) if "time_axis_ns" in npz else None
            depth_axis = np.asarray(npz["depth_axis_m"], dtype=np.float32) if "depth_axis_m" in npz else None
            # save_npz 把 to_metadata() 序列化为 metadata 成员（0 维 Unicode 数组）；
            # 读取端必须还原 dielectric_constant 与 metadata（速度分析证据等），
            # 否则写回测线的 ε 与证据在下次加载时静默丢失。
            stored_meta: dict[str, Any] = {}
            if "metadata" in npz:
                try:
                    parsed = json.loads(str(npz["metadata"]))
                    if isinstance(parsed, dict):
                        stored_meta = parsed
                except (json.JSONDecodeError, TypeError, ValueError):
                    stored_meta = {}
        ds = GPRDataSet.from_matrix(
            line_id, matrix,
            length_m=float(length_axis[-1]) if length_axis is not None and len(length_axis) else length_m,
            time_window_ns=float(stored_meta.get("time_window_ns", 250.0)),
            dielectric_constant=float(stored_meta.get("dielectric_constant", dielectric_constant)),
            source_path=str(src), format_name="npz-dataset",
            metadata=dict(stored_meta.get("metadata") or {"source_name": src.name}),
        )
        if length_axis is not None:
            ds.distance_axis_m = length_axis
        if time_axis is not None:
            ds.time_axis_ns = time_axis
            ds.time_window_ns = float(time_axis[-1]) if len(time_axis) else ds.time_window_ns
        if depth_axis is not None:
            ds.depth_axis_m = depth_axis
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
        line_id, matrix, length_m=length_m, dielectric_constant=dielectric_constant,
        source_path=str(src), format_name=fmt, metadata={"source_name": src.name},
    )


def load_gpr_dataset_for_import(
    path: str | Path,
    *,
    line_id: str,
    staging_dir: str | Path,
    length_m: float | None = None,
    dielectric_constant: float = 9.0,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> GPRDataSet:
    """Load a source file through a chunked, cancellable staging matrix."""
    from core.chunked_gpr_io import (
        load_hdf5_for_import,
        load_npy_for_import,
        load_npz_for_import,
        load_numeric_csv_to_memmap,
    )

    src = Path(path).resolve()
    stage = Path(staging_dir)
    stage.mkdir(parents=True, exist_ok=True)
    matrix_path = stage / "matrix.npy"
    suffix = src.suffix.lower()
    axes: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"source_name": src.name, "staging_matrix_path": str(matrix_path)}
    if suffix in {".csv", ".txt"}:
        sidecar_header = detect_mygpr_airborne_sidecar_csv(src)
        if sidecar_header is not None:
            # The legacy sidecar parser has a special trace-major contract.  It
            # is kept for compatibility; ordinary matrix CSVs use the bounded
            # two-pass path below.
            ds = _load_mygpr_airborne_sidecar_csv(src, line_id=line_id, dielectric_constant=dielectric_constant)
            if ds.matrix.nbytes >= 64 * 1024 * 1024:
                mm = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=ds.matrix.shape)
                for start_row in range(0, ds.sample_count, 256):
                    if cancel_requested and cancel_requested():
                        from core.chunked_gpr_io import ImportCancelled
                        raise ImportCancelled("用户取消了当前文件导入。")
                    end_row = min(ds.sample_count, start_row + 256)
                    mm[start_row:end_row] = ds.matrix[start_row:end_row]
                mm.flush()
                ds.matrix = mm
                ds.metadata["staging_matrix_path"] = str(matrix_path)
            return ds
        matrix = load_numeric_csv_to_memmap(src, matrix_path, cancel_requested=cancel_requested, progress_callback=progress_callback)
        fmt = "csv-matrix-chunked"
    elif suffix == ".npy":
        matrix = load_npy_for_import(src, matrix_path, cancel_requested=cancel_requested, progress_callback=progress_callback)
        fmt = "npy-matrix-chunked"
        metadata["source_npy_fast_path"] = bool(np.asarray(matrix).dtype == np.float32)
    elif suffix == ".npz":
        matrix, axes = load_npz_for_import(src, matrix_path, cancel_requested=cancel_requested, progress_callback=progress_callback)
        fmt = "npz-dataset-chunked"
    elif suffix in {".h5", ".hdf5"}:
        matrix = load_hdf5_for_import(src, matrix_path, cancel_requested=cancel_requested, progress_callback=progress_callback)
        fmt = "hdf5-matrix-chunked"
    else:
        raise ValueError(f"Unsupported GPR data format: {src.suffix}")
    ds = GPRDataSet.from_matrix(
        line_id, matrix, length_m=length_m, dielectric_constant=dielectric_constant,
        source_path=str(src), format_name=fmt, metadata=metadata,
    )
    if "distance_axis_m" in axes:
        ds.distance_axis_m = axes["distance_axis_m"]
    if "time_axis_ns" in axes:
        ds.time_axis_ns = axes["time_axis_ns"]
        ds.time_window_ns = float(ds.time_axis_ns[-1]) if len(ds.time_axis_ns) else ds.time_window_ns
    if "depth_axis_m" in axes:
        ds.depth_axis_m = axes["depth_axis_m"]
    return ds


__all__ = ["GPR_DATA_SCHEMA", "GPRDataSet", "detect_mygpr_airborne_sidecar_csv", "load_gpr_dataset", "load_gpr_dataset_for_import", "time_to_depth_axis"]

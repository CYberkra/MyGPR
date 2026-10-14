"""CDUT acquisition HDF5: trace-major time-domain data and aligned sensors.

This adapter handles completed acquisition files. It does not tail a live writer
or infer a frequency sweep, spatial spacing, calibration or amplitude units.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

FORMAT_NAME = "cdut-acquisition-hdf5"
SENSOR_NAMES = ("lon", "lat", "alt", "height")


def acquisition_trace_metadata(metadata: dict) -> dict[str, np.ndarray]:
    """Restore JSON null sensor gaps as NaN in the processing contract."""
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        metadata = nested
    if metadata.get("source_format") != FORMAT_NAME:
        return {}
    sensors = metadata.get("trace_meta", {})
    names = {"lon": "longitude", "lat": "latitude", "alt": "altitude_m", "height": "flight_height_m"}
    result = {target: np.asarray(sensors[source], dtype=float) for source, target in names.items() if source in sensors}
    expected = int(metadata["trace_count"])
    if any(value.shape != (expected,) for value in result.values()):
        raise ValueError("acquisition trace metadata length mismatch")
    return result


def _hard_object(group, key):
    if not isinstance(group.get(key, getlink=True), h5py.HardLink):
        raise ValueError(f"Acquisition HDF5 external/soft links are not allowed: {key}")
    return group[key]


def inspect_acquisition_hdf5(handle) -> dict[str, Any] | None:
    """Recognize and validate the schema without reading dataset payloads."""
    fields = ("samples_per_trace", "time_window_ns", "store_component")
    if "traces" not in handle or not any(key in handle.attrs for key in fields):
        return None
    if not all(key in handle.attrs for key in fields):
        raise ValueError("Acquisition HDF5 is missing samples_per_trace/time_window_ns/store_component")
    traces = _hard_object(handle, "traces")
    if not isinstance(traces, h5py.Dataset) or traces.ndim != 2 or min(traces.shape) <= 0:
        raise ValueError("Acquisition HDF5 traces must be a non-empty 2D dataset")
    if traces.dtype != np.dtype("float32"):
        raise ValueError("Acquisition HDF5 traces must use float32")
    sample_count = int(handle.attrs["samples_per_trace"])
    if sample_count != traces.shape[1] or sample_count != handle.attrs["samples_per_trace"]:
        raise ValueError("Acquisition HDF5 samples_per_trace does not match traces columns")
    window = float(handle.attrs["time_window_ns"])
    if not np.isfinite(window) or window < 0:
        raise ValueError("Acquisition HDF5 time_window_ns must be finite and nonnegative")
    component = handle.attrs["store_component"]
    if isinstance(component, bytes):
        component = component.decode("utf-8")
    if component not in {"real", "magnitude"}:
        raise ValueError(f"Unsupported acquisition store_component: {component}")
    if "trace_meta" in handle:
        group = _hard_object(handle, "trace_meta")
        if not isinstance(group, h5py.Group):
            raise ValueError("Acquisition HDF5 trace_meta must be a group")
        for name in SENSOR_NAMES:
            if name not in group:
                continue
            dataset = _hard_object(group, name)
            if (not isinstance(dataset, h5py.Dataset) or dataset.shape != (traces.shape[0],)
                    or dataset.dtype.kind not in "fiu"):
                raise ValueError(f"Acquisition HDF5 trace_meta/{name} must align with traces")
    return {
        "sample_count": sample_count, "trace_count": int(traces.shape[0]),
        "time_window_ns": window, "store_component": str(component),
        "prefix": str(handle.attrs.get("prefix", "")), "created": str(handle.attrs.get("created", "")),
    }


def load_acquisition_hdf5(handle, info, *, source: Path, line_id: str,
                          length_m: float | None, dielectric_constant: float,
                          output_npy: Path | None = None, cancel_requested=None, progress_callback=None):
    """Normalize axes and retain sensor gaps; staging reads bounded trace blocks."""
    from core.chunked_gpr_io import check_cancel, emit_progress, _require_output_space, DEFAULT_CHUNK_BYTES
    from core.gpr_data_model import GPRDataSet, time_to_depth_axis
    from core.hdf5_array_proxy import HDF5ArrayProxy

    samples, count = info["sample_count"], info["trace_count"]
    if length_m is not None and (not np.isfinite(length_m) or length_m <= 0):
        raise ValueError("length_m must be a positive finite measured length")
    metadata = {
        "source_name": source.name, "source_format": FORMAT_NAME,
        **info, "source_matrix_order": "traces_by_samples",
        "time_axis_status": "known" if info["time_window_ns"] > 0 else "unknown",
        "distance_axis_status": "provided_length" if length_m is not None else "unknown",
        "time_axis_endpoint": False,
        "trace_meta": {name: [] for name in SENSOR_NAMES},
    }
    block_traces = max(1, min(256, DEFAULT_CHUNK_BYTES // (samples * 4)))
    matrix = None
    try:
        check_cancel(cancel_requested)
        if output_npy is not None:
            _require_output_space(output_npy, samples * count * 4)
            matrix = np.lib.format.open_memmap(output_npy, mode="w+", dtype=np.float32, shape=(samples, count))
            metadata["staging_matrix_path"] = str(output_npy)
        for start in range(0, count, block_traces):
            check_cancel(cancel_requested)
            end = min(start + block_traces, count)
            for name in SENSOR_NAMES:
                key = f"trace_meta/{name}"
                values = handle[key][start:end] if key in handle else np.full(end - start, np.nan)
                metadata["trace_meta"][name].extend(float(v) if np.isfinite(v) else None for v in values)
            if matrix is not None:
                matrix[:, start:end] = handle["traces"][start:end].T
            emit_progress(progress_callback, "copy_hdf5", end, count)
        check_cancel(cancel_requested)
        if matrix is not None:
            matrix.flush()
        else:
            matrix = HDF5ArrayProxy(source, "/traces", swmr=False).T
        window = info["time_window_ns"]
        # Acquisition convention excludes the endpoint: dt = window / samples.
        # An unknown window stays zero; QC asks for timing rather than inventing it.
        time = (np.arange(samples, dtype=np.float64) * (window / samples)).astype(np.float32)
        distance = (np.linspace(0, length_m, count, dtype=np.float32)
                    if length_m is not None else np.zeros(count, dtype=np.float32))
        return GPRDataSet(
            line_id=line_id, matrix=matrix, distance_axis_m=distance, time_axis_ns=time,
            depth_axis_m=time_to_depth_axis(time, dielectric_constant), source_path=str(source),
            time_window_ns=window, dielectric_constant=dielectric_constant,
            format_name=FORMAT_NAME, metadata=metadata,
        )
    except Exception:
        if matrix is not None:
            del matrix
        if output_npy is not None:
            output_npy.unlink(missing_ok=True)
        raise

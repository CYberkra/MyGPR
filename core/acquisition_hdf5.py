"""CDUT acquisition HDF5: trace-major waveforms and aligned sensors.

This adapter handles completed acquisition files. It does not tail a live writer
or infer a frequency sweep, spatial spacing, calibration or amplitude units.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import uuid

import h5py
import numpy as np

FORMAT_NAME = "cdut-acquisition-hdf5"
SENSOR_NAMES = ("lon", "lat", "alt", "height")
MOTION_NAMES = ("roll", "pitch", "yaw", "ax", "ay", "az", "gx", "gy", "gz")
TRACE_FIELDS = SENSOR_NAMES + MOTION_NAMES + ("timestamp_ms", "trace_id")


def _sfcw_companions(handle, info: dict[str, Any]) -> dict[str, Any]:
    """Read only the bounded, identity-bound manifest; never scan the S21 journal.

    The producer's Bluestein ICZT uses linspace(0, t_max, n_time).  A matching
    original manifest is evidence for that axis; frequency input alone is not.
    """
    source = Path(handle.filename)
    manifest = source.with_name(source.name + '.sfcw.json')
    raw = source.with_name(source.name + '.s21.jsonl')
    if not manifest.exists() and info.get('input_domain') != 'frequency':
        return {}
    result: dict[str, Any] = {'sfcw_companions': {
        'manifest': 'missing', 'raw_s21': 'present_unverified' if raw.is_file() else 'missing'}}
    if not manifest.exists():
        return result
    try:
        with manifest.open('rb') as stream:
            contents = stream.read(65537)
        if len(contents) > 65536:
            raise ValueError('manifest exceeds 64 KiB')
        record = json.loads(contents)
        original_id = uuid.UUID(info.get('acquisition_id', ''))
        if original_id.int == 0 or uuid.UUID(record['acquisition_id']) != original_id:
            raise ValueError('acquisition identity mismatch')
        settings = record['sfcw']
        params = settings['iczt']
        _validate_iczt_parameters(params, info)
        component = settings['store_component']
        def canonical(value):
            return 'magnitude' if value == 'mag' else value
        if component not in ('real', 'mag') or canonical(component) != canonical(info['store_component']):
            raise ValueError('storage component mismatch')
        if info.get('input_domain', 'frequency') != 'frequency' or info.get('signal_domain', 'time') != 'time':
            raise ValueError('signal domain mismatch')
        if 'time_axis_endpoint' in handle.attrs and not info['time_axis_endpoint']:
            raise ValueError('explicit endpoint conflicts with original ICZT axis')
        if not isinstance(settings.get('initiate'), bool):
            raise ValueError('invalid initiate policy')
        interval = settings.get('interval')
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not np.isfinite(interval) or not 0 <= interval <= 3600):
            raise ValueError('invalid capture interval')
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f'SFCW companion is invalid or conflicts with acquisition: {exc}') from exc
    result.update(sfcw=settings, sfcw_manifest_sha256=hashlib.sha256(contents).hexdigest(),
                  time_axis_endpoint=True, time_axis_convention='sfcw_manifest',
                  input_domain='frequency', signal_domain='time')
    result['sfcw_companions']['manifest'] = 'validated'
    return result


def _validate_iczt_parameters(params: dict, info: dict) -> None:
    """Validate the bounded ICZT settings independently of file identity."""
    required = {'f_start', 'f_stop', 'n_freq', 't_max', 'n_time', 'window', 'method'}
    if set(params) != required:
        raise ValueError('incomplete or unknown ICZT parameters')
    for key in ('f_start', 'f_stop', 't_max'):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'invalid {key}')
    for key in ('n_freq', 'n_time'):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 100000:
            raise ValueError(f'invalid {key}')
    if not 0 <= params['f_start'] < params['f_stop'] or params['t_max'] <= 0:
        raise ValueError('invalid frequency or time bounds')
    if params['method'] != 'bluestein' or params['window'] not in (
            None, 'kaiser', 'kaiser6', 'kaiser8', 'hamming', 'hann', 'hanning',
            'blackman', 'rect', 'rectangular', 'none'):
        raise ValueError('unsupported ICZT method or window')
    if params['n_time'] != info['sample_count'] or not np.isclose(
            params['t_max'] * 1e9, info['time_window_ns'], rtol=1e-9, atol=1e-6):
        raise ValueError('sample count or time window mismatch')


def acquisition_trace_metadata(metadata: dict) -> dict[str, np.ndarray]:
    """Restore JSON null sensor gaps as NaN in the processing contract."""
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        metadata = nested
    if metadata.get("source_format") != FORMAT_NAME:
        return {}
    sensors = metadata.get("trace_meta", {})
    names = {"lon": "longitude", "lat": "latitude", "alt": "altitude_m", "height": "flight_height_m",
             "roll": "roll_deg", "pitch": "pitch_deg", "yaw": "yaw_deg",
             "ax": "accel_x_g", "ay": "accel_y_g", "az": "accel_z_g",
             "gx": "angular_rate_x_dps", "gy": "angular_rate_y_dps", "gz": "angular_rate_z_dps",
             "timestamp_ms": "timestamp_ms"}
    result = {target: np.asarray(sensors[source], dtype=float) for source, target in names.items() if source in sensors}
    if "timestamp_ms" in result:
        result["trace_timestamp_s"] = result["timestamp_ms"] / 1000.0
    if "trace_id" in sensors:
        result["trace_id"] = np.asarray(sensors["trace_id"], dtype=np.uint64)
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
    if component not in {"real", "mag", "magnitude", "fdata"}:
        raise ValueError(f"Unsupported acquisition store_component: {component}")
    endpoint = handle.attrs.get("time_axis_endpoint", False)
    if not isinstance(endpoint, (bool, np.bool_)):
        raise ValueError("Acquisition HDF5 time_axis_endpoint must be a boolean")
    committed = handle.attrs.get("committed_trace_count", traces.shape[0])
    if (isinstance(committed, (bool, np.bool_)) or not isinstance(committed, (int, np.integer))
            or committed != traces.shape[0]):
        raise ValueError("Acquisition HDF5 committed_trace_count does not match traces; finish/recover acquisition first")
    present = []
    if "trace_meta" in handle:
        group = _hard_object(handle, "trace_meta")
        if not isinstance(group, h5py.Group):
            raise ValueError("Acquisition HDF5 trace_meta must be a group")
        for name in TRACE_FIELDS:
            if name not in group:
                continue
            dataset = _hard_object(group, name)
            if (not isinstance(dataset, h5py.Dataset) or dataset.shape != (traces.shape[0],)
                    or dataset.dtype.kind not in ("iu" if name == "trace_id" else "fiu")):
                raise ValueError(f"Acquisition HDF5 trace_meta/{name} must align with traces")
            present.append(name)
    provenance = {}
    for name in ("acquisition_id", "data_source"):
        if name in handle.attrs:
            value = handle.attrs[name]
            provenance[name] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if "simulation" in handle.attrs:
        value = handle.attrs["simulation"]
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("Acquisition HDF5 simulation must be a boolean")
        provenance["simulation"] = bool(value)
    domains = {}
    for name, allowed in (("input_domain", {"time", "frequency"}), ("signal_domain", {"time"})):
        if name in handle.attrs:
            value = handle.attrs[name]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f'Unsupported acquisition {name}: {value}')
            domains[name] = value
    info = {
        "sample_count": sample_count, "trace_count": int(traces.shape[0]),
        "time_window_ns": window, "store_component": str(component),
        "prefix": str(handle.attrs.get("prefix", "")), "created": str(handle.attrs.get("created", "")),
        "time_axis_endpoint": bool(endpoint),
        "time_axis_convention": "declared" if "time_axis_endpoint" in handle.attrs else "legacy_window_over_samples",
        "trace_meta_fields": list(dict.fromkeys(SENSOR_NAMES + tuple(present))),
        **provenance, **domains,
    }
    info.update(_sfcw_companions(handle, info))
    return info


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
        "trace_meta": {name: [] for name in info["trace_meta_fields"]},
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
            for name in metadata["trace_meta"]:
                key = f"trace_meta/{name}"
                values = handle[key][start:end] if key in handle else np.full(end - start, np.nan)
                if name == "trace_id":
                    if np.any(values < 0):
                        raise ValueError("Acquisition HDF5 trace_id must be nonnegative")
                    metadata["trace_meta"][name].extend(int(v) for v in values)
                else:
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
        # Keep the legacy convention unless the producer explicitly declares an endpoint.
        # An unknown window stays zero; QC asks for timing rather than inventing it.
        divisor = max(1, samples - int(info["time_axis_endpoint"]))
        time = (np.arange(samples, dtype=np.float64) * (window / divisor)).astype(np.float32)
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

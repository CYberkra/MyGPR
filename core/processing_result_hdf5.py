"""Read exported processing matrices without replacing their persisted physical axes."""
from __future__ import annotations

import json

import h5py
import numpy as np

from mygpr.domain.project.result_package import RESULT_H5_SCHEMA


def is_processing_result(handle) -> bool:
    return str(handle.attrs.get('schema') or '') == RESULT_H5_SCHEMA


def _dataset(handle, name, ndim):
    if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f'Processing result requires a local dataset: {name}')
    value = handle[name]
    if (not isinstance(value, h5py.Dataset) or value.ndim != ndim or value.size == 0
            or value.dtype.kind not in 'fiu' or value.is_virtual or value.external):
        raise ValueError(f'Invalid processing result dataset: {name}')
    return value


def load_processing_result(handle, *, source, line_id, output_npy=None,
                           cancel_requested=None, progress_callback=None):
    from core.gpr_data_model import GPRDataSet
    from core.chunked_gpr_io import _copy_array_rows_to_npy

    matrix = _dataset(handle, 'matrix', 2)
    axes = {}
    for name in ('distance_axis_m', 'time_axis_ns', 'depth_axis_m'):
        axis = _dataset(handle, name, 1)
        expected = matrix.shape[1] if name == 'distance_axis_m' else matrix.shape[0]
        if axis.shape != (expected,):
            raise ValueError(f'Processing result axis length mismatch: {name}')
        axes[name] = axis[:]
        if not np.all(np.isfinite(axes[name])):
            raise ValueError(f'Processing result axis is nonfinite: {name}')
    metadata = json.loads(str(handle.attrs.get('metadata_json') or '{}'))
    if not isinstance(metadata, dict):
        raise ValueError('Invalid processing result metadata')
    window = float(handle.attrs['time_window_ns'])
    dielectric = float(handle.attrs['dielectric_constant'])
    if not np.isfinite(window) or window < 0 or not np.isfinite(dielectric) or dielectric <= 0:
        raise ValueError('Invalid processing result time window or dielectric constant')
    if output_npy is None:
        values = matrix[:]
    else:
        values = _copy_array_rows_to_npy(matrix, output_npy, cancel_requested=cancel_requested,
                                         progress_callback=progress_callback)
        metadata['staging_matrix_path'] = str(output_npy)
    metadata['processing_result_artifact_id'] = str(handle.attrs.get('artifact_id') or '')
    return GPRDataSet(line_id=line_id, matrix=values, **axes, source_path=str(source),
                      time_window_ns=window, dielectric_constant=dielectric,
                      format_name='mygpr-processing-result-hdf5', metadata=metadata)

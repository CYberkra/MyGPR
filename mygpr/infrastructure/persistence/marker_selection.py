"""Bounded, cancellable trace selection using the existing artifact transaction."""
from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path
from uuid import uuid4

import numpy as np

from mygpr.domain.markers.mapping import SEGMENT_KEY
from mygpr.application.project.processing_service import source_sha256


def commit_selection(session, view, preview, context):
    line_id = view.document.line_id
    info = (session.get_artifact_dataset_info(line_id, view.artifact_id)
            if view.artifact_id else session.get_dataset_info(line_id))
    if not session.summary.storage_backend.startswith('hybrid'):
        raise ValueError('此项目存储不能保存道映射，请先迁移项目格式')
    indices = np.concatenate([np.arange(start, stop, dtype=np.int64)
                              for start, stop in preview['selected_input_spans']])
    if info.shape[1] != view.count or not indices.size:
        raise ValueError('输入道数已变化或输出为空')
    trace = {key: np.asarray(values)[indices].copy() for key, values in view.trace_metadata.items()}
    old_segments = np.asarray(view.trace_metadata[SEGMENT_KEY])
    trace[SEGMENT_KEY] = np.cumsum(np.r_[False, (np.diff(indices) != 1)
                                           | (np.diff(old_segments[indices]) != 0)], dtype=np.int64)
    input_header = {key: value for key, value in view.header.items() if key != '_marker_track'}
    header = {**input_header, 'num_traces': len(indices), 'source_trace_offset': None}
    header['marker_maps'] = {source: {**entry, 'kind': 'resampled' if entry['kind'] == 'resampled' else 'selection'}
                             for source, entry in header.get('marker_maps', {}).items()}
    axis = header.get('distance_axis_m')
    if axis is not None:
        header['distance_axis_m'] = np.asarray(axis)[indices].tolist()
    params = {'mode': preview['mode'], 'ranges': preview['ranges']}
    steps = [{'method_id': 'trace_range_select', 'params': params, 'enabled': True}]
    lineage = [{'method_id': 'trace_range_select', 'params': params,
                'metadata': {'selected_count': preview['affected_count'], 'output_segments': preview['output_segments']},
                'implementation_version': '1', 'output_shape': [info.shape[0], len(indices)],
                'output_dtype': 'float32', 'runtime_warnings': []}]
    input_sha = source_sha256(info)
    title = ('保留' if preview['mode'] == 'keep' else '剔除') + '事件区间'
    run_id = uuid4().hex
    with TemporaryDirectory(prefix='mygpr-marker-selection-') as directory:
        matrix = np.lib.format.open_memmap(Path(directory) / 'selected.npy', mode='w+', dtype=np.float32,
                                          shape=(info.shape[0], len(indices)))
        try:
            block_rows = max(1, min(256, (16 * 1024 * 1024) // max(4 * view.count, 1)))
            blocks = (session.iter_artifact_blocks(line_id, view.artifact_id, block_rows=block_rows)
                      if view.artifact_id else session.iter_dataset_blocks(line_id, block_rows=block_rows))
            for start, stop, block in blocks:
                context.raise_if_cancelled()
                matrix[start:stop] = block[:, indices]
                context.report_progress(stop, info.shape[0], title)
            matrix.flush()
            context.raise_if_cancelled()
            artifact = session.save_processing_artifact(
                line_id, matrix, name=title, method_id='trace_range_select', method_name=title,
                params={**params, 'marker_selection': preview, 'output_header': header,
                        'pipeline': steps, 'lineage': lineage, 'execution_mode': 'block',
                        'parent_artifact_id': view.artifact_id, 'run_input_artifact_id': view.artifact_id,
                        'step_input_artifact_id': view.artifact_id, 'run_group_id': run_id,
                        'run_input_sha256': input_sha, 'step_input_sha256': input_sha,
                        'run_step_index': 1, 'artifact_kind': 'processing', 'lineage_version': 2},
                pipeline=[{**steps[0], 'lineage': lineage[0]}],
                branch_id=f'{line_id}:{run_id}', input_dataset=input_header, trace_metadata=trace, context=context)
            context.emit_artifact({'artifact_id': artifact.artifact_id, 'line_id': line_id,
                                   'data_reference': artifact.data_reference})
            return artifact
        finally:
            getattr(matrix, '_mmap').close()

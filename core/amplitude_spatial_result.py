"""Persist an amplitude slice in the existing versioned spatial-result store."""
from __future__ import annotations

import shutil
import uuid

import numpy as np

from core.field_project_models import local_now, validate_line_id
from core.spatial_result_versions import SPATIAL_RESULT_SCHEMA, _check_cancel
from core.storage_primitives import fsync_directory, fsync_file


def save_amplitude_result(service, result, *, name: str, cancel_requested=None):
    """Atomically publish a raster, exact coverage counts and source manifest."""
    import rasterio
    from rasterio.transform import from_origin
    from core.processing_artifact_index import index_processing_artifacts

    project = service.project
    project.assert_writable()
    sources = tuple(result.sources)
    line_ids = [validate_line_id(source.line_id) for source in sources]
    if not line_ids or len(set(line_ids)) != len(line_ids):
        raise ValueError('振幅切片测线列表无效')
    if not result.source_state_token or service._current_source_state_token(line_ids) != result.source_state_token:
        raise ValueError('测线或轨迹已变化，请重新生成切片后保存')
    grid = result.grid
    values = np.asarray(grid.values, dtype=np.float32)
    trace_counts = np.asarray(grid.trace_counts, dtype=np.int64)
    line_counts = np.asarray(grid.line_counts, dtype=np.int64)
    if (values.ndim != 2 or not values.size or not np.isfinite(values).any()
            or trace_counts.shape != values.shape or line_counts.shape != values.shape):
        raise ValueError('切片网格或覆盖计数无效')
    source_manifest = {}
    warnings = []
    for source in sources:
        _check_cancel(cancel_requested)
        digest = str(source.content_sha256 or '')
        if source.artifact_id and digest:
            artifacts = [item for item in index_processing_artifacts(project.root, source.line_id)
                         if item.artifact_id == source.artifact_id]
            if len(artifacts) != 1 or str(artifacts[0].output_data_sha256 or '') != digest:
                raise ValueError(f'{source.line_id} 的处理成果已变化，请重新生成切片')
        elif not source.artifact_id and digest:
            dataset = project.load_gpr_dataset(source.line_id)
            if str(dataset.metadata.get('data_sha256') or '') != digest:
                raise ValueError(f'{source.line_id} 的原始数据已变化，请重新生成切片')
        if not digest:
            warnings.append(f'{source.line_id}：来源未提供内容哈希')
        source_manifest[source.line_id] = {
            'artifact_id': source.artifact_id, 'component': source.component, 'content_sha256': digest,
            'position_sha256': str(source.position_sha256 or ''),
        }
    base_name = 'amplitude_slice'
    revision = service._next_revision(base_name)
    result_id = f'{base_name}_v{revision:03d}'
    staging = service.root / f'.{result_id}.{uuid.uuid4().hex}.staging'
    destination = service.root / result_id
    if destination.exists():
        raise FileExistsError(destination)
    staging.mkdir(parents=True, exist_ok=False)
    committed = False
    try:
        transform = from_origin(float(grid.x_origin_m) - grid.cell_size_m / 2.,
                                float(grid.y_origin_m) + grid.cell_size_m / 2.,
                                float(grid.cell_size_m), float(grid.cell_size_m))
        with rasterio.open(staging / 'amplitude.tif', 'w', driver='GTiff',
                           width=values.shape[1], height=values.shape[0], count=1,
                           dtype='float32', crs=result.coordinate_system, transform=transform,
                           nodata=np.nan, compress='deflate') as raster:
            raster.write(values, 1)
        np.savez_compressed(staging / 'coverage.npz', trace_counts=trace_counts, line_counts=line_counts)
        for filename in ('amplitude.tif', 'coverage.npz'):
            fsync_file(staging / filename)
        payload = {
            'schema': SPATIAL_RESULT_SCHEMA, 'result_id': result_id, 'name': str(name or '振幅切片'),
            'revision': revision, 'created_at': local_now(),
            'status': 'generated_with_warnings' if warnings else 'generated',
            'coordinate_system': result.coordinate_system, 'vertical_datum': '', 'line_ids': line_ids,
            'options': {'kind': 'amplitude_slice', 'axis': 'acquisition_time_ns',
                        'time_start_ns': result.start_ns, 'time_end_ns': result.end_ns,
                        'attribute': result.attribute, 'cell_size_m': grid.cell_size_m,
                        'aggregation': 'mean_per_line_then_mean_of_lines', 'interpolation': 'none'},
            'sources': {'line_sources': source_manifest, 'source_state_token': result.source_state_token},
            'source_state_token': result.source_state_token,
            'files': {'amplitude_geotiff': 'amplitude.tif', 'coverage': 'coverage.npz'},
            'summary': {'kind': 'amplitude_slice', 'line_count': len(line_ids),
                        'valid_cell_count': int(np.isfinite(values).sum()), 'preflight_warnings': warnings},
        }
        service._atomic_json(staging / 'manifest.json', payload)
        fsync_file(staging / 'manifest.json')
        fsync_directory(staging)
        _check_cancel(cancel_requested)
        if service._current_source_state_token(line_ids) != result.source_state_token:
            raise ValueError('测线或轨迹已变化，请重新生成切片后保存')
        staging.replace(destination)
        fsync_directory(service.root)
        committed = True
        return service.load_result(result_id)
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)

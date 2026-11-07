import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

from core.amplitude_spatial_result import save_amplitude_result
from core.spatial_result_versions import SpatialResultVersionService
from mygpr.application.spatial.amplitude import AmplitudeResult, AmplitudeSource
from mygpr.domain.spatial.amplitude_slice import grid_line_values


def test_saves_geotiff_nodata_coverage_and_reproducible_manifest(tmp_path, monkeypatch):
    project = SimpleNamespace(root=tmp_path, manifest=SimpleNamespace(vertical_datum=''),
                              assert_writable=lambda: None)
    service = SpatialResultVersionService(project)
    monkeypatch.setattr(service, '_current_source_state_token', lambda ids: 'state-1')
    grid = grid_line_values([.1, 2.1], [.1, .1], [1., 3.], ['L1', 'L1'], cell_size_m=1.)
    result = AmplitudeResult(grid, (AmplitudeSource('L1', component='mag'),),
                             'EPSG:32648', 10., 20., 'rms', (), 'state-1')
    record = save_amplitude_result(service, result, name='振幅切片')
    output = tmp_path / 'spatial' / 'results' / record.result_id
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['options']['time_start_ns'] == 10.
    assert manifest['sources']['line_sources']['L1']['component'] == 'mag'
    with rasterio.open(output / 'amplitude.tif') as dataset:
        assert dataset.crs.to_epsg() == 32648
        np.testing.assert_allclose(dataset.read(1)[0, [0, 2]], [1., 3.])
        assert np.isnan(dataset.read(1)[0, 1])
        assert dataset.transform.c == 0. and dataset.transform.f == 1.
    with np.load(output / 'coverage.npz') as coverage:
        np.testing.assert_array_equal(coverage['trace_counts'], [[1, 0, 1]])
    exported = service.export_result(record.result_id, tmp_path / 'export.tif', format_name='geotiff')
    assert exported.is_file()


def test_refuses_to_save_when_sources_changed(tmp_path, monkeypatch):
    project = SimpleNamespace(root=tmp_path, manifest=SimpleNamespace(vertical_datum=''),
                              assert_writable=lambda: None)
    service = SpatialResultVersionService(project)
    monkeypatch.setattr(service, '_current_source_state_token', lambda ids: 'changed')
    grid = grid_line_values([0.1], [0.1], [1.], ['L1'], cell_size_m=1.)
    result = AmplitudeResult(grid, (AmplitudeSource('L1', component='mag'),),
                             'EPSG:32648', 10., 20., 'rms', (), 'old')
    with pytest.raises(ValueError, match='变化'):
        save_amplitude_result(service, result, name='切片')
    assert not service.list_results()

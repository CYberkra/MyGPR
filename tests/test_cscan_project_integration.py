"""Synthetic HDF import through slice, persistence, reopen and deletion."""
import hashlib

import h5py
import numpy as np

from mygpr.application.jobs.models import JobStatus
from mygpr.application.spatial.amplitude import AmplitudeSource, LineObservations
from mygpr.application.spatial.amplitude_projection import project_observations
from mygpr.interfaces.backend import MyGPRBackend


def test_import_slice_save_reopen_delete_preserves_original(tmp_path):
    source = tmp_path / 'source.h5'
    waveforms = np.array([[1.] * 16, [2.] * 16, [3.] * 16], dtype=np.float32)
    with h5py.File(source, 'w') as h5:
        h5.attrs.update(samples_per_trace=16, time_window_ns=16., store_component='mag')
        h5.create_dataset('traces', data=waveforms, chunks=(1, 16), maxshape=(None, 16))
        meta = h5.create_group('trace_meta')
        for key, values in {'lon': [105., 105.00001, 105.00002], 'lat': [30., 30., 30.],
                            'alt': [500., 501., 502.], 'height': [2., 2., 2.]}.items():
            meta.create_dataset(key, data=values)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    root = tmp_path / 'project'
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(root, name='slice', coordinate_system='EPSG:32648')
        backend.projects.import_line_source(project.project_id, 'L01', source)
        track, = backend.spatial.load_tracks(project.project_id)
        src = AmplitudeSource('L01')
        coords = LineObservations(src, track.coordinate_system,
            np.array([p.trace_index for p in track.points]), np.array([p.x for p in track.points]),
            np.array([p.y for p in track.points]), np.zeros(len(track.points)))
        projected, _ = project_observations([coords])
        job = backend.submit_amplitude_slice(project.project_id, [src], projected, 0., 4., cell_size_m=1.)
        done = backend.jobs.wait(job, timeout=60)
        assert done.status is JobStatus.COMPLETED, done.error_message
        result = done.result
        assert result.sources[0].component == 'mag'
        assert len(result.curtains) == 1
        assert np.isfinite(result.grid.values).sum() > 0
        saved = backend.jobs.wait(backend.submit_save_amplitude_slice(project.project_id, result), timeout=60)
        assert saved.status is JobStatus.COMPLETED, saved.error_message
        result_id = saved.result.result_id
        backend.projects.close_project(project.project_id)
        reopened = backend.projects.open_project(root)
        records = backend.spatial.list_results(reopened.project_id)
        assert any(item.result_id == result_id and 'amplitude_geotiff' in item.files for item in records)
        opened = backend.jobs.wait(backend.submit_open_amplitude_slice(reopened.project_id, result_id), timeout=60)
        assert opened.status is JobStatus.COMPLETED, opened.error_message
        assert not opened.result.stale and len(opened.result.result.curtains) == 1
        np.testing.assert_allclose(opened.result.result.grid.values, result.grid.values, equal_nan=True)
        backend.projects.update_line_properties(reopened.project_id, 'L01', length_m=None,
                                                view_reversed=False, time_window_ns=20.)
        stale = backend.jobs.wait(backend.submit_open_amplitude_slice(reopened.project_id, result_id), timeout=60)
        assert stale.status is JobStatus.COMPLETED, stale.error_message
        assert stale.result.stale and not stale.result.result.curtains
        deleted = backend.jobs.wait(backend.submit_spatial_delete(reopened.project_id, result_id), timeout=60)
        assert deleted.status is JobStatus.COMPLETED, deleted.error_message
        assert not (root / 'spatial' / 'results' / result_id).exists()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    finally:
        backend.shutdown()

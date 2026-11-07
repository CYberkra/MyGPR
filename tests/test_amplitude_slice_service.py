from types import SimpleNamespace

import numpy as np
import pytest

from mygpr.application.spatial.amplitude import AmplitudeSource, LineObservations, collect_observations, build_amplitude_grid
from mygpr.domain.spatial.models import SpatialTrack, SpatialTrackPoint
from mygpr.application.spatial.amplitude_identity import track_position_fingerprint


class Projects:
    def __init__(self, *, artifact_ids=None):
        self.data = np.array([[1., 10., 100.], [1., 10., 100.],
                              [1., 10., 100.], [1., 10., 100.]])
        self.artifact_ids = artifact_ids
        self.track = SpatialTrack('L1', 'line', (
            SpatialTrackPoint(0, 10., 20.), SpatialTrackPoint(2, 30., 40.)), 'EPSG:32648')

    def load_spatial_tracks(self, project_id):
        return (self.track,)

    def get_dataset_info(self, project_id, line_id):
        return SimpleNamespace(shape=self.data.shape, metadata={'time_axis_ns': [0., 1., 2., 3.],
                               'store_component': 'mag'})

    def get_artifact_dataset_info(self, project_id, line_id, artifact_id):
        return self.get_dataset_info(project_id, line_id)

    def read_trace_metadata(self, project_id, line_id):
        return {'trace_id': np.array([11, 12, 13])}

    def read_artifact_trace_metadata(self, project_id, line_id, artifact_id):
        return {} if self.artifact_ids is None else {'trace_id': np.array(self.artifact_ids)}

    def iter_dataset_blocks(self, project_id, line_id, **kwargs):
        start, end = kwargs['trace_start'], kwargs['trace_end']
        yield 0, self.data.shape[0], self.data[:, start:end]

    def iter_artifact_blocks(self, project_id, line_id, artifact_id, **kwargs):
        yield from self.iter_dataset_blocks(project_id, line_id, **kwargs)


def test_raw_observations_match_trace_indices_not_compacted_track_positions():
    result = collect_observations(Projects(), 'p', [AmplitudeSource('L1', component='mag')], 0., 4.)
    np.testing.assert_array_equal(result[0].trace_indices, [0, 2])
    np.testing.assert_array_equal(result[0].x, [10., 30.])
    np.testing.assert_array_equal(result[0].values, [1., 100.])


def test_artifact_mapping_uses_trace_ids_and_fails_when_missing():
    source = AmplitudeSource('L1', artifact_id='a', component='mag')
    result = collect_observations(Projects(artifact_ids=[13, 12, 11]), 'p', [source], 0., 4.)
    np.testing.assert_array_equal(result[0].trace_indices, [0, 2])
    np.testing.assert_array_equal(result[0].x, [30., 10.])
    with pytest.raises(ValueError, match='道标识'):
        collect_observations(Projects(), 'p', [source], 0., 4.)


def test_rejects_mislabeled_component_and_unknown_axis():
    projects = Projects()
    with pytest.raises(ValueError, match='分量'):
        collect_observations(projects, 'p', [AmplitudeSource('L1', component='real')], 0., 4.)
    original = projects.get_dataset_info
    projects.get_dataset_info = lambda *args: SimpleNamespace(
        shape=(4, 3), metadata={**original(*args).metadata, 'time_axis_status': 'unknown'})
    with pytest.raises(ValueError, match='时间轴'):
        collect_observations(projects, 'p', [AmplitudeSource('L1', component='mag')], 0., 4.)


def test_nested_acquisition_metadata_is_enforced():
    projects = Projects()
    projects.get_dataset_info = lambda *args: SimpleNamespace(
        shape=(4, 3), metadata={'time_axis_ns': [0., 1., 2., 3.],
                                'metadata': {'store_component': 'fdata'}})
    with pytest.raises(ValueError, match='分量'):
        collect_observations(projects, 'p', [AmplitudeSource('L1', component='real')], 0., 4.)


def test_preprojected_positions_remain_aligned_to_original_trace_numbers():
    source = AmplitudeSource('L1', component='mag')
    coords = LineObservations(source, 'EPSG:32648', np.array([0, 2]),
                              np.array([100., 300.]), np.array([200., 400.]), np.zeros(2))
    result = collect_observations(Projects(), 'p', [source], 0., 4., positions=[coords])
    np.testing.assert_array_equal(result[0].x, [100., 300.])
    assert result[0].coordinate_system == 'EPSG:32648'


def test_component_can_be_selected_from_declared_acquisition_metadata():
    result = collect_observations(Projects(), 'p', [AmplitudeSource('L1')], 0., 4.)
    assert result[0].source.component == 'mag'


def test_rejects_projected_positions_from_an_old_track_revision():
    projects = Projects()
    src = AmplitudeSource('L1', component='mag')
    old_hash = track_position_fingerprint(projects.track)
    coords = LineObservations(src, 'EPSG:32648', np.array([0, 2]),
                              np.array([10., 30.]), np.array([20., 40.]), np.zeros(2), old_hash)
    projects.track = SpatialTrack('L1', 'line', (SpatialTrackPoint(0, 99., 20.),), 'EPSG:32648')
    with pytest.raises(ValueError, match='轨迹已变化'):
        collect_observations(projects, 'p', [src], 0., 4., positions=[coords])


def test_grid_uses_projected_positions_and_records_actual_source():
    source = AmplitudeSource('L1')
    coords = LineObservations(source, 'EPSG:32648', np.array([0, 2]),
                              np.array([0.1, 2.1]), np.array([0.1, 0.1]), np.zeros(2))
    result = build_amplitude_grid(Projects(), 'p', [source], [coords], 0., 4., cell_size_m=1.)
    assert result.coordinate_system == 'EPSG:32648'
    assert result.sources[0].component == 'mag'
    np.testing.assert_allclose(result.grid.values[0, [0, 2]], [1., 100.])
    assert np.isnan(result.grid.values[0, 1])

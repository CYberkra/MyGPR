from types import SimpleNamespace

import numpy as np
import pytest

from mygpr.application.spatial.amplitude import AmplitudeSource, collect_observations
from mygpr.domain.spatial.models import SpatialTrack, SpatialTrackPoint


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

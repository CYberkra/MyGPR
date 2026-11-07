import numpy as np
from types import SimpleNamespace

from mygpr.application.spatial.amplitude import AmplitudeSource
from mygpr.application.spatial.curtain_preview import read_curtain_previews
from mygpr.domain.spatial.models import SpatialTrack, SpatialTrackPoint
class PreviewProjects:
    def __init__(self):
        self.data = np.ones((4, 3))
        self.track = SpatialTrack('L1', 'line', (SpatialTrackPoint(0, 0., 0.),
                                  SpatialTrackPoint(2, 2., 0.)), 'EPSG:32648')

    def load_spatial_tracks(self, project_id):
        return (self.track,)

    def get_dataset_info(self, project_id, line_id):
        return SimpleNamespace(shape=self.data.shape, metadata={'time_axis_ns': [0., 1., 2., 3.]})

    def read_window(self, project_id, line_id, **kwargs):
        # Only first and last columns are displayed; the missing middle
        # trajectory point must still break the rendered connection.
        return self.data[:, [0, 2]], np.arange(4), np.array([0, 2])


def test_decimation_does_not_bridge_a_missing_intermediate_location():
    projects = PreviewProjects()
    preview, = read_curtain_previews(projects, 'p', [AmplitudeSource('L1', component='mag')])
    np.testing.assert_array_equal(preview.trace_indices, [0, 2])
    np.testing.assert_array_equal(preview.connected, [False])
    assert preview.matrix.shape == (4, 2)


def test_decimation_keeps_connection_when_all_intermediate_locations_exist():
    projects = PreviewProjects()
    projects.track = SpatialTrack('L1', 'line', tuple(SpatialTrackPoint(i, i, 0) for i in range(3)), 'EPSG:32648')
    preview, = read_curtain_previews(projects, 'p', [AmplitudeSource('L1', component='mag')])
    np.testing.assert_array_equal(preview.connected, [True])

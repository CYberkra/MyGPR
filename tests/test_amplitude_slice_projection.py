import numpy as np
import pytest

from mygpr.application.spatial.amplitude import AmplitudeSource, LineObservations
from mygpr.application.spatial.amplitude_projection import project_observations


def _line(crs, x, y):
    return LineObservations(AmplitudeSource('L1', component='mag'), crs,
                            np.array([0]), np.array([x]), np.array([y]), np.array([1.]))


def test_geographic_coordinates_are_projected_to_metric_utm():
    result, crs = project_observations([_line('EPSG:4326', 105., 30.)])
    assert crs == 'EPSG:32648'
    assert abs(result[0].x[0] - 500000.) < 0.01
    assert 3_000_000. < result[0].y[0] < 4_000_000.


def test_unknown_and_non_metric_coordinates_are_not_silently_assumed():
    with pytest.raises(ValueError, match='坐标'):
        project_observations([_line('', 1., 2.)])
    with pytest.raises(ValueError, match='米'):
        project_observations([_line('EPSG:4326', 105., 30.)], target_crs='EPSG:4326')


def test_one_metric_crs_is_preserved():
    result, crs = project_observations([_line('EPSG:32648', 500000., 3320000.)])
    assert crs == 'EPSG:32648'
    assert result[0].x[0] == 500000.

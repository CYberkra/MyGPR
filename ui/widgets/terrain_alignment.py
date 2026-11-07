"""Report potential vertical-datum discrepancies without moving the DEM."""
import numpy as np

from ui.widgets.terrain_tiles import sample_bilinear


def datum_notice(payload, ground_points, extent):
    if ground_points is None or len(ground_points) < 4:
        return ''
    points = np.asarray(ground_points, dtype=float)
    terrain = sample_bilinear(payload['z'], payload['gx'], payload['gy'], points[:, 0], points[:, 1])
    valid = np.isfinite(terrain) & np.isfinite(points[:, 2])
    if np.count_nonzero(valid) < 4:
        return ''
    difference = float(np.median(points[valid, 2] - terrain[valid]))
    if abs(difference) <= max(5., float(extent) * .1):
        return ''
    return f'地表估计与地形高程中位差 {difference:+.1f} m，请核对高程基准；未自动平移'

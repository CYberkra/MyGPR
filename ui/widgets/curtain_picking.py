"""CPU-side screen-space picking for a projected curtain triangle mesh."""
from __future__ import annotations

import numpy as np


def pick_mesh(vertices, faces, matrix, width, height, point):
    """Return (depth, nearest vertex) for the frontmost triangle under point."""
    xyz = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError('幕帘选择几何无效')
    if width <= 0 or height <= 0 or not len(triangles):
        return None
    if np.any(triangles < 0) or np.any(triangles >= len(xyz)):
        raise ValueError('幕帘三角形索引无效')
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    clip = np.column_stack((xyz, np.ones(len(xyz)))) @ transform.T
    good = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0)
    ndc = np.full((len(xyz), 3), np.nan)
    ndc[good] = clip[good, :3] / clip[good, 3, None]
    screen = np.column_stack(((ndc[:, 0] + 1.) * width / 2., (1. - ndc[:, 1]) * height / 2.))
    a, b, c = (screen[triangles[:, i]] for i in range(3))
    denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    valid &= good[triangles].all(axis=1)
    safe = np.where(valid, denominator, 1.)
    x, y = float(point[0]), float(point[1])
    wa = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / safe
    wb = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / safe
    wc = 1. - wa - wb
    weights = np.column_stack((wa, wb, wc))
    valid &= np.all(weights >= -1e-9, axis=1)
    depth = np.sum(ndc[triangles, 2] * weights, axis=1)
    valid &= np.isfinite(depth) & (depth >= -1.) & (depth <= 1.)
    if not valid.any():
        return None
    indices = np.flatnonzero(valid)
    selected = indices[int(np.argmin(depth[indices]))]
    vertex = int(triangles[selected, int(np.argmax(weights[selected]))])
    return float(depth[selected]), vertex

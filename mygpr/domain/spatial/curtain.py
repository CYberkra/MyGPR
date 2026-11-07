"""Geometry for survey-aligned B-scan curtains; no interpolation between lines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CurtainGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    values: np.ndarray


def curtain_geometry(x, y, anchor_z, vertical_axis, matrix, *, vertical_scale: float) -> CurtainGeometry:
    """Make a triangle mesh with no faces across missing traces or samples.

    ``vertical_axis`` is a nonnegative relative sample axis. ``vertical_scale``
    is explicit: 1 for an already justified depth axis in metres, or a display
    scale for time. The caller must label time-scale curtains as schematic.
    """
    x, y, anchor = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (x, y, anchor_z))
    axis = np.asarray(vertical_axis, dtype=np.float64).reshape(-1)
    data = np.asarray(matrix, dtype=np.float64)
    if (len(x) < 2 or len(axis) < 2 or len(y) != len(x) or len(anchor) != len(x)
            or data.shape != (len(axis), len(x))):
        raise ValueError('剖面波形与坐标形状不一致')
    scale = float(vertical_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('垂直显示比例必须为正')
    if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0) or axis[0] < 0:
        raise ValueError('剖面纵轴必须非负且严格递增')
    rows, cols = data.shape
    xx = np.broadcast_to(x, (rows, cols))
    yy = np.broadcast_to(y, (rows, cols))
    zz = anchor[None, :] - scale * axis[:, None]
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    valid = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(zz) & np.isfinite(data)
    quads = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    row, col = np.nonzero(quads)
    top_left = row * cols + col
    faces = np.empty((len(top_left) * 2, 3), dtype=np.int32)
    faces[0::2] = np.column_stack((top_left, top_left + cols, top_left + 1))
    faces[1::2] = np.column_stack((top_left + 1, top_left + cols, top_left + cols + 1))
    return CurtainGeometry(vertices, faces, data.ravel())

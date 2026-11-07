"""Small display-only DJI Inspire 1 / T600-inspired low-poly mesh.

Local +X is the illustrative forward direction; coordinates are not a CAD
model and are intentionally scaled by the viewer for legibility.
"""
from __future__ import annotations

import numpy as np


def _ellipsoid(center, radii, color, *, sides=12, bands=6):
    vertices = []
    for latitude in np.linspace(-np.pi / 2, np.pi / 2, bands + 1):
        for theta in np.linspace(0, 2 * np.pi, sides, endpoint=False):
            vertices.append([center[0] + radii[0] * np.cos(latitude) * np.cos(theta),
                             center[1] + radii[1] * np.cos(latitude) * np.sin(theta),
                             center[2] + radii[2] * np.sin(latitude)])
    faces = []
    for row in range(bands):
        for col in range(sides):
            a, b = row * sides + col, row * sides + (col + 1) % sides
            faces.extend(((a, b, a + sides), (b, b + sides, a + sides)))
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32), color


def _beam(start, end, radius, color, *, sides=8):
    start, end = np.asarray(start, float), np.asarray(end, float)
    direction = end - start
    direction /= np.linalg.norm(direction)
    ref = np.array([0., 0., 1.]) if abs(direction[2]) < .9 else np.array([1., 0., 0.])
    u = np.cross(direction, ref)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    vertices = np.concatenate((start + ring, end + ring))
    faces = []
    for index in range(sides):
        nxt = (index + 1) % sides
        faces.extend(((index, nxt, index + sides), (nxt, nxt + sides, index + sides)))
    return vertices, np.asarray(faces, dtype=np.int32), color


def t600_mesh():
    """Return vertices, triangular faces and vertex colors for one model."""
    white = (.88, .90, .92, 1.)
    graphite = (.12, .15, .18, 1.)
    silver = (.46, .49, .52, 1.)
    parts = [_ellipsoid((0., 0., 0.), (.28, .13, .09), white),
             _ellipsoid((.15, 0., .05), (.13, .105, .055), white),
             _ellipsoid((.15, 0., -.13), (.065, .065, .06), graphite)]
    for side in (-1, 1):
        pivot = np.array([0., side * .10, .01])
        shoulder = np.array([0., side * .25, .17])
        parts.append(_beam(pivot, shoulder, .025, graphite))
        for front in (-1, 1):
            motor = np.array([front * .31, side * .31, .19])
            parts.append(_beam(shoulder, motor, .017, graphite))
            parts.append(_ellipsoid(motor, (.035, .035, .024), silver))
            # Two opposed slim blades meeting at the motor hub.
            for sign in (-1, 1):
                blade_center = motor + np.array([sign * .085, 0., .027])
                parts.append(_ellipsoid(blade_center, (.095, .018, .005), graphite, bands=4))
    vertices, faces, colors = [], [], []
    offset = 0
    for xyz, triangles, color in parts:
        vertices.append(xyz)
        faces.append(triangles + offset)
        colors.append(np.tile(color, (len(xyz), 1)))
        offset += len(xyz)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces), np.concatenate(colors).astype(np.float32)


def place_t600(vertices, track, origin, extent, vertical_exaggeration=1.):
    """Place an enlarged symbol at the first valid point of one trajectory."""
    track = np.asarray(track, dtype=float)
    if track.ndim != 2 or track.shape[1] != 3 or len(track) < 2:
        return None
    valid = np.isfinite(track).all(axis=1)
    track = track[valid]
    if len(track) < 2:
        return None
    delta = track[1:, :2] - track[0, :2]
    usable = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-6)
    if not len(usable):
        return None
    direction = delta[usable[0]]
    angle = np.arctan2(direction[1], direction[0])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    center = track[0] - np.asarray(origin, dtype=float)
    center[2] *= vertical_exaggeration
    scale = max(float(extent) * .12, 2.)
    return np.asarray(vertices, dtype=float) @ rotation.T * scale + center

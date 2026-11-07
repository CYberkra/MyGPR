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


def _hull(sections, color, *, sides=16):
    """Build an oval-section fuselage from nose-to-tail stations."""
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    vertices = []
    for x, width, z_low, z_high in sections:
        z_mid = (z_low + z_high) / 2
        z_radius = (z_high - z_low) / 2
        vertices.extend(np.column_stack((np.full(sides, x), width * np.cos(theta),
                                         z_mid + z_radius * np.sin(theta))))
    faces = []
    for row in range(len(sections) - 1):
        for col in range(sides):
            a, b = row * sides + col, row * sides + (col + 1) % sides
            faces.extend(((a, a + sides, b), (b, a + sides, b + sides)))
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32), color


def _propeller(center, angle, color):
    """Two tapered, swept blades; the stationary shape does not imply RPM."""
    # Upper and lower skins meet at the hub; lengths are illustrative.
    outline = np.array([[.014, -.009], [.065, -.018], [.125, -.023], [.17, -.014],
                        [.182, -.004], [.174, .004], [.125, .012], [.06, .011], [.014, .009]])
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    vertices, faces = [], []
    for sign in (-1, 1):
        xy = (outline * sign) @ rot.T + np.asarray(center[:2])
        count = len(xy)
        top = np.column_stack((xy, np.full(count, center[2] + .004)))
        bottom = np.column_stack((xy, np.full(count, center[2] - .002)))
        offset = len(vertices)
        vertices.extend(np.concatenate((top, bottom)))
        for i in range(1, count - 1):
            faces.extend(((offset, offset + i, offset + i + 1),
                          (offset + count, offset + count + i + 1, offset + count + i)))
        for i in range(count):
            j = (i + 1) % count
            faces.extend(((offset + i, offset + j, offset + count + i),
                          (offset + j, offset + count + j, offset + count + i)))
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32), color


def _shade_vertices(vertices, faces, colors):
    """Bake subtle shape shading so GL display works without a custom shader."""
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    magnitude = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals /= np.maximum(magnitude, 1e-12)
    light = np.array([-.35, -.45, .82])
    light /= np.linalg.norm(light)
    factor = .70 + .30 * np.abs(vertex_normals @ light)
    shaded = colors.copy()
    shaded[:, :3] *= factor[:, None]
    return shaded


def t600_mesh():
    """Return vertices, triangular faces and vertex colors for one model."""
    white = (.94, .95, .96, 1.)
    graphite = (.15, .17, .19, 1.)
    dark = (.07, .09, .11, 1.)
    silver = (.61, .64, .66, 1.)
    # Tapered central pod with a darker underside and a distinct rear battery.
    parts = [_hull([(-.29, .018, -.035, .035), (-.23, .095, -.075, .075),
                    (-.10, .13, -.085, .095), (.10, .125, -.083, .10),
                    (.24, .09, -.065, .085), (.32, .018, -.025, .035)], white),
             _ellipsoid((-.07, 0., -.058), (.21, .10, .045), graphite),
             _ellipsoid((-.23, 0., .055), (.085, .075, .035), silver),
             # Hanging gimbal mount and round front-facing lens.
             _beam((.12, 0., -.07), (.12, 0., -.135), .028, silver),
             _ellipsoid((.12, 0., -.17), (.073, .070, .067), silver),
             _beam((.18, 0., -.17), (.205, 0., -.17), .042, dark, sides=12),
             _ellipsoid((.207, 0., -.17), (.007, .030, .030), graphite)]
    for side in (-1, 1):
        pivot = np.array([-.005, side * .11, .005])
        shoulder = np.array([-.005, side * .275, .17])
        parts.append(_beam(pivot, shoulder, .023, graphite))
        parts.append(_ellipsoid(pivot, (.04, .025, .035), silver, sides=10, bands=4))
        for front in (-1, 1):
            motor = np.array([front * .31, side * .31, .19])
            parts.append(_beam(shoulder, motor, .014, graphite))
            parts.append(_beam(motor + [0., 0., -.025], motor + [0., 0., .012], .028, silver))
            parts.append(_ellipsoid(motor + [0., 0., .016], (.025, .025, .012), dark, sides=10, bands=4))
            parts.append(_propeller(motor + [0., 0., .032], side * front * .42, graphite))
    vertices, faces, colors = [], [], []
    offset = 0
    for xyz, triangles, color in parts:
        vertices.append(xyz)
        faces.append(triangles + offset)
        colors.append(np.tile(color, (len(xyz), 1)))
        offset += len(xyz)
    vertices = np.concatenate(vertices).astype(np.float32)
    faces = np.concatenate(faces)
    colors = np.concatenate(colors).astype(np.float32)
    return vertices, faces, _shade_vertices(vertices, faces, colors).astype(np.float32)


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

"""Render bounded time-domain B-scan curtains in an existing 3-D scene."""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QEvent, Qt

from ui.desktop_backend_facade import build_curtain_geometry
from ui.widgets.map_tiles import extract_epsg
from ui.widgets.terrain_tiles import sample_bilinear
from ui.widgets.curtain_picking import pick_mesh

_LOGGER = logging.getLogger(__name__)


class TrajectoryCurtainsMixin:
    def set_curtains(self, previews, start_ns, end_ns):
        self._curtain_previews = tuple(previews or ())
        self._curtain_window = (float(start_ns), float(end_ns))
        self._render_curtains()

    def clear_curtains(self):
        self._curtain_previews = ()
        self._clear_curtain_items()
        self.curtain_notice.emit('')

    def _clear_curtain_items(self):
        if self._gl_view is not None:
            for item in self._curtain_items:
                self._gl_view.removeItem(item)
        self._curtain_items = []
        self._curtain_pick_data = []

    def _render_curtains(self):
        self._clear_curtain_items()
        if (self._gl_view is None or self._origin is None or not self._curtain_previews
                or not self._track_data):
            return
        # These are display-only previews. A time-domain vertical display scale
        # is deliberately unrelated to actual electromagnetic depth.
        from ui.widgets.trajectory_3d_view import _gl_state, _lonlat_xyz_to_mercator
        gl, _ = _gl_state()
        if gl is None:
            return
        tracks = {line_id for line_id, _color, _xyz in self._track_data}
        previews = [p for p in self._curtain_previews if p.source.line_id in tracks]
        if not previews:
            return
        scene_epsg = self._terrain_src[0] if self._terrain_src is not None else None
        tmin = min(float(p.time_ns[0]) for p in previews)
        tmax = max(float(p.time_ns[-1]) for p in previews)
        scale = max(float(self._extent), 1.) * .35 / max(tmax - tmin, 1e-9)
        skipped = 0
        drawn = 0
        for preview in previews:
            try:
                epsg = extract_epsg(preview.coordinate_system)
                if scene_epsg is None or epsg != scene_epsg:
                    skipped += 1
                    continue
                xyz = np.column_stack((preview.x, preview.y, preview.anchor_z)).astype(float)
                if epsg == 4326:
                    xyz = _lonlat_xyz_to_mercator(xyz)
                local = xyz - self._origin
                payload = self._terrain_payload
                if self._drape and payload is not None:
                    zs = sample_bilinear(payload['z'], payload['gx'], payload['gy'], xyz[:, 0], xyz[:, 1])
                    lift = max(self._extent * .05, 2.)
                    zt = np.where(np.isfinite(zs), zs + lift, xyz[:, 2])
                    local[:, 2] = (zt - float(np.mean(payload['z']))) * self._exag
                axis = np.asarray(preview.time_ns, dtype=float) - tmin
                mesh = build_curtain_geometry(local[:, 0], local[:, 1], local[:, 2], axis,
                                              preview.matrix, vertical_scale=scale,
                                              connected=preview.connected)
                if not len(mesh.faces):
                    skipped += 1
                    continue
                finite = mesh.values[np.isfinite(mesh.values)]
                if not len(finite):
                    skipped += 1
                    continue
                low, high = np.percentile(finite, [2., 98.])
                level = (np.full_like(mesh.values, .5) if high <= low else
                         np.clip((mesh.values - low) / (high - low), 0., 1.))
                level = np.nan_to_num(level, nan=.5)
                colors = np.column_stack((level, level, level, np.ones(len(level)))).astype(np.float32)
                item = gl.GLMeshItem(vertexes=np.nan_to_num(mesh.vertices).astype(np.float32),
                                     faces=mesh.faces, vertexColors=colors, smooth=True,
                                     computeNormals=False, glOptions='opaque')
                self._gl_view.addItem(item)
                self._curtain_items.append(item)
                self._curtain_pick_data.append((preview, mesh.vertices, mesh.faces))
                drawn += 1
                center = sum(self._curtain_window) / 2.
                if preview.time_ns[0] <= center <= preview.time_ns[-1]:
                    marker = local.copy()
                    marker[:, 2] -= (center - tmin) * scale
                    self._render_curtain_marker(gl, marker, preview.connected)
            except Exception as exc:  # noqa: BLE001 - one invalid preview must not break the scene
                skipped += 1
                _LOGGER.warning('三维剖面 %s 无法显示: %s', preview.source.line_id, exc)
        notice = '时间域示意，垂直比例不代表地下深度'
        if skipped:
            notice += f'；{skipped} 条剖面无法显示，请检查坐标和轨迹'
        if not drawn:
            notice += '；当前没有可显示的剖面'
        self.curtain_notice.emit(notice)

    def _render_curtain_marker(self, gl, xyz, connected):
        valid = np.isfinite(xyz).all(axis=1)
        links = np.asarray(connected, dtype=bool) & valid[:-1] & valid[1:]
        indices = np.flatnonzero(links)
        if not len(indices):
            return
        vertices = xyz[np.column_stack((indices, indices + 1)).ravel()]
        item = gl.GLLinePlotItem(pos=vertices, color=(.1, .45, 1., 1.),
                                 width=3., mode='lines', glOptions='opaque')
        self._gl_view.addItem(item)
        self._curtain_items.append(item)

    def eventFilter(self, watched, event):
        if watched is self._gl_view and self._curtain_pick_data:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._curtain_press = event.position()
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                press = self._curtain_press
                self._curtain_press = None
                if press is not None and (event.position() - press).manhattanLength() <= 4:
                    self._pick_curtain(event.position())
        return super().eventFilter(watched, event)

    def _pick_curtain(self, point):
        view = self._gl_view
        width, height = view.width(), view.height()
        if width <= 0 or height <= 0:
            return
        viewport = view.getViewport()
        transform = view.projectionMatrix(viewport, viewport) * view.viewMatrix()
        matrix = np.asarray(transform.copyDataTo(), dtype=float).reshape(4, 4)
        best = None
        for preview, vertices, faces in self._curtain_pick_data:
            found = pick_mesh(vertices, faces, matrix, width, height, (point.x(), point.y()))
            if found is not None and (best is None or found[0] < best[0]):
                best = (found[0], preview, found[1])
        if best is None:
            return
        _depth, preview, vertex = best
        sample, trace = divmod(vertex, preview.matrix.shape[1])
        self.curtain_selected.emit(preview.source.line_id, preview.source.artifact_id,
                                   int(preview.trace_indices[trace]), float(preview.time_ns[sample]))

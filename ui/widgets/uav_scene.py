"""Display a single illustrative aircraft in the spatial 3-D scene."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QLabel

from ui.widgets.terrain_tiles import sample_bilinear
from ui.widgets.uav_t600_model import place_t600, t600_mesh


class UAVSceneMixin:
    def _init_uav(self):
        self._uav_visible = True
        self._uav_item = None
        self._uav_mesh = None
        self._uav_caption = QLabel('T600 示意（位置、朝向非实测）', self)
        self._uav_caption.setVisible(False)
        self.layout().addWidget(self._uav_caption)

    def set_uav_visible(self, visible):
        self._uav_visible = bool(visible)
        self._render_uav()

    def _clear_uav(self):
        self._uav_caption.setVisible(False)
        if self._uav_item is not None and self._gl_view is not None:
            self._gl_view.removeItem(self._uav_item)
        self._uav_item = None

    def _render_uav(self):
        self._clear_uav()
        if not self._uav_visible or self._gl_view is None or self._origin is None or not self._track_data:
            return
        from ui.widgets.trajectory_3d_view import _gl_state
        gl, _ = _gl_state()
        if gl is None or not hasattr(gl, 'GLMeshItem'):
            return
        if self._uav_mesh is None:
            self._uav_mesh = t600_mesh()
        for _line_id, _color, xyz in self._track_data:
            track = np.asarray(xyz, dtype=float).copy()
            if self._drape and self._terrain_payload is not None:
                payload = self._terrain_payload
                ground = sample_bilinear(payload['z'], payload['gx'], payload['gy'],
                                         track[:, 0], track[:, 1])
                track[:, 2] = np.where(np.isfinite(ground), ground + max(self._extent * .05, 2.), track[:, 2])
            vertices, faces, colors = self._uav_mesh
            positioned = place_t600(vertices, track, self._origin, self._extent, self._exag)
            if positioned is None:
                continue
            self._uav_item = gl.GLMeshItem(vertexes=positioned.astype(np.float32), faces=faces,
                                           vertexColors=colors, smooth=True, computeNormals=False,
                                           glOptions='opaque')
            self._gl_view.addItem(self._uav_item)
            self._uav_caption.setVisible(True)
            return

"""Synchronize compatible B-scan workspace views."""

from ui.widgets.bscan_link import map_point, map_range


class BScanWorkspaceLink:
    def _compatible_groups(self, source):
        record = self._sources.get(source.current_key(), {})
        identity = (record.get('project_id'), record.get('line_id'))
        if not all(identity):
            return []
        return [group for group in self._groups.values() if group is not source and group.view._matrix is not None
                and (self._sources.get(group.current_key(), {}).get('project_id'),
                     self._sources.get(group.current_key(), {}).get('line_id')) == identity]

    def _set_link_enabled(self, enabled):
        self._link_enabled = bool(enabled)
        if not self._link_enabled:
            for group in self._groups.values():
                group.view._hide_crosshair()

    def _link_range(self, source):
        if not self._link_enabled or self._link_busy or source.view._image_shape is None:
            return
        view = source.view
        (left, right), (top, bottom) = view._plot.vb.viewRange()
        width, height = view._image_shape
        left, right = max(0., left), min(float(width), right)
        top, bottom = max(0., top), min(float(height), bottom)
        self._link_busy = True
        try:
            for target in self._compatible_groups(source):
                mapped = map_range(view, target.view, ((left, right), (top, bottom)))
                if mapped is not None:
                    target.view._plot.vb.setRange(xRange=mapped[0], yRange=mapped[1], padding=0)
        finally:
            self._link_busy = False

    def _link_cursor(self, source, pos):
        if not self._link_enabled or source.view._matrix is None or not source.view._plot.sceneBoundingRect().contains(pos):
            return
        point = source.view._plot.vb.mapSceneToView(pos)
        for target in self._compatible_groups(source):
            mapped = map_point(source.view, target.view, point.x(), point.y())
            if mapped is None:
                target.view._hide_crosshair()
                continue
            target.view._vline.setPos(mapped[0])
            target.view._hline.setPos(mapped[1])
            target.view._vline.setVisible(True)
            target.view._hline.setVisible(True)

"""Marker overlay lifecycle, independent of terrain and depth rendering."""
class SpatialMarkersMixin:
    def set_marker_rows(self, rows):
        from ui.widgets.marker_map_overlay import MarkerMapOverlay
        if not hasattr(self, '_marker_overlay'):
            self._marker_overlay = MarkerMapOverlay(self._map_view)
            self._marker_overlay.selected.connect(lambda line_id, uid:
                self.marker_browser.selection_changed.emit(self.marker_browser.project_id, line_id, uid))
        self._marker_overlay.set_rows(rows)

    def locate_marker(self, row):
        existing = getattr(getattr(self, '_marker_overlay', None), 'rows', ())
        if not any(item['uid'] == row['uid'] for item in existing):
            self.set_marker_rows((*existing, row))
        if not self._marker_overlay.select(row['uid'], locate=True):
            self._crs_label.setText('该记录无有效空间位置；保留在标记列表中')

    def select_marker(self, uid):
        if hasattr(self, '_marker_overlay'):
            self._marker_overlay.select(uid)

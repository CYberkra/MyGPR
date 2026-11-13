"""Spatial-page line object menus and their captured-target guards.

The host owns tracks, the list and navigation signals. This mixin keeps mouse
and keyboard menu binding with the menu implementation, while preserving the
page's existing _line_menu and _line_context interfaces.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ui.widgets.object_context_menu import ObjectMenuBinding, object_action, object_menu


class SpatialContextMenuMixin:
    """Object-menu behavior shared only with the SpatialPage host."""

    def _bind_line_context(self, map_view_key):
        self._line_menu_map_key = map_view_key
        self._line_context = ObjectMenuBinding(self, self._line_list, self._line_menu)
        self._map_view.add_line_menu_actions = self._add_line_menu_actions

    def _line_menu(self, pos):
        menu = object_menu(self)
        item = self._line_list.itemAt(pos)
        line_id = str(item.data(Qt.ItemDataRole.UserRole) or '') if item else ''
        if line_id:
            self._add_line_menu_actions(menu, line_id)
            menu.addSeparator()
        object_action(menu, 'fit-tracks', '适应全部测线', self._map_view.fit_to_tracks,
                      enabled=bool(self._tracks) and self._active_view_key() == self._line_menu_map_key,
                      reason='请先切换有轨迹的平面地图')
        return menu

    def _add_line_menu_actions(self, menu, line_id):
        track = self._tracks_by_id.get(line_id)

        def current_item():
            if not line_id or self._tracks_by_id.get(line_id) is not track:
                return None
            return next((self._line_list.item(row) for row in range(self._line_list.count())
                         if self._line_list.item(row).data(Qt.ItemDataRole.UserRole) == line_id), None)

        item = current_item()
        if item is not None:
            object_action(menu, 'set-current', '设为当前测线',
                          lambda: self.current_line_requested.emit(line_id) if current_item() is not None else None)
            object_action(menu, 'copy-id', '复制测线标识',
                          lambda: QApplication.clipboard().setText(line_id) if current_item() is not None else None)
            checked = item.checkState() == Qt.CheckState.Checked

            def toggle():
                target = current_item()
                if target is not None:
                    target.setCheckState(Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked)

            visibility = object_action(menu, 'toggle-visible', '显示测线', toggle)
            visibility.setCheckable(True)
            visibility.setChecked(checked)
            object_action(menu, 'zoom-line', '缩放到此测线',
                          lambda: self._map_view.fit_to_line(line_id) if current_item() is not None else None,
                          enabled=self._map_view.can_fit_line(line_id)
                          and self._active_view_key() == self._line_menu_map_key,
                          reason='需要平面地图中的有效轨迹')

    def _on_set_current_clicked(self) -> None:
        line_id = self._selected_line_id()
        if line_id:
            self.current_line_requested.emit(line_id)

    def _on_prefetch_view_clicked(self):
        if self._active_view_key() != self._line_menu_map_key or self._map_view.tracks_bbox_lonlat() is None:
            self._prefetch_label.setText('请先打开包含已配准轨迹的平面地图')
            return
        count = self._map_view.prefetch_current_view()
        self._prefetch_label.setText(f'正在下载 {count} 张瓦片…' if count else '当前视野没有待下载瓦片')
        self.basemap_prefetch_requested.emit()

    def _on_depth_slider_changed(self, value: int) -> None:
        depth_m = value / 100.0
        self._depth_value_label.setText(f'深度: {depth_m:.2f} m')
        self._depth_view.set_isoline(depth_m)

    def _on_depth_cell_changed(self, _value):
        self.clear_depth_grid()
        checked = [str(t.line_id) for t in self._checked_tracks()]
        if checked:
            self.depth_preview_requested.emit(checked, float(self._depth_cell_spin.value()))

    def _on_save_depth_layer_clicked(self) -> None:
        line_ids = list(self._depth_payload_line_ids)
        if not line_ids or self._depth_view.value_range() is None:
            return
        self.save_depth_layer_requested.emit(
            line_ids, float(self._depth_cell_spin.value()))

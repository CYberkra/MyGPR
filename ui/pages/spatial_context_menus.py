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

    def _line_menu(self, pos):
        menu = object_menu(self)
        item = self._line_list.itemAt(pos)
        line_id = str(item.data(Qt.ItemDataRole.UserRole) or '') if item else ''
        track = self._tracks_by_id.get(line_id)

        def current_item():
            if not line_id or self._tracks_by_id.get(line_id) is not track:
                return None
            return next((self._line_list.item(row) for row in range(self._line_list.count())
                         if self._line_list.item(row).data(Qt.ItemDataRole.UserRole) == line_id), None)

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
            object_action(menu, 'zoom-line', '缩放到此测线', lambda: None,
                          enabled=False, reason='当前仅支持全部测线范围')
            menu.addSeparator()
        object_action(menu, 'fit-tracks', '适应全部测线', self._map_view.fit_to_tracks,
                      enabled=bool(self._tracks) and self._active_view_key() == self._line_menu_map_key,
                      reason='请先切换有轨迹的平面地图')
        return menu

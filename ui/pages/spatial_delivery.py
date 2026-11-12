"""Spatial result selection in the existing map tools sidebar."""
from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import PushButton
from qfluentwidgets import FluentIcon as FIF

from ui.pages.delivery_page import DeliveryPage
from ui.widgets.marker_browser import MarkerBrowserButton


class SpatialDeliveryMixin:
    def _build_spatial_header(self, layout):
        row = QHBoxLayout()
        self.marker_browser = MarkerBrowserButton(self)
        row.addWidget(self.marker_browser)
        self._spatial_results_btn = PushButton('空间成果', self, FIF.GLOBE)
        self._spatial_results_btn.setCheckable(True)
        self._spatial_results_btn.toggled.connect(self._show_spatial_delivery)
        row.addWidget(self._spatial_results_btn)
        row.addStretch(1)
        layout.addLayout(row)

    def _build_spatial_delivery(self, layout, map_route):
        self._delivery_map_route = map_route
        self._spatial_tools_stack = QStackedWidget(self)
        layout.addWidget(self._spatial_tools_stack, 1)
        details = QWidget(self._spatial_tools_stack)
        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self._spatial_tools_stack.addWidget(details)
        self.delivery_tools = DeliveryPage(self._spatial_tools_stack)
        self.delivery_tools.compact_spatial_table()
        self.delivery_tools.spatial_locate_requested.connect(self._locate_result_tracks)
        self._spatial_tools_stack.addWidget(self.delivery_tools)
        return detail_layout

    def _show_spatial_delivery(self, visible):
        self._spatial_tools_stack.setCurrentIndex(int(visible))
        if visible:
            self._right_panel.set_collapsed(False)

    def open_spatial_result(self, result_id):
        if not self.delivery_tools.focus_spatial_result(result_id):
            return False
        self._spatial_results_btn.setChecked(True)
        self._show_spatial_delivery(True)
        return self.delivery_tools.focus_spatial_result(result_id, locate=True)

    def _locate_result_tracks(self, line_ids):
        located = set(line_ids) & self._tracks_by_id.keys()
        if not located:
            return
        with QSignalBlocker(self._line_list):
            for row in range(self._line_list.count()):
                item = self._line_list.item(row)
                matched = str(item.data(Qt.ItemDataRole.UserRole) or '') in located
                item.setCheckState(Qt.CheckState.Checked if matched else Qt.CheckState.Unchecked)
        self._view_segment.setCurrentItem(self._delivery_map_route)
        self._switch_view(self._delivery_map_route)
        self._refresh_views()
        self.clear_depth_grid()
        self._map_view.fit_to_tracks()

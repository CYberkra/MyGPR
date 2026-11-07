"""Saved amplitude slices and 3-D time-window navigation for SpatialPage."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, PushButton, Slider

from ui.desktop_backend_facade import prepare_amplitude_sources
from ui.widgets.curtain_bscan_panel import CurtainBScanPanel

_SEG_3D = 'trajectory3d'
_SEG_AMPLITUDE = 'amplitudeSlice'
_SEG_DEPTH = 'depthSlice'


class SpatialAmplitudeMixin:
    def _connect_curtain_controls(self):
        self._3d_view.curtain_notice.connect(self._on_curtain_notice)
        self._3d_view.curtain_selected.connect(self._amplitude_view.select_curtain)
        self._amplitude_view.curtains_ready.connect(self._set_curtains)
        self._amplitude_view.show_3d_requested.connect(self._show_amplitude_3d)
        self._curtain_slider.valueChanged.connect(
            lambda value: self._curtain_time.setText(f'时间: {value / 10:g} ns'))
        self._curtain_slider.sliderReleased.connect(
            lambda: self._amplitude_view.set_center_and_request(self._curtain_slider.value() / 10.))
        self._curtain_back.clicked.connect(self._show_amplitude_2d)
        self._curtain_bscan.line_selected.connect(self._amplitude_view.select_line)
        self._amplitude_view.current_line_changed.connect(self._curtain_bscan.select_line)
        self._amplitude_view.bscan_changed.connect(self._curtain_bscan.set_bundle)

    def _sync_view_affordances(self, route_key):
        self._depth_row_widget.setVisible(route_key == _SEG_DEPTH)
        self._curtain_notice.setVisible(route_key == _SEG_3D and bool(self._curtain_notice.text()))
        self._curtain_row.setVisible(route_key == _SEG_3D and self._curtain_session)
        showing_curtains = route_key == _SEG_3D and self._curtain_session
        self._curtain_bscan.setVisible(showing_curtains)
        self._spatial_detail_card.setVisible(not showing_curtains)
        self._spatial_coverage_card.setVisible(not showing_curtains)
        if route_key == _SEG_3D and self._view3d_card.is_collapsed():
            self._view3d_card.set_collapsed(False, animate=True)

    def _build_curtain_details(self, layout, detail_card, coverage_card):
        self._spatial_detail_card = detail_card
        self._spatial_coverage_card = coverage_card
        self._curtain_bscan = CurtainBScanPanel(self)
        self._curtain_bscan.setVisible(False)
        layout.addWidget(self._curtain_bscan, 1)

    def _build_curtain_controls(self, parent, layout):
        self._curtain_notice = CaptionLabel('', parent)
        self._curtain_notice.setVisible(False)
        layout.addWidget(self._curtain_notice)
        self._curtain_row = QWidget(parent)
        row = QHBoxLayout(self._curtain_row)
        row.setContentsMargins(0, 0, 0, 0)
        self._curtain_time = CaptionLabel('时间: --', self._curtain_row)
        self._curtain_slider = Slider(Qt.Orientation.Horizontal, self._curtain_row)
        self._curtain_slider.setRange(0, 10000)
        self._curtain_back = PushButton('平面切片', self._curtain_row)
        row.addWidget(self._curtain_time)
        row.addWidget(self._curtain_slider, 1)
        row.addWidget(self._curtain_back)
        layout.addWidget(self._curtain_row)
        self._curtain_row.setVisible(False)
        self._has_curtains = False
        self._curtain_session = False

    def _show_amplitude_3d(self):
        self._curtain_session = True
        self._curtain_bscan.set_lines(self._amplitude_view._tracks,
                                      self._amplitude_view._line.currentData())
        if self._right_panel.is_collapsed():
            self._right_panel.set_collapsed(False, animate=True)
        self._view_segment.setCurrentItem(_SEG_3D)
        self._switch_view(_SEG_3D)
        self._3d_view._reset_camera()

    def _show_amplitude_2d(self):
        self._view_segment.setCurrentItem(_SEG_AMPLITUDE)
        self._switch_view(_SEG_AMPLITUDE)

    def show_saved_amplitude(self, loaded):
        result = loaded.result
        selected_ids = {source.line_id for source in result.sources}
        tracks = [track for track in self._tracks if str(track.line_id) in selected_ids]
        self._line_list.blockSignals(True)
        try:
            for row in range(self._line_list.count()):
                item = self._line_list.item(row)
                item.setCheckState(Qt.CheckState.Checked if str(item.data(Qt.ItemDataRole.UserRole)) in selected_ids
                                   else Qt.CheckState.Unchecked)
        finally:
            self._line_list.blockSignals(False)
        self._refresh_views()
        positions, warning = (), ''
        if not loaded.stale and len(tracks) == len(selected_ids):
            try:
                source_ids = {source.line_id: source.artifact_id for source in result.sources}
                _sources, positions = prepare_amplitude_sources(
                    tracks, source_ids, '', target_crs=result.coordinate_system)
                hashes = {source.line_id: source.position_sha256 for source in result.sources}
                if any(hashes.get(item.source.line_id) and hashes[item.source.line_id] != item.source_position_sha256
                       for item in positions):
                    positions = ()
                    warning = '轨迹已变化，仅显示保存时的切片'
            except (ValueError, OSError, RuntimeError) as exc:
                warning = f'已打开保存的切片，轨迹不可用：{exc}'
        elif not loaded.stale:
            warning = '部分测线不可用，仅显示保存时的切片'
        self._amplitude_view.show_saved(loaded, tracks, self._colors, positions, warning)
        self._views_applied[_SEG_AMPLITUDE] = self._tracks_fingerprint(tracks, self._colors)
        self._dirty_views.discard(_SEG_AMPLITUDE)
        self._view_segment.setCurrentItem(_SEG_AMPLITUDE)
        self._switch_view(_SEG_AMPLITUDE)

    def _set_curtains(self, previews, start_ns, end_ns):
        self._has_curtains = bool(previews)
        if self._has_curtains:
            bounds = self._amplitude_view.center_bounds()
            self._curtain_slider.setEnabled(bounds is not None)
            if bounds is not None:
                self._curtain_slider.setRange(*bounds)
            self._curtain_slider.setValue(round((float(start_ns) + float(end_ns)) * 5))
            self._curtain_notice.setText('时间域示意，垂直比例不代表地下深度')
        elif self._curtain_session:
            self._curtain_notice.setText('切片已变化，请等待结果或返回平面切片')
        else:
            self._curtain_notice.setText('')
        self._3d_view.set_curtains(previews, start_ns, end_ns)
        self._sync_view_affordances(self._view_segment.currentRouteKey())

    def _on_curtain_notice(self, text):
        self._curtain_notice.setText(str(text))
        self._curtain_notice.setVisible(bool(text) and self._view_stack.currentWidget() is self._3d_view)

    def amplitude_preview_failed(self, message):
        if self._curtain_session and self._view_stack.currentWidget() is self._3d_view:
            self._curtain_notice.setText(str(message))
            self._curtain_notice.setVisible(True)

"""Sensor synchronization form and application payload contracts."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('PyQt6')


def test_advanced_sensor_settings_are_mapped_without_manual_json(qapp):
    from ui.pages.sensor_sync_form import SensorSyncForm

    form = SensorSyncForm()
    form._fields['rtk_time_offset_s'].setValue(1.25)
    form._fields['maximum_nearest_residual_s'].setValue(.5)
    form._fields['lever_arm_x_m'].setValue(.2)
    form._fields['lever_arm_y_m'].setValue(-.1)
    form._fields['lever_arm_z_m'].setValue(1.5)
    form._fields['radar_trigger_delay_s'].setValue(.03)
    form._fields['antenna_height_offset_m'].setValue(.4)
    form._fields['allow_extrapolation'].setChecked(True)
    form._fields['project_crs'].setText('EPSG:4547')
    values = form.settings()
    assert values['rtk_time_offset_s'] == 1.25
    assert values['maximum_nearest_residual_s'] == .5
    assert values['lever_arm_y_m'] == -.1
    assert values['lever_arm_z_m'] == 1.5
    assert values['radar_trigger_delay_s'] == .03
    assert values['allow_extrapolation'] is True
    assert values['project_crs'] == 'EPSG:4547'
    form.set_settings(values)
    assert form.settings() == values
    form.close()
    form.deleteLater()


def test_extrapolation_option_describes_endpoint_and_limit(qapp):
    from ui.pages.sensor_sync_form import SensorSyncForm
    form = SensorSyncForm()
    text = form._fields['allow_extrapolation'].text()
    assert '端点' in text and '残差' in text
    form.close()


def test_project_sync_request_contains_form_settings(qapp):
    from ui.pages.project_page import ProjectPage

    page = ProjectPage()
    page.set_project_info(NS(read_only=False))
    page.set_lines([NS(line_id='L01')])
    page.select_line('L01')
    page._sensor_edits['rtk'].setText('D:/data/rtk.csv')
    page._sync_form._fields['rtk_time_offset_s'].setValue(.75)
    requests = []
    page.sync_requested.connect(requests.append)
    page._emit_sync_request()
    assert requests[0]['settings']['rtk_time_offset_s'] == .75
    page.close()
    page.deleteLater()


def test_sync_dialog_allows_scrolling_to_project_crs(qapp):
    from PyQt6.QtWidgets import QScrollArea
    from ui.pages.project_page import ProjectPage

    page = ProjectPage()
    scroll = page._sync_dialog.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.widget().isAncestorOf(page._sync_form._fields['project_crs'])
    page.close()


def test_unfocused_wheel_does_not_change_sync_parameter(qapp):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from ui.pages.sensor_sync_form import SensorSyncForm

    form = SensorSyncForm()
    spin = form._fields['rtk_time_offset_s']
    event = QWheelEvent(QPointF(2, 2), QPointF(2, 2), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    qapp.sendEvent(spin, event)
    assert spin.value() == 0.0


def test_sync_result_shows_warning_before_raw_diagnostics(qapp):
    from ui.widgets.sensor_details_panel import SensorDetailsPanel

    panel = SensorDetailsPanel()
    panel.set_sync_result(NS(summary='已同步', diagnostics={
        'warnings': ['RTK 固定解覆盖率低于建议值 90%'], 'fixed_solution_ratio': 0.0}, config={}))
    assert 'RTK 固定解覆盖率低于建议值' in panel._warning_label.text()


def test_project_metadata_is_shown_for_current_project_only(qapp):
    from ui.pages.project_page import ProjectPage

    page = ProjectPage()
    page.set_project_info(NS(project_id='P1', name='Test', root_path='D:/test'))
    page.set_project_metadata(NS(project_id='P1', coordinate_system='EPSG:4523',
                                 vertical_datum='1985 国家高程基准'))
    assert page._info_values['coordinate_system'].text() == 'EPSG:4523'
    page.set_project_metadata(NS(project_id='old', coordinate_system='EPSG:4547'))
    assert page._info_values['coordinate_system'].text() == 'EPSG:4523'


def test_switching_projects_clears_previous_sync_result(qapp):
    from ui.pages.project_page import ProjectPage

    page = ProjectPage()
    page.set_project_info(NS(project_id='P1', root_path='D:/p1'))
    page.set_sensor_sync_result('L01', NS(diagnostics={'warnings': ['old project']}))
    page.set_project_info(NS(project_id='P2', root_path='D:/p2'))
    assert not page._sensor_sync_results


def test_import_form_submits_opt_in_matrix_axes(qapp):
    from ui.pages.project_page import ProjectPage

    page = ProjectPage()
    page.file_edit.setText('D:/plain.npy')
    page._matrix_axes_check.setChecked(True)
    page._matrix_time_window.setValue(48.0)
    page._matrix_trace_interval.setValue(.25)
    requests = []
    page.import_requested.connect(requests.append)
    page._emit_import_request(preflight=False)
    assert requests[0]['time_window_ns'] == 48.0
    assert requests[0]['trace_interval_m'] == .25

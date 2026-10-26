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
    assert values['lever_arm_z_m'] == 1.5
    assert values['radar_trigger_delay_s'] == .03
    assert values['allow_extrapolation'] is True
    assert values['project_crs'] == 'EPSG:4547'
    form.set_settings(values)
    assert form.settings() == values
    form.close()
    form.deleteLater()


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

"""The sensor panel exposes availability, counts and failures without parsing text."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import Qt


def _page():
    from ui.pages.project_page import ProjectPage
    page = ProjectPage()
    page.set_project_info(NS(project_id='A', root_path='A', name='A'))
    page.set_lines([NS(line_id='L01', name='L01', trace_count=3, sample_count=501),
                    NS(line_id='L02', name='L02', trace_count=3, sample_count=501)])
    page.select_line('L01')
    return page


def test_panel_shows_partial_missing_and_unstatistical_rows(qapp):
    from ui.sensor_summary import sensor_completeness_report
    page = _page()
    try:
        report = sensor_completeness_report({
            'longitude': [104, 104, 104], 'latitude': [30, 30, 30],
            'flight_height_m': [2, np.nan, 0],
            'roll_deg': [0], 'pitch_deg': [0], 'yaw_deg': [0],
        }, 3)
        page.set_line_sensor_report('L01', report)
        cells = page._sensor_detail._cells
        assert [cell.text() for cell in cells[0]] == ['RTK 经纬度', '完整', '3 / 3（100.0%）', '0 道']
        assert [cell.text() for cell in cells[1]] == ['高度计', '部分有效', '1 / 3（33.3%）', '2 道']
        assert [cell.text() for cell in cells[2]] == ['IMU 姿态', '长度不符', '无法统计', '无法统计']
        assert [cell.text() for cell in cells[3]] == ['IMU 加速度', '未提供', '—', '3 道']
        assert '固定解' in page._sensor_detail._note.text()
        assert cells[1][2].textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    finally:
        page.close()


def test_loading_error_and_new_project_never_leave_previous_counts(qapp):
    from ui.sensor_summary import sensor_completeness_report, SensorCompletenessReport
    page = _page()
    try:
        assert '正在检查' in page._sensor_detail.text()
        page.set_line_sensor_report('L01', sensor_completeness_report({'flight_height_m': [2, 2, 2]}, 3))
        page.select_line('L02')
        assert '正在检查' in page._sensor_detail.text()
        assert all(not cell.text() for cells in page._sensor_detail._cells for cell in cells)
        error = SensorCompletenessReport(3, (), error='传感器信息读取失败：损坏；完整性尚未确认。')
        page.set_line_sensor_report('L02', error)
        assert '读取失败' in page._sensor_detail.text()
        assert '未提供' not in page._sensor_detail.text()
        assert all(not cell.text() for cells in page._sensor_detail._cells for cell in cells)
        page.set_project_info(NS(project_id='B', root_path='B', name='B'))
        page.set_lines([NS(line_id='L01')])
        page.select_line('L01')
        assert '正在检查' in page._sensor_detail.text()
        assert '损坏' not in page._sensor_detail.text()
    finally:
        page.close()


def test_invalid_values_are_distinct_from_missing_and_zero_traces(qapp):
    from ui.sensor_summary import sensor_completeness_report
    page = _page()
    try:
        page.set_line_sensor_report('L01', sensor_completeness_report({'flight_height_m': [0, -1, np.nan]}, 3))
        assert [cell.text() for cell in page._sensor_detail._cells[1]][1:] == [
            '无有效值', '0 / 3（0.0%）', '3 道']
        page.set_line_sensor_report('L01', sensor_completeness_report({}, 0))
        assert '无雷达道' in page._sensor_detail.text()
        assert '100.0%' not in page._sensor_detail.text()
        assert all(not cell.text() for cells in page._sensor_detail._cells for cell in cells)
    finally:
        page.close()


def test_theme_roundtrip_and_short_pane_preserve_all_sensor_information(qapp):
    from ui.design_tokens import color
    from ui.sensor_summary import sensor_completeness_report
    page = _page()
    try:
        page.set_line_sensor_report('L01', sensor_completeness_report({'flight_height_m': [2, np.nan, 0]}, 3))
        panel = page._sensor_detail
        page._detail_tabs.setCurrentIndex(1)
        page.resize(1200, 700)
        page.show()
        scroll = panel.parentWidget().parentWidget().parentWidget()
        scroll.setFixedSize(440, 130)
        for dark in (False, True, False):
            page.apply_theme(dark)
            qapp.processEvents()
            assert color('warning', dark) in panel._cells[1][1].styleSheet()
            assert color('text_secondary', dark) in panel._cells[1][2].styleSheet()
            assert scroll.horizontalScrollBar().maximum() == 0
            assert scroll.verticalScrollBar().maximum() > 0
            scroll.ensureWidgetVisible(panel._note)
            qapp.processEvents()
            assert scroll.verticalScrollBar().value() > 0
            assert '固定解' in panel._note.text()
    finally:
        page.close()

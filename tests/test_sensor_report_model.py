"""Structured sensor availability without parsing display strings or changing validity."""
from dataclasses import FrozenInstanceError
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest


def test_report_is_immutable_and_preserves_joint_validity_and_legacy_text():
    from ui.sensor_summary import sensor_completeness_report, sensor_completeness_text
    metadata = {
        'longitude': [104, 104, 200], 'latitude': [30, np.nan, 30],
        'flight_height_m': [2, np.nan, 0],
        'roll_deg': [1, 2, np.nan], 'pitch_deg': [1, 2, 3], 'yaw_deg': [1, 2, 3],
        'trace_timestamp_s': [1700000000, np.nan, 1700000001],
    }
    report = sensor_completeness_report(metadata, 3)
    rows = {row.label: row for row in report.rows}
    assert report.total == 3 and len(report.rows) == 6
    assert (rows['RTK经纬度'].state, rows['RTK经纬度'].valid_count,
            rows['RTK经纬度'].missing_count) == ('partial', 1, 2)
    assert rows['高度'].valid_count == 1
    assert rows['IMU姿态'].valid_count == 2
    assert rows['IMU加速度'].state == 'missing'
    assert rows['IMU加速度'].valid_count is None
    assert rows['IMU加速度'].missing_count == 3
    assert rows['采集时间'].valid_count == 2
    assert report.as_text() == sensor_completeness_text(metadata, 3)
    assert 'RTK经纬度：有效 1/3 道（33.3%），缺失/无效 2 道' in report.as_text()
    with pytest.raises(FrozenInstanceError):
        report.total = 1
    with pytest.raises(FrozenInstanceError):
        report.rows[0].state = 'complete'


@pytest.mark.parametrize(('metadata', 'expected_state', 'valid', 'missing'), [
    ({'flight_height_m': [1, 2, 3]}, 'complete', 3, 0),
    ({'flight_height_m': [1, 0, np.nan]}, 'partial', 1, 2),
    ({'flight_height_m': [0, -1, np.nan]}, 'invalid', 0, 3),
    ({}, 'missing', None, 3),
    ({'flight_height_m': ['unreadable', 2, 3]}, 'unreadable', None, None),
    ({'flight_height_m': [1]}, 'mismatch', None, None),
    ({'flight_height_m': [[1, 2, 3]]}, 'mismatch', None, None),
])
def test_height_states_preserve_unknown_counts(metadata, expected_state, valid, missing):
    from ui.sensor_summary import sensor_completeness_report
    row = sensor_completeness_report(metadata, 3).rows[1]
    assert (row.state, row.total, row.valid_count, row.missing_count) == (expected_state, 3, valid, missing)


def test_incomplete_group_is_not_claimed_valid_and_height_agl_keeps_priority():
    from ui.sensor_summary import sensor_completeness_report
    report = sensor_completeness_report({
        'longitude': [100, 101], 'height_agl_m': [0, np.nan], 'flight_height_m': [3, 4],
    }, 2)
    assert (report.rows[0].state, report.rows[0].valid_count, report.rows[0].missing_count) == (
        'incomplete', None, 2)
    assert report.rows[1].state == 'invalid'
    assert 'RTK经纬度：字段不全，缺失/无效 2 道' in report.as_text()


def test_empty_and_read_failure_reports_do_not_claim_sensor_absence():
    from ui.sensor_summary import SensorCompletenessReport, sensor_completeness_report
    empty = sensor_completeness_report({}, 0)
    assert empty.rows == ()
    assert empty.as_text() == '传感器完整性：无雷达道，无法统计。'
    error = '传感器信息读取失败：存储故障；完整性尚未确认。'
    failed = SensorCompletenessReport(total=3, rows=(), error=error)
    assert failed.as_text() == error and '未提供' not in failed.as_text()


def _controller(monkeypatch, read_metadata):
    from ui.controllers import project_controller as module
    controller = module.ProjectController()
    controller._current = NS(project_id='A')
    controller.set_backend(NS(backend=NS(projects=NS(
        read_trace_metadata=read_metadata,
        list_lines=lambda *args: [NS(line_id='L01', trace_count=3)],
    ))))
    commands = []
    monkeypatch.setattr(module, 'run_command', lambda command, **kwargs: commands.append(command))
    texts, reports = [], []
    controller.line_sensor_summary_ready.connect(lambda lid, text: texts.append((lid, text)))
    controller.line_sensor_report_ready.connect(lambda lid, report: reports.append((lid, report)))
    return controller, commands, texts, reports


def test_worker_emits_report_and_compatible_text_from_one_statistics_pass(qapp, monkeypatch):
    from ui.controllers import project_controller as module
    original = module.sensor_completeness_report
    passes = []
    def count_pass(metadata, total):
        passes.append((metadata, total))
        return original(metadata, total)
    monkeypatch.setattr(module, 'sensor_completeness_report', count_pass)
    controller, commands, texts, reports = _controller(monkeypatch, lambda *a: {'flight_height_m': [2, 2, 2]})
    controller.line_sensor_summary('L01')
    commands[-1].execute()
    assert len(passes) == len(texts) == len(reports) == 1
    assert texts[0] == ('L01', reports[0][1].as_text())
    assert reports[0][1].rows[1].state == 'complete'


@pytest.mark.parametrize('change', ['project', 'same_line', 'generation'])
def test_queued_reports_and_text_are_both_rejected_after_context_changes(qapp, monkeypatch, change):
    controller, commands, texts, reports = _controller(monkeypatch, lambda *a: {'flight_height_m': [2, 2, 2]})
    controller.line_sensor_summary('L01', 3)
    worker = threading.Thread(target=commands[-1].execute)
    worker.start()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert not texts and not reports
    if change == 'project':
        controller._current = NS(project_id='B')
    elif change == 'same_line':
        controller.line_sensor_summary('L01', 3)
    else:
        controller._project_context_generation += 1
    qapp.processEvents()
    assert not texts and not reports


def test_worker_read_failure_has_full_friendly_error_and_no_fake_missing_rows(qapp, monkeypatch):
    def fail(*args):
        raise OSError('broken storage')
    controller, commands, texts, reports = _controller(monkeypatch, fail)
    controller.line_sensor_summary('L01', 3)
    commands[-1].execute()
    assert len(texts) == len(reports) == 1
    report = reports[0][1]
    assert report.total == 3 and report.rows == ()
    assert report.error.startswith('传感器信息读取失败：')
    assert report.error.endswith('；完整性尚未确认。')
    assert texts[0][1] == report.as_text() == report.error
    assert '未提供' not in report.as_text()


class _Endpoint:
    def __init__(self):
        self.slots = []
        self.calls = []
    def connect(self, slot):
        self.slots.append(slot)
    def __call__(self, *args):
        self.calls.append(args)
    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


class _ProtocolStub:
    def __init__(self, missing=()):
        self._missing = missing
    def __getattr__(self, name):
        if name in self._missing:
            raise AttributeError(name)
        endpoint = _Endpoint()
        setattr(self, name, endpoint)
        return endpoint


@pytest.mark.parametrize(('controller_missing', 'page_missing', 'use_report'), [
    ((), (), True),
    (('line_sensor_report_ready',), (), False),
    ((), ('set_line_sensor_report',), False),
])
def test_coordinator_prefers_one_report_connection_and_falls_back_for_legacy_peers(
        controller_missing, page_missing, use_report):
    from ui.coordinator_project import ProjectChain
    project = _ProtocolStub(page_missing)
    pages = {'projectInterface': project}
    controller = _ProtocolStub(controller_missing)
    co = _ProtocolStub()
    co.page = lambda name: pages.setdefault(name, _ProtocolStub())
    co.project_controller = controller
    co.processing_controller = _ProtocolStub()
    co.interpretation_controller = None
    co.delivery_controller = None
    co.file_tree = lambda: None
    chain = ProjectChain(co)
    chain.connect_all()
    if use_report:
        assert controller.line_sensor_report_ready.slots == [project.set_line_sensor_report]
        assert controller.line_sensor_summary_ready.slots == []
    else:
        assert controller.line_sensor_summary_ready.slots == [project.set_line_sensor_summary]
        if not controller_missing:
            assert controller.line_sensor_report_ready.slots == []

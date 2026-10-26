"""User-visible sensor gaps and processing step outcomes, without real hardware."""
from types import SimpleNamespace as NS
import threading

import h5py
import numpy as np
import pytest


def test_summary_counts_joint_validity_and_keeps_missing_distinct():
    from ui.sensor_summary import sensor_completeness_text
    meta = {'longitude': [104, 104, 200], 'latitude': [30, np.nan, 30],
            'flight_height_m': [2, np.nan, 0],
            'roll_deg': [1, 2, np.nan], 'pitch_deg': [1, 2, 3], 'yaw_deg': [1, 2, 3],
            'trace_timestamp_s': [1700000000, np.nan, 1700000001]}
    text = sensor_completeness_text(meta, 3)
    assert 'RTK经纬度：有效 1/3 道（33.3%），缺失/无效 2 道' in text
    assert '高度：有效 1/3 道（33.3%），缺失/无效 2 道' in text
    assert 'IMU姿态：有效 2/3 道（66.7%），缺失/无效 1 道' in text
    assert 'IMU加速度：未提供' in text and 'IMU角速度：未提供' in text
    assert '采集时间：有效 2/3 道' in text
    assert '固定解' in text and '同步' in text


def test_summary_never_resizes_mismatched_fields_or_claims_empty_is_complete():
    from ui.sensor_summary import sensor_completeness_text
    assert '长度不符' in sensor_completeness_text({'flight_height_m': [2]}, 3)
    assert '无法统计' in sensor_completeness_text({}, 0)
    assert '100.0%' not in sensor_completeness_text({}, 0)


def test_project_sensor_details_are_visible_and_cleared_on_project_switch(qapp):
    from ui.pages.project_page import ProjectPage
    page = ProjectPage()
    try:
        page.set_project_info(NS(project_id='A', root_path='A', name='A'))
        page.set_lines([NS(line_id='L01', name='line', trace_count=3, sample_count=32)])
        assert page.select_line('L01')
        page.set_line_sensor_summary('L01', '高度：有效 1/3 道（33.3%），缺失/无效 2 道')
        assert '缺失/无效 2 道' in page._sensor_detail.text()
        page.set_project_info(NS(project_id='B', root_path='B', name='B'))
        page.set_lines([NS(line_id='L01', name='other', trace_count=3, sample_count=32)])
        page.select_line('L01')
        assert '缺失/无效 2 道' not in page._sensor_detail.text()
    finally:
        page.close()


def test_sensor_query_discards_old_project_and_same_line_superseded_results(qapp, monkeypatch):
    from ui.controllers import project_controller as module
    controller = module.ProjectController()
    controller._current = NS(project_id='A')
    controller.set_backend(NS(backend=NS(projects=NS(read_trace_metadata=lambda *a: {'flight_height_m': [2, 2, 2]}))))
    commands = []
    monkeypatch.setattr(module, 'run_command', lambda command, **kw: commands.append(command))
    received = []
    controller.line_sensor_summary_ready.connect(lambda lid, text: received.append((lid, text)))
    controller.line_sensor_summary('L01', 3)
    old = commands[-1]
    controller.line_sensor_summary('L01', 3)
    old.execute()
    assert not received
    commands[-1].execute()
    assert received and '有效 3/3 道' in received[-1][1]
    received.clear()
    controller.line_sensor_summary('L01', 3)
    worker = threading.Thread(target=commands[-1].execute)
    worker.start()
    worker.join(timeout=3)
    assert not worker.is_alive()
    controller._current = NS(project_id='B')
    qapp.processEvents()
    assert not received


def test_sensor_query_failure_is_not_reported_as_absent_sensors(qapp, monkeypatch):
    from ui.controllers import project_controller as module
    def fail(*args):
        raise OSError('broken storage')
    controller = module.ProjectController()
    controller._current = NS(project_id='A')
    controller.set_backend(NS(backend=NS(projects=NS(read_trace_metadata=fail))))
    commands = []
    monkeypatch.setattr(module, 'run_command', lambda command, **kw: commands.append(command))
    messages = []
    controller.line_sensor_summary_ready.connect(lambda lid, text: messages.append(text))
    controller.line_sensor_summary('L01', 3)
    commands[-1].execute()
    assert '读取失败' in messages[0] and '未提供' not in messages[0]


def test_bridge_delivers_warning_before_success_and_reports_skipped_steps(qapp):
    from PyQt6.QtTest import QTest
    from mygpr.application.jobs.runner import InMemoryJobRunner
    from ui.controllers.backend_controller import JobBridge
    runner = InMemoryJobRunner(max_workers=1)
    def run(context):
        context.emit_warning({'code': 'processing_step_status', 'level': 'warning',
                              'message': '高度补偿：已跳过（高度包含NaN）',
                              'details': {'method_id': 'motion_compensation_height', 'execution_status': 'skipped'}})
        return {'ok': True}
    job = runner.submit('missing height', run)
    runner.wait(job, timeout=3)
    bridge = JobBridge(NS(jobs=runner))
    events = []
    bridge.notice_received.connect(lambda jid, payload: events.append(('notice', payload['message'])))
    bridge.job_completed.connect(lambda jid, ok, text, result: events.append(('done', ok, text)))
    try:
        bridge.watch(job)
        for _ in range(200):
            QTest.qWait(10)
            if any(item[0] == 'done' for item in events):
                break
        assert events[0][0] == 'notice'
        assert events[-1][0] == 'done' and events[-1][1] is True
        assert '1步已跳过' in events[-1][2]
    finally:
        bridge.shutdown()
        runner.shutdown()


def test_task_message_shows_skip_and_final_summary_without_resetting_progress():
    from ui.coordinator_jobs import JobHub
    updates, logs = [], []
    view = NS(upsert_job=lambda *a: None, set_status=lambda *a: None,
              update_progress=lambda *args: updates.append(args))
    hub = JobHub(NS(job_bridge=lambda: None, log_message=logs.append,
                    processing=NS(processing_job_id='')))
    hub._views = lambda: (view,)
    hub._mini_views = lambda: ()
    hub.on_progress('job', 1, 3, '处理中')
    hub.on_notice('job', {'message': '高度补偿：已跳过（高度缺失）', 'level': 'warning'})
    assert updates[-1] == ('job', 1, 3, '高度补偿：已跳过（高度缺失）')
    hub.on_completed('job', True, '任务完成（1步已跳过，详见日志）', None)
    assert updates[-1][1:3] == (1, 3)
    assert '1步已跳过' in updates[-1][3]
    assert any('WARNING' in text and '已跳过' in text for text in logs)


def test_nested_skip_metadata_does_not_claim_execution():
    from mygpr.domain.processing.warnings import build_processing_step_notice
    notice = build_processing_step_notice('frequency_filter_1d', '频域滤波', {
        'effective_params': {'skipped': True},
        'runtime_warnings': [{'message': '有效频带为空，已保持输入不变'}]})
    assert notice['details']['execution_status'] == 'skipped'
    assert '已跳过' in notice['message'] and '频带为空' in notice['message']


def test_progress_flood_cannot_erase_early_skip_notice(qapp):
    from PyQt6.QtTest import QTest
    from mygpr.application.jobs.runner import InMemoryJobRunner
    from ui.controllers.backend_controller import JobBridge
    runner = InMemoryJobRunner(max_workers=1)
    def run(context):
        context.emit_warning({'code': 'processing_step_status', 'level': 'warning',
            'message': '高度补偿：已跳过（高度缺失）', 'details': {'execution_status': 'skipped'}})
        for index in range(350):
            context.report_progress(index, 350, '数据块')
        return None
    job = runner.submit('flood', run)
    runner.wait(job, timeout=3)
    assert len(runner.events(job)) <= 256
    bridge = JobBridge(NS(jobs=runner))
    notices, results = [], []
    bridge.notice_received.connect(lambda jid, payload: notices.append(payload))
    bridge.job_completed.connect(lambda jid, ok, text, result: results.append(text))
    try:
        bridge.watch(job)
        for _ in range(200):
            QTest.qWait(10)
            if results:
                break
        assert any('高度缺失' in item.get('message', '') for item in notices)
        assert results and '1步已跳过' in results[0]
    finally:
        bridge.shutdown()
        runner.shutdown()


def test_notice_retention_is_bounded_but_totals_survive_truncation():
    from mygpr.application.jobs.runner import InMemoryJobRunner
    from mygpr.application.jobs.models import JobRetentionPolicy
    runner = InMemoryJobRunner(max_workers=1, retention_policy=JobRetentionPolicy(max_events_per_job=8))
    def run(context):
        for _ in range(20):
            context.emit_warning({'code': 'processing_step_status', 'level': 'warning',
                                  'details': {'execution_status': 'skipped'}})
        for index in range(40):
            context.report_progress(index, 40, '块')
    try:
        job = runner.submit('retention', run)
        runner.wait(job, timeout=3)
        notices, warnings, skipped, total = runner.notice_state(job)
        assert len(notices) == 8 and (warnings, skipped, total) == (20, 20, 20)
        assert len(runner.events(job)) <= 8
        assert not runner.notice_state(job, after_sequence=notices[-1].sequence)[0]
        assert runner.forget(job)
        with pytest.raises(KeyError):
            runner.notice_state(job)
    finally:
        runner.shutdown()


@pytest.mark.parametrize('height,expected', [([2, 2, 2, 2], 'executed'), ([2, np.nan, 2, 2], 'skipped')])
def test_native_compensation_emits_explicit_step_outcome(height, expected):
    from mygpr.application.jobs.context import ExecutionContext
    from mygpr.domain.processing.models import ProcessingRequest
    from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor
    notices = []
    result = NativeProcessingExecutor().execute(ProcessingRequest(
        data=np.ones((32, 4), dtype=np.float32), method_id='motion_compensation_height',
        trace_metadata={'flight_height_m': np.asarray(height)}, header_info={'time_window_ns': 100.}),
        ExecutionContext(warning_callback=notices.append))
    status = [item for item in notices if item.get('code') == 'processing_step_status']
    assert len(status) == 1
    assert status[0]['details']['execution_status'] == expected
    if expected == 'skipped':
        np.testing.assert_array_equal(result.data, np.ones((32, 4)))


@pytest.mark.parametrize('save_intermediates', [True, False])
def test_real_project_pipeline_reports_skip_with_both_execution_paths(tmp_path, save_intermediates):
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.application.jobs.models import JobEventType, JobStatus
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    path = tmp_path / 'sensor-gaps.h5'
    with h5py.File(path, 'w') as handle:
        handle.attrs.update(samples_per_trace=32, time_window_ns=100., store_component='fdata')
        handle.create_dataset('traces', data=np.arange(16 * 32, dtype=np.float32).reshape(16, 32))
        handle.create_dataset('trace_meta/height', data=np.full(16, np.nan))
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='gaps')
        backend.projects.import_line_source(project.project_id, 'L01', path)
        job = backend.submit_project_pipeline(project.project_id, 'L01',
            PipelineDefinition(name='height', steps=(PipelineStep('motion_compensation_height', {}),)),
            save_intermediates=save_intermediates)
        snapshot = backend.jobs.wait(job, timeout=20)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        events = [event for event in backend.jobs.events(job) if event.event_type is JobEventType.WARNING
                  and event.payload.get('code') == 'processing_step_status']
        assert len(events) == 1
        assert events[0].payload['details']['execution_status'] == 'skipped'
    finally:
        backend.shutdown()

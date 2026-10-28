"""Job status agrees with durable sensor commits; small data prove no field accuracy."""
from dataclasses import asdict
from pathlib import Path

import pytest

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.jobs.models import JobStatus
from mygpr.application.jobs.runner import InMemoryJobRunner
from tests.test_sensor_sync_source_provenance import sync_project  # noqa: F401


@pytest.mark.parametrize('stage', [6, 7])
@pytest.mark.parametrize('had_previous', [False, True])
def test_sensor_sync_cancel_respects_persisted_commit_boundary(sync_project, monkeypatch, stage, had_previous):  # noqa: F811
    backend, project, paths = sync_project
    sensors = Path(project.root_path) / 'raw' / 'L01' / 'sensors'
    store = backend.projects._session(project.project_id)._store
    if had_previous:
        backend.acquisition.synchronize_project_line(project.project_id, 'L01', **paths)
        paths['rtk_path'].write_text(paths['rtk_path'].read_text(encoding='utf-8').replace('104', '105'), encoding='utf-8')
    previous_line = asdict(store.get_line('L01'))
    previous_files = {path.relative_to(sensors).as_posix(): path.read_bytes()
                      for path in sensors.rglob('*') if path.is_file()}
    observed = []
    original = ExecutionContext.report_progress

    def cancel_at_stage(context, completed, total, message=''):
        original(context, completed, total, message)
        if completed == stage and total == 7:
            observed.append(context)
            assert backend.jobs.cancel(context.metadata['job_id'])

    monkeypatch.setattr(ExecutionContext, 'report_progress', cancel_at_stage)
    job_id = backend.submit_sensor_sync(project.project_id, 'L01', **paths)
    snapshot = backend.jobs.wait(job_id, timeout=10)
    assert len(observed) == 1
    assert observed[0].is_cancelled()  # commit must not clear or replace the token
    assert backend.projects.active_lease_count(project.project_id) == 0
    if stage == 7:
        assert (sensors / 'L01_sensor_sync_manifest.json').is_file()
        assert store.get_line('L01').sensor_sync_status.startswith('已同步')
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        assert (Path(project.root_path) / snapshot.result.manifest_path).is_file()
        backend.projects.close_project(project.project_id)
        reopened = backend.projects.open_project(project.root_path)
        loaded = backend.projects._session(reopened.project_id)._store
        assert loaded.get_line('L01').sensor_sync_status.startswith('已同步')
    else:
        assert snapshot.status is JobStatus.CANCELLED, snapshot.error_message
        assert {path.relative_to(sensors).as_posix(): path.read_bytes()
                for path in sensors.rglob('*') if path.is_file()} == previous_files
        assert asdict(store.get_line('L01')) == previous_line
        backend.projects.close_project(project.project_id)


def test_uncommitted_plain_job_still_cancels_when_operation_returns():
    runner = InMemoryJobRunner(max_workers=1)

    def operation(context):
        runner.cancel(context.metadata['job_id'])
        return {'computed': True}

    try:
        job_id = runner.submit('uncommitted', operation)
        snapshot = runner.wait(job_id, timeout=5)
        assert snapshot.status is JobStatus.CANCELLED
        assert snapshot.result is None
    finally:
        runner.shutdown()


def test_committed_job_does_not_hide_later_operation_error():
    runner = InMemoryJobRunner(max_workers=1)

    def operation(context):
        context.mark_committed()
        runner.cancel(context.metadata['job_id'])
        raise ValueError('postcommit operation error')

    try:
        job_id = runner.submit('committed but failed', operation)
        snapshot = runner.wait(job_id, timeout=5)
        assert snapshot.status is JobStatus.FAILED
        assert snapshot.error_type == 'ValueError'
        assert snapshot.error_message == 'postcommit operation error'
    finally:
        runner.shutdown()

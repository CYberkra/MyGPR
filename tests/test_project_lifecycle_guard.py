"""Switching/closing a project must never abandon active project work."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ui.coordinator_project import ProjectChain


@pytest.fixture
def context():
    controller = SimpleNamespace(open_project=Mock(), close_current=Mock())
    co = SimpleNamespace(
        backend_ready=True, project_controller=controller,
        interpretation_controller=None, infobar=Mock(), require_project=lambda: True,
        processing=SimpleNamespace(processing_job_id=None),
        jobs=SimpleNamespace(import_job_ids=set(), spatial_job_ids=set()))
    return co, ProjectChain(co)


@pytest.mark.parametrize('job_kind', ['processing', 'import', 'spatial', 'watched', 'command'])
def test_project_switch_and_close_are_blocked_during_work(context, job_kind):
    co, chain = context
    if job_kind == 'processing':
        co.processing.processing_job_id = 'running'
    elif job_kind == 'import':
        co.jobs.import_job_ids.add('running')
    elif job_kind == 'spatial':
        co.jobs.spatial_job_ids.add('running')
    elif job_kind == 'watched':
        chain.on_job_status('running', 'queued')
    else:
        chain.set_operation_busy('project', True)
    chain.on_open_recent_project('B')
    chain.on_close_project_requested()
    co.project_controller.open_project.assert_not_called()
    co.project_controller.close_current.assert_not_called()
    assert '任务' in co.infobar.call_args.args[2]


def test_terminal_job_releases_switch_guard(context):
    co, chain = context
    chain.on_job_status('job', 'running')
    chain.on_job_status('job', 'completed')
    chain.on_open_recent_project('B')
    co.project_controller.open_project.assert_called_once_with('B')


def test_close_uses_existing_controller_after_operations_finish(context):
    co, chain = context
    chain.set_operation_busy('project', True)
    chain.set_operation_busy('project', False)
    chain.on_close_project_requested()
    co.project_controller.close_current.assert_called_once_with()

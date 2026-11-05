"""Permanent deletion protects original imports and rejects stale confirmations."""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core.field_project_errors import FieldProjectOperationError
from core.permanent_delete import permanently_delete_paths
from mygpr.application.project.errors import ProjectApplicationError, ProjectBusyError
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


@pytest.fixture
def project(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(tmp_path / 'project', name='Deletion scope')
    try:
        yield backend, summary
    finally:
        backend.shutdown()


@pytest.mark.parametrize('inside', [False, True])
@pytest.mark.parametrize('kind', ['project', 'lines', 'artifacts'])
def test_original_import_survives_each_deletion(project, tmp_path, inside, kind):
    backend, summary = project
    root = Path(summary.root_path)
    source = (root / 'raw' / 'L01' if inside else tmp_path / 'originals') / 'survey.npy'
    source.parent.mkdir(parents=True, exist_ok=True)
    np.save(source, np.arange(128, dtype=np.float32).reshape(16, 8))
    original = source.read_bytes()
    backend.projects.import_line_source(summary.project_id, 'L01', source,
        time_window_ns=50.0, trace_interval_m=0.1)
    saved = backend.project_processing.execute_pipeline(summary.project_id, 'L01',
        PipelineDefinition(steps=(PipelineStep('dewow', {'window': 3}),)))
    service = backend.maintenance.deletion
    targets = [saved.artifact_id] if kind == 'artifacts' else ['L01']
    plan = service.preview(kind, summary.project_id, targets, line_id='L01',
                           root=str(root) if kind == 'project' else '')
    service.execute(plan)
    assert source.read_bytes() == original
    assert not (root / '.trash').exists()
    if kind == 'project':
        assert not (root / 'project.json').exists()
    elif kind == 'lines':
        assert backend.projects.list_lines(summary.project_id) == ()
        assert not (root / 'data' / 'lines' / 'L01.h5').exists()
    else:
        assert backend.projects.list_artifacts(summary.project_id, 'L01') == ()
        assert backend.projects.read_dataset(summary.project_id, 'L01').data.shape == (16, 8)


def test_original_protection_survives_line_then_project_delete(project):
    backend, summary = project
    root = Path(summary.root_path)
    source = root / 'raw' / 'L01' / 'original.npy'
    source.parent.mkdir(parents=True)
    np.save(source, np.ones((16, 8), dtype=np.float32))
    original = source.read_bytes()
    backend.projects.import_line_source(summary.project_id, 'L01', source)
    service = backend.maintenance.deletion
    service.execute(service.preview('lines', summary.project_id, ['L01']))
    service.execute(service.preview('project', summary.project_id, root=str(root)))
    assert source.read_bytes() == original


def test_live_task_blocks_preflight(project):
    backend, summary = project
    with backend.projects.acquire_lease(summary.project_id):
        with pytest.raises(ProjectBusyError):
            backend.maintenance.deletion.preview('project', summary.project_id, root=summary.root_path)
    assert Path(summary.root_path, 'project.json').is_file()


def test_closed_project_delete_does_not_close_another_project(project, tmp_path):
    backend, current = project
    other = backend.projects.create_project(tmp_path / 'other', name='Other')
    backend.projects.close_project(other.project_id)
    service = backend.maintenance.deletion
    plan = service.preview('project', current.project_id, root=other.root_path)
    service.execute(plan)
    assert backend.projects.get_summary(current.project_id).project_id == current.project_id
    assert not Path(other.root_path).exists()


def test_read_only_project_is_rejected(project):
    backend, summary = project
    backend.projects.close_project(summary.project_id)
    backend.projects.open_project(summary.root_path, read_only=True)
    with pytest.raises(ProjectApplicationError, match='只读'):
        backend.maintenance.deletion.preview('project', summary.project_id, root=summary.root_path)
    assert Path(summary.root_path, 'project.json').is_file()


def test_changed_confirmation_never_deletes_new_data(project):
    backend, summary = project
    service = backend.maintenance.deletion
    plan = service.preview('project', summary.project_id, root=summary.root_path)
    backend.projects.save_line_dataset(summary.project_id, 'L01', np.ones((16, 8), dtype=np.float32))
    with pytest.raises(ProjectApplicationError, match='变化'):
        service.execute(plan)
    assert backend.projects.read_dataset(summary.project_id, 'L01').data.shape == (16, 8)


def test_failed_unlink_retains_progress_and_can_resume(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    first, second = root / 'first.npy', root / 'second.npy'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    unlink = Path.unlink
    finalized = []
    def fail(path, *args, **kwargs):
        if path == second:
            raise PermissionError('occupied')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail)
    with pytest.raises(FieldProjectOperationError, match='未完成'):
        permanently_delete_paths(root, [first, second], key='test', finalize=lambda: finalized.append(True))
    assert not first.exists() and second.is_file() and finalized == []
    assert (root / '.delete_operations' / 'test.json').is_file()
    monkeypatch.undo()
    permanently_delete_paths(root, [first, second], key='test', finalize=lambda: finalized.append(True))
    assert finalized == [True] and not second.exists()
    assert not (root / '.delete_operations' / 'test.json').exists()


def test_path_escape_is_rejected_before_any_delete(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    owned, original = root / 'owned.npy', tmp_path / 'original.npy'
    owned.write_bytes(b'owned')
    original.write_bytes(b'original')
    with pytest.raises(FieldProjectOperationError):
        permanently_delete_paths(root, [owned, original], key='escape', finalize=lambda: None)
    assert owned.read_bytes() == b'owned' and original.read_bytes() == b'original'


def test_tree_uses_current_method_instead_of_generated_result_name():
    from ui.file_tree import artifact_tree_name
    artifact = NS(name='处理结果_20261105', method_name='Processing pipeline', manifest={'params': {
        'lineage_version': 2, 'result_label': '处理结果_20261105', 'run_step_index': 2,
        'lineage': [{'method_id': 'dewow'}, {'method_id': 'agcGain'}]}})
    label = artifact_tree_name(artifact)
    assert label and '处理结果' not in label and 'Processing pipeline' not in label


def test_cancelled_confirmation_has_no_file_mutation(qapp, project, monkeypatch):
    from ui.controllers.project_controller import ProjectController
    from ui.controllers import permanent_delete_controller as module
    backend, summary = project
    pc = ProjectController()
    pc.set_backend(NS(backend=backend, job_bridge=None))
    pc._current = summary
    monkeypatch.setattr(module, 'run_command', lambda command, **kwargs: command.execute())
    plans = []
    pc.deletion.confirmation_requested.connect(plans.append)
    assert pc.deletion.request('project', root=summary.root_path)
    assert pc._busy and len(plans) == 1
    pc.deletion.confirm(plans[0], False)
    assert not pc._busy and Path(summary.root_path, 'project.json').is_file()

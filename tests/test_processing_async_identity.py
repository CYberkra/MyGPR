"""Real controller signal paths retain preview and submitted-job identity."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QObject, pyqtSignal

from ui.controllers.processing_controller import ProcessingController
from ui.controllers.project_controller import ProjectController, _PreviewArtifactCommand, _PreviewLineCommand


class Bridge(QObject):
    job_completed = pyqtSignal(str, bool, str, object)

    def __init__(self):
        super().__init__()
        self.watched = []

    def watch(self, job_id, *, title):
        self.watched.append(job_id)


@pytest.mark.parametrize('result', [{'artifact_id': 'actual'}, SimpleNamespace(artifact_id='actual')])
def test_pipeline_watcher_reports_only_its_exact_final_artifact(qapp, result):
    bridge = Bridge()
    submissions = []
    backend = SimpleNamespace(submit_project_pipeline=lambda *args, **kwargs:
                              submissions.append((args, kwargs)) or 'mine')
    controller = ProcessingController()
    controller.set_backend(SimpleNamespace(backend=backend, job_bridge=bridge))
    ready, finished = [], []
    controller.run_result_ready.connect(lambda job_id, aid: ready.append((job_id, aid)))
    controller.run_finished.connect(lambda success, message: finished.append((success, message)))
    assert controller.run_pipeline('P', 'L01', {'steps': [{'method_id': 'dewow', 'params': {'window': 23}}]},
                                   'result', input_artifact_id='input-A') == 'mine'
    assert submissions[0][1]['input_artifact_id'] == 'input-A'
    assert bridge.watched == ['mine']
    bridge.job_completed.emit('other', True, 'other complete', {'artifact_id': 'unrelated'})
    assert ready == finished == []
    bridge.job_completed.emit('mine', True, 'saved', result)
    assert ready == [('mine', 'actual')]
    assert finished == [(True, 'saved')]
    bridge.job_completed.emit('mine', True, 'duplicate', {'artifact_id': 'duplicate'})
    assert ready == [('mine', 'actual')]


def failing_controller():
    def fail(*args, **kwargs):
        raise OSError('cannot read saved data')
    controller = ProjectController()
    controller._current = SimpleNamespace(project_id='P')
    controller.set_backend(SimpleNamespace(backend=SimpleNamespace(projects=SimpleNamespace(
        get_artifact_dataset_info=fail, get_dataset_info=fail))))
    return controller


def test_preview_failure_signal_contains_captured_context(qapp):
    controller = failing_controller()
    scoped, legacy = [], []
    controller.preview_context_failed.connect(lambda *args: scoped.append(args))
    controller.preview_failed.connect(legacy.append)
    _PreviewArtifactCommand(controller, 'P', 'L01', 'A', 0, False).execute()
    assert scoped == [('P', 'L01', 'A', 'cannot read saved data')]
    assert len(legacy) == 1
    _PreviewLineCommand(controller, 'P', 'L01', 0).execute()
    assert scoped[-1] == ('P', 'L01', '', 'cannot read saved data')


def test_comparison_artifact_preview_uses_reference_channel(qapp):
    controller = failing_controller()
    comparison, regular = [], []
    controller.comparison_artifact_preview_ready.connect(lambda *args: comparison.append(args))
    controller.artifact_preview_ready.connect(lambda *args: regular.append(args))
    _PreviewArtifactCommand(controller, 'P', 'L02', 'A2', 0, False, reference_request='token2').execute()
    assert comparison == [('P', 'L02', 'A2', 'token2', None, 'cannot read saved data')]
    assert regular == []


def test_preview_failure_from_old_context_is_discarded_even_without_strict_generation(qapp):
    controller = failing_controller()
    scoped, legacy = [], []
    controller.preview_context_failed.connect(lambda *args: scoped.append(args))
    controller.preview_failed.connect(legacy.append)
    command = _PreviewArtifactCommand(controller, 'P', 'L01', 'A', 0, False)
    controller.invalidate_artifact_previews()
    command.execute()
    assert scoped == legacy == []


def test_preview_failure_from_previous_project_is_discarded(qapp):
    controller = failing_controller()
    scoped = []
    controller.preview_context_failed.connect(lambda *args: scoped.append(args))
    command = _PreviewArtifactCommand(controller, 'P', 'L01', 'A', 0, False)
    controller._current = SimpleNamespace(project_id='replacement-project')
    command.execute()
    assert scoped == []


def test_pipeline_completion_contains_job_identity_for_coordinator(qapp):
    from ui.coordinator_processing import ProcessingChain
    bridge = Bridge()
    job_ids = iter(['mine', 'other'])
    controller = ProcessingController()
    controller.set_backend(SimpleNamespace(backend=SimpleNamespace(
        submit_project_pipeline=lambda *args, **kwargs: next(job_ids)), job_bridge=bridge))
    chain = ProcessingChain(SimpleNamespace())
    chain.processing_job_id = 'mine'
    completed = []
    chain.on_run_finished = lambda success, message: completed.append((success, message))
    controller.run_context_finished.connect(chain.on_run_context_finished)
    definition = {'steps': [{'method_id': 'dewow', 'params': {}}]}
    controller.run_pipeline('P', 'L01', definition, 'mine')
    controller.run_pipeline('P', 'L02', definition, 'other')
    bridge.job_completed.emit('other', True, 'other saved', {'artifact_id': 'other-output'})
    assert completed == []
    bridge.job_completed.emit('mine', True, 'mine saved', {'artifact_id': 'mine-output'})
    assert completed == [(True, 'mine saved')]


def test_pipeline_watcher_observes_completion_raised_during_watch(qapp):
    class ImmediateBridge(Bridge):
        def watch(self, job_id, *, title):
            super().watch(job_id, title=title)
            self.job_completed.emit(job_id, True, 'already complete', {'artifact_id': 'actual'})
    bridge = ImmediateBridge()
    controller = ProcessingController()
    controller.set_backend(SimpleNamespace(backend=SimpleNamespace(
        submit_project_pipeline=lambda *args, **kwargs: 'mine'), job_bridge=bridge))
    ready = []
    controller.run_result_ready.connect(lambda job_id, aid: ready.append((job_id, aid)))
    controller.run_pipeline('P', 'L01', {'steps': [{'method_id': 'dewow', 'params': {}}]}, 'mine')
    assert ready == [('mine', 'actual')]


def test_coordinator_freezes_submission_before_immediate_watcher_completion(qapp):
    from ui.coordinator_processing import ProcessingChain
    from ui.pages.processing_page import ProcessingPage
    from tests.test_processing_workspace_flow import artifact, bundle
    class ImmediateBridge(Bridge):
        def watch(self, job_id, *, title):
            self.job_completed.emit(job_id, True, 'saved', {'artifact_id': 'actual'})
    page = ProcessingPage()
    page.set_line_label('L01')
    page.set_original_bundle(bundle())
    page._pipeline_list.add_step('dewow', 'Dewow', {})
    bridge = ImmediateBridge()
    controller = ProcessingController()
    controller.set_backend(SimpleNamespace(backend=SimpleNamespace(
        submit_project_pipeline=lambda *args, **kwargs: 'mine'), job_bridge=bridge))
    coordinator = SimpleNamespace(processing_controller=controller,
        project_controller=SimpleNamespace(refresh_lines=lambda: None,
            refresh_artifacts=lambda line_id: page.set_artifacts([artifact('actual')])),
        require_line=lambda: 'L01', current_project_id=lambda: '', page=lambda key: page,
        infobar=lambda *args: None, file_tree=lambda: None,
        project=SimpleNamespace(current_line_id='L01'))
    chain = ProcessingChain(coordinator)
    controller.run_submitted.connect(chain.on_run_submitted)
    controller.run_result_ready.connect(page.set_run_result)
    controller.run_context_finished.connect(chain.on_run_context_finished)
    chain.on_run_requested(page.current_pipeline())
    assert not page._running
    assert not chain.processing_job_id
    assert page.current_pipeline()['input_artifact_id'] == 'actual'
    page.close()
    page.deleteLater()


@pytest.mark.parametrize('artifact_id', ['', 'saved-input'])
def test_autotune_submission_preserves_source_header_and_trace_metadata(qapp, artifact_id):
    import numpy as np
    from ui.controllers.processing_controller import _AutotuneSubmitCommand

    bridge = Bridge()
    dataset = SimpleNamespace(data=np.zeros((32, 8), dtype=np.float32),
        header_info={'total_time_ns': 512., 'data_context': 'gprmax_impulse'},
        trace_metadata={'trace_axis_m': np.arange(8, dtype=np.float64)})
    reads, submissions = [], []

    def read(*args):
        reads.append(args)
        return dataset

    controller = ProcessingController()
    controller.set_backend(SimpleNamespace(backend=SimpleNamespace(
        projects=SimpleNamespace(read_dataset=read, read_artifact_dataset=read),
        submit_autotune=lambda *args, **kwargs: submissions.append(kwargs) or 'tune'), job_bridge=bridge))
    _AutotuneSubmitCommand(controller, 'P', 'L01', 'dewow', {'window': 9}, artifact_id).execute()
    assert reads == [('P', 'L01', artifact_id) if artifact_id else ('P', 'L01')]
    assert submissions[0]['header_info'] == dataset.header_info
    assert np.array_equal(submissions[0]['trace_metadata']['trace_axis_m'], dataset.trace_metadata['trace_axis_m'])


def test_pipeline_defaults_to_final_only_block_friendly_execution(qapp):
    bridge = Bridge()
    submissions = []
    backend = SimpleNamespace(submit_project_pipeline=lambda *args, **kwargs:
                              submissions.append(kwargs) or 'job')
    controller = ProcessingController()
    controller.set_backend(SimpleNamespace(backend=backend, job_bridge=bridge))
    controller.run_pipeline('P', 'L01', {'steps': [{'method_id': 'dewow', 'params': {}}]}, 'result')
    assert submissions[0]['save_intermediates'] is False


def test_batch_uses_per_line_inputs_and_never_shares_artifact_ids(qapp):
    bridge = Bridge()
    submissions = []
    job_ids = iter(['J1', 'J2'])
    backend = SimpleNamespace(submit_project_pipeline=lambda *args, **kwargs:
                              submissions.append((args, kwargs)) or next(job_ids))
    controller = ProcessingController()
    controller.set_backend(SimpleNamespace(backend=backend, job_bridge=bridge))
    jobs = controller.run_pipeline_batch(
        'P', ['L01', 'L02'], {'steps': [{'method_id': 'dewow', 'params': {}}]}, 'batch',
        input_artifact_ids={'L01': 'A1', 'L02': 'A2'}, save_intermediates=True)
    assert jobs == ['J1', 'J2']
    assert [call[1]['input_artifact_id'] for call in submissions] == ['A1', 'A2']
    assert all(call[1]['save_intermediates'] is True for call in submissions)

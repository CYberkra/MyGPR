from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QObject, pyqtSignal

from ui.controllers.project_controller import ProjectController


class FakeBridge(QObject):
    job_completed = pyqtSignal(str, bool, str, object)

    def __init__(self):
        super().__init__()
        self.cancelled = []
        self.watched = []

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def watch(self, job_id, title=''):
        self.watched.append(job_id)


def test_new_request_cancels_old_and_discards_late_result(qapp, monkeypatch):
    import ui.controllers.project_controller as module
    ids = iter(['old', 'new'])
    backend = SimpleNamespace(submit_amplitude_slice=lambda *args, **kw: next(ids))
    bridge = FakeBridge()
    controller = ProjectController()
    controller._current = SimpleNamespace(project_id='P')
    controller._backend_controller = SimpleNamespace(backend=backend, job_bridge=bridge)
    monkeypatch.setattr(module, 'prepare_amplitude_sources', lambda *a: (('source',), ('positions',)))
    received = []
    controller.amplitude_preview_ready.connect(lambda result, pos: received.append((result, pos)))
    controller.request_amplitude_preview([], {}, '', 0., 1., 1., 'rms')
    controller.request_amplitude_preview([], {}, '', 1., 2., 1., 'rms')
    assert bridge.cancelled == ['old']
    bridge.job_completed.emit('old', True, '', 'obsolete')
    assert received == []
    bridge.job_completed.emit('new', True, '', 'current')
    assert received == [('current', ('positions',))]


def test_project_switch_discards_queued_slice_result(qapp, monkeypatch):
    import ui.controllers.project_controller as module
    backend = SimpleNamespace(submit_amplitude_slice=lambda *args, **kw: 'job')
    bridge = FakeBridge()
    controller = ProjectController()
    controller._current = SimpleNamespace(project_id='P')
    controller._backend_controller = SimpleNamespace(backend=backend, job_bridge=bridge)
    monkeypatch.setattr(module, 'prepare_amplitude_sources', lambda *a: (('source',), ('positions',)))
    received = []
    controller.amplitude_preview_ready.connect(lambda *args: received.append(args))
    controller.request_amplitude_preview([], {}, '', 0., 1., 1., 'rms')
    controller._current = SimpleNamespace(project_id='Q')
    controller._project_context_generation += 1
    bridge.job_completed.emit('job', True, '', 'obsolete')
    assert received == []


def test_result_too_large_reports_actionable_error(qapp, monkeypatch):
    import ui.controllers.project_controller as module
    backend = SimpleNamespace(submit_amplitude_slice=lambda *args, **kw: 'job')
    bridge = FakeBridge()
    controller = ProjectController()
    controller._current = SimpleNamespace(project_id='P')
    controller._backend_controller = SimpleNamespace(backend=backend, job_bridge=bridge)
    monkeypatch.setattr(module, 'prepare_amplitude_sources', lambda *a: (('source',), ('positions',)))
    errors = []
    controller.amplitude_preview_failed.connect(errors.append)
    controller.request_amplitude_preview([], {}, '', 0., 1., 1., 'rms')
    bridge.job_completed.emit('job', True, '任务完成', None)
    assert errors and '过大' in errors[0] and '网格' in errors[0]

from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('qfluentwidgets')

from ui.controllers import project_controller as module


def test_only_current_import_preflight_is_delivered(qapp, monkeypatch):
    pending = []
    monkeypatch.setattr(module, 'run_command', lambda command, **kwargs: pending.append(command))
    controller = module.ProjectController()
    controller._current = SimpleNamespace(project_id='project')
    acquisition = SimpleNamespace(preflight=lambda source, **kwargs: source)
    controller.set_backend(SimpleNamespace(backend=SimpleNamespace(acquisition=acquisition)))
    delivered = []
    controller.preflight_ready.connect(delivered.append)
    controller.preflight_import('old.npy', 'L01', 9.0)
    controller.preflight_import('new.npy', 'L01', 9.0)
    pending[1].execute()
    pending[0].execute()
    qapp.processEvents()
    assert delivered == ['new.npy']
    controller.preflight_import('edited.npy', 'L01', 9.0)
    controller.invalidate_preflight()
    pending[2].execute()
    # Also cover a result queued before invalidation but delivered afterwards.
    controller._deliver_project_read(pending[2], ('preflight_ready', ('edited.npy',)))
    qapp.processEvents()
    assert delivered == ['new.npy']

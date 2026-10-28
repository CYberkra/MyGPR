"""Page warmup must not compete with backend composition during startup."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('qfluentwidgets')

from PyQt6.QtCore import QObject, pyqtSignal
from ui import main_window as module


class BackendSignals(QObject):
    backend_ready = pyqtSignal()
    backend_failed = pyqtSignal(str)


@pytest.mark.parametrize('outcome', ['ready', 'failed'])
def test_warmup_waits_for_backend_outcome_and_schedules_once(monkeypatch, outcome):
    queued = []
    monkeypatch.setattr(module, 'defer_call', lambda host, fn: queued.append(fn))
    backend = BackendSignals()
    state = SimpleNamespace(backend_controller=backend, _backend_ready=False,
                            _warmup_next_page=Mock())
    state._schedule_warmup = lambda *args: module.MyGPRMainWindow._schedule_warmup(state)
    module.MyGPRMainWindow._start_warmup(state)
    assert queued == [], 'page construction started while backend was initializing'
    if outcome == 'ready':
        backend.backend_ready.emit()
    else:
        backend.backend_failed.emit('initialization failed')
    backend.backend_ready.emit()
    assert len(queued) == 1
    queued[0]()
    state._warmup_next_page.assert_called_once_with()


@pytest.mark.parametrize('ready', [False, True])
def test_warmup_without_pending_backend_starts_immediately(monkeypatch, ready):
    queued = []
    monkeypatch.setattr(module, 'defer_call', lambda host, fn: queued.append(fn))
    state = SimpleNamespace(backend_controller=BackendSignals() if ready else None,
                            _backend_ready=ready, _warmup_next_page=Mock())
    state._schedule_warmup = lambda *args: module.MyGPRMainWindow._schedule_warmup(state)
    module.MyGPRMainWindow._start_warmup(state)
    assert len(queued) == 1

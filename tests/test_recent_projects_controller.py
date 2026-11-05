"""Advisory directory probes must not block Qt or resurrect stale records."""
from threading import Event, Thread

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtTest import QSignalSpy
from ui.controllers.recent_projects_controller import RecentProjectsController, _ProbeDirectoriesCommand


@pytest.fixture
def controller(qapp, monkeypatch):
    commands = []
    monkeypatch.setattr('ui.controllers.recent_projects_controller.run_command',
                        lambda command, **kwargs: commands.append(command))
    controller = RecentProjectsController()
    yield controller, commands
    controller.cancel()
    controller.deleteLater()


def test_requests_coalesce_and_discard_queued_old_results(controller, tmp_path):
    probe, commands = controller
    received = []
    probe.availability_changed.connect(lambda *args: received.append(args))
    probe.check([str(tmp_path / 'old')])
    old = commands[0]
    probe.check([str(tmp_path / 'discarded')])
    probe.check([str(tmp_path)])
    assert len(commands) == 1
    old.result(old.generation, old.paths[0], '')
    assert received == []
    old.execute()
    assert len(commands) == 2
    commands[1].execute()
    assert received == [(str(tmp_path), '')]


def test_probe_io_and_delivery_are_separated_by_qt_queue(controller, tmp_path, qapp):
    probe, commands = controller
    spy = QSignalSpy(probe.availability_changed)
    probe.check([str(tmp_path), str(tmp_path / 'missing')])
    worker = Thread(target=commands[0].execute)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(spy) == 0
    qapp.processEvents()
    assert len(spy) == 2
    assert spy[0] == [str(tmp_path), '']
    assert '不存在' in spy[1][1]


def test_cancelled_probe_does_not_deliver_after_blocked_stat_returns(monkeypatch):
    entered, release, cancelled = Event(), Event(), Event()
    received, finished = [], []

    def slow_stat(_path):
        entered.set()
        assert release.wait(2)
        return ''

    monkeypatch.setattr('ui.controllers.recent_projects_controller._directory_error', slow_stat)
    command = _ProbeDirectoriesCommand(1, ('network-path',), cancelled,
                                       lambda *args: received.append(args), finished.append)
    worker = Thread(target=command.execute)
    worker.start()
    try:
        assert entered.wait(2)
        cancelled.set()
    finally:
        release.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert received == []
    assert finished == [1]

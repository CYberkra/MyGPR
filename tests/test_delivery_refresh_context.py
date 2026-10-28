"""Delivery lists must retain project ownership until Qt consumes the result."""
from threading import Event, Thread
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from ui.controllers.delivery_controller import DeliveryController
from ui.coordinator_project import ProjectChain


@pytest.fixture
def lifecycle(monkeypatch):
    from ui.controllers import delivery_controller as module

    commands = []
    monkeypatch.setattr(module, "run_command", lambda command, **kwargs: commands.append(command))
    controller = DeliveryController()
    controller.set_backend(NS(backend=NS(
        spatial=NS(list_results=lambda pid: [NS(owner=pid)]),
        projects=NS(list_report_packages=lambda pid: [NS(owner=pid)]),
    )))
    state = {"project_id": "A"}
    co = MagicMock()
    co.delivery_controller = controller
    co.current_project_id.side_effect = lambda: state["project_id"]
    co.settings.get.return_value = {}
    pages = {}
    co.page.side_effect = lambda name: pages.setdefault(name, MagicMock())
    chain = ProjectChain(co)

    def open_project(pid):
        state["project_id"] = pid
        chain.on_project_opened(NS(project_id=pid, root_path="", name=pid))

    open_project("A")
    yield controller, chain, commands, open_project, state


@pytest.mark.parametrize("kind", ["spatial", "reports"])
@pytest.mark.parametrize("transition", ["switch", "close", "reopen_same", "open_same"])
@pytest.mark.parametrize("queued", [False, True])
def test_old_delivery_refresh_is_discarded_after_lifecycle_change(qapp, lifecycle, kind, transition, queued):
    controller, chain, commands, open_project, state = lifecycle
    command = commands[0 if kind == "spatial" else 1]
    received = []
    signal = controller.spatial_results_updated if kind == "spatial" else controller.report_list_updated
    signal.connect(received.append)
    if queued:
        worker = Thread(target=command.execute)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
    if transition == "switch":
        open_project("B")
    elif transition == "open_same":
        open_project("A")
    else:
        state["project_id"] = ""
        chain.on_project_closed()
        if transition == "reopen_same":
            open_project("A")
    if not queued:
        command.execute()
    qapp.processEvents()
    assert received == []


@pytest.mark.parametrize("kind", ["spatial", "reports"])
def test_current_delivery_refresh_reaches_gui_thread(qapp, lifecycle, kind):
    from PyQt6.QtCore import QThread

    controller, chain, commands, open_project, state = lifecycle
    received = []
    threads = []
    signal = controller.spatial_results_updated if kind == "spatial" else controller.report_list_updated
    signal.connect(lambda rows: (received.append(rows), threads.append(QThread.currentThread())))
    worker = Thread(target=commands[0 if kind == "spatial" else 1].execute)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    qapp.processEvents()
    assert len(received) == 1 and received[0][0].owner == "A"
    assert threads == [qapp.thread()]


@pytest.mark.parametrize("kind", ["spatial", "reports"])
def test_delivery_read_finishing_after_project_switch_is_discarded(qapp, lifecycle, kind):
    controller, chain, commands, open_project, state = lifecycle
    entered = Event()
    release = Event()

    def finish_old_read(pid):
        entered.set()
        assert release.wait(timeout=5)
        return [NS(owner=pid)]

    backend = controller._backend_controller.backend
    if kind == "spatial":
        backend.spatial.list_results = finish_old_read
    else:
        backend.projects.list_report_packages = finish_old_read
    received = []
    signal = controller.spatial_results_updated if kind == "spatial" else controller.report_list_updated
    signal.connect(received.append)
    worker = Thread(target=commands[0 if kind == "spatial" else 1].execute)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        open_project("B")
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    qapp.processEvents()
    assert received == []


def _report_bridge(controller):
    from PyQt6.QtCore import QObject, pyqtSignal

    class Bridge(QObject):
        job_completed = pyqtSignal(str, bool, str, object)

        def watch(self, job_id, title):
            pass

    bridge = Bridge()
    controller._backend_controller.job_bridge = bridge
    controller._backend_controller.backend.submit_project_report = lambda *args, **kwargs: "report-job"
    return bridge


@pytest.mark.parametrize("transition", ["switch", "close", "reopen_same", "open_same"])
@pytest.mark.parametrize("success", [False, True])
def test_queued_report_completion_cannot_deliver_after_project_lifecycle_change(qapp, lifecycle, transition, success):
    controller, chain, commands, open_project, state = lifecycle
    bridge = _report_bridge(controller)
    received = []
    controller.report_generated.connect(received.append)
    assert controller.generate_report("A") == "report-job"
    logs = []
    controller.log_message.connect(logs.append)
    worker = Thread(target=lambda: bridge.job_completed.emit("report-job", success, "old report result", NS(owner="A")))
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    if transition == "switch":
        open_project("B")
    elif transition == "open_same":
        open_project("A")
    else:
        state["project_id"] = ""
        chain.on_project_closed()
        if transition == "reopen_same":
            open_project("A")
    qapp.processEvents()
    assert received == []
    assert logs == []
    # Even a stale completion must disconnect its matching job callback.
    assert bridge.receivers(bridge.job_completed) == 0


def test_current_report_completion_delivers_once_and_disconnects_by_job_id(qapp, lifecycle):
    controller, chain, commands, open_project, state = lifecycle
    bridge = _report_bridge(controller)
    received = []
    controller.report_generated.connect(received.append)
    assert controller.generate_report("A") == "report-job"
    result = NS(owner="A")
    bridge.job_completed.emit("other-job", True, "unrelated", NS(owner="other"))
    assert received == []
    bridge.job_completed.emit("report-job", True, "complete", result)
    bridge.job_completed.emit("report-job", True, "duplicate", result)
    assert received == [result]
    assert bridge.receivers(bridge.job_completed) == 0

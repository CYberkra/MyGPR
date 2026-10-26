from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QCoreApplication, QThread  # noqa: E402

from ui.controllers import interpretation_controller as module  # noqa: E402


class EditService:
    def __init__(self):
        self.opened = []
        self.closed = []
        self.edited = []
        self.saved = []
        self.open_hook = None
        self.edit_hook = None
        self.save_hook = None

    def open_session(self, project_id, line_id, **kwargs):
        self.opened.append((project_id, line_id))
        if self.open_hook:
            self.open_hook(line_id)
        return SimpleNamespace(project_id=project_id, line_id=line_id, session_id=f"S-{line_id}", dirty=False)

    def close_session(self, session_id):
        self.closed.append(session_id)

    def replace_points(self, session_id, points):
        self.edited.append((session_id, points))
        if self.edit_hook:
            self.edit_hook()
        return SimpleNamespace(project_id="P1", line_id=session_id[2:], session_id=session_id, dirty=True)

    def save_session(self, session_id, **kwargs):
        self.saved.append((session_id, kwargs))
        if self.save_hook:
            self.save_hook()


@pytest.fixture
def harness(qapp, monkeypatch):
    queued = []
    monkeypatch.setattr(module, "run_command", lambda command, **kwargs: queued.append(command))
    service = EditService()
    controller = module.InterpretationController()
    controller.set_backend(SimpleNamespace(backend=SimpleNamespace(interpretation_edit=service)))
    signals = {name: [] for name in ("session_opened", "session_updated", "saved", "session_failed")}
    for name, values in signals.items():
        getattr(controller, name).connect(values.append)
    return controller, service, queued, signals


def _execute(command):
    thread = threading.Thread(target=command.execute)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _drain():
    QCoreApplication.processEvents()


def _open(harness, line_id="L01"):
    controller, _, queued, _ = harness
    controller.open_session("P1", line_id)
    _execute(queued.pop())
    _drain()


def test_close_invalidates_open_that_has_not_started(harness):
    controller, service, queued, signals = harness
    controller.open_session("P1", "L01")
    command = queued.pop()
    controller.close_session()
    _execute(command)
    _drain()
    assert service.opened == []
    assert signals["session_opened"] == []
    assert controller._session_id is None


def test_late_open_is_closed_and_does_not_block_new_context(harness):
    controller, service, queued, signals = harness
    entered, release = threading.Event(), threading.Event()

    def block_old(line_id):
        if line_id == "L01":
            entered.set()
            assert release.wait(5)

    service.open_hook = block_old
    controller.open_session("P1", "L01")
    thread = threading.Thread(target=queued.pop().execute)
    thread.start()
    try:
        assert entered.wait(5)
        controller.close_session()
        _open(harness, "L02")
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    _drain()
    while queued:
        _execute(queued.pop(0))
    _drain()
    assert controller._session_id == "S-L02"
    assert [snapshot.line_id for snapshot in signals["session_opened"]] == ["L02"]
    assert "S-L01" in service.closed


@pytest.mark.parametrize("operation", ["edit", "save"])
def test_queued_operation_cannot_rebind_to_new_session(harness, operation):
    controller, service, queued, signals = harness
    _open(harness)
    if operation == "edit":
        controller.replace_points([(1, 2)])
    else:
        controller.save()
    old_command = queued.pop()
    controller.close_session()
    _open(harness, "L02")
    _execute(old_command)
    _drain()
    assert service.edited == []
    assert service.saved == []
    assert signals["session_updated"] == []
    assert signals["saved"] == []
    assert signals["session_failed"] == []


@pytest.mark.parametrize("outcome", ["edit", "save", "edit_failure", "save_failure"])
def test_completion_already_queued_is_discarded_after_close(harness, outcome):
    controller, service, queued, signals = harness
    _open(harness)

    def fail():
        raise RuntimeError("old operation failed")

    if outcome.startswith("edit"):
        service.edit_hook = fail if outcome.endswith("failure") else None
        controller.replace_points([(1, 2)])
    else:
        service.save_hook = fail if outcome.endswith("failure") else None
        controller.save()
    _execute(queued.pop())  # worker returned; UI has not consumed its signals
    controller.close_session()
    _open(harness, "L02")
    assert controller._session_id == "S-L02"
    assert signals["session_updated"] == []
    assert signals["saved"] == []
    assert signals["session_failed"] == []


def test_snapshot_guard_checks_project_line_and_session(harness):
    controller, _, _, signals = harness
    _open(harness)
    snapshot = signals["session_opened"][0]
    assert controller.is_current_session(snapshot)
    for field in ("project_id", "line_id", "session_id"):
        stale = SimpleNamespace(**{**vars(snapshot), field: "other"})
        assert not controller.is_current_session(stale)
    assert not controller.is_current_session(SimpleNamespace())
    controller.close_session()
    assert not controller.is_current_session(snapshot)


@pytest.mark.parametrize("outcome", ["edit", "save", "edit_failure", "save_failure"])
def test_in_flight_operation_cannot_update_new_context(harness, outcome):
    controller, service, queued, signals = harness
    _open(harness)
    entered, release = threading.Event(), threading.Event()

    def block_old():
        entered.set()
        assert release.wait(5)
        if outcome.endswith("failure"):
            raise RuntimeError("old in-flight operation failed")

    if outcome.startswith("edit"):
        service.edit_hook = block_old
        controller.replace_points([(1, 2)])
    else:
        service.save_hook = block_old
        controller.save()
    thread = threading.Thread(target=queued.pop().execute)
    thread.start()
    try:
        assert entered.wait(5)
        controller.close_session()
        _open(harness, "L02")
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    _drain()
    assert controller._session_id == "S-L02"
    assert signals["session_updated"] == []
    assert signals["saved"] == []
    assert signals["session_failed"] == []
    assert all(sid == "S-L01" for sid, _ in service.edited + service.saved)


@pytest.mark.parametrize("outcome", ["save", "failure"])
def test_string_completion_is_published_from_controller_qt_thread(harness, outcome):
    controller, service, queued, signals = harness
    _open(harness)
    threads = []
    signal = controller.saved if outcome == "save" else controller.session_failed
    signal.connect(lambda _: threads.append(QThread.currentThread()))

    def fail():
        raise RuntimeError("save failed")

    service.save_hook = fail if outcome == "failure" else None
    controller.save()
    _execute(queued.pop())
    assert threads == []
    assert signals["saved"] == []
    assert signals["session_failed"] == []
    _drain()
    assert threads == [controller.thread()]


@pytest.mark.parametrize("outcome", ["save", "failure"])
def test_reentrant_context_change_during_log_cannot_publish_old_string(harness, outcome):
    controller, service, queued, signals = harness
    _open(harness)

    def fail():
        raise RuntimeError("save failed")

    service.save_hook = fail if outcome == "failure" else None
    controller.log_message.connect(lambda _: controller.close_session())
    controller.save()
    _execute(queued.pop())
    _drain()
    assert signals["saved"] == []
    assert signals["session_failed"] == []


def test_unsaved_changes_include_pending_edit_until_successful_save(harness):
    controller, _, queued, _ = harness
    _open(harness)
    assert not controller.has_unsaved_changes()
    controller.replace_points([(1, 2)])
    assert controller.has_unsaved_changes()
    _execute(queued.pop())
    _drain()
    assert controller.has_unsaved_changes()
    controller.save()
    assert controller.has_unsaved_changes()
    _execute(queued.pop())
    _drain()
    assert not controller.has_unsaved_changes()


def test_failed_edit_keeps_preexisting_dirty_state(harness):
    controller, service, queued, _ = harness
    _open(harness)
    controller.replace_points([(1, 2)])
    _execute(queued.pop())
    _drain()

    def fail():
        raise RuntimeError("edit failed")

    service.edit_hook = fail
    controller.replace_points([(3, 4)])
    _execute(queued.pop())
    _drain()
    assert controller.has_unsaved_changes()
    controller.close_session()
    assert not controller.has_unsaved_changes()


def test_old_save_completion_cannot_clear_new_session_dirty_state(harness):
    controller, service, queued, _ = harness
    _open(harness)
    entered, release = threading.Event(), threading.Event()

    def block_save():
        entered.set()
        assert release.wait(5)

    service.save_hook = block_save
    controller.save()
    thread = threading.Thread(target=queued.pop().execute)
    thread.start()
    try:
        assert entered.wait(5)
        controller.close_session()
        _open(harness, "L02")
        controller.replace_points([(1, 2)])
        _execute(queued.pop())
        _drain()
        assert controller.has_unsaved_changes()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    _drain()
    assert controller.has_unsaved_changes()


def test_failed_edit_from_clean_session_releases_pending_dirty_state(harness):
    controller, service, queued, _ = harness
    _open(harness)

    def fail():
        raise RuntimeError("edit failed")

    service.edit_hook = fail
    controller.replace_points([(1, 2)])
    assert controller.has_unsaved_changes()
    _execute(queued.pop())
    _drain()
    assert not controller.has_unsaved_changes()


def test_failed_save_keeps_dirty_state(harness):
    controller, service, queued, _ = harness
    _open(harness)
    controller.replace_points([(1, 2)])
    _execute(queued.pop())
    _drain()

    def fail():
        raise RuntimeError("save failed")

    service.save_hook = fail
    controller.save()
    _execute(queued.pop())
    _drain()
    assert controller.has_unsaved_changes()


def test_clean_edit_snapshot_clears_dirty_state(harness):
    controller, service, queued, _ = harness
    _open(harness)
    controller.replace_points([(1, 2)])
    _execute(queued.pop())
    _drain()
    assert controller.has_unsaved_changes()
    service.replace_points = lambda sid, _: SimpleNamespace(
        project_id="P1", line_id="L01", session_id=sid, dirty=False)
    controller.replace_points([])
    _execute(queued.pop())
    _drain()
    assert not controller.has_unsaved_changes()

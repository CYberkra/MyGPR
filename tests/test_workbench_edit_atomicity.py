"""Failed interactive edits preserve scientific data, history and durable draft."""
import copy

import numpy as np
import pytest

from mygpr.interfaces.backend import MyGPRBackend
from mygpr.domain.processing.workbench import WorkbenchStep


@pytest.fixture
def workbench(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    project = backend.projects.create_project(tmp_path / "project", name="atomic edits")
    matrix = np.arange(64 * 12, dtype=np.float32).reshape(64, 12)
    backend.projects.save_line_dataset(
        project.project_id, "L01", matrix,
        time_window_ns=512.0, length_m=11.0,
    )
    service = backend.processing_workbench
    session = service.open_session(project.project_id, "L01", restore_draft=False)
    state = service._session(session.session_id)
    state.original_trace_metadata["distance_m"] = np.arange(12, dtype=float)
    state.current_trace_metadata["distance_m"] = np.arange(12, dtype=float)
    service.append_step(session.session_id, "dewow", {"window": 9})
    service.append_step(session.session_id, "agcGain", {"window": 11})
    yield service, session.session_id
    backend.shutdown()


def capture(service, session_id):
    session = service._session(session_id)
    return {
        "snapshot": service.get_session(session_id),
        "matrix": np.array(session.current_data, copy=True),
        "header": copy.deepcopy(session.current_header),
        "traces": copy.deepcopy(session.current_trace_metadata),
        "undo": copy.deepcopy([[dict(step.params) for step in steps] for steps in session.undo_stack]),
        "redo": copy.deepcopy([[dict(step.params) for step in steps] for steps in session.redo_stack]),
        "draft": service._draft_path(session).read_bytes(),
    }


def assert_unchanged(service, session_id, before):
    after = capture(service, session_id)
    assert after["snapshot"] == before["snapshot"]
    np.testing.assert_array_equal(after["matrix"], before["matrix"])
    np.testing.assert_equal(after["header"], before["header"])
    np.testing.assert_equal(after["traces"], before["traces"])
    for key in ("undo", "redo", "draft"):
        assert after[key] == before[key]


@pytest.mark.parametrize("edit", ["update", "remove", "move", "undo", "redo", "template"])
def test_replay_failure_preserves_full_session(workbench, monkeypatch, edit):
    service, session_id = workbench
    template_id = service.save_template(service._session(session_id).project_id, name="test", session_id=session_id).template_id
    if edit == "redo":
        service.undo(session_id)
    before = capture(service, session_id)
    original = service._processing.execute_method

    def fail_replay(request, context):
        if request.method_id == "agcGain" or edit in {"remove", "undo"}:
            raise ValueError("injected replay failure")
        return original(request, context)

    monkeypatch.setattr(service._processing, "execute_method", fail_replay)
    edits = {
        "update": lambda: service.update_step(session_id, 0, params={"window": 15}),
        "remove": lambda: service.remove_step(session_id, 0),
        "move": lambda: service.move_step(session_id, 0, 1),
        "undo": lambda: service.undo(session_id),
        "redo": lambda: service.redo(session_id),
        "template": lambda: service.apply_template(session_id, template_id),
    }
    with pytest.raises(ValueError, match="injected replay failure"):
        edits[edit]()
    assert_unchanged(service, session_id, before)
    monkeypatch.setattr(service._processing, "execute_method", original)
    service.append_step(session_id, "dewow", {"window": 7})


def test_failed_append_preserves_redo_history(workbench):
    service, session_id = workbench
    service.undo(session_id)
    before = capture(service, session_id)
    with pytest.raises(Exception, match="unknown processing method"):
        service.append_step(session_id, "unknown-method")
    assert_unchanged(service, session_id, before)
    assert service.redo(session_id).can_undo


def test_executor_mutation_before_failure_does_not_leak(workbench, monkeypatch):
    service, session_id = workbench
    before = capture(service, session_id)

    def fail(request, _context):
        request.data[:] = -999
        request.header_info["total_time_ns"] = 1.0
        request.trace_metadata["distance_m"][:] = -999
        raise ValueError("injected mutating executor")

    monkeypatch.setattr(service._processing, "execute_method", fail)
    with pytest.raises(ValueError, match="injected mutating executor"):
        service.append_step(session_id, "dewow", {"window": 7})
    assert_unchanged(service, session_id, before)


def test_atomic_writer_replace_failure_keeps_draft_and_session(workbench, monkeypatch):
    service, session_id = workbench
    before = capture(service, session_id)

    def fail(*_args, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr("core.storage_primitives.os.replace", fail)
    with pytest.raises(OSError, match="injected replace failure"):
        service.update_step(session_id, 0, params={"window": 15})
    assert_unchanged(service, session_id, before)


def test_nested_step_parameters_are_isolated_on_failed_replay(workbench, monkeypatch):
    service, session_id = workbench
    state = service._session(session_id)
    first = state.steps[0]
    state.steps[0] = WorkbenchStep(first.step_id, first.method_id, first.label,
                                  {"window": 9, "nested": {"values": [1, 2]}})
    before = capture(service, session_id)

    def fail(request, _context):
        request.params["nested"]["values"].append(99)
        raise ValueError("injected nested mutation")

    monkeypatch.setattr(service._processing, "execute_method", fail)
    with pytest.raises(ValueError, match="injected nested mutation"):
        service.update_step(session_id, 1, label="changed")
    assert_unchanged(service, session_id, before)
    assert state.steps[0].params["nested"]["values"] == [1, 2]


@pytest.mark.parametrize("operation", ["undo", "redo", "move", "invalid"])
def test_noop_or_invalid_edit_does_not_copy_or_replace_current_matrix(workbench, monkeypatch, operation):
    service, session_id = workbench
    if operation == "undo":
        service._session(session_id).undo_stack.clear()
    state = service._session(session_id)
    matrix = state.current_data
    original_array = np.array

    def guarded_array(value, *args, **kwargs):
        assert value is not matrix, "no-op edit copied the entire current matrix"
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr("mygpr.application.processing.workbench_service.np.array", guarded_array)
    if operation == "invalid":
        with pytest.raises(IndexError):
            service.update_step(session_id, 999)
    elif operation == "move":
        service.move_step(session_id, 0, 0)
    else:
        getattr(service, operation)(session_id)
    assert service._session(session_id) is state
    assert state.current_data is matrix


@pytest.mark.parametrize("edit", ["append", "update", "reset"])
def test_draft_failure_preserves_full_session(workbench, monkeypatch, edit):
    service, session_id = workbench
    before = capture(service, session_id)

    def fail(*_args, **_kwargs):
        raise OSError("injected draft failure")

    monkeypatch.setattr(service, "_clear_draft" if edit == "reset" else "_save_draft", fail)
    edits = {
        "append": lambda: service.append_step(session_id, "dewow", {"window": 7}),
        "update": lambda: service.update_step(session_id, 0, params={"window": 15}),
        "reset": lambda: service.reset(session_id),
    }
    with pytest.raises(OSError, match="injected draft failure"):
        edits[edit]()
    assert_unchanged(service, session_id, before)

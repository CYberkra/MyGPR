"""Concurrent comparison reads have independent identity and bounded I/O."""

from types import SimpleNamespace as NS
from threading import Thread, get_ident

import numpy as np
import pytest

from ui.controllers import project_controller as module


@pytest.fixture
def context(qapp, monkeypatch):
    queued, calls = [], []
    pc = module.ProjectController()
    pc._current = NS(project_id="P")

    def info(pid, line):
        return NS(shape=(1000, 4000), time_window_ns=900.0, length_m=80.0)

    def read(pid, line, **kwargs):
        calls.append((pid, line, kwargs))
        return np.ones((8, 6)), np.arange(8) * 100, np.arange(6) * 600

    pc.set_backend(NS(backend=NS(projects=NS(get_dataset_info=info, read_window=read))))
    monkeypatch.setattr(pc, "_line_elevation", lambda *a: None)
    monkeypatch.setattr(module, "run_command", lambda command, **kw: queued.append(command))
    return pc, queued, calls


def test_references_survive_other_preview_requests_and_emit_on_gui_thread(context, qapp):
    pc, queued, calls = context
    got, ordinary = [], []
    pc.comparison_preview_ready.connect(lambda *args: got.append((get_ident(), args)))
    pc.dataset_preview_ready.connect(ordinary.append)
    pc.preview_comparison_line("P", "L02", "first")
    pc.preview_comparison_line("P", "L03", "second")
    assert pc._preview_generation == 0
    pc.preview_line("L01")
    worker = Thread(target=lambda: [c.execute() for c in reversed(queued[:2])])
    worker.start()
    worker.join()
    assert got == []  # queued to QObject's GUI thread
    qapp.processEvents()
    assert [item[1][:3] for item in got] == [("P", "L03", "second"), ("P", "L02", "first")]
    assert all(item[0] == get_ident() for item in got)
    assert not ordinary
    assert all(c[2] == dict(max_samples=900, max_traces=1800) for c in calls)
    assert got[0][1][3].matrix.shape == (8, 6)


def test_project_roundtrip_also_fences_already_queued_read(context, qapp):
    pc, queued, _ = context
    got = []
    pc.comparison_preview_ready.connect(lambda *args: got.append(args))
    pc.preview_comparison_line("P", "L02", "old")
    worker = Thread(target=queued[0].execute)
    worker.start()
    worker.join()
    pc._project_context_generation += 2  # P -> Q -> P before Qt delivery
    qapp.processEvents()
    assert got == []
    queued[0].execute()
    assert got == []


def test_reference_failure_does_not_poison_current_line(context):
    pc, queued, _ = context
    got, errors = [], []
    pc.comparison_preview_ready.connect(lambda *args: got.append(args))
    pc.preview_failed.connect(errors.append)

    def fail(*a, **kw):
        raise OSError("fixture unreadable")

    pc._backend().projects.read_window = fail
    pc.preview_comparison_line("P", "L02", "failure")
    queued[0].execute()
    assert got[0][:4] == ("P", "L02", "failure", None)
    assert got[0][4] and not errors


def test_missing_backend_returns_reference_failure_without_work(context):
    pc, queued, _ = context
    pc.set_backend(None)
    got = []
    pc.comparison_preview_ready.connect(lambda *args: got.append(args))
    pc.preview_comparison_line("P", "L02", "unavailable")
    assert got and got[0][:4] == ("P", "L02", "unavailable", None)
    assert not queued

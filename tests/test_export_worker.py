from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QSignalSpy

from ui.export_worker import ExportTaskWorker


def _qapp():
    return QApplication.instance() or QApplication([])


def test_export_task_worker_emits_finished():
    _qapp()
    worker = ExportTaskWorker(lambda value: {"value": value}, 7)
    spy = QSignalSpy(worker.finished)
    worker.run()
    assert len(spy) == 1
    assert spy[0] == [{"value": 7}]


def test_export_task_worker_emits_failed():
    _qapp()

    def boom():
        raise RuntimeError("export failed")

    worker = ExportTaskWorker(boom)
    spy = QSignalSpy(worker.failed)
    worker.run()
    assert len(spy) == 1
    assert "export failed" in spy[0][0]

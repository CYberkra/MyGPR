# -*- coding: utf-8 -*-
"""Background export helpers for MyGPR GUI actions.

The worker executes pure file-writing/export callables away from the Qt main
thread.  GUI state must be snapshotted before the task starts; the callable must
not access Qt widgets directly.  This keeps long ZIP/PNG/JSON export operations
from blocking interaction while preserving deterministic output payloads.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class ExportTaskWorker(QObject):
    """Run a single export callable in a ``QThread``."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._func(*self._args, **self._kwargs)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)


def start_export_task(parent: QObject, func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[QThread, ExportTaskWorker]:
    """Create and start a background export task.

    The caller owns the returned thread/worker references and should keep them
    alive until one of the worker signals fires.
    """

    thread = QThread(parent)
    worker = ExportTaskWorker(func, *args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker

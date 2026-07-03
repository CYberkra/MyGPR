#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small QThread worker used by workbench copy/hash/QC operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class WorkbenchTaskWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.finished.emit(self.operation(*self.args, **self.kwargs))
        except Exception as exc:
            self.failed.emit(str(exc))


__all__ = ["WorkbenchTaskWorker"]

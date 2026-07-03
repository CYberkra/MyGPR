#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qt signal adapter for the pure MyGPR shared data model."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from core.shared_data_model import SharedDataModel


class SharedDataQtAdapter(QObject, SharedDataModel):
    """QObject wrapper around :class:`core.shared_data_model.SharedDataModel`.

    The underlying state and history logic stays in the pure core model. This
    adapter is intentionally thin: it preserves the historical ``changed`` Qt
    signal used by the GUI while keeping ``core`` usable without PyQt6.
    """

    changed = pyqtSignal(object)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        SharedDataModel.__init__(self)

    def _notify_changed(self, event: dict) -> None:  # type: ignore[override]
        SharedDataModel._notify_changed(self, event)
        self.changed.emit(dict(event))


# Historical public name used by app_qt.py and existing scripts/tests.
SharedDataState = SharedDataQtAdapter

__all__ = ["SharedDataQtAdapter", "SharedDataState"]

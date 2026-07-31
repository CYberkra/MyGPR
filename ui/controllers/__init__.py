#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qt controller layer bridging the MyGPR backend services and the UI.

Controllers are plain ``QObject`` subclasses: no QWidget imports, all
long-running work happens on worker threads and every cross-thread
notification travels through ``pyqtSignal``.
"""
from __future__ import annotations

__all__ = [
    "BackendController",
    "JobBridge",
    "ProjectController",
    "ProcessingController",
    "InterpretationController",
    "DeliveryController",
]


def __getattr__(name: str):
    if name in ("BackendController", "JobBridge"):
        from ui.controllers.backend_controller import BackendController, JobBridge

        return {"BackendController": BackendController, "JobBridge": JobBridge}[name]
    if name == "ProjectController":
        from ui.controllers.project_controller import ProjectController

        return ProjectController
    if name == "ProcessingController":
        from ui.controllers.processing_controller import ProcessingController

        return ProcessingController
    if name == "InterpretationController":
        from ui.controllers.interpretation_controller import InterpretationController

        return InterpretationController
    if name == "DeliveryController":
        from ui.controllers.delivery_controller import DeliveryController

        return DeliveryController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

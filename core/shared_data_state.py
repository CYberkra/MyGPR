#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible pure shared-data alias.

Core callers now receive the Qt-free model.  Desktop code that needs a Qt
``changed`` signal imports ``ui.shared_data_qt_adapter.SharedDataQtAdapter``
directly.  This removes the former core -> ui dependency.
"""
from __future__ import annotations

from core.shared_data_model import SharedDataModel

SharedDataState = SharedDataModel

__all__ = ["SharedDataModel", "SharedDataState"]

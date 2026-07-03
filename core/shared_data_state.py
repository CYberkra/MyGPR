#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim for the GUI shared data state.

Use ``core.shared_data_model.SharedDataModel`` for headless/core logic.  The
historical ``SharedDataState`` name is kept here for GUI-facing imports and
for older scripts; it resolves to the Qt adapter in ``ui.shared_data_qt_adapter``.
"""

from __future__ import annotations

from ui.shared_data_qt_adapter import SharedDataQtAdapter, SharedDataState

__all__ = ["SharedDataQtAdapter", "SharedDataState"]

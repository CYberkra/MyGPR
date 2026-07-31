#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for the migrated AutoTune application slice.

New production code should construct :class:`mygpr.interfaces.backend.MyGPRBackend`.
Historical public and selected private imports remain available during migration.
"""
from __future__ import annotations

from mygpr.application.autotune import candidate_generators as _candidates
from mygpr.application.autotune import diagnostics as _diagnostics
from mygpr.application.autotune import legacy_engine as _entrypoints
from mygpr.application.autotune import scoring as _scoring

for _module in (_entrypoints, _candidates, _diagnostics, _scoring):
    for _name in dir(_module):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_module, _name)

__all__ = list(_entrypoints.__all__)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for the shared MyGPR error taxonomy."""
from mygpr.domain.common import errors as _implementation

for _name in dir(_implementation):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_implementation, _name)

__all__ = list(_implementation.__all__)

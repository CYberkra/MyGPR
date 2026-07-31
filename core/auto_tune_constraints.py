#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for AutoTune parameter constraints."""
from mygpr.domain.autotune import constraints as _implementation

for _name in dir(_implementation):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_implementation, _name)

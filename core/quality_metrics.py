#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for AutoTune quality metrics."""
from mygpr.domain.autotune import quality_metrics as _implementation

for _name in dir(_implementation):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_implementation, _name)

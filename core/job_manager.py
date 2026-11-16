#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cooperative cancellation contract shared by long-running core operations.

Execution is owned by the Qt/application adapter.  Core modules only need the
exception type: long-running routines raise :class:`JobCancelled` as soon as
they observe a cancellation request, and the application layer maps it onto its
own job state.  Job lifecycle records, journals and resource arbitration live in
the application layer, not here.
"""
from __future__ import annotations


class JobCancelled(RuntimeError):
    """Raised by a worker when cooperative cancellation is observed."""


__all__ = ["JobCancelled"]

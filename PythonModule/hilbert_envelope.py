"""Compatibility facade for the native processing implementation."""
from __future__ import annotations

from PythonModule._compat_facade import reexport

reexport(globals(), 'mygpr.infrastructure.processing.algorithms.extended.hilbert')
del reexport

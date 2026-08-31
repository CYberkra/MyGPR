"""Compatibility facade for the native motion-processing implementation."""
from __future__ import annotations

from PythonModule._compat_facade import reexport

reexport(globals(), 'mygpr.infrastructure.processing.algorithms.motion.v2')
del reexport

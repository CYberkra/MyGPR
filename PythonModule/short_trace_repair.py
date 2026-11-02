"""Compatibility facade for the native short anomalous trace method."""
from PythonModule._compat_facade import reexport

reexport(globals(), 'mygpr.infrastructure.processing.algorithms.extended.short_trace_repair')
del reexport

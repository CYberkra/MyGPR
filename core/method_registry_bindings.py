#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Callable bindings used by the legacy processing-method metadata registry.

Canonical numerical implementations live in ``mygpr.infrastructure`` and the
``PythonModule`` compatibility facades.  Keeping bindings separate prevents the
large declarative registry from mixing import/fallback logic with metadata.
"""


from typing import Any

from PythonModule.svd_background import method_svd_background
from PythonModule.fk_filter import method_fk_filter
from PythonModule.frequency_filter_1d import method_frequency_filter_1d
from PythonModule.hankel_svd import method_hankel_svd
from PythonModule.hilbert_envelope import method_hilbert_envelope
from PythonModule.kirchhoff_migration import method_kirchhoff_migration
from PythonModule.stolt_migration import method_stolt_migration
from PythonModule.time_cut import method_time_cut
from PythonModule.time_to_depth import method_time_to_depth
from PythonModule.trace_qc import method_trace_qc
from PythonModule.sec_gain import method_sec_gain
from PythonModule.sliding_average import method_sliding_average
from PythonModule.rpca_background import method_rpca_background
from PythonModule.ccbs_filter import method_ccbs
from PythonModule.median_background_2D import method_median_background_2d
from PythonModule.trace_median_filter import method_trace_median_filter
from PythonModule.trace_savgol_filter import method_trace_savgol_filter
from PythonModule.svd_subspace import method_svd_subspace

_method_wavelet_2d: Any
_method_wavelet_svd: Any

try:
    import pywt as _pywt  # noqa: F401
    from PythonModule.wavelet_2d import method_wavelet_2d as _imported_method_wavelet_2d
    from PythonModule.wavelet_svd import method_wavelet_svd as _imported_method_wavelet_svd

    HAS_PYWAVELETS = True
    _method_wavelet_2d = _imported_method_wavelet_2d
    _method_wavelet_svd = _imported_method_wavelet_svd
except ModuleNotFoundError as e:
    if e.name != "pywt":
        raise

    HAS_PYWAVELETS = False

    def _missing_wavelet_2d(*args, **kwargs):
        raise ImportError(
            "Wavelet 2D 去噪需要安装 PyWavelets。请执行: pip install PyWavelets"
        )

    def _missing_wavelet_svd(*args, **kwargs):
        raise ImportError(
            "Wavelet-SVD 需要安装 PyWavelets。请执行: pip install PyWavelets"
        )

    # Preserve the historical compatibility identity exposed by core.methods_registry.
    _missing_wavelet_2d.__module__ = "core.methods_registry"
    _missing_wavelet_svd.__module__ = "core.methods_registry"
    _method_wavelet_2d = _missing_wavelet_2d
    _method_wavelet_svd = _missing_wavelet_svd


from PythonModule.dewow import method_dewow
from PythonModule.equidistant_trace_resample import method_equidistant_trace_resample
from PythonModule.energy_decay_gain import method_energy_decay_gain
from PythonModule.set_zero_time import method_set_zero_time
from PythonModule.motion_compensation_height import method_motion_compensation_height
from PythonModule.motion_compensation_speed import method_motion_compensation_speed  # type: ignore[import]
from PythonModule.trajectory_smoothing import method_trajectory_smoothing
from PythonModule.motion_compensation_attitude import (  # type: ignore[import]
    method_motion_compensation_attitude,
)
from PythonModule.motion_compensation_vibration import (  # type: ignore[import]
    method_motion_compensation_vibration,
)
from PythonModule.motion_compensation_v2 import method_motion_compensation_v2
from PythonModule.amplitude_scale import method_amplitude_scale


__all__ = [
    name
    for name in globals()
    if name.startswith("method_") or name.startswith("_method_") or name == "HAS_PYWAVELETS"
]

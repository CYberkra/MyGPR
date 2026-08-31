#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Callable bindings used by the legacy processing-method metadata registry.

Canonical numerical implementations live in ``mygpr.infrastructure`` and the
``PythonModule`` compatibility facades.  Keeping bindings separate prevents the
large declarative registry from mixing import/fallback logic with metadata.
"""


from typing import Any

from PythonModule.svd_background import method_svd_background as method_svd_background  # explicit re-export
from PythonModule.fk_filter import method_fk_filter as method_fk_filter  # explicit re-export
from PythonModule.frequency_filter_1d import method_frequency_filter_1d as method_frequency_filter_1d  # explicit re-export
from PythonModule.hankel_svd import method_hankel_svd as method_hankel_svd  # explicit re-export
from PythonModule.hilbert_envelope import method_hilbert_envelope as method_hilbert_envelope  # explicit re-export
from PythonModule.kirchhoff_migration import method_kirchhoff_migration as method_kirchhoff_migration  # explicit re-export
from PythonModule.stolt_migration import method_stolt_migration as method_stolt_migration  # explicit re-export
from PythonModule.time_cut import method_time_cut as method_time_cut  # explicit re-export
from PythonModule.time_to_depth import method_time_to_depth as method_time_to_depth  # explicit re-export
from PythonModule.trace_qc import method_trace_qc as method_trace_qc  # explicit re-export
from PythonModule.sec_gain import method_sec_gain as method_sec_gain  # explicit re-export
from PythonModule.sliding_average import method_sliding_average as method_sliding_average  # explicit re-export
from PythonModule.rpca_background import method_rpca_background as method_rpca_background  # explicit re-export
from PythonModule.ccbs_filter import method_ccbs as method_ccbs  # explicit re-export
from PythonModule.median_background_2D import method_median_background_2d as method_median_background_2d  # explicit re-export
from PythonModule.trace_median_filter import method_trace_median_filter as method_trace_median_filter  # explicit re-export
from PythonModule.trace_savgol_filter import method_trace_savgol_filter as method_trace_savgol_filter  # explicit re-export
from PythonModule.svd_subspace import method_svd_subspace as method_svd_subspace  # explicit re-export

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


from PythonModule.dewow import method_dewow as method_dewow  # explicit re-export
from PythonModule.equidistant_trace_resample import method_equidistant_trace_resample as method_equidistant_trace_resample  # explicit re-export
from PythonModule.energy_decay_gain import method_energy_decay_gain as method_energy_decay_gain  # explicit re-export
from PythonModule.set_zero_time import method_set_zero_time as method_set_zero_time  # explicit re-export
from PythonModule.motion_compensation_height import method_motion_compensation_height as method_motion_compensation_height  # explicit re-export
from PythonModule.motion_compensation_speed import method_motion_compensation_speed as method_motion_compensation_speed  # explicit re-export
from PythonModule.trajectory_smoothing import method_trajectory_smoothing as method_trajectory_smoothing  # explicit re-export
from PythonModule.motion_compensation_attitude import method_motion_compensation_attitude as method_motion_compensation_attitude  # type: ignore[import]  # explicit re-export
from PythonModule.motion_compensation_vibration import method_motion_compensation_vibration as method_motion_compensation_vibration  # type: ignore[import]  # explicit re-export
from PythonModule.motion_compensation_v2 import method_motion_compensation_v2 as method_motion_compensation_v2  # explicit re-export
from PythonModule.amplitude_scale import method_amplitude_scale as method_amplitude_scale  # explicit re-export


__all__ = [
    name
    for name in globals()
    if name.startswith("method_") or name.startswith("_method_") or name == "HAS_PYWAVELETS"
]

"""Native UAV-GPR motion-compensation algorithms and adapters."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mygpr.infrastructure.processing.algorithms.motion.attitude import method_motion_compensation_attitude
from mygpr.infrastructure.processing.algorithms.motion.height import method_motion_compensation_height
from mygpr.infrastructure.processing.algorithms.motion.speed import method_motion_compensation_speed
from mygpr.infrastructure.processing.algorithms.motion.trajectory import method_trajectory_smoothing
from mygpr.infrastructure.processing.algorithms.motion.v2 import method_motion_compensation_v2
from mygpr.infrastructure.processing.algorithms.motion.vibration import method_motion_compensation_vibration

MotionFunction = Callable[..., tuple[np.ndarray, dict[str, Any]]]


def _runtime_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(params or {})
    header = dict(runtime.pop("_header_info", runtime.pop("header_info", {})) or {})
    trace = dict(runtime.pop("_trace_metadata", runtime.pop("trace_metadata", {})) or {})
    runtime.pop("_execution_context", None)
    runtime["header_info"] = header
    runtime["trace_metadata"] = trace
    return runtime


def _execute(function: MotionFunction, data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return function(data, **_runtime_kwargs(params))


def native_motion_height(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _execute(method_motion_compensation_height, data, params)


def native_motion_speed(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _execute(method_motion_compensation_speed, data, params)


def native_trajectory_smoothing(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _execute(method_trajectory_smoothing, data, params)


def native_motion_attitude(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _execute(method_motion_compensation_attitude, data, params)


def native_motion_vibration(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _execute(method_motion_compensation_vibration, data, params)


def native_motion_v2(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _execute(method_motion_compensation_v2, data, params)


__all__ = [
    "native_motion_attitude",
    "native_motion_height",
    "native_motion_speed",
    "native_motion_v2",
    "native_motion_vibration",
    "native_trajectory_smoothing",
]

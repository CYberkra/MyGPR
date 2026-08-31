#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default AutoTune composition for historical function-style callers.

The implementation is dependency-injected in :mod:`service`.  This module is a
small composition bridge retained until all callers construct ``MyGPRBackend``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mygpr.application.autotune.errors import AutoTuneCancelled, AutoTuneError
from mygpr.application.autotune.ports import AutoTuneDependencies
from mygpr.application.autotune.service import (
    auto_select_method_group_with_dependencies,
    auto_tune_method_with_dependencies,
)
from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.autotune.models import (
    AutoTuneContext,
    FAILURE_PENALTY,
    INVALID_TRIAL_SCORE,
    OuterSelectionScore,
    PROFILE_LABELS,
    TrialScore,
)
from mygpr.infrastructure.processing.autotune_adapter import DomainAutoTuneConstraintPolicy
from mygpr.infrastructure.processing.legacy_adapter import LegacyProcessingExecutor
from mygpr.infrastructure.processing.native_adapter import (
    CompositeProcessingExecutor,
    NativeProcessingCatalog,
    NativeProcessingExecutor,
)

_CATALOG = NativeProcessingCatalog()
_LEGACY_EXECUTOR = LegacyProcessingExecutor()
_NATIVE_EXECUTOR = NativeProcessingExecutor()
_DEFAULT_DEPENDENCIES = AutoTuneDependencies(
    catalog=_CATALOG,
    executor=CompositeProcessingExecutor(_NATIVE_EXECUTOR, _LEGACY_EXECUTOR),
    constraints=DomainAutoTuneConstraintPolicy(),
)


def default_autotune_dependencies() -> AutoTuneDependencies:
    """Return the process-wide immutable default dependency bundle."""
    return _DEFAULT_DEPENDENCIES


def auto_tune_method(
    data: np.ndarray,
    method_key: str,
    candidate_params: list[dict[str, Any]] | None = None,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    base_params: dict[str, Any] | None = None,
    roi_spec: dict[str, Any] | None = None,
    search_mode: str = "standard",
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    execution_context: ExecutionContext | None = None,
) -> dict[str, Any]:
    """Run AutoTune through native methods with controlled legacy fallback."""
    return auto_tune_method_with_dependencies(
        _DEFAULT_DEPENDENCIES,
        data,
        method_key,
        candidate_params=candidate_params,
        header_info=header_info,
        trace_metadata=trace_metadata,
        base_params=base_params,
        roi_spec=roi_spec,
        search_mode=search_mode,
        progress_callback=progress_callback,
        cancel_checker=cancel_checker,
        execution_context=execution_context,
    )


def auto_select_method_group(
    data: np.ndarray,
    method_keys: list[str],
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    base_params_map: dict[str, dict[str, Any]] | None = None,
    roi_spec: dict[str, Any] | None = None,
    search_mode: str = "standard",
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    execution_context: ExecutionContext | None = None,
) -> dict[str, Any]:
    """Compare methods through native methods with controlled legacy fallback."""
    return auto_select_method_group_with_dependencies(
        _DEFAULT_DEPENDENCIES,
        data,
        method_keys,
        header_info=header_info,
        trace_metadata=trace_metadata,
        base_params_map=base_params_map,
        roi_spec=roi_spec,
        search_mode=search_mode,
        progress_callback=progress_callback,
        cancel_checker=cancel_checker,
        execution_context=execution_context,
    )


__all__ = [
    "AutoTuneCancelled",
    "AutoTuneContext",
    "AutoTuneDependencies",
    "AutoTuneError",
    "FAILURE_PENALTY",
    "INVALID_TRIAL_SCORE",
    "OuterSelectionScore",
    "PROFILE_LABELS",
    "TrialScore",
    "auto_select_method_group",
    "auto_tune_method",
    "default_autotune_dependencies",
]

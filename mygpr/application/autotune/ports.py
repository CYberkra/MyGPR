#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ports and dependency bundle for the AutoTune application service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from mygpr.application.processing.ports import ProcessingCatalogPort, ProcessingExecutorPort
from mygpr.domain.autotune.constraints import ParameterConstraintResult

ProgressCallback = Callable[[int, int, str], None]
CancelChecker = Callable[[], bool]


class AutoTuneConstraintPort(Protocol):
    """Data-shape-aware parameter constraint policy."""

    def constrain(
        self,
        method_id: str,
        params: dict[str, Any],
        data_shape: tuple[int, int],
        header_info: dict[str, Any] | None = None,
    ) -> ParameterConstraintResult: ...


@dataclass(frozen=True, slots=True)
class AutoTuneDependencies:
    """All concrete services required by AutoTune orchestration."""

    catalog: ProcessingCatalogPort
    executor: ProcessingExecutorPort
    constraints: AutoTuneConstraintPort


class AutoTuneUseCase(Protocol):
    """Callable contract used by presentation controllers."""

    def __call__(self, *args: object, **kwargs: object) -> dict[str, object]: ...


__all__ = [
    "AutoTuneConstraintPort",
    "AutoTuneDependencies",
    "AutoTuneUseCase",
    "CancelChecker",
    "ProgressCallback",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concrete AutoTune constraint adapter."""
from __future__ import annotations

from typing import Any

from mygpr.application.autotune.ports import AutoTuneConstraintPort
from mygpr.domain.autotune.constraints import (
    ParameterConstraintResult,
    constrain_auto_tune_params,
)


class DomainAutoTuneConstraintPolicy(AutoTuneConstraintPort):
    """Apply the migrated domain parameter constraints."""

    def constrain(
        self,
        method_id: str,
        params: dict[str, Any],
        data_shape: tuple[int, int],
        header_info: dict[str, Any] | None = None,
    ) -> ParameterConstraintResult:
        return constrain_auto_tune_params(method_id, params, data_shape, header_info)


__all__ = ["DomainAutoTuneConstraintPolicy"]

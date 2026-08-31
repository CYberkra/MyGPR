#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable application entry points for automatic parameter tuning."""
from __future__ import annotations

from mygpr.application.autotune.errors import AutoTuneCancelled, AutoTuneError
from mygpr.application.autotune.legacy_engine import auto_select_method_group, auto_tune_method
from mygpr.application.autotune.ports import AutoTuneDependencies
from mygpr.application.autotune.service import (
    AutoTuneService,
    auto_select_method_group_with_dependencies,
    auto_tune_method_with_dependencies,
)
from mygpr.domain.autotune.models import (
    AutoTuneContext,
    FAILURE_PENALTY,
    INVALID_TRIAL_SCORE,
    OuterSelectionScore,
    PROFILE_LABELS,
    TrialScore,
)

__all__ = [
    "AutoTuneCancelled",
    "AutoTuneContext",
    "AutoTuneDependencies",
    "AutoTuneError",
    "AutoTuneService",
    "FAILURE_PENALTY",
    "INVALID_TRIAL_SCORE",
    "OuterSelectionScore",
    "PROFILE_LABELS",
    "TrialScore",
    "auto_select_method_group",
    "auto_select_method_group_with_dependencies",
    "auto_tune_method",
    "auto_tune_method_with_dependencies",
]

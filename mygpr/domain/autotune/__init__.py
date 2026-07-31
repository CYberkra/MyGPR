"""Domain rules and models for automatic parameter tuning."""

from mygpr.domain.autotune.models import (
    AutoTuneContext,
    FAILURE_PENALTY,
    INVALID_TRIAL_SCORE,
    OuterSelectionScore,
    PROFILE_LABELS,
    TrialScore,
)

__all__ = [
    "AutoTuneContext",
    "FAILURE_PENALTY",
    "INVALID_TRIAL_SCORE",
    "OuterSelectionScore",
    "PROFILE_LABELS",
    "TrialScore",
]

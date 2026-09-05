"""Velocity analysis domain package: hyperbola fitting for (UAV-)GPR."""
from mygpr.domain.velocity.errors import VelocityAnalysisError
from mygpr.domain.velocity.fitting import fit_hyperbola
from mygpr.domain.velocity.models import (
    VELOCITY_ANALYSIS_EVIDENCE_SCHEMA,
    HyperbolaFit,
    VelocityPick,
)

__all__ = [
    "VELOCITY_ANALYSIS_EVIDENCE_SCHEMA",
    "HyperbolaFit",
    "VelocityAnalysisError",
    "VelocityPick",
    "fit_hyperbola",
]

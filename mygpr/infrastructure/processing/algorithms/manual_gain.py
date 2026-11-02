"""Piecewise dB gain: same numerical rule as the acquisition display, saved output."""
from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.domain.processing.manual_gain import apply_manual_gain_curve, normalize_gain_nodes
from mygpr.infrastructure.processing.algorithms.common import normalize_output


def method_manual_gain(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    nodes = normalize_gain_nodes(params.get("nodes"))
    result, db, warnings = apply_manual_gain_curve(data, nodes)
    return normalize_output("manual_gain", result, {
        "method": "manual_gain", "nodes": nodes,
        "position_basis": "current_input_samples", "interpolation": "linear_db",
        "amplitude_rule": "10**(db/20)",
        "gain_min_db": float(db.min()), "gain_max_db": float(db.max()),
    }, warnings)

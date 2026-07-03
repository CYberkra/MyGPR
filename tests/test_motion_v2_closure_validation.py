#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for Motion V2 closure validation wiring."""

from __future__ import annotations

from scripts import validate_motion_v2_closure as closure


def test_motion_v2_closure_atomic_pipeline_preserves_equal_distance_axis() -> None:
    """Speed compensation must run after attitude/APC footprint updates."""
    assert closure.ATOMIC_PIPELINE == (
        "trajectory_smoothing",
        "motion_compensation_attitude",
        "motion_compensation_speed",
        "motion_compensation_height",
    )

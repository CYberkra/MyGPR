#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motion-compensation workflow contracts for airborne GPR."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep


@dataclass(frozen=True, slots=True)
class MotionCompensationProfile:
    """Versioned backend profile; frontends only choose a profile and parameters."""

    mode: str = "integrated_v2"
    integrated_params: Mapping[str, Any] = field(default_factory=dict)
    height_params: Mapping[str, Any] = field(default_factory=dict)
    speed_params: Mapping[str, Any] = field(default_factory=dict)
    attitude_params: Mapping[str, Any] = field(default_factory=dict)
    vibration_params: Mapping[str, Any] = field(default_factory=dict)
    include_vibration_cleanup: bool = False

    def __post_init__(self) -> None:
        mode = str(self.mode or "integrated_v2")
        if mode not in {"integrated_v2", "atomic"}:
            raise ValueError(f"unsupported motion compensation mode: {mode}")
        object.__setattr__(self, "mode", mode)
        for field_name in (
            "integrated_params", "height_params", "speed_params",
            "attitude_params", "vibration_params",
        ):
            object.__setattr__(self, field_name, MappingProxyType(dict(getattr(self, field_name) or {})))


def build_motion_pipeline(profile: MotionCompensationProfile | None = None) -> PipelineDefinition:
    selected = profile or MotionCompensationProfile()
    if selected.mode == "integrated_v2":
        steps = [
            PipelineStep(
                "motion_compensation_v2",
                dict(selected.integrated_params),
                label="Integrated UAV motion compensation v2",
            )
        ]
    else:
        steps = [
            PipelineStep("motion_compensation_speed", dict(selected.speed_params), label="Speed/spacing compensation"),
            PipelineStep("motion_compensation_attitude", dict(selected.attitude_params), label="Attitude compensation"),
            PipelineStep("motion_compensation_height", dict(selected.height_params), label="Height compensation"),
        ]
    if selected.include_vibration_cleanup:
        steps.append(
            PipelineStep(
                "motion_compensation_vibration",
                dict(selected.vibration_params),
                label="Vibration and rotor-interference suppression",
            )
        )
    return PipelineDefinition(
        name="UAV-GPR motion compensation",
        schema_version="mygpr.motion_pipeline.v1",
        steps=tuple(steps),
    )


__all__ = ["MotionCompensationProfile", "build_motion_pipeline"]

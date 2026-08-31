#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed processing-plugin contracts and production/research catalog split."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

ALGORITHM_API_VERSION = "mygpr.processing_plugin.v1"


@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm_id: str
    display_name: str
    callable: Callable[..., Any] | str | None
    api_version: str = ALGORITHM_API_VERSION
    category: str = "experimental"
    maturity: str = "experimental"
    visibility: str = "hidden"
    field_approved: bool = False
    research_only: bool = True
    input_contract: str = "bscan.samples_x_traces"
    output_contract: str = "bscan.samples_x_traces"
    preserves_trace_axis: bool = True
    preserves_sample_axis: bool = True
    memory_model: str = "in_memory"
    cancellation_points: str = "method_defined"
    validated_devices: tuple[str, ...] = ()
    validated_frequency_range_mhz: tuple[float, float] | None = None
    parameter_schema: tuple[dict[str, Any], ...] = ()
    known_risks: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.algorithm_id or self.algorithm_id.startswith("_"):
            raise ValueError("Algorithm id must be a public stable identifier")
        if self.api_version != ALGORITHM_API_VERSION:
            raise ValueError(f"Unsupported algorithm API: {self.api_version}")
        if self.maturity not in {"stable", "experimental", "research", "deprecated"}:
            raise ValueError(f"Unsupported maturity: {self.maturity}")
        if self.field_approved and (self.research_only or self.visibility != "public"):
            raise ValueError("Field-approved algorithms must be public and not research-only")
        if self.callable is None:
            raise ValueError(f"Algorithm {self.algorithm_id} has no implementation")


def spec_from_legacy(algorithm_id: str, record: Mapping[str, Any]) -> AlgorithmSpec:
    maturity = str(record.get("maturity") or "experimental")
    visibility = str(record.get("visibility") or "public")
    field_approved = bool(record.get("field_approved", maturity == "stable" and visibility == "public"))
    shape_changed = bool(record.get("shape_changed", False))
    axis_transform = str(record.get("axis_transform") or "identity")
    preserves_trace = not shape_changed and axis_transform not in {"trace_remove", "trace_resample"}
    preserves_sample = not shape_changed and axis_transform not in {"sample_crop", "time_to_depth"}
    function = record.get("func") or record.get("module")
    frequency_values = record.get("validated_frequency_range_mhz")
    frequency_range: tuple[float, float] | None = None
    if frequency_values:
        values = tuple(float(v) for v in frequency_values)
        if len(values) != 2:
            raise ValueError(f"{algorithm_id}: validated frequency range must contain two values")
        frequency_range = (values[0], values[1])
    spec = AlgorithmSpec(
        algorithm_id=algorithm_id,
        display_name=str(record.get("display_name") or record.get("name") or algorithm_id),
        callable=function,
        category=str(record.get("category") or "experimental"),
        maturity=maturity,
        visibility=visibility,
        field_approved=field_approved,
        research_only=not field_approved,
        input_contract=str(record.get("input_contract") or "bscan.samples_x_traces"),
        output_contract=str(record.get("output_contract") or "bscan.samples_x_traces"),
        preserves_trace_axis=bool(record.get("preserves_trace_axis", preserves_trace)),
        preserves_sample_axis=bool(record.get("preserves_sample_axis", preserves_sample)),
        memory_model=str(record.get("memory_model") or "in_memory"),
        cancellation_points=str(record.get("cancellation_points") or "method_defined"),
        validated_devices=tuple(record.get("validated_devices") or ()),
        validated_frequency_range_mhz=frequency_range,
        parameter_schema=tuple(dict(item) for item in record.get("params", ())),
        known_risks=tuple(str(item) for item in record.get("known_risks", ())),
        metadata={key: value for key, value in record.items() if key not in {"func", "params"}},
    )
    spec.validate()
    return spec


class AlgorithmCatalog:
    def __init__(self, specs: list[AlgorithmSpec]) -> None:
        self._specs = {spec.algorithm_id: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate algorithm id")

    @classmethod
    def from_legacy(cls, registry: Mapping[str, Mapping[str, Any]]) -> "AlgorithmCatalog":
        specs = [spec_from_legacy(key, record) for key, record in registry.items() if not key.startswith("_")]
        return cls(specs)

    def get(self, algorithm_id: str) -> AlgorithmSpec:
        return self._specs[algorithm_id]

    def production(self) -> tuple[AlgorithmSpec, ...]:
        return tuple(spec for spec in self._specs.values() if spec.field_approved)

    def research(self) -> tuple[AlgorithmSpec, ...]:
        return tuple(spec for spec in self._specs.values() if not spec.field_approved)

    def all(self) -> tuple[AlgorithmSpec, ...]:
        return tuple(self._specs.values())


__all__ = ["ALGORITHM_API_VERSION", "AlgorithmCatalog", "AlgorithmSpec", "spec_from_legacy"]

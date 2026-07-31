#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central parameter-schema validation for every processing entry point."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mygpr.domain.common.errors import ParameterValidationError
from mygpr.domain.processing.models import ProcessingMethodDescriptor


_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "bool": (bool, np.bool_),
    "int": (int, np.integer),
    "float": (int, float, np.integer, np.floating),
    "str": (str,),
}


def validate_parameters(
    descriptor: ProcessingMethodDescriptor,
    params: Mapping[str, Any] | None,
    *,
    reject_unknown: bool = True,
) -> dict[str, Any]:
    """Validate names, types, ranges and common cross-field invariants.

    Defaults remain owned by the algorithm implementation; this function only
    validates user-supplied values so existing numerical behavior is preserved.
    """
    values = {str(key): value for key, value in dict(params or {}).items()}
    schema = {str(key): dict(spec or {}) for key, spec in descriptor.parameter_schema.items()}
    if reject_unknown and schema:
        unknown = sorted(key for key in values if key not in schema and not key.startswith("_"))
        if unknown:
            raise ParameterValidationError(
                f"unknown parameter(s) for {descriptor.method_id}: {', '.join(unknown)}",
                context={"method_id": descriptor.method_id, "unknown_parameters": unknown},
            )

    validated: dict[str, Any] = {}
    for key, value in values.items():
        if key.startswith("_") or key not in schema:
            validated[key] = value
            continue
        spec = schema[key]
        type_name = str(spec.get("type", "")).strip().lower()
        expected = _TYPE_MAP.get(type_name)
        if expected is not None:
            if type_name in {"int", "float"} and isinstance(value, (bool, np.bool_)):
                valid_type = False
            else:
                valid_type = isinstance(value, expected)
            if not valid_type:
                raise ParameterValidationError(
                    f"parameter {key!r} for {descriptor.method_id} must be {type_name}",
                    context={
                        "method_id": descriptor.method_id,
                        "parameter": key,
                        "expected_type": type_name,
                        "actual_type": type(value).__name__,
                    },
                )
        normalized = value.item() if isinstance(value, np.generic) else value
        if "min" in spec and float(normalized) < float(spec["min"]):
            raise ParameterValidationError(
                f"parameter {key!r} for {descriptor.method_id} is below minimum {spec['min']}",
                context={"method_id": descriptor.method_id, "parameter": key, "minimum": spec["min"]},
            )
        if "max" in spec and float(normalized) > float(spec["max"]):
            raise ParameterValidationError(
                f"parameter {key!r} for {descriptor.method_id} exceeds maximum {spec['max']}",
                context={"method_id": descriptor.method_id, "parameter": key, "maximum": spec["max"]},
            )
        choices = spec.get("choices", spec.get("options"))
        if choices is not None and normalized not in choices:
            raise ParameterValidationError(
                f"parameter {key!r} for {descriptor.method_id} must be one of {tuple(choices)!r}",
                context={"method_id": descriptor.method_id, "parameter": key, "choices": list(choices)},
            )
        validated[key] = normalized

    _validate_cross_fields(descriptor.method_id, validated)
    return validated


def _validate_cross_fields(method_id: str, values: Mapping[str, Any]) -> None:
    ordered_pairs = (
        ("rank_start", "rank_end"),
        ("gain_min", "gain_max"),
        ("angle_low", "angle_high"),
        ("time_start_ns", "time_end_ns"),
    )
    for lower, upper in ordered_pairs:
        if lower in values and upper in values:
            low = float(values[lower])
            high = float(values[upper])
            if upper == "time_end_ns" and high == 0.0:
                continue
            if low > high:
                raise ParameterValidationError(
                    f"parameter {lower!r} must not exceed {upper!r} for {method_id}",
                    context={"method_id": method_id, lower: low, upper: high},
                )
    if "low_freq_mhz" in values and "high_freq_mhz" in values:
        if float(values["low_freq_mhz"]) >= float(values["high_freq_mhz"]):
            raise ParameterValidationError(
                f"low_freq_mhz must be lower than high_freq_mhz for {method_id}",
                context={"method_id": method_id},
            )


__all__ = ["validate_parameters"]

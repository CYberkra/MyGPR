#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical horizontal/vertical CRS contracts and precision budgets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover
    CRS = None
    Transformer = None


@dataclass(frozen=True)
class CRSDefinition:
    authority: str
    wkt: str
    axis_unit: str
    is_geographic: bool

    @classmethod
    def parse(cls, value: Any) -> "CRSDefinition":
        if CRS is None:
            raise RuntimeError("CRS validation requires pyproj")
        crs = CRS.from_user_input(value)
        authority = ":".join(crs.to_authority()) if crs.to_authority() else ""
        unit = crs.axis_info[0].unit_name if crs.axis_info else ""
        return cls(authority=authority, wkt=crs.to_wkt(), axis_unit=unit, is_geographic=bool(crs.is_geographic))


@dataclass(frozen=True)
class CoordinatePrecisionBudget:
    horizontal_tolerance_m: float = 0.10
    vertical_tolerance_m: float = 0.15
    roundtrip_tolerance_m: float = 0.01


def verify_roundtrip(source_crs: Any, target_crs: Any, x: float, y: float, *, budget: CoordinatePrecisionBudget | None = None) -> float:
    if Transformer is None:
        raise RuntimeError("CRS validation requires pyproj")
    selected = budget or CoordinatePrecisionBudget()
    forward = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    reverse = Transformer.from_crs(target_crs, source_crs, always_xy=True)
    tx, ty = forward.transform(x, y)
    rx, ry = reverse.transform(tx, ty)
    error = ((rx - x) ** 2 + (ry - y) ** 2) ** 0.5
    if error > selected.roundtrip_tolerance_m:
        raise ValueError(f"CRS roundtrip error {error:.6f} exceeds {selected.roundtrip_tolerance_m:.6f}")
    return float(error)


__all__ = ["CRSDefinition", "CoordinatePrecisionBudget", "verify_roundtrip"]

"""Immutable spatial contracts exposed by the backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SpatialTrackPoint:
    trace_index: int
    x: float
    y: float
    elevation_m: float = 0.0
    interface_elevation_m: float | None = None
    interface_depth_m: float | None = None


@dataclass(frozen=True, slots=True)
class SpatialTrack:
    line_id: str
    name: str
    points: tuple[SpatialTrackPoint, ...]
    coordinate_system: str = ""
    source: str = ""


@dataclass(frozen=True, slots=True)
class SpatialResult:
    result_id: str
    name: str
    revision: int
    status: str
    line_ids: tuple[str, ...]
    created_at: str = ""
    coordinate_system: str = ""
    vertical_datum: str = ""
    stale: bool = False
    summary: Mapping[str, Any] = field(default_factory=dict)
    files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary or {})))
        object.__setattr__(self, "files", MappingProxyType(dict(self.files or {})))


__all__ = ["SpatialResult", "SpatialTrack", "SpatialTrackPoint"]

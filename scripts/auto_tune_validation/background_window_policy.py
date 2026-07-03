#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relative trace-count-aware background window candidate policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_RATIO_CANDIDATES = (0.05, 0.10, 0.20, 0.40, 0.70, 1.00)


@dataclass(frozen=True)
class BackgroundWindowCandidate:
    """One background-window candidate with absolute and relative metadata."""

    ntraces: int
    ntraces_ratio: float
    label: str
    window_length_m: float | None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "ntraces": int(self.ntraces),
            "ntraces_ratio": float(self.ntraces_ratio),
            "label": self.label,
            "window_length_m": None if self.window_length_m is None else float(self.window_length_m),
        }


def policy_label_for_ratio(ratio: float) -> str:
    """Map relative window ratio to a readable policy label."""
    x = max(0.0, float(ratio))
    if x <= 0.10:
        return "local"
    if x <= 0.25:
        return "medium"
    if x <= 0.50:
        return "large"
    if x < 1.00:
        return "near_full_line"
    return "full_line"


def generate_relative_background_candidates(
    *,
    trace_count: int,
    trace_spacing_m: float | None = None,
    ratio_candidates: Iterable[float] = DEFAULT_RATIO_CANDIDATES,
    explicit_ntraces: Iterable[int] | None = None,
    max_fraction_of_trace_count: float = 1.0,
    include_full_line_candidate: bool = True,
    min_ntraces: int = 3,
    max_ntraces: int | None = None,
) -> list[BackgroundWindowCandidate]:
    """Generate odd, deduplicated, clamped candidates from relative ratios."""
    tcount = int(max(1, trace_count))
    min_allowed = _oddize_up(int(max(1, min_ntraces)))
    max_by_fraction = _oddize_down(max(1, int(round(tcount * max_fraction_of_trace_count))))
    max_allowed = _oddize_down(int(max_ntraces if max_ntraces is not None else tcount))
    max_allowed = min(max_allowed, max_by_fraction)
    max_allowed = max(max_allowed, min_allowed)

    values: set[int] = set()
    for ratio in ratio_candidates:
        n = _oddize_nearest(max(1, int(round(float(ratio) * tcount))))
        n = _clamp(n, min_allowed, max_allowed)
        values.add(n)
    if include_full_line_candidate:
        values.add(_clamp(_oddize_down(tcount), min_allowed, max_allowed))
    if explicit_ntraces is not None:
        for n in explicit_ntraces:
            v = _clamp(_oddize_nearest(int(n)), min_allowed, max_allowed)
            values.add(v)

    ordered = sorted(values)
    if not ordered:
        ordered = [min_allowed]

    candidates: list[BackgroundWindowCandidate] = []
    for n in ordered:
        ratio = float(n) / float(tcount)
        label = policy_label_for_ratio(ratio)
        wlen = None if trace_spacing_m is None else float(n) * float(trace_spacing_m)
        candidates.append(
            BackgroundWindowCandidate(
                ntraces=int(n),
                ntraces_ratio=ratio,
                label=label,
                window_length_m=wlen,
            )
        )
    return candidates


def _clamp(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def _oddize_nearest(value: int) -> int:
    if value % 2 == 1:
        return value
    down = value - 1
    up = value + 1
    if down >= 1:
        return down
    return up


def _oddize_down(value: int) -> int:
    v = int(value)
    if v % 2 == 1:
        return v
    return v - 1


def _oddize_up(value: int) -> int:
    v = int(value)
    if v % 2 == 1:
        return v
    return v + 1


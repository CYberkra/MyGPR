# -*- coding: utf-8 -*-
"""Small performance instrumentation helpers for MyGPR.

The helpers are deliberately lightweight and optional: they record wall-clock
durations for UI/rendering paths without changing numerical processing results.
They are safe to keep enabled in normal builds because each record operation is
O(1) and stores only aggregate counts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Dict, Iterator


@dataclass
class PerfCounter:
    name: str
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0

    def add(self, elapsed_ms: float) -> None:
        value = float(elapsed_ms)
        self.count += 1
        self.total_ms += value
        self.last_ms = value
        if value > self.max_ms:
            self.max_ms = value

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "last_ms": round(self.last_ms, 3),
        }


class PerfMonitor:
    """Aggregate named wall-clock timings for UI performance audits."""

    def __init__(self) -> None:
        self._counters: Dict[str, PerfCounter] = {}

    def record(self, name: str, elapsed_ms: float) -> None:
        key = str(name)
        counter = self._counters.get(key)
        if counter is None:
            counter = self._counters[key] = PerfCounter(name=key)
        counter.add(elapsed_ms)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def snapshot(self) -> dict:
        return {name: counter.to_dict() for name, counter in sorted(self._counters.items())}

    def reset(self) -> None:
        self._counters.clear()

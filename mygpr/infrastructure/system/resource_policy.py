"""Host resource guard for memory-heavy processing tasks."""
from __future__ import annotations

import os
from typing import Any

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.ports import ProcessingResourcePolicyPort
from mygpr.domain.processing.models import ResourceEstimate


class InsufficientProcessingMemory(RuntimeError):
    """Raised before a task would exceed the configured memory budget."""


class LocalProcessingResourcePolicy(ProcessingResourcePolicyPort):
    """Validate estimated memory against host availability and explicit limits."""

    def __init__(self, *, memory_fraction: float | None = None) -> None:
        configured = os.environ.get("MYGPR_PROCESSING_MEMORY_FRACTION")
        value = float(configured) if configured not in (None, "") else memory_fraction
        self._memory_fraction = min(0.95, max(0.10, float(value if value is not None else 0.75)))

    def validate(
        self,
        estimate: ResourceEstimate,
        *,
        context: ExecutionContext | None = None,
        operation: str = "processing",
    ) -> None:
        explicit = _explicit_memory_limit(context)
        available = available_memory_bytes()
        host_budget = int(available * self._memory_fraction) if available > 0 else 0
        limits = [value for value in (explicit, host_budget) if value > 0]
        budget = min(limits) if limits else 0
        if budget > 0 and estimate.memory_bytes > budget:
            raise InsufficientProcessingMemory(
                f"{operation} requires approximately {estimate.memory_bytes} bytes of memory, "
                f"but the permitted budget is {budget} bytes"
            )


def _explicit_memory_limit(context: ExecutionContext | None) -> int:
    if context is None:
        return 0
    raw: Any = context.metadata.get("max_memory_bytes", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError, OverflowError):
        return 0


def available_memory_bytes() -> int:
    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available))
    except ImportError:
        return _sysconf_available_memory()


def _sysconf_available_memory() -> int:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return max(0, page_size * available_pages)
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


__all__ = [
    "InsufficientProcessingMemory",
    "LocalProcessingResourcePolicy",
    "available_memory_bytes",
]

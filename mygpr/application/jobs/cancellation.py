#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cooperative cancellation primitives independent of any UI toolkit."""
from __future__ import annotations

from threading import Event
from typing import Callable


class JobCancelledError(RuntimeError):
    """Raised when a cooperative operation observes cancellation."""


class CancellationToken:
    """Read-only cancellation view passed to application and algorithm code."""

    __slots__ = ("_event", "_checker")

    def __init__(
        self,
        event: Event | None = None,
        checker: Callable[[], bool] | None = None,
    ) -> None:
        self._event = event or Event()
        self._checker = checker

    @property
    def is_cancelled(self) -> bool:
        if self._event.is_set():
            return True
        return bool(self._checker()) if self._checker is not None else False

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise JobCancelledError("job cancelled")

    def as_checker(self):
        """Return the legacy ``Callable[[], bool]`` cancellation shape."""
        return lambda: self.is_cancelled


class CancellationTokenSource:
    """Owner capable of requesting cancellation."""

    __slots__ = ("_event", "token")

    def __init__(self) -> None:
        self._event = Event()
        self.token = CancellationToken(self._event)

    def cancel(self) -> None:
        self._event.set()

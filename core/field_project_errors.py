#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared project-operation exceptions without importing the large operation facade."""
from __future__ import annotations

from mygpr.domain.common.errors import MyGPRError


class FieldProjectOperationError(MyGPRError):
    """Raised when a user-facing project operation cannot be completed."""


__all__ = ["FieldProjectOperationError"]

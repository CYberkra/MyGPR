#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured runtime warning helpers for processing and I/O flows."""

from __future__ import annotations

from typing import Any


def build_runtime_warning(
    code: str,
    message: str,
    *,
    level: str = "warning",
    **details: Any,
) -> dict[str, Any]:
    """Build a structured runtime warning payload."""
    return {
        "code": str(code),
        "level": str(level),
        "message": str(message),
        "details": dict(details),
    }


def merge_runtime_warnings(*warning_groups: Any) -> list[dict[str, Any]]:
    """Merge multiple warning lists while preserving order and uniqueness."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    for group in warning_groups:
        if not group:
            continue
        if isinstance(group, dict):
            iterable = [group]
        else:
            iterable = list(group)
        for item in iterable:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("code", "")),
                str(item.get("message", "")),
                tuple(
                    sorted(
                        (str(k), str(v))
                        for k, v in (item.get("details", {}) or {}).items()
                    )
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "code": str(item.get("code", "warning")),
                    "level": str(item.get("level", "warning")),
                    "message": str(item.get("message", "")),
                    "details": dict(item.get("details", {}) or {}),
                }
            )
    return merged


def format_runtime_warning_text(warning: dict[str, Any]) -> str:
    """Format a warning for logs and reports."""
    if not warning:
        return ""
    details = warning.get("details", {}) or {}
    if details:
        details_text = ", ".join(f"{k}={v}" for k, v in details.items())
        return f"[{warning.get('code', 'warning')}] {warning.get('message', '')} ({details_text})"
    return f"[{warning.get('code', 'warning')}] {warning.get('message', '')}"

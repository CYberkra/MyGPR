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


def build_processing_step_notice(method_id: str, name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Report execution separately from job success; preserve the reason for a no-op."""
    effective = metadata.get('effective_params')
    effective = effective if isinstance(effective, dict) else {}
    skipped = bool(metadata.get('skipped', False) or effective.get('skipped', False))
    warnings = metadata.get('runtime_warnings') or []
    warning_reason = next((str(item.get('message', '')) for item in warnings if isinstance(item, dict)), '')
    reason = str(metadata.get('reason') or effective.get('reason') or warning_reason or '所需输入不可用')
    status = 'skipped' if skipped else 'executed'
    outcome = f'已跳过（{reason}）' if skipped else '已执行'
    if not skipped and metadata.get('runtime_warnings'):
        outcome += '（有警告，详见日志）'
    return build_runtime_warning('processing_step_status', f'{name}：{outcome}',
                                 level='warning' if skipped else 'info',
                                 method_id=method_id, execution_status=status)


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
            iterable = group
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



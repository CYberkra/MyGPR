#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability-based workbench command policy, independent of UI pages."""

from __future__ import annotations


STATE_CAPABILITIES: dict[str, frozenset[str]] = {
    "empty": frozenset({"project_create", "project_open", "import"}),
    "temporary_preview": frozenset({"display", "qc", "formalize"}),
    "qc_review_required": frozenset({"display", "qc", "formalize", "evidence_export"}),
    "qc_blocked": frozenset({"display", "qc", "formalize", "evidence_export"}),
    "formal_ready": frozenset(
        {"display", "qc", "processing", "interpretation", "spatial", "report", "evidence_export"}
    ),
}


def assert_command_allowed(state: str, command: str) -> None:
    allowed = STATE_CAPABILITIES.get(str(state), frozenset())
    if command not in allowed:
        raise PermissionError(f"Command {command!r} is not allowed in state {state!r}")


__all__ = ["STATE_CAPABILITIES", "assert_command_allowed"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application path helpers for user-writable data directories."""

from __future__ import annotations

import os


APP_DIR_NAME = "MyGPR"


def get_app_data_dir() -> str:
    """Return the root writable directory for app settings/data."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, APP_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def get_settings_dir() -> str:
    path = os.path.join(get_app_data_dir(), "settings")
    os.makedirs(path, exist_ok=True)
    return path


def get_output_dir() -> str:
    path = os.path.join(get_app_data_dir(), "output")
    os.makedirs(path, exist_ok=True)
    return path


def get_logs_dir() -> str:
    path = os.path.join(get_output_dir(), "logs")
    os.makedirs(path, exist_ok=True)
    return path


def get_favorites_dir() -> str:
    path = os.path.join(get_app_data_dir(), "favorites")
    os.makedirs(path, exist_ok=True)
    return path


def get_workflow_templates_dir() -> str:
    path = os.path.join(get_app_data_dir(), "workflow_templates")
    os.makedirs(path, exist_ok=True)
    return path


def get_repo_root() -> str:
    """Return the MyGPR source/package root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_default_evidence_root() -> str:
    """Return the preferred Evidence root without hard-coding a workstation path.

    Priority:
    1. MYGPR_EVIDENCE_ROOT environment variable
    2. Sibling ../MyGPR-Evidence directory next to this repository/package
    3. User-writable MyGPR app data Evidence directory
    """
    env = os.environ.get("MYGPR_EVIDENCE_ROOT", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(env)))
    sibling = os.path.abspath(os.path.join(get_repo_root(), os.pardir, "MyGPR-Evidence"))
    if os.path.isdir(sibling):
        return sibling
    return os.path.join(get_app_data_dir(), "Evidence")


def get_default_gpr_result_runs_dir() -> str:
    """Return the optional local gprMax runs directory.

    This is deliberately environment/config driven. It should not default to a
    developer workstation path.
    """
    env = os.environ.get("MYGPR_GPR_RESULT_RUNS", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(env)))
    return os.path.join(get_output_dir(), "gpr_result_runs")


def expand_path_template(raw_path: str, *, base_dir: str | os.PathLike[str] | None = None) -> str:
    """Expand MyGPR path placeholders and environment variables.

    Supported placeholders:
    - ${MYGPR_REPO_ROOT}
    - ${MYGPR_EVIDENCE_ROOT}
    - ${MYGPR_GPR_RESULT_RUNS}

    Standard environment variables are also expanded. Relative paths are resolved
    against ``base_dir`` when provided.
    """
    text = str(raw_path or "").strip()
    replacements = {
        "${MYGPR_REPO_ROOT}": get_repo_root(),
        "$MYGPR_REPO_ROOT": get_repo_root(),
        "${MYGPR_EVIDENCE_ROOT}": get_default_evidence_root(),
        "$MYGPR_EVIDENCE_ROOT": get_default_evidence_root(),
        "${MYGPR_GPR_RESULT_RUNS}": get_default_gpr_result_runs_dir(),
        "$MYGPR_GPR_RESULT_RUNS": get_default_gpr_result_runs_dir(),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    text = os.path.expanduser(os.path.expandvars(text))
    if base_dir and text and not os.path.isabs(text):
        text = os.path.join(os.fspath(base_dir), text)
    return os.path.abspath(text) if text else text

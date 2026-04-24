#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application path helpers for user-writable data directories."""

from __future__ import annotations

import os


APP_DIR_NAME = "GPR_GUI"


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

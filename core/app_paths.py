#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application path helpers for user-writable data directories."""

from __future__ import annotations

import os


APP_DIR_NAME = "MyGPR"


def get_app_data_dir(*, create: bool = True) -> str:
    """Return the root writable directory for app settings/data."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, APP_DIR_NAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def get_output_dir() -> str:
    path = os.path.join(get_app_data_dir(), "output")
    os.makedirs(path, exist_ok=True)
    return path


def get_logs_dir() -> str:
    override = os.environ.get("MYGPR_LOG_DIR", "").strip()
    path = (os.path.abspath(os.path.expanduser(os.path.expandvars(override)))
            if override else os.path.join(get_output_dir(), "logs"))
    os.makedirs(path, exist_ok=True)
    return path


def get_tile_cache_dir(*, create: bool = True) -> str:
    """Return the shared, user-writable basemap and terrain tile cache."""
    path = os.path.join(get_app_data_dir(create=False), "tile_cache")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def get_repo_root() -> str:
    """Return the MyGPR source/package root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

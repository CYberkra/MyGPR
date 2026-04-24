#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the canonical top-level read_file_data module."""

from __future__ import annotations

import importlib.util
import os


_ROOT_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "read_file_data.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_gpr_root_read_file_data", _ROOT_MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"无法加载根目录 read_file_data 模块: {_ROOT_MODULE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

readcsv = _MODULE.readcsv
savecsv = _MODULE.savecsv
save_image = _MODULE.save_image
show_image = _MODULE.show_image

__all__ = ["readcsv", "savecsv", "save_image", "show_image"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility re-export for the canonical top-level ``read_file_data`` module.

Using a normal import keeps source and PyInstaller-frozen execution identical;
loading the source file by physical path breaks in one-file bundles.
"""

from __future__ import annotations

from read_file_data import readcsv, save_image, savecsv, show_image

__all__ = ["readcsv", "savecsv", "save_image", "show_image"]

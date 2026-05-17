#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theme manager regression tests."""

from __future__ import annotations

from core.theme_manager import ThemeManager


def test_theme_manager_logs_invalid_config_and_falls_back(tmp_path, caplog):
    manager = ThemeManager(base_dir=str(tmp_path))
    broken_config = tmp_path / "theme_config.json"
    broken_config.write_text("{not valid json", encoding="utf-8")
    manager.config_file = str(broken_config)

    assert manager._load_config() == "light"
    assert "加载主题配置失败" in caplog.text

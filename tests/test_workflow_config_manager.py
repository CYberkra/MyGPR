#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for workflow configuration persistence behavior."""

import logging

from core.workflow_data import WorkflowConfig, WorkflowConfigManager


def test_workflow_config_manager_logs_invalid_config(tmp_path, caplog):
    manager = WorkflowConfigManager(config_dir=str(tmp_path))
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="core.workflow_data"):
        assert manager.load_config("broken") is None

    assert "加载配置失败" in caplog.text


def test_workflow_config_manager_skips_invalid_list_entries(tmp_path, caplog):
    manager = WorkflowConfigManager(config_dir=str(tmp_path))
    good = WorkflowConfig(name="可用流程")
    manager.save_config(good, "good.json")
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="core.workflow_data"):
        configs = manager.list_configs()

    assert [item["filename"] for item in configs] == ["good.json"]
    assert "跳过无效工作流配置 broken.json" in caplog.text

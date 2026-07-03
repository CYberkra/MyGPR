#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Security regression tests for workflow config path handling."""

from __future__ import annotations

from core.workflow_data import WorkflowConfig, WorkflowConfigManager


def test_workflow_config_manager_rejects_path_traversal(tmp_path):
    manager = WorkflowConfigManager(config_dir=str(tmp_path / "workflows"))
    config = WorkflowConfig(name="safe")

    try:
        manager.save_config(config, "../escape")
    except ValueError:
        pass
    else:  # pragma: no cover - explicit failure branch for readability
        raise AssertionError("path traversal save should be rejected")

    assert not (tmp_path / "escape.json").exists()
    assert manager.load_config("../escape") is None
    assert manager.delete_config("../escape") is False


def test_workflow_config_manager_keeps_valid_names_inside_dir(tmp_path):
    manager = WorkflowConfigManager(config_dir=str(tmp_path / "workflows"))
    config = WorkflowConfig(name="safe")

    saved = manager.save_config(config, "safe_config")

    assert str(tmp_path / "workflows") in saved
    assert manager.load_config("safe_config") is not None
    assert manager.delete_config("safe_config") is True

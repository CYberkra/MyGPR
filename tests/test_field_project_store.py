#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from core.field_project_store import FieldProjectStore, FIELD_PROJECT_SCHEMA


def test_create_project_structure_and_manifest(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    assert (store.root / "project.json").exists()
    for folder in ["raw", "processed", "targets", "spatial", "reports", "logs"]:
        assert (store.root / folder).is_dir()
    assert store.manifest.schema == FIELD_PROJECT_SCHEMA
    assert len(store.list_lines()) == 8


def test_targets_and_spatial_exports_are_persistent(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    targets = store.default_targets("L03")[:2]
    store.save_targets("L03", targets)
    reopened = FieldProjectStore.open(store.root)
    loaded = reopened.load_targets("L03")
    assert len(loaded) == 2
    assert loaded[0]["name"] == "T-01"
    assert (reopened.root / "spatial" / "L03_targets_xy.csv").exists()


def test_processed_result_updates_project_json(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    data_path, params_path = store.save_processed_line("L03", np.ones((6, 8)), {"gain": {"factor": 1.8}})
    assert data_path.exists()
    assert params_path.exists()
    reopened = FieldProjectStore.open(store.root)
    line = reopened.get_line("L03")
    assert line.processing_status == "已完成"
    assert line.processed_result.endswith(".npy")

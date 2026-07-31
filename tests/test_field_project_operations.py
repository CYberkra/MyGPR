from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.field_project_operations import (
    FieldProjectOperationError,
    RecentProjectsStore,
    create_project,
    import_line_data,
    next_line_id,
    open_project,
    validate_import_source,
)


def test_create_and_open_formal_project_updates_recent_list(tmp_path: Path) -> None:
    recent = RecentProjectsStore(tmp_path / "recent_projects.json")
    store = create_project(tmp_path, name="现场测试项目", location="测试区", recent_store=recent)

    assert store.root.name == "现场测试项目"
    assert (store.root / "project.json").exists()
    assert (store.root / "raw").is_dir()
    assert store.manifest.name == "现场测试项目"
    assert store.list_lines() == []

    opened = open_project(store.root, recent_store=recent)
    assert opened.manifest.project_id == store.manifest.project_id
    records = recent.load()
    assert records and Path(records[0].path) == store.root


def test_import_line_data_creates_normalized_dataset(tmp_path: Path) -> None:
    recent = RecentProjectsStore(tmp_path / "recent_projects.json")
    store = create_project(tmp_path, name="导入测试", recent_store=recent)
    matrix_path = tmp_path / "line_matrix.npy"
    np.save(matrix_path, np.arange(96 * 48, dtype=np.float32).reshape(96, 48))

    line = import_line_data(store, matrix_path, name="导入线")

    assert line.line_id == "L01"
    assert line.gpr_dataset_path == "data/lines/L01.h5"
    assert (store.root / line.gpr_dataset_path).exists()
    assert next_line_id(store) == "L02"


def test_validate_import_source_rejects_recognized_but_unreadable_vendor_format(tmp_path: Path) -> None:
    dzt = tmp_path / "profile.dzt"
    dzt.write_bytes(b"not a readable demo file")

    with pytest.raises(FieldProjectOperationError) as excinfo:
        validate_import_source(dzt)
    assert "尚未直接解码" in str(excinfo.value)

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.field_project_operations import (
    RecentProjectsStore,
    check_project_source_files,
    create_project,
    delete_project_permanently,
    import_line_data,
    preflight_project_delete,
    prune_missing_recent_projects,
    relink_project_line_source,
)
from core.source_file_registry import (
    export_source_file_manifest_csv,
    get_line_source_record,
    load_source_registry,
    source_status_label_for_line,
)


def test_import_records_source_file_and_checks_status(tmp_path: Path) -> None:
    recent = RecentProjectsStore(tmp_path / "recent.json")
    store = create_project(tmp_path, name="source-provenance", recent_store=recent)
    source = tmp_path / "external" / "line.npy"
    source.parent.mkdir()
    np.save(source, np.arange(80 * 40, dtype=np.float32).reshape(80, 40))

    line = import_line_data(store, source, name="外部源测线")

    record = get_line_source_record(store, line.line_id)
    assert record is not None
    assert record.source_path == str(source.resolve())
    assert record.project_raw_path == line.raw_path
    assert source_status_label_for_line(store, line.line_id) == "正常"
    assert (store.root / "metadata" / "source_files.json").exists()

    source.unlink()
    rows = check_project_source_files(store)
    assert rows[0].status == "missing"
    assert source_status_label_for_line(store, line.line_id) == "缺失"


def test_relink_source_file_and_export_manifest(tmp_path: Path) -> None:
    store = create_project(tmp_path, name="source-relink")
    source = tmp_path / "external" / "line.npy"
    source.parent.mkdir()
    arr = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    np.save(source, arr)
    line = import_line_data(store, source, name="外部源测线")
    moved = tmp_path / "moved" / "line.npy"
    moved.parent.mkdir()
    source.replace(moved)

    record = relink_project_line_source(store, line.line_id, moved)
    assert record.status == "available"
    assert record.source_path == str(moved.resolve())

    out = export_source_file_manifest_csv(store)
    text = out.read_text(encoding="utf-8-sig")
    assert "line.npy" in text
    assert line.line_id in text


def test_project_delete_preflight_and_recent_cleanup_preserve_external_sources(tmp_path: Path) -> None:
    recent = RecentProjectsStore(tmp_path / "recent.json")
    store = create_project(tmp_path, name="delete-preflight", recent_store=recent)
    external = tmp_path / "external" / "line.npy"
    external.parent.mkdir()
    np.save(external, np.arange(32 * 16, dtype=np.float32).reshape(32, 16))
    import_line_data(store, external, name="外部源测线")
    stale = tmp_path / "stale_project"
    recent.save([*recent.load(), type(recent.load()[0])(path=str(stale), name="stale")])

    preview = preflight_project_delete(store, recent_store=recent)

    assert preview.line_count == 1
    assert preview.file_count > 0
    assert preview.external_source_count == 1
    assert preview.missing_recent_count == 1

    delete_project_permanently(store, recent_store=recent)
    removed = prune_missing_recent_projects(recent_store=recent)

    assert removed == 1
    assert external.exists(), "删除项目不能删除项目目录之外的原始来源文件"
    assert not store.root.exists()
    assert recent.load() == []

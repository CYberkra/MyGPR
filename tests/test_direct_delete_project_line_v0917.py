from __future__ import annotations

from pathlib import Path

from core.field_project_models import FieldLineRecord
from core.field_project_operations import (
    RecentProjectsStore,
    delete_project_line,
    delete_project_permanently,
    remove_recent_project,
)
from core.field_project_store import FieldProjectStore


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_delete_project_line_removes_project_local_artifacts_only(tmp_path: Path) -> None:
    external_source = _write(tmp_path / "external_inputs" / "source.csv", "a,b\n1,2\n")
    store = FieldProjectStore.create_empty(tmp_path / "project", name="delete-line-test")
    line = FieldLineRecord(
        line_id="L01",
        name="1号测线",
        length_m=12.3,
        raw_path=str(external_source),
        gpr_dataset_path="raw/L01/L01_gpr_dataset.npz",
        trajectory_path="raw/L01/L01_trajectory.csv",
        processed_result="processed/L01/L01_processed.npy",
        params_path="processed/L01/L01_params.json",
        target_count=1,
    )
    store.upsert_line(line)
    _write(store.root / "raw" / "L01" / "source_copy.csv", "project copy")
    _write(store.root / "raw" / "L01" / "L01_gpr_dataset.npz", "npz")
    _write(store.root / "processed" / "L01" / "L01_processed.npy", "npy")
    _write(store.root / "targets" / "L01_targets.csv", "target_id,line_id\nT-01,L01\n")
    _write(store.root / "spatial" / "L01_targets_xy.csv", "line_id,x,y\nL01,1,2\n")
    _write(store.root / "spatial" / "project_spatial_coordinates.csv", "line_id,x,y\nL01,1,2\n")

    result = delete_project_line(store, "L01")

    assert result.line_id == "L01"
    assert result.remaining_line_count == 0
    assert len(result.deleted_paths) >= 5
    assert store.list_lines() == []
    assert not (store.root / "raw" / "L01").exists()
    assert not (store.root / "processed" / "L01").exists()
    assert not (store.root / "targets" / "L01_targets.csv").exists()
    assert not (store.root / "spatial" / "L01_targets_xy.csv").exists()
    assert not (store.root / "trash").exists()
    assert external_source.exists(), "删除测线不能删除项目目录之外的原始导入来源文件"
    assert store.manifest.reports["status"] == "需重新生成"
    assert "已删除" in store.manifest.reports["stale_reason"]


def test_recent_project_remove_does_not_touch_project_files(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="recent-remove-test")
    recent = RecentProjectsStore(tmp_path / "recent.json")
    recent.add(store)

    removed = remove_recent_project(store.root, recent_store=recent)

    assert removed == 1
    assert recent.load() == []
    assert (store.root / FieldProjectStore.MANIFEST_NAME).exists()


def test_delete_project_permanently_removes_project_folder_not_external_sources(tmp_path: Path) -> None:
    external_source = _write(tmp_path / "external_inputs" / "source.csv", "a,b\n1,2\n")
    store = FieldProjectStore.create_empty(tmp_path / "project", name="delete-project-test")
    _write(store.root / "raw" / "L01" / "source_copy.csv", "project copy")
    _write(store.root / "project_notes.txt", "note")
    recent = RecentProjectsStore(tmp_path / "recent.json")
    recent.add(store)
    original_root = store.root

    result = delete_project_permanently(store, recent_store=recent)

    assert result.project_name == "delete-project-test"
    assert result.original_path == str(original_root)
    assert result.deleted_path == str(original_root)
    assert not original_root.exists()
    assert not (tmp_path / ".mygpr_trash").exists()
    assert external_source.exists(), "删除项目不能删除项目目录之外的原始导入来源文件"
    assert recent.load() == []

from __future__ import annotations

from pathlib import Path

from core.field_project_operations import RecentProjectsStore, create_project, open_project, update_project_metadata
from core.field_project_status import build_project_status_snapshot
from ui.field_panels.project_dialogs import ProjectCreateDialog, ProjectSettingsDialog


def test_create_project_persists_engineering_metadata(tmp_path: Path) -> None:
    recent = RecentProjectsStore(tmp_path / "recent.json")
    store = create_project(
        tmp_path,
        name="地下管线复核",
        location="测试测区B",
        operator="张工",
        project_no="PROJ-2026-0001",
        device_model="Mala CX12",
        coordinate_system="CGCS2000 / UTM 50N",
        vertical_datum="1985 国家高程基准",
        recent_store=recent,
    )
    reopened = open_project(store.root, recent_store=recent)

    assert reopened.manifest.project_no == "PROJ-2026-0001"
    assert reopened.manifest.device_model == "Mala CX12"
    assert reopened.manifest.coordinate_system == "CGCS2000 / UTM 50N"
    assert reopened.manifest.vertical_datum == "1985 国家高程基准"
    assert reopened.manifest.operator == "张工"

    snapshot = build_project_status_snapshot(reopened)
    assert snapshot.project_no == "PROJ-2026-0001"
    assert snapshot.operator == "张工"
    assert snapshot.coordinate_system == "CGCS2000 / UTM 50N"
    assert snapshot.vertical_datum == "1985 国家高程基准"


def test_update_project_metadata_updates_manifest_and_recent_name(tmp_path: Path) -> None:
    recent = RecentProjectsStore(tmp_path / "recent.json")
    store = create_project(tmp_path, name="旧项目名", recent_store=recent)

    update_project_metadata(
        store,
        name="新项目名",
        location="新测区",
        operator="李工",
        project_no="PROJ-EDIT-001",
        device_model="IDS Stream DP",
        coordinate_system="WGS84 / UTM zone 50N",
        vertical_datum="椭球高",
        recent_store=recent,
    )

    reopened = open_project(store.root, recent_store=recent)
    assert reopened.manifest.name == "新项目名"
    assert reopened.manifest.location == "新测区"
    assert reopened.manifest.operator == "李工"
    assert reopened.manifest.project_no == "PROJ-EDIT-001"
    assert reopened.manifest.coordinate_system == "WGS84 / UTM zone 50N"
    assert recent.load()[0].name == "新项目名"


def test_project_dialogs_expose_engineering_metadata() -> None:
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    create_dialog = ProjectCreateDialog(default_dir="/tmp")
    values = create_dialog.values()
    assert "coordinate_system" in values
    assert "vertical_datum" in values
    assert "device_model" in values

    store = type("Manifest", (), {
        "name": "项目A",
        "project_no": "NO-1",
        "location": "测区",
        "operator": "操作员",
        "device_model": "设备",
        "coordinate_system": "坐标",
        "vertical_datum": "基准",
    })()
    settings = ProjectSettingsDialog(manifest=store)
    values = settings.values()
    assert values["name"] == "项目A"
    assert values["project_no"] == "NO-1"
    assert values["coordinate_system"] == "坐标"

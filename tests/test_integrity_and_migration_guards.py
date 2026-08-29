# -*- coding: utf-8 -*-
"""证据链篡改检出与 schema 迁移链的负向测试。

证据链的核心承诺是"篡改可检出"：本文件对封存/清单/数据三类哈希
分别验证 verify() 在被篡改时返回 False；对 schema 迁移链验证
逐版本升级、缺失迁移步报错、更新版本只读保护。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.project_catalog import ProjectCatalog
from core.reporting_model import ReportSealer, ReportSnapshot
from core.schema_registry import (
    DEFAULT_SCHEMA_REGISTRY,
    LoadedDocument,
    SchemaDefinition,
    SchemaRegistry,
    UnsupportedSchemaError,
)


# ---------------------------------------------------------------------------
# 报告封存（ReportSealer）：篡改可检出
# ---------------------------------------------------------------------------

def _snapshot() -> ReportSnapshot:
    return ReportSnapshot.capture(
        project_id="P-TAMPER",
        project_revision=1,
        software_version="0.9.37",
        template_version="field-industry-v1",
    )


def _sealed_package(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "report_v1"
    package.mkdir()
    (package / "project_report.pdf").write_bytes(b"%PDF-1.4 evidence")
    (package / "summary.json").write_text("{}", encoding="utf-8")
    seal_path = ReportSealer().seal(
        package, snapshot=_snapshot(), approval={"approver": "tester"}
    )
    return package, seal_path


def test_seal_verifies_clean_package(tmp_path: Path) -> None:
    _, seal_path = _sealed_package(tmp_path)
    ok, errors = ReportSealer().verify(seal_path)
    assert ok is True
    assert errors == []


def test_tampered_file_content_is_detected(tmp_path: Path) -> None:
    package, seal_path = _sealed_package(tmp_path)
    (package / "project_report.pdf").write_bytes(b"%PDF-1.4 TAMPERED")
    ok, errors = ReportSealer().verify(seal_path)
    assert ok is False
    assert "hash:project_report.pdf" in errors


def test_tampered_manifest_hash_is_detected(tmp_path: Path) -> None:
    _, seal_path = _sealed_package(tmp_path)
    payload = json.loads(seal_path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    seal_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    ok, errors = ReportSealer().verify(seal_path)
    assert ok is False
    assert "hash:seal-manifest" in errors


def test_deleted_sealed_file_is_detected(tmp_path: Path) -> None:
    package, seal_path = _sealed_package(tmp_path)
    (package / "summary.json").unlink()
    ok, errors = ReportSealer().verify(seal_path)
    assert ok is False
    assert any(err.startswith("unsafe-or-missing:") for err in errors)


# ---------------------------------------------------------------------------
# 项目目录（ProjectCatalog）：物理损坏可检出
# ---------------------------------------------------------------------------

def test_catalog_integrity_check_detects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "catalog.db"
    catalog = ProjectCatalog(path)
    catalog.initialize(project_id="P-001", project_name="integrity")
    assert catalog.integrity_check() == (True, "ok")

    # 覆写为垃圾字节后，SQLite 的 PRAGMA integrity_check 必须报损坏。
    path.write_bytes(b"not a sqlite database at all" * 8)
    ok, message = catalog.integrity_check()
    assert ok is False
    assert message


# ---------------------------------------------------------------------------
# Schema 迁移链：逐版本升级 / 缺步报错 / 新版本只读
# ---------------------------------------------------------------------------

def _field_project_v1() -> dict:
    return {
        "schema": "mygpr.field_project.v1",
        "project_id": "P-001",
        "name": "migration-chain",
        "lines": [],
    }


def test_field_project_migrates_v1_to_v3_stepwise() -> None:
    document = DEFAULT_SCHEMA_REGISTRY.migrate(_field_project_v1(), family="mygpr.field_project")
    assert isinstance(document, LoadedDocument)
    assert document.migrated is True
    assert document.current_schema == "mygpr.field_project.v3"
    # v1 -> v2 注入 CRS 与 revision；v2 -> v3 注入存储后端与策略
    assert document.payload["revision"] == 0
    assert document.payload["storage_backend"] == "legacy_files_v2"
    assert document.payload["storage_policy"]["bounded_memory"] is True
    assert document.payload["storage_policy"]["atomic_commit"] is True


def test_sensor_sync_migrates_v1_to_v2() -> None:
    document = DEFAULT_SCHEMA_REGISTRY.migrate(
        {"schema": "mygpr.sensor_sync.v1", "config": {}}, family="mygpr.sensor_sync"
    )
    assert document.migrated is True
    assert document.current_schema == "mygpr.sensor_sync.v2"
    for name in ("radar", "rtk", "imu", "altimeter"):
        assert document.payload["config"][f"{name}_clock"]["epoch"] == "relative_start"


def test_newer_schema_is_read_only_and_untouched() -> None:
    future = dict(_field_project_v1())
    future["schema"] = "mygpr.field_project.v9"
    future["future_field"] = {"keep": True}
    document = DEFAULT_SCHEMA_REGISTRY.migrate(future, family="mygpr.field_project")
    assert document.read_only is True
    assert document.payload["future_field"] == {"keep": True}
    # 不得伪造成当前版本
    assert document.payload["schema"] == "mygpr.field_project.v9"


def test_wrong_family_is_rejected() -> None:
    with pytest.raises(UnsupportedSchemaError):
        DEFAULT_SCHEMA_REGISTRY.migrate(
            {"schema": "mygpr.sensor_sync.v1"}, family="mygpr.field_project"
        )


def test_missing_migration_step_raises() -> None:
    registry = SchemaRegistry()

    def _noop(payload: dict) -> dict:
        return dict(payload)

    # current=3 但只登记了 v1 -> v2：v2 -> v3 缺失，必须显式报错而不是静默跳过。
    registry.register(
        SchemaDefinition(
            "mygpr.test_gap", 3, migrations={1: _noop}, owner="project"
        )
    )
    with pytest.raises(UnsupportedSchemaError, match="Missing migration"):
        registry.migrate({"schema": "mygpr.test_gap.v1"}, family="mygpr.test_gap")


def test_every_registered_multi_version_chain_is_complete() -> None:
    # 对注册表中所有多版本家族做一次 v1 起点的迁移演练，任何缺步都会在此暴露。
    families_with_migrations = [
        ("mygpr.field_project", _field_project_v1()),
        ("mygpr.sensor_sync", {"schema": "mygpr.sensor_sync.v1", "config": {}}),
        ("mygpr.gis_layers", {"schema": "mygpr.gis_layers.v1", "layers": []}),
        ("mygpr.source_files", {"schema": "mygpr.source_files.v1", "sources": []}),
        ("mygpr.job_journal", {"schema": "mygpr.job_journal.v1", "jobs": []}),
    ]
    for family, payload in families_with_migrations:
        document = DEFAULT_SCHEMA_REGISTRY.migrate(payload, family=family)
        assert document.migrated is True
        assert document.read_only is False
        assert str(document.payload["schema"]).startswith(family)

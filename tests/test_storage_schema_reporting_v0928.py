from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import pytest

from core.project_repository import ProjectAccessMode, ProjectRepository
from core.reporting_model import ReportSealer, ReportSnapshot
from core.schema_registry import (
    DEFAULT_SCHEMA_REGISTRY,
    DocumentCorruptionError,
)
from core.source_file_registry import (
    SourceFileRecord,
    complete_source_hash,
    fingerprint_source,
    load_source_registry,
    save_source_registry,
)
from core.storage_primitives import ProjectReadOnlyError, atomic_write_json, utc_now


def _field_project_v1() -> dict:
    return {
        "schema": "mygpr.field_project.v1",
        "project_id": "P-001",
        "name": "migration-case",
        "lines": [],
    }


def test_schema_migration_snapshots_and_newer_schema_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    atomic_write_json(path, _field_project_v1())

    loaded = DEFAULT_SCHEMA_REGISTRY.load_path(path, family="mygpr.field_project")
    assert loaded.migrated is True
    assert loaded.read_only is False
    assert loaded.payload["schema"] == "mygpr.field_project.v3"
    assert loaded.payload["revision"] == 0
    assert loaded.payload["storage_backend"] == "legacy_files_v2"
    assert loaded.payload["legacy_layout"] is True
    assert list((tmp_path / ".schema_snapshots").glob("project.json.*.json"))

    atomic_write_json(path, {**loaded.payload, "schema": "mygpr.field_project.v99"})
    newer = DEFAULT_SCHEMA_REGISTRY.load_path(path, family="mygpr.field_project")
    assert newer.read_only is True
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "mygpr.field_project.v99"



def test_field_store_opens_newer_schema_read_only_without_leaking_lock_or_writing(tmp_path: Path) -> None:
    from core.field_project_store import FieldProjectStore

    root = tmp_path / "newer-project"
    root.mkdir()
    payload = {
        **_field_project_v1(),
        "schema": "mygpr.field_project.v99",
        "coordinate_crs_wkt": "",
        "vertical_crs_wkt": "",
        "revision": 4,
        "storage_policy": {"single_writer": True, "atomic_commit": True},
    }
    atomic_write_json(root / "project.json", payload)
    before = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    store = FieldProjectStore.open(root, access_mode=ProjectAccessMode.WRITE)
    try:
        assert store.read_only is True
        assert store.manifest.schema == "mygpr.field_project.v99"
        with pytest.raises(ProjectReadOnlyError):
            store.save_manifest()
    finally:
        store.close()
    after = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    assert before == after
    assert not (root / ".mygpr.lock").exists()

def test_schema_corruption_is_quarantined_instead_of_reset(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(DocumentCorruptionError) as caught:
        DEFAULT_SCHEMA_REGISTRY.load_path(path, family="mygpr.field_project")
    quarantine = caught.value.quarantine_path
    assert quarantine is not None and quarantine.exists()
    assert quarantine.with_suffix(quarantine.suffix + ".reason.json").exists()
    assert path.read_text(encoding="utf-8") == "{not-json"


def test_project_session_read_only_guard_and_transaction_rollback(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    lock_payload = {
        "schema": "mygpr.project_lock.v1",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "token": "external-writer",
        "created_at": utc_now(),
    }
    atomic_write_json(root / ".mygpr.lock", lock_payload)
    with ProjectRepository.open_session(root, mode=ProjectAccessMode.AUTO) as session:
        assert session.read_only is True
        with pytest.raises(ProjectReadOnlyError):
            session.assert_writable()
    (root / ".mygpr.lock").unlink()

    target = root / "metadata.json"
    target.write_text("before", encoding="utf-8")
    with ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE) as session:
        with pytest.raises(RuntimeError):
            with session.transaction("rollback-contract") as transaction:
                transaction.track(target)
                target.write_text("after", encoding="utf-8")
                raise RuntimeError("force rollback")
    assert target.read_text(encoding="utf-8") == "before"


def test_large_source_identity_can_be_completed_to_full_sha256(tmp_path: Path) -> None:
    root = tmp_path / "project"
    source = tmp_path / "source.bin"
    source.write_bytes((b"MYGPR-source-identity" * 4096) + b"tail")
    fp = fingerprint_source(source, hash_limit_bytes=1)
    assert fp["full_hash_status"] == "pending"
    assert not fp["source_sha256"]
    assert fp["quick_sha256"] and fp["merkle_root"]

    save_source_registry(
        root,
        [SourceFileRecord(line_id="L01", role="gpr", **fp)],
    )
    completed = complete_source_hash(root, "L01")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert completed.full_hash_status == "complete"
    assert completed.source_sha256 == expected
    assert load_source_registry(root)[0].source_sha256 == expected


def test_report_seal_detects_post_seal_mutation(tmp_path: Path) -> None:
    package = tmp_path / "report"
    package.mkdir()
    (package / "project_report.pdf").write_bytes(b"pdf-payload")
    (package / "report_manifest.json").write_text('{"schema":"mygpr.report_package.v3"}', encoding="utf-8")
    snapshot = ReportSnapshot.capture(
        project_id="P-001",
        project_revision=7,
        software_version="0.9.28",
        template_version="field-industry-v1",
    )
    sealer = ReportSealer()
    seal_path = sealer.seal(package, snapshot=snapshot, approval={"approver": "reviewer"})
    ok, errors = sealer.verify(seal_path)
    assert ok is True and errors == []

    (package / "project_report.pdf").write_bytes(b"tampered")
    ok, errors = sealer.verify(seal_path)
    assert ok is False
    assert "hash:project_report.pdf" in errors

"""Names change metadata only; relocation preserves one complete project."""

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.jobs.cancellation import CancellationTokenSource, JobCancelledError
from mygpr.application.project.service import ProjectBusyError
from mygpr.interfaces.backend import MyGPRBackend


@pytest.fixture
def project(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(tmp_path / "原始项目", name="校园勘探")
    matrix = np.arange(360, dtype="float32").reshape(18, 20)
    backend.projects.save_line_dataset(
        summary.project_id, "L01", matrix, name="第一条", time_window_ns=900.0, length_m=10.0
    )
    yield backend, summary, matrix
    backend.shutdown()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_rename_line_and_project_keep_identity_data_and_reopen(project):
    backend, before, matrix = project
    root = Path(before.root_path)
    data_hash = digest(root / "data/lines/L01.h5")
    line = backend.projects.rename_line(before.project_id, "L01", "  南北向·垂直极化  ")
    assert line.line_id == "L01" and line.name == "南北向·垂直极化"
    changed = backend.projects.rename_project(before.project_id, "  校园复测  ")
    assert changed.project_id == before.project_id and changed.root_path == before.root_path
    assert changed.name == "校园复测"
    assert digest(root / "data/lines/L01.h5") == data_hash
    with sqlite3.connect(root / "catalog.sqlite") as db:
        assert db.execute("select name from lines where line_id=?", ("L01",)).fetchone()[0] == line.name
        assert db.execute("select value from catalog_meta where key=?", ("project_name",)).fetchone()[0] == changed.name
    backend.projects.close_project(before.project_id)
    reopened = backend.projects.open_project(root)
    assert reopened.name == changed.name
    assert backend.projects.get_line(before.project_id, "L01").name == line.name
    np.testing.assert_array_equal(backend.projects.read_dataset(before.project_id, "L01").data, matrix)


@pytest.mark.parametrize("name", ["", "  ", "名字\n换行", "名字\x00", "a" * 81, None, 42])
def test_invalid_names_do_not_mutate_project(project, name):
    backend, before, _ = project
    manifest = (Path(before.root_path) / "project.json").read_bytes()
    with pytest.raises(ValueError):
        backend.projects.rename_line(before.project_id, "L01", name)
    with pytest.raises(ValueError):
        backend.projects.rename_project(before.project_id, name)
    assert (Path(before.root_path) / "project.json").read_bytes() == manifest


def test_rename_failure_rolls_back_memory_manifest_and_catalog(project, monkeypatch):
    backend, before, _ = project
    store = backend.projects._session(before.project_id)._store
    manifest = (store.root / "project.json").read_bytes()

    def fail(*a, **kw):
        raise OSError("injected catalog failure")

    monkeypatch.setattr(store.storage.catalog, "upsert_line", fail)
    with pytest.raises(OSError):
        backend.projects.rename_line(before.project_id, "L01", "失败名称")
    assert (store.root / "project.json").read_bytes() == manifest
    assert backend.projects.get_line(before.project_id, "L01").name == "第一条"
    monkeypatch.setattr(store.storage.catalog, "set_meta", fail)
    with pytest.raises(OSError):
        backend.projects.rename_project(before.project_id, "失败项目")
    assert (store.root / "project.json").read_bytes() == manifest
    assert backend.projects.get_summary(before.project_id).name == before.name


def test_identity_changes_reject_active_jobs_and_read_only_project(project, tmp_path):
    backend, before, _ = project
    lease = backend.projects.acquire_lease(before.project_id)
    try:
        with pytest.raises(ProjectBusyError):
            backend.projects.rename_project(before.project_id, "不允许")
        with pytest.raises(ProjectBusyError):
            backend.projects.rename_line(before.project_id, "L01", "不允许")
        with pytest.raises(ProjectBusyError):
            backend.projects.relocate_project(before.project_id, tmp_path / "new")
    finally:
        lease.release()
    backend.projects.close_project(before.project_id)
    backend.projects.open_project(before.root_path, read_only=True)
    with pytest.raises(PermissionError):
        backend.projects.rename_line(before.project_id, "L01", "不允许")
    with pytest.raises(PermissionError):
        backend.projects.rename_project(before.project_id, "不允许")
    with pytest.raises(PermissionError):
        backend.projects.relocate_project(before.project_id, tmp_path / "new")


def test_relocate_keeps_old_project_and_opens_verified_new_root(project, tmp_path):
    backend, before, matrix = project
    root = Path(before.root_path)
    (root / "attachments" / "测试.txt").write_text("test", encoding="utf8")
    old_hash = digest(root / "data/lines/L01.h5")
    target = tmp_path / "新磁盘" / "校园复测"
    target.parent.mkdir()
    got = backend.projects.relocate_project(before.project_id, target)
    assert got.project_id == before.project_id and got.name == before.name
    assert Path(got.root_path) == target.resolve()
    assert Path(backend.projects.get_summary(before.project_id).root_path) == target.resolve()
    assert digest(root / "data/lines/L01.h5") == digest(target / "data/lines/L01.h5") == old_hash
    assert (target / "attachments" / "测试.txt").read_text(encoding="utf8") == "test"
    assert backend.projects.audit_project(before.project_id).healthy
    np.testing.assert_array_equal(backend.projects.read_dataset(before.project_id, "L01").data, matrix)
    assert not (root / ".mygpr.lock").exists() and (target / ".mygpr.lock").exists()
    backend.projects.close_project(before.project_id)
    assert backend.projects.open_project(target).project_id == before.project_id


def test_relocation_cancel_restores_original_and_removes_partial_target(project, tmp_path):
    backend, before, matrix = project
    cts = CancellationTokenSource()

    def progress(done, total, message):
        if "复制" in message:
            cts.cancel()

    context = ExecutionContext(cancellation_token=cts.token, progress_callback=progress)
    target = tmp_path / "cancelled"
    with pytest.raises(JobCancelledError):
        backend.projects.relocate_project(before.project_id, target, context=context)
    assert not target.exists()
    assert backend.projects.get_summary(before.project_id).root_path == before.root_path
    np.testing.assert_array_equal(backend.projects.read_dataset(before.project_id, "L01").data, matrix)
    assert not list(tmp_path.glob(".cancelled.mygpr-move-*"))


def test_relocation_does_not_overwrite_or_move_into_itself(project, tmp_path):
    backend, before, _ = project
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("safe")
    for target in [existing, Path(before.root_path), Path(before.root_path) / "child"]:
        with pytest.raises((ValueError, FileExistsError)):
            backend.projects.relocate_project(before.project_id, target)
    assert (existing / "keep").read_text() == "safe"
    assert backend.projects.get_summary(before.project_id).root_path == before.root_path


def test_relocation_blocks_new_leases_and_close_while_copying(project, tmp_path):
    backend, before, _ = project
    checked = []

    def progress(*args):
        if checked:
            return
        with pytest.raises(ProjectBusyError):
            backend.projects.acquire_lease(before.project_id)
        with pytest.raises(ProjectBusyError):
            backend.projects.close_project(before.project_id)
        checked.append(True)

    backend.projects.relocate_project(
        before.project_id, tmp_path / "target", context=ExecutionContext(progress_callback=progress)
    )
    assert checked


def test_relocation_rejects_external_raw_dependency(project, tmp_path):
    backend, before, _ = project
    source = tmp_path / "external.npy"
    np.save(source, np.ones((18, 20), dtype="float32"))
    backend.projects.import_line_source(before.project_id, "L02", source, copy_into_project=False)
    with pytest.raises(ValueError, match="外部"):
        backend.projects.relocate_project(before.project_id, tmp_path / "target")
    assert not (tmp_path / "target").exists()
    assert backend.projects.get_summary(before.project_id).root_path == before.root_path


def test_relocation_rejects_dangling_target_link(project, tmp_path):
    backend, before, _ = project
    target = tmp_path / "target-link"
    other = tmp_path / "other-project"
    try:
        target.symlink_to(other, target_is_directory=True)
    except OSError as exc:
        pytest.skip("Host cannot create test symlinks: " + str(exc))
    with pytest.raises(ValueError, match="链接|联接"):
        backend.projects.relocate_project(before.project_id, target)
    assert not other.exists()


def test_relocation_aborts_on_attachment_enumeration_failure(project, tmp_path, monkeypatch):
    from mygpr.infrastructure.persistence import project_relocation as module

    backend, before, _ = project
    private = Path(before.root_path) / "attachments" / "private"
    private.mkdir()
    (private / "keep").write_text("payload")
    original = module.os.walk

    def walk(root, **kw):
        for base, dirs, files in original(root, followlinks=False):
            if Path(base) == private:
                if kw.get("onerror"):
                    kw["onerror"](PermissionError("injected enumeration failure"))
                continue
            yield base, dirs, files

    monkeypatch.setattr(module.os, "walk", walk)
    with pytest.raises(PermissionError, match="enumeration"):
        backend.projects.relocate_project(before.project_id, tmp_path / "target")
    assert (private / "keep").read_text() == "payload" and not (tmp_path / "target").exists()


def test_relocation_rejects_uncheckpointed_catalog_wal(project, tmp_path, monkeypatch):
    backend, before, _ = project
    store = backend.projects._session(before.project_id)._store
    wal = store.root / "catalog.sqlite-wal"
    monkeypatch.setattr(store.storage.catalog, "checkpoint", lambda: wal.write_bytes(b"injected pending WAL"))
    try:
        with pytest.raises((OSError, RuntimeError), match="WAL|日志|占用"):
            backend.projects.relocate_project(before.project_id, tmp_path / "target")
        assert not (tmp_path / "target").exists()
    finally:
        wal.unlink(missing_ok=True)


def test_relocation_copy_failure_preserves_original_and_removes_staging(project, tmp_path, monkeypatch):
    from mygpr.infrastructure.persistence import project_relocation as module

    backend, before, matrix = project

    def fail(*args):
        raise OSError("injected copy error")

    monkeypatch.setattr(module, "_copy_file", fail)
    with pytest.raises(OSError, match="copy error"):
        backend.projects.relocate_project(before.project_id, tmp_path / "target")
    assert not (tmp_path / "target").exists() and not list(tmp_path.glob(".target.mygpr-move-*"))
    np.testing.assert_array_equal(backend.projects.read_dataset(before.project_id, "L01").data, matrix)


@pytest.mark.parametrize("addition", ["attachments/late.txt", "catalog.sqlite-wal"])
def test_relocation_detects_files_or_wal_added_during_copy(project, tmp_path, monkeypatch, addition):
    from mygpr.infrastructure.persistence import project_relocation as module

    backend, before, _ = project
    source = Path(before.root_path)
    extra = source / addition
    original = module._copy_file

    def copy(*args):
        original(*args)
        extra.write_bytes(b"concurrent external write")

    monkeypatch.setattr(module, "_copy_file", copy)
    try:
        with pytest.raises(OSError, match="变化|WAL"):
            backend.projects.relocate_project(before.project_id, tmp_path / "target")
        assert not (tmp_path / "target").exists()
        assert backend.projects.get_summary(before.project_id).root_path == before.root_path
    finally:
        extra.unlink(missing_ok=True)


def test_target_link_check_does_not_depend_on_destination_existing(tmp_path, monkeypatch):
    from mygpr.infrastructure.persistence import project_relocation as module

    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "dangling"
    actual = module.os.path.lexists
    monkeypatch.setattr(module.os.path, "lexists", lambda p: Path(p) == target or actual(p))
    check = module._is_link
    monkeypatch.setattr(module, "_is_link", lambda p: Path(p) == target or check(p))
    with pytest.raises(ValueError, match="链接|联接"):
        module.relocation_target(source, target)


def test_relocation_retains_processing_results_and_parameters(project, tmp_path):
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.application.jobs.models import JobStatus

    backend, before, _ = project
    job = backend.submit_project_pipeline(
        before.project_id,
        "L01",
        PipelineDefinition(
            name="Processing pipeline",
            steps=(PipelineStep("dewow", {"window": 5}), PipelineStep("agcGain", {"window": 5})),
        ),
    )
    done = backend.jobs.wait(job, timeout=30)
    assert done.status is JobStatus.COMPLETED, done.error_message
    artifacts = backend.projects.list_artifacts(before.project_id, "L01")
    assert len(artifacts) == 2
    expected = {
        a.artifact_id: backend.projects.read_artifact_dataset(before.project_id, "L01", a.artifact_id).data.copy()
        for a in artifacts
    }
    backend.projects.rename_line(before.project_id, "L01", "正式测线")
    moved = backend.projects.relocate_project(before.project_id, tmp_path / "正式项目")
    assert moved.project_id == before.project_id
    for a in backend.projects.list_artifacts(before.project_id, "L01"):
        assert a.manifest == next(old.manifest for old in artifacts if old.artifact_id == a.artifact_id)
        np.testing.assert_array_equal(
            backend.projects.read_artifact_dataset(before.project_id, "L01", a.artifact_id).data,
            expected[a.artifact_id],
        )


def test_relocation_rejects_report_external_output(project, tmp_path):
    backend, before, _ = project
    report = Path(before.root_path) / "reports/versions/report_v001"
    report.mkdir(parents=True)
    outside = tmp_path / "outside.html"
    outside.write_text("outside")
    (report / "report_manifest.json").write_text(
        json.dumps(
            dict(
                report_id="report_v001", result=dict(package_dir="reports/versions/report_v001", html_path=str(outside))
            )
        ),
        encoding="utf8",
    )
    with pytest.raises(ValueError, match="外部|相对"):
        backend.projects.relocate_project(before.project_id, tmp_path / "target")
    assert not (tmp_path / "target").exists() and outside.read_text() == "outside"

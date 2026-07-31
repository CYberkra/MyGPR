#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable, checksummed MyGPR project backup and restore operations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.field_project_errors import FieldProjectOperationError
from core.field_project_store import FieldProjectStore
from core.project_root_guard import ensure_project_root_marker
from core.security_paths import UnsafeManagedPathError, resolve_managed_path, safe_relative_path
from core.storage_primitives import atomic_output_path, utc_now

PROJECT_MANIFEST_NAME = FieldProjectStore.MANIFEST_NAME


@dataclass(frozen=True)
class ProjectBackupResult:
    """Result of creating and verifying a portable project backup archive."""

    archive_path: str
    file_count: int
    size_mb: float
    manifest_sha256: str = ""
    verified: bool = False
    external_device: bool = False
    recovery_tested: bool = False


@dataclass(frozen=True)
class ProjectRestoreResult:
    project_path: str
    file_count: int
    verified: bool
    source_archive: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise FieldProjectOperationError(f"无法创建唯一目录：{path}")


def backup_project_archive(
    store: FieldProjectStore,
    destination_dir: str | Path | None = None,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    require_external_device: bool = False,
) -> ProjectBackupResult:
    """Create a checksummed backup and perform a restore-style verification."""
    backup_dir = (
        Path(destination_dir).expanduser().resolve()
        if destination_dir is not None
        else Path(os.environ.get("MYGPR_BACKUP_ROOT", str(Path.home() / "MyGPR_Backups"))) / store.root.name
    )
    try:
        backup_dir.relative_to(store.root.resolve())
    except ValueError:
        pass
    else:
        raise FieldProjectOperationError("备份目录必须位于项目目录之外。")
    backup_dir.mkdir(parents=True, exist_ok=True)
    # Flush SQLite WAL before enumerating files.  HDF5 handles are short-lived,
    # so this creates a transactionally complete snapshot without copying
    # volatile -wal/-shm files into the archive.
    if getattr(getattr(store, "storage", None), "is_hybrid", False):
        store.storage.catalog.checkpoint(truncate=True)
    try:
        external_device = os.stat(backup_dir).st_dev != os.stat(store.root).st_dev
    except OSError:
        external_device = False
    if require_external_device and not external_device:
        raise FieldProjectOperationError("正式灾难恢复备份必须选择不同物理设备。")

    stamp = utc_now().replace(":", "").replace("+", "_")
    archive = backup_dir / f"{store.root.name}_backup_{stamp}.zip"
    excluded_parts = {".git", ".venv", "__pycache__", "backups", ".transactions", ".trash"}
    candidates: list[tuple[Path, Path]] = []
    for path in store.root.rglob("*"):
        if path.is_symlink():
            raise FieldProjectOperationError(f"项目包含符号链接，拒绝备份：{path.relative_to(store.root)}")
        if not path.is_file():
            continue
        rel = path.relative_to(store.root)
        if any(part in excluded_parts for part in rel.parts) or path.name == ".mygpr.lock":
            continue
        if path.name in {"catalog.sqlite-wal", "catalog.sqlite-shm"}:
            continue
        candidates.append((path, rel))

    file_rows: list[dict[str, Any]] = []
    backup_manifest: dict[str, Any]
    with atomic_output_path(archive, suffix=".backup.tmp") as temporary:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive_file:
            for index, (path, rel) in enumerate(candidates, start=1):
                if cancel_requested is not None and cancel_requested():
                    from core.job_manager import JobCancelled

                    raise JobCancelled("项目备份已取消")
                digest = _sha256_path(path)
                archive_file.write(path, rel.as_posix())
                file_rows.append({"path": rel.as_posix(), "size_bytes": path.stat().st_size, "sha256": digest})
                if progress_callback is not None:
                    progress_callback(index, max(len(candidates), 1), f"备份 {rel.as_posix()}")
            backup_manifest = {
                "schema": "mygpr.project_backup.v1",
                "project_id": store.manifest.project_id,
                "project_name": store.manifest.name,
                "project_schema": store.manifest.schema,
                "created_at": utc_now(),
                "source_device": str(getattr(os.stat(store.root), "st_dev", "")),
                "backup_device": str(getattr(os.stat(backup_dir), "st_dev", "")),
                "external_device": external_device,
                "files": file_rows,
            }
            archive_file.writestr(
                "backup_manifest.json",
                json.dumps(backup_manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        with zipfile.ZipFile(temporary, "r") as archive_file:
            bad = archive_file.testzip()
            if bad is not None:
                raise FieldProjectOperationError(f"备份 ZIP 校验失败：{bad}")
            manifest_payload = json.loads(archive_file.read("backup_manifest.json").decode("utf-8"))
            for row in manifest_payload.get("files", []):
                digest = hashlib.sha256()
                with archive_file.open(row["path"], "r") as member:
                    for block in iter(lambda: member.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != row["sha256"]:
                    raise FieldProjectOperationError(f"备份内容哈希不一致：{row['path']}")

    manifest_sha = hashlib.sha256(
        json.dumps(backup_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    size_mb = round(archive.stat().st_size / (1024 * 1024), 3)
    store.append_log(f"创建并验证项目备份: {archive}, files={len(file_rows)}, size={size_mb:.3f}MB")
    return ProjectBackupResult(str(archive), len(file_rows), size_mb, manifest_sha, True, external_device, True)


def _zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    unix_mode = (member.external_attr >> 16) & 0xFFFF
    return (unix_mode & 0o170000) == 0o120000


def _validated_backup_members(source: zipfile.ZipFile) -> tuple[dict[str, Any], dict[str, zipfile.ZipInfo]]:
    infos = source.infolist()
    if len(infos) > 200_000:
        raise FieldProjectOperationError("备份文件成员数量异常，拒绝恢复。")
    members: dict[str, zipfile.ZipInfo] = {}
    folded: set[str] = set()
    for info in infos:
        if info.flag_bits & 0x1:
            raise FieldProjectOperationError(f"不支持加密备份成员：{info.filename}")
        if _zip_member_is_symlink(info):
            raise FieldProjectOperationError(f"备份包含符号链接：{info.filename}")
        if info.is_dir():
            continue
        try:
            name = safe_relative_path(info.filename).as_posix()
        except UnsafeManagedPathError as exc:
            raise FieldProjectOperationError(f"备份包含不安全路径：{info.filename}") from exc
        if name in members or name.casefold() in folded:
            raise FieldProjectOperationError(f"备份包含重复或大小写冲突成员：{name}")
        members[name] = info
        folded.add(name.casefold())
    manifest_info = members.get("backup_manifest.json")
    if manifest_info is None or manifest_info.file_size > 16 * 1024 * 1024:
        raise FieldProjectOperationError("备份缺少有效 backup_manifest.json。")
    try:
        manifest = json.loads(source.read(manifest_info).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, OSError) as exc:
        raise FieldProjectOperationError("备份清单不可读取。") from exc
    if manifest.get("schema") != "mygpr.project_backup.v1" or not isinstance(manifest.get("files"), list):
        raise FieldProjectOperationError("备份清单格式无效。")
    declared: dict[str, dict[str, Any]] = {}
    declared_folded: set[str] = set()
    for raw_row in manifest["files"]:
        if not isinstance(raw_row, dict):
            raise FieldProjectOperationError("备份清单包含无效文件记录。")
        try:
            rel = safe_relative_path(str(raw_row.get("path") or "")).as_posix()
        except UnsafeManagedPathError as exc:
            raise FieldProjectOperationError("备份清单包含不安全文件路径。") from exc
        if rel == "backup_manifest.json" or rel in declared or rel.casefold() in declared_folded:
            raise FieldProjectOperationError(f"备份清单包含重复文件：{rel}")
        size = raw_row.get("size_bytes")
        digest = str(raw_row.get("sha256") or "")
        if not isinstance(size, int) or size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise FieldProjectOperationError(f"备份清单文件属性无效：{rel}")
        info = members.get(rel)
        if info is None or info.file_size != size:
            raise FieldProjectOperationError(f"备份成员与清单不一致：{rel}")
        declared[rel] = raw_row
        declared_folded.add(rel.casefold())
    actual = set(members) - {"backup_manifest.json"}
    if actual != set(declared):
        missing = sorted(set(declared) - actual)
        extra = sorted(actual - set(declared))
        raise FieldProjectOperationError(f"备份内容与清单不一致：missing={missing[:3]}, extra={extra[:3]}")
    return manifest, {name: members[name] for name in declared}


def _extract_verified_member(
    source: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    digest = hashlib.sha256()
    completed = False
    try:
        with source.open(member, "r") as reader, destination.open("xb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                written += len(block)
                if written > expected_size:
                    raise FieldProjectOperationError(f"备份成员展开大小超过清单：{member.filename}")
                writer.write(block)
                digest.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        if written != expected_size or digest.hexdigest() != expected_sha256:
            raise FieldProjectOperationError(f"备份成员校验失败：{member.filename}")
        completed = True
    finally:
        if not completed:
            destination.unlink(missing_ok=True)


def restore_project_archive(
    archive_path: str | Path,
    destination_root: str | Path,
    *,
    project_dir_name: str | None = None,
    read_only_verify: bool = True,
) -> ProjectRestoreResult:
    """Restore a backup as a new project after strict manifest verification."""
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file() or archive.is_symlink():
        raise FieldProjectOperationError(f"备份文件不存在或不安全：{archive}")
    destination_parent = Path(destination_root).expanduser().resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as source:
        manifest, members = _validated_backup_members(source)
        required_bytes = sum(int(row["size_bytes"]) for row in manifest["files"])
        free_bytes = shutil.disk_usage(destination_parent).free
        reserve = max(64 * 1024 * 1024, required_bytes // 20)
        if free_bytes < required_bytes + reserve:
            raise FieldProjectOperationError("恢复目标磁盘剩余空间不足。")
        requested_name = project_dir_name or f"{archive.stem}_restored"
        safe_name = safe_relative_path(requested_name)
        if len(safe_name.parts) != 1:
            raise FieldProjectOperationError("恢复项目目录名必须是单个安全名称。")
        target = _unique_destination(destination_parent / safe_name.name)
        staging = destination_parent / f".{target.name}.{uuid.uuid4().hex}.restore"
        staging.mkdir(parents=True, exist_ok=False)
        staging_moved = False
        try:
            rows = {str(row["path"]): row for row in manifest["files"]}
            for rel, info in members.items():
                destination = resolve_managed_path(staging, rel)
                row = rows[rel]
                _extract_verified_member(
                    source,
                    info,
                    destination,
                    expected_size=int(row["size_bytes"]),
                    expected_sha256=str(row["sha256"]),
                )
            project_manifest_path = staging / PROJECT_MANIFEST_NAME
            if not project_manifest_path.is_file():
                raise FieldProjectOperationError("备份不包含 project.json。")
            project_payload = json.loads(project_manifest_path.read_text(encoding="utf-8"))
            project_id = str(project_payload.get("project_id") or "")
            if not project_id or project_id != str(manifest.get("project_id") or ""):
                raise FieldProjectOperationError("备份工程身份与 project.json 不一致。")
            ensure_project_root_marker(staging, project_id)
            staging.replace(target)
            staging_moved = True
        finally:
            if not staging_moved:
                shutil.rmtree(staging, ignore_errors=True)
    if read_only_verify:
        restored = FieldProjectStore.open(target, access_mode="read_only")
        restored.close()
    return ProjectRestoreResult(str(target), len(manifest.get("files", [])), True, str(archive))


__all__ = [
    "ProjectBackupResult",
    "ProjectRestoreResult",
    "backup_project_archive",
    "restore_project_archive",
]

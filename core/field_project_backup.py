#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable, checksummed MyGPR project backup and restore operations."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.field_project_errors import FieldProjectOperationError
from core.field_project_store import FieldProjectStore
from core.project_root_guard import ensure_project_root_marker
from core.security_paths import UnsafeManagedPathError, resolve_managed_path, safe_relative_path
from core.storage_primitives import atomic_output_path, utc_now

_LOGGER = logging.getLogger(__name__)

PROJECT_MANIFEST_NAME = FieldProjectStore.MANIFEST_NAME
BACKUP_SCHEMA = "mygpr.project_backup.v2"
SUPPORTED_BACKUP_SCHEMAS = ("mygpr.project_backup.v1", BACKUP_SCHEMA)

# 增量备份：基准 manifest 内的文件名约定与保留策略默认值。
INCREMENTAL_BASE_MANIFEST = "backup_manifest.json"
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


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        from core.job_manager import JobCancelled

        raise JobCancelled("项目备份已取消")


def _sha256_path(path: Path, *, cancel_requested: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            _raise_if_cancelled(cancel_requested)
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
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


def _latest_backup_manifest(backup_dir: Path, root_name: str, project_id: str) -> tuple[dict[str, Any], str] | None:
    """Return (newest same-project backup manifest, archive file name).

    档案名前缀为 ``{root_name}_backup_``（与 backup_project_archive 一致），
    并以 manifest 内 project_id 复核，防止误用其它项目的档案。
    """
    best: tuple[str, dict[str, Any], str] | None = None
    try:
        archives = sorted(backup_dir.glob(f"{root_name}_backup_*.zip"))
    except OSError:
        return None
    for archive in archives:
        try:
            with zipfile.ZipFile(archive, "r") as handle:
                payload = json.loads(handle.read(INCREMENTAL_BASE_MANIFEST).decode("utf-8"))
        except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError):
            continue
        if str(payload.get("project_id") or "") != project_id:
            continue
        created = str(payload.get("created_at") or "")
        if best is None or created > best[0]:
            best = (created, payload, archive.name)
    return None if best is None else (best[1], best[2])


def _prune_old_backups(backup_dir: Path, root_name: str, project_id: str, keep: int, *, current: Path) -> list[str]:
    """Retain ``keep`` restore points and every archive they depend on."""
    if keep <= 0:
        return []
    prunable: list[Path] = []
    for path in backup_dir.glob(f"{root_name}_backup_*.zip"):
        if path == current:
            continue
        try:
            with zipfile.ZipFile(path, "r") as handle:
                payload = json.loads(handle.read(INCREMENTAL_BASE_MANIFEST).decode("utf-8"))
        except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError):
            continue
        if str(payload.get("project_id") or "") == project_id:
            prunable.append(path)
    prunable.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    protected = {current}
    try:
        for retained in [current, *prunable[:max(keep - 1, 0)]]:
            with _open_backup_chain(retained) as layers:
                protected.update(path for path, *_ in layers)
    except (OSError, ValueError, zipfile.BadZipFile, FieldProjectOperationError):
        # Incomplete/invalid dependency metadata must never authorize deletion.
        _LOGGER.warning("备份依赖链无法核验，已跳过保留策略清理", exc_info=True)
        return []
    removed: list[str] = []
    for stale in prunable[max(keep - 1, 0):]:
        if stale in protected:
            continue
        try:
            stale.unlink()
            removed.append(stale.name)
        except OSError:
            continue
    return removed


def backup_project_archive(
    store: FieldProjectStore,
    destination_dir: str | Path | None = None,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    require_external_device: bool = False,
    incremental: bool = False,
    retention_keep: int | None = None,
) -> ProjectBackupResult:
    """Create a checksummed backup and perform a restore-style verification.

    ``incremental=True`` 只打包自同目录最近一次同项目备份以来变化/新增的文件
    （对比路径 + sha256）；manifest 记录基准档案与被省略的未变文件数。
    ``retention_keep`` 保留最近 N 个恢复点及其全部依赖档案。
    """
    incremental_base: dict[str, Any] | None = None
    _raise_if_cancelled(cancel_requested)
    backup_dir = (
        Path(destination_dir).expanduser().resolve()
        if destination_dir is not None
        else Path(os.environ.get("MYGPR_BACKUP_ROOT", str(Path.home() / "MyGPR_Backups"))) / store.root.name
    ).expanduser().resolve()
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

    if incremental:
        found = _latest_backup_manifest(backup_dir, store.root.name, str(store.manifest.project_id))
        if found is None:
            incremental = False  # 无基准可依，退化为全量
            incremental_base, incremental_base_name = None, ""
        else:
            incremental_base, incremental_base_name = found
    else:
        incremental_base, incremental_base_name = None, ""
    base_files: dict[str, dict[str, Any]] = {}
    if incremental and incremental_base is not None:
        with _open_backup_chain(backup_dir / incremental_base_name, cancel_requested=cancel_requested) as layers:
            base_files = {rel: item[0] for rel, item in _combined_backup_files(layers).items()}

    stamp = utc_now().replace(":", "").replace("+", "_")
    archive = backup_dir / f"{store.root.name}_backup_{stamp}.zip"
    excluded_parts = {".git", ".venv", "__pycache__", "backups", ".transactions", ".trash"}

    def _enumerate_candidates() -> tuple[list[tuple[Path, Path]], int, list[str]]:
        # Windows 下全量并发测试/备份期间偶发源文件枚举缺漏（句柄延迟释放、
        # 索引服务短暂锁定），这里每次重试都重新枚举，而不是复用陈旧清单。
        found: list[tuple[Path, Path]] = []
        unchanged = 0
        current_paths: set[str] = set()
        for path in store.root.rglob("*"):
            _raise_if_cancelled(cancel_requested)
            if path.is_symlink():
                raise FieldProjectOperationError(f"项目包含符号链接，拒绝备份：{path.relative_to(store.root)}")
            if not path.is_file():
                continue
            rel = path.relative_to(store.root)
            if any(part in excluded_parts for part in rel.parts) or path.name == ".mygpr.lock":
                continue
            if path.name in {"catalog.sqlite-wal", "catalog.sqlite-shm"}:
                continue
            current_paths.add(rel.as_posix())
            if incremental and base_files:
                row = base_files.get(rel.as_posix())
                if (
                    row is not None
                    and int(row.get("size_bytes", -1)) == path.stat().st_size
                ):
                    # 大小相同再比对内容哈希，避免误跳过同尺寸变更文件。
                    if _sha256_path(path, cancel_requested=cancel_requested) == str(row.get("sha256", "")):
                        unchanged += 1
                        continue
            found.append((path, rel))
        return found, unchanged, sorted(set(base_files) - current_paths)

    def _write_backup_archive(candidates: list[tuple[Path, Path]], unchanged_count: int, deleted_files: list[str]) -> dict[str, Any]:
        file_rows: list[dict[str, Any]] = []
        with atomic_output_path(archive, suffix=".backup.tmp") as temporary:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive_file:
                for index, (path, rel) in enumerate(candidates, start=1):
                    _raise_if_cancelled(cancel_requested)
                    info = zipfile.ZipInfo.from_file(path, rel.as_posix())
                    info.compress_type = zipfile.ZIP_DEFLATED
                    digest = hashlib.sha256()
                    written = 0
                    with path.open("rb") as reader, archive_file.open(info, "w", force_zip64=True) as writer:
                        while True:
                            _raise_if_cancelled(cancel_requested)
                            block = reader.read(8 * 1024 * 1024)
                            if not block:
                                break
                            writer.write(block)
                            digest.update(block)
                            written += len(block)
                    file_rows.append({"path": rel.as_posix(), "size_bytes": written, "sha256": digest.hexdigest()})
                    if progress_callback is not None:
                        progress_callback(index, max(len(candidates), 1), f"备份 {rel.as_posix()}")
                    _raise_if_cancelled(cancel_requested)
                _raise_if_cancelled(cancel_requested)
                backup_manifest = {
                    "schema": BACKUP_SCHEMA,
                    "project_id": store.manifest.project_id,
                    "project_name": store.manifest.name,
                    "project_schema": store.manifest.schema,
                    "created_at": utc_now(),
                    "source_device": str(getattr(os.stat(store.root), "st_dev", "")),
                    "backup_device": str(getattr(os.stat(backup_dir), "st_dev", "")),
                    "external_device": external_device,
                    "incremental": bool(incremental),
                    "incremental_base_archive": incremental_base_name if incremental else "",
                    "unchanged_files_omitted": unchanged_count if incremental else 0,
                    # v2 deletion semantics require a reader that understands
                    # replay order; legacy v1 readers must reject this archive.
                    "deleted_files": deleted_files if incremental else [],
                    "files": file_rows,
                }
                archive_file.writestr(
                    "backup_manifest.json",
                    json.dumps(backup_manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                )
            # 写后校验：ZIP 自检 + 逐成员哈希，再与源目录现状交叉核对。
            # 枚举后源目录仍可能新增文件（并发写场景），那属于内容演进而非
            # 漏文件，只提示缺失方向的不一致。
            _raise_if_cancelled(cancel_requested)
            with zipfile.ZipFile(temporary, "r") as archive_file:
                # Reading each member through EOF verifies ZIP CRC as well as
                # the declared SHA256, with a cancellation point per block.
                manifest_payload = json.loads(archive_file.read("backup_manifest.json").decode("utf-8"))
                for row in manifest_payload.get("files", []):
                    _raise_if_cancelled(cancel_requested)
                    digest = hashlib.sha256()
                    with archive_file.open(row["path"], "r") as member:
                        while True:
                            _raise_if_cancelled(cancel_requested)
                            block = member.read(8 * 1024 * 1024)
                            if not block:
                                break
                            digest.update(block)
                    if digest.hexdigest() != row["sha256"]:
                        raise FieldProjectOperationError(f"备份内容哈希不一致：{row['path']}")
                archived = {row["path"] for row in manifest_payload.get("files", [])}
                current_sources = {rel.as_posix() for _, rel in _enumerate_candidates()[0]}
                missing_in_archive = sorted(current_sources - archived)
                if missing_in_archive:
                    raise FieldProjectOperationError(
                        f"备份内容与源目录不一致：missing={missing_in_archive[:3]}"
                    )
            # Verify the complete restore point before atomic publication and
            # retention pruning. ZIP member hashes alone do not test recovery.
            _raise_if_cancelled(cancel_requested)
            with tempfile.TemporaryDirectory(prefix=".mygpr-backup-verify-", dir=backup_dir) as verification_root:
                _restore_project_archive(temporary, verification_root, cancel_requested=cancel_requested)
            _raise_if_cancelled(cancel_requested)
            return backup_manifest

    backup_manifest: dict[str, Any] | None = None
    last_incomplete: Exception | None = None
    for _attempt in range(2):
        try:
            backup_manifest = _write_backup_archive(*_enumerate_candidates())
            break
        except (OSError, FieldProjectOperationError) as exc:
            # 只自动重试"源目录与档案不一致"这一类瞬时缺漏；哈希不符等
            # 内容损坏不重试（重试同样损坏，徒增时长）。
            if "备份内容与源目录不一致" not in str(exc):
                raise
            last_incomplete = exc
            _LOGGER.warning("备份与源目录不一致，第 %d 次重试：%s", _attempt + 1, exc)
    if backup_manifest is None:
        raise FieldProjectOperationError(
            f"备份两次尝试均与源目录不一致（疑似并发写入或句柄延迟释放）：{last_incomplete}"
        )

    manifest_sha = hashlib.sha256(
        json.dumps(backup_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    file_count = len(backup_manifest.get("files", []))
    size_mb = round(archive.stat().st_size / (1024 * 1024), 3)
    if retention_keep is not None:
        _prune_old_backups(backup_dir, store.root.name, str(store.manifest.project_id), int(retention_keep), current=archive)
    store.append_log(
        f"创建并验证项目备份: {archive}, files={file_count}, size={size_mb:.3f}MB, "
        f"incremental={bool(incremental)}"
    )
    return ProjectBackupResult(str(archive), file_count, size_mb, manifest_sha, True, external_device, True)


def _zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    unix_mode = (member.external_attr >> 16) & 0xFFFF
    return (unix_mode & 0o170000) == 0o120000


def _validated_backup_members(source: zipfile.ZipFile, *, cancel_requested: Callable[[], bool] | None = None) -> tuple[dict[str, Any], dict[str, zipfile.ZipInfo]]:
    infos = source.infolist()
    if len(infos) > 200_000:
        raise FieldProjectOperationError("备份文件成员数量异常，拒绝恢复。")
    members: dict[str, zipfile.ZipInfo] = {}
    folded: set[str] = set()
    for info in infos:
        _raise_if_cancelled(cancel_requested)
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
        _raise_if_cancelled(cancel_requested)
        manifest = json.loads(source.read(manifest_info).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, OSError) as exc:
        raise FieldProjectOperationError("备份清单不可读取。") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise FieldProjectOperationError("备份清单格式无效。")
    if manifest.get("schema") not in SUPPORTED_BACKUP_SCHEMAS:
        raise FieldProjectOperationError(f"不支持备份清单版本：{manifest.get('schema')}")
    declared: dict[str, dict[str, Any]] = {}
    declared_folded: set[str] = set()
    for raw_row in manifest["files"]:
        _raise_if_cancelled(cancel_requested)
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
        declared[rel] = {**raw_row, "path": rel}
        declared_folded.add(rel.casefold())
    actual = set(members) - {"backup_manifest.json"}
    if actual != set(declared):
        missing = sorted(set(declared) - actual)
        extra = sorted(actual - set(declared))
        raise FieldProjectOperationError(f"备份内容与清单不一致：missing={missing[:3]}, extra={extra[:3]}")
    deleted_files = manifest.get("deleted_files", [])
    if not isinstance(deleted_files, list):
        raise FieldProjectOperationError("备份删除清单格式无效。")
    deleted: set[str] = set()
    normalized_deleted: list[str] = []
    for raw_path in deleted_files:
        _raise_if_cancelled(cancel_requested)
        try:
            rel = safe_relative_path(raw_path).as_posix() if isinstance(raw_path, str) else ""
        except UnsafeManagedPathError as exc:
            raise FieldProjectOperationError("备份删除清单包含不安全路径。") from exc
        # Deleting an old spelling before adding the renamed member is valid
        # on Windows. Same-spelling delete/add and duplicate deletes are not.
        if not rel or rel.casefold() in deleted or rel in declared or rel == "backup_manifest.json":
            raise FieldProjectOperationError("备份删除清单包含无效或冲突路径。")
        deleted.add(rel.casefold())
        normalized_deleted.append(rel)
    manifest["files"] = list(declared.values())
    manifest["deleted_files"] = normalized_deleted
    return manifest, {name: members[name] for name in declared}


@contextmanager
def _open_backup_chain(archive: Path, *, cancel_requested: Callable[[], bool] | None = None):
    """Open/validate every sibling ancestor, oldest first, closing on all exits."""
    layers = []
    seen: set[Path] = set()
    expected_project_id = None
    with ExitStack() as stack:
        current = archive
        while True:
            _raise_if_cancelled(cancel_requested)
            if current.is_symlink() or not current.is_file():
                raise FieldProjectOperationError(f"增量备份缺少基准档案或档案不安全：{current.name}")
            current = current.resolve()
            if current in seen:
                raise FieldProjectOperationError("增量备份依赖链存在循环。")
            seen.add(current)
            handle = stack.enter_context(zipfile.ZipFile(current, "r"))
            manifest, members = _validated_backup_members(handle, cancel_requested=cancel_requested)
            project_id = str(manifest.get("project_id") or "")
            if not project_id or (expected_project_id is not None and project_id != expected_project_id):
                raise FieldProjectOperationError("备份依赖链的项目身份不一致。")
            expected_project_id = project_id
            layers.append((current, handle, manifest, members))
            if not bool(manifest.get("incremental")):
                break
            base_name = str(manifest.get("incremental_base_archive") or "")
            try:
                relative = safe_relative_path(base_name)
            except UnsafeManagedPathError as exc:
                raise FieldProjectOperationError("增量备份基准档案路径不安全。") from exc
            if len(relative.parts) != 1:
                raise FieldProjectOperationError("增量备份基准档案必须是同目录安全文件名。")
            current = current.parent / relative.name
        yield list(reversed(layers))


def _combined_backup_files(layers):
    combined = {}
    for _path, handle, manifest, members in layers:
        for rel in manifest.get("deleted_files", []):
            combined.pop(rel, None)
        for row in manifest["files"]:
            rel = str(row["path"])
            combined[rel] = (row, members[rel], handle)
    folded: set[str] = set()
    for rel in combined:
        if rel.casefold() in folded:
            raise FieldProjectOperationError("备份依赖链存在大小写冲突文件。")
        folded.add(rel.casefold())
    return combined


def _extract_verified_member(
    source: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    _raise_if_cancelled(cancel_requested)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    digest = hashlib.sha256()
    completed = False
    try:
        with source.open(member, "r") as reader, destination.open("xb") as writer:
            while True:
                _raise_if_cancelled(cancel_requested)
                block = reader.read(8 * 1024 * 1024)
                if not block:
                    break
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
    """Restore a backup as a new project after strict manifest verification.

    增量备份（manifest ``incremental=true``）自动沿 ``incremental_base_archive``
    链在同目录回溯合并基准内容；缺失任何一层基准即报错。
    """
    return _restore_project_archive(
        archive_path, destination_root,
        project_dir_name=project_dir_name, read_only_verify=read_only_verify,
    )


def _restore_project_archive(
    archive_path: str | Path,
    destination_root: str | Path,
    *,
    project_dir_name: str | None = None,
    read_only_verify: bool = True,
    cancel_requested: Callable[[], bool] | None = None,
) -> ProjectRestoreResult:
    """Shared restore implementation; backup verification may cancel its I/O."""
    _raise_if_cancelled(cancel_requested)
    archive = Path(archive_path).expanduser()
    if not archive.is_file() or archive.is_symlink():
        raise FieldProjectOperationError(f"备份文件不存在或不安全：{archive}")
    archive = archive.resolve()
    destination_parent = Path(destination_root).expanduser().resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)

    with _open_backup_chain(archive, cancel_requested=cancel_requested) as layers:
        manifest = layers[-1][2]
        combined = _combined_backup_files(layers)
        manifest = {**manifest, "files": [item[0] for item in combined.values()]}
        members = {rel: item[1] for rel, item in combined.items()}
        member_handles = {rel: item[2] for rel, item in combined.items()}
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
                _raise_if_cancelled(cancel_requested)
                destination = resolve_managed_path(staging, rel)
                row = rows[rel]
                layer_handle = member_handles.get(rel)
                if layer_handle is None:
                    raise FieldProjectOperationError(f"备份链缺少成员：{rel}")
                _extract_verified_member(
                    layer_handle,
                    info,
                    destination,
                    expected_size=int(row["size_bytes"]),
                    expected_sha256=str(row["sha256"]),
                    cancel_requested=cancel_requested,
                )
            _raise_if_cancelled(cancel_requested)
            project_manifest_path = staging / PROJECT_MANIFEST_NAME
            if not project_manifest_path.is_file():
                raise FieldProjectOperationError("备份不包含 project.json。")
            project_payload = json.loads(project_manifest_path.read_text(encoding="utf-8"))
            project_id = str(project_payload.get("project_id") or "")
            if not project_id or project_id != str(manifest.get("project_id") or ""):
                raise FieldProjectOperationError("备份工程身份与 project.json 不一致。")
            ensure_project_root_marker(staging, project_id)
            if read_only_verify:
                _raise_if_cancelled(cancel_requested)
                restored = FieldProjectStore.open(staging, access_mode="read_only")
                restored.close()
            _raise_if_cancelled(cancel_requested)
            staging.replace(target)
            staging_moved = True
        finally:
            if not staging_moved:
                shutil.rmtree(staging, ignore_errors=True)
    return ProjectRestoreResult(str(target), len(manifest.get("files", [])), True, str(archive))


__all__ = [
    "ProjectBackupResult",
    "ProjectRestoreResult",
    "backup_project_archive",
    "restore_project_archive",
]

"""Bounded, checksummed staging copy; the original writer lock stays held."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import stat
from tempfile import TemporaryDirectory


def _is_link(path):
    attributes = path.lstat()
    return path.is_symlink() or bool(
        getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def relocation_target(source, destination):
    raw = Path(destination).expanduser()
    if not raw.is_absolute():
        raise ValueError("请选择绝对路径作为目标项目目录。")
    # Inspect before resolve: resolving a junction would hide its provenance.
    if any(os.path.lexists(p) and _is_link(p) for p in (raw, *raw.parents)):
        raise ValueError("目标目录不能经过符号链接或目录联接。")
    target, source = raw.resolve(), Path(source).resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("目标目录不能是当前项目、其子目录或上级目录。")
    if target.exists():
        raise FileExistsError("目标目录已存在；请选择一个新目录。")
    if not target.parent.is_dir():
        raise ValueError("目标项目目录的父目录不存在。")
    return target


def _files(root, context):
    result = []
    ignored = {".mygpr.lock", ".mygpr.lock.guard", "catalog.sqlite-wal", "catalog.sqlite-shm"}

    def walk_error(error):
        raise error

    for parent, directories, files in os.walk(root, followlinks=False, onerror=walk_error):
        context.raise_if_cancelled()
        base = Path(parent)
        for name in (*directories, *files):
            if _is_link(base / name):
                raise ValueError(f"项目包含外部链接，请先收纳到项目中：{(base / name).relative_to(root)}")
        if base == root:
            directories[:] = [d for d in directories if d != ".transactions"]
        for name in files:
            relative = (base / name).relative_to(root)
            if len(relative.parts) == 1 and name in ignored:
                continue
            result.append((relative, (base / name).stat()))
    return result


def _copy_file(source, target, context, progress):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as src, target.open("xb") as dst:
        while True:
            context.raise_if_cancelled()
            block = src.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            dst.write(block)
            progress(len(block))
        dst.flush()
        os.fsync(dst.fileno())
    actual = hashlib.sha256()
    with target.open("rb") as stream:
        while True:
            context.raise_if_cancelled()
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            actual.update(block)
    if actual.digest() != digest.digest():
        raise OSError(f"迁移文件校验失败：{target.name}")
    shutil.copystat(source, target, follow_symlinks=False)


@contextmanager
def staged_project_copy(store, destination, context):
    store.assert_writable()
    source = store.root.resolve()
    target = relocation_target(source, destination)
    if getattr(store.storage, "is_hybrid", False):
        store.storage.catalog.checkpoint()
        _check_wal(store)
    files = _files(source, context)
    total = max(1, sum(info.st_size for _, info in files))
    if shutil.disk_usage(target.parent).free < total + 4 * 1024 * 1024:
        raise OSError("目标磁盘空间不足，无法创建完整项目副本。")
    copied = 0

    def progress(count):
        nonlocal copied
        copied += count
        context.report_progress(copied, total, "复制并校验项目文件")

    with TemporaryDirectory(
        prefix=f".{target.name}.mygpr-move-", dir=target.parent, ignore_cleanup_errors=True
    ) as folder:
        temporary = Path(folder).resolve()
        assert temporary.parent == target.parent.resolve() and temporary != source
        stage = temporary / "project"
        stage.mkdir()
        for relative, before in files:
            _copy_file(source / relative, stage / relative, context, progress)
            after = (source / relative).stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise OSError(f"复制期间源文件发生变化：{relative}")
        _check_wal(store)
        current_files = _files(source, context)
        before_index = {str(p): (s.st_size, s.st_mtime_ns) for p, s in files}
        after_index = {str(p): (s.st_size, s.st_mtime_ns) for p, s in current_files}
        if before_index != after_index:
            raise OSError("复制期间源文件列表或内容发生变化，请暂停外部写入后重试。")
        context.raise_if_cancelled()
        yield stage, target


def _check_wal(store):
    if not getattr(store.storage, "is_hybrid", False):
        return
    wal = Path(str(store.storage.catalog.path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise OSError("项目数据库 WAL 日志仍被占用，请关闭外部数据库程序后重试。")

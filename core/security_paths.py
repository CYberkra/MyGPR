#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path-validation helpers for project-managed files.

Project manifests and registries are user-writable JSON files.  Treat every path
read from them as untrusted and resolve it through this module before accessing,
removing, exporting, or hashing a file.
"""
from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_WINDOWS_INVALID_RE = re.compile(r'[<>:"|?*\x00-\x1f]')


class UnsafeManagedPathError(ValueError):
    """Raised when metadata points outside its managed storage root."""


def _normalised_relative_text(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value).strip()
    if not text or "\x00" in text:
        raise UnsafeManagedPathError("受管理路径不能为空或包含 NUL 字符。")
    win = PureWindowsPath(text)
    posix = PurePosixPath(text.replace("\\", "/"))
    if win.is_absolute() or win.drive or text.startswith(("\\\\", "//")) or posix.is_absolute():
        raise UnsafeManagedPathError(f"受管理路径必须是相对路径：{text!r}")
    parts = posix.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafeManagedPathError(f"受管理路径包含非法目录段：{text!r}")
    for part in parts:
        if part.endswith((" ", ".")):
            raise UnsafeManagedPathError(f"路径段不能以空格或句点结尾：{part!r}")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED or _WINDOWS_INVALID_RE.search(part):
            raise UnsafeManagedPathError(f"路径段在 Windows 上非法或保留：{part!r}")
    return PurePosixPath(*parts).as_posix()


def safe_relative_path(value: str | os.PathLike[str]) -> Path:
    """Return a validated cross-platform relative path."""
    return Path(_normalised_relative_text(value))


def resolve_managed_path(
    root: str | Path,
    value: str | os.PathLike[str],
    *,
    allow_root: bool = False,
    require_exists: bool = False,
    require_file: bool = False,
    require_dir: bool = False,
    reject_symlink: bool = True,
) -> Path:
    """Resolve an untrusted relative path and prove it remains under ``root``."""
    root_path = Path(root).expanduser().resolve()
    relative = safe_relative_path(value)
    candidate_lexical = root_path / relative
    if reject_symlink:
        current = root_path
        for part in relative.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise UnsafeManagedPathError(f"受管理路径不能经过符号链接：{relative.as_posix()}")
    candidate = candidate_lexical.resolve(strict=False)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise UnsafeManagedPathError(f"路径越过受管理目录：{value!r}") from exc
    if candidate == root_path and not allow_root:
        raise UnsafeManagedPathError("路径不能指向受管理目录本身。")
    if require_exists and not candidate.exists():
        raise FileNotFoundError(candidate)
    if require_file and not candidate.is_file():
        raise FileNotFoundError(candidate)
    if require_dir and not candidate.is_dir():
        raise NotADirectoryError(candidate)
    return candidate


def ensure_direct_child(root: str | Path, child: str | Path) -> Path:
    """Validate that ``child`` is exactly one directory level below ``root``."""
    root_path = Path(root).resolve()
    candidate = Path(child).resolve(strict=False)
    if candidate.parent != root_path or candidate == root_path or candidate.is_symlink():
        raise UnsafeManagedPathError(f"目录不是预期的直接子目录：{candidate}")
    return candidate


__all__ = [
    "UnsafeManagedPathError",
    "ensure_direct_child",
    "resolve_managed_path",
    "safe_relative_path",
]

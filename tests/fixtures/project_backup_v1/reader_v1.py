"""Unchanged v1 reader validator frozen from commit e379a5ef602459fef166539f96d3904e7cd9f76a.

Used to prove old readers reject new v2 archives without runtime Git access.
"""
import json
import re
from typing import Any
import zipfile
from core.field_project_errors import FieldProjectOperationError
from core.security_paths import UnsafeManagedPathError, safe_relative_path

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

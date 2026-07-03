#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-file provenance registry for MyGPR field projects.

The registry tracks where imported line data came from without making those
external source files part of the project deletion scope.  UI code should show
only the small status label; the full provenance stays in
``metadata/source_files.json`` for audit/reproduction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.field_project_models import atomic_write_json, local_now, validate_line_id

SOURCE_FILES_SCHEMA = "mygpr.source_files.v1"
HASH_SIZE_LIMIT_BYTES = 128 * 1024 * 1024
STATUS_LABELS = {
    "available": "正常",
    "missing": "缺失",
    "modified": "已变更",
    "unchecked": "未检查",
    "untracked": "未记录",
}


@dataclass
class SourceFileRecord:
    line_id: str
    role: str = "gpr"
    source_path: str = ""
    source_filename: str = ""
    source_size_bytes: int = 0
    source_mtime_ns: int = 0
    source_sha256: str = ""
    hash_policy: str = ""
    imported_at: str = field(default_factory=local_now)
    import_mode: str = "copied_to_project"
    project_raw_path: str = ""
    status: str = "unchecked"
    last_checked_at: str = ""
    warning: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceFileRecord":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: payload.get(key, getattr(cls, key, "")) for key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status or "未记录")


def source_registry_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "metadata" / "source_files.json"


def _project_root(store_or_root: Any) -> Path:
    if isinstance(store_or_root, Path):
        return store_or_root.resolve()
    if isinstance(store_or_root, str):
        return Path(store_or_root).resolve()
    return Path(getattr(store_or_root, "root", store_or_root)).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_source(path: str | Path, *, hash_limit_bytes: int = HASH_SIZE_LIMIT_BYTES) -> dict[str, Any]:
    src = Path(path).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(src)
    stat = src.stat()
    sha = ""
    policy = "size_mtime_only"
    if stat.st_size <= hash_limit_bytes:
        sha = _sha256(src)
        policy = "sha256"
    return {
        "source_path": str(src),
        "source_filename": src.name,
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_sha256": sha,
        "hash_policy": policy,
    }


def load_source_registry(store_or_root: Any) -> list[SourceFileRecord]:
    path = source_registry_path(_project_root(store_or_root))
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("sources", []) if isinstance(payload, dict) else []
    return [SourceFileRecord.from_dict(row) for row in rows if isinstance(row, dict)]


def save_source_registry(store_or_root: Any, records: Iterable[SourceFileRecord]) -> Path:
    root = _project_root(store_or_root)
    path = source_registry_path(root)
    payload = {
        "schema": SOURCE_FILES_SCHEMA,
        "updated_at": local_now(),
        "sources": [record.to_dict() for record in records],
    }
    atomic_write_json(path, payload)
    return path


def _dedupe(records: Iterable[SourceFileRecord]) -> list[SourceFileRecord]:
    by_key: dict[tuple[str, str], SourceFileRecord] = {}
    order: list[tuple[str, str]] = []
    for record in records:
        key = (record.line_id, record.role or "gpr")
        if key not in by_key:
            order.append(key)
        by_key[key] = record
    return [by_key[key] for key in order]


def record_source_file(
    store_or_root: Any,
    line_id: str,
    source: str | Path,
    *,
    role: str = "gpr",
    import_mode: str = "copied_to_project",
    project_raw_path: str = "",
) -> SourceFileRecord:
    safe_line_id = validate_line_id(line_id)
    fp = fingerprint_source(source)
    record = SourceFileRecord(
        line_id=safe_line_id,
        role=role,
        import_mode=import_mode,
        project_raw_path=str(project_raw_path or ""),
        status="available",
        last_checked_at=local_now(),
        **fp,
    )
    records = [row for row in load_source_registry(store_or_root) if not (row.line_id == safe_line_id and row.role == role)]
    records.append(record)
    save_source_registry(store_or_root, _dedupe(records))
    return record


def remove_line_source_records(store_or_root: Any, line_id: str) -> int:
    safe_line_id = validate_line_id(line_id)
    records = load_source_registry(store_or_root)
    kept = [record for record in records if record.line_id != safe_line_id]
    removed = len(records) - len(kept)
    if removed:
        save_source_registry(store_or_root, kept)
    return removed


def get_line_source_record(store_or_root: Any, line_id: str, *, role: str = "gpr") -> SourceFileRecord | None:
    safe_line_id = validate_line_id(line_id)
    for record in load_source_registry(store_or_root):
        if record.line_id == safe_line_id and record.role == role:
            return record
    return None


def evaluate_source_record(record: SourceFileRecord) -> SourceFileRecord:
    path = Path(record.source_path).expanduser()
    checked = local_now()
    if not path.exists() or not path.is_file():
        record.status = "missing"
        record.last_checked_at = checked
        record.warning = "源文件不存在，可能已移动、改名或所在磁盘未连接。"
        return record
    try:
        fp = fingerprint_source(path)
    except Exception as exc:
        record.status = "missing"
        record.last_checked_at = checked
        record.warning = f"源文件无法读取：{exc}"
        return record
    changed = False
    reasons: list[str] = []
    if int(fp["source_size_bytes"]) != int(record.source_size_bytes):
        changed = True
        reasons.append("文件大小变化")
    if int(fp["source_mtime_ns"]) != int(record.source_mtime_ns):
        changed = True
        reasons.append("修改时间变化")
    if record.source_sha256 and fp.get("source_sha256") and fp["source_sha256"] != record.source_sha256:
        changed = True
        reasons.append("SHA256 不一致")
    record.status = "modified" if changed else "available"
    record.last_checked_at = checked
    record.warning = "；".join(reasons)
    return record


def check_all_source_files(store_or_root: Any) -> list[SourceFileRecord]:
    records = [evaluate_source_record(record) for record in load_source_registry(store_or_root)]
    if records:
        save_source_registry(store_or_root, records)
    return records


def check_line_source_file(store_or_root: Any, line_id: str, *, role: str = "gpr") -> SourceFileRecord | None:
    records = load_source_registry(store_or_root)
    target: SourceFileRecord | None = None
    for idx, record in enumerate(records):
        if record.line_id == validate_line_id(line_id) and record.role == role:
            records[idx] = evaluate_source_record(record)
            target = records[idx]
            break
    if target is not None:
        save_source_registry(store_or_root, records)
    return target


def source_status_label_for_line(store_or_root: Any, line_id: str) -> str:
    record = get_line_source_record(store_or_root, line_id)
    return record.status_label if record is not None else STATUS_LABELS["untracked"]


def relink_line_source_file(
    store_or_root: Any,
    line_id: str,
    new_source: str | Path,
    *,
    role: str = "gpr",
    allow_mismatch: bool = False,
) -> SourceFileRecord:
    safe_line_id = validate_line_id(line_id)
    records = load_source_registry(store_or_root)
    existing = next((record for record in records if record.line_id == safe_line_id and record.role == role), None)
    fp = fingerprint_source(new_source)
    warning = ""
    if existing is not None:
        mismatches: list[str] = []
        if int(fp["source_size_bytes"]) != int(existing.source_size_bytes):
            mismatches.append("文件大小不一致")
        if existing.source_sha256 and fp.get("source_sha256") and fp["source_sha256"] != existing.source_sha256:
            mismatches.append("SHA256 不一致")
        if mismatches and not allow_mismatch:
            raise ValueError("重新定位的源文件与原记录不匹配：" + "；".join(mismatches))
        warning = "；".join(mismatches)
    updated = SourceFileRecord(
        line_id=safe_line_id,
        role=role,
        imported_at=existing.imported_at if existing else local_now(),
        import_mode=existing.import_mode if existing else "relinked",
        project_raw_path=existing.project_raw_path if existing else "",
        status="available" if not warning else "modified",
        last_checked_at=local_now(),
        warning=warning,
        **fp,
    )
    kept = [record for record in records if not (record.line_id == safe_line_id and record.role == role)]
    kept.append(updated)
    save_source_registry(store_or_root, _dedupe(kept))
    return updated


def export_source_file_manifest_csv(store_or_root: Any, destination: str | Path | None = None) -> Path:
    root = _project_root(store_or_root)
    out = Path(destination) if destination is not None else root / "reports" / "source_file_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = load_source_registry(root)
    headers = [
        "line_id",
        "role",
        "status",
        "source_filename",
        "source_path",
        "source_size_bytes",
        "source_mtime_ns",
        "source_sha256",
        "hash_policy",
        "imported_at",
        "last_checked_at",
        "import_mode",
        "project_raw_path",
        "warning",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            row["status"] = record.status_label
            writer.writerow({key: row.get(key, "") for key in headers})
    return out


def source_summary(records: Iterable[SourceFileRecord]) -> dict[str, int]:
    summary = {"total": 0, "available": 0, "missing": 0, "modified": 0, "unchecked": 0, "untracked": 0}
    for record in records:
        summary["total"] += 1
        summary[record.status if record.status in summary else "unchecked"] += 1
    return summary


__all__ = [
    "HASH_SIZE_LIMIT_BYTES",
    "SOURCE_FILES_SCHEMA",
    "STATUS_LABELS",
    "SourceFileRecord",
    "check_all_source_files",
    "check_line_source_file",
    "evaluate_source_record",
    "export_source_file_manifest_csv",
    "fingerprint_source",
    "get_line_source_record",
    "load_source_registry",
    "record_source_file",
    "relink_line_source_file",
    "remove_line_source_records",
    "save_source_registry",
    "source_registry_path",
    "source_status_label_for_line",
    "source_summary",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Format-independent report snapshot, document and sealing model."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.storage_primitives import atomic_write_json, utc_now
from core.security_paths import UnsafeManagedPathError, resolve_managed_path, safe_relative_path


@dataclass(frozen=True)
class ReportSnapshot:
    snapshot_id: str
    project_id: str
    project_revision: int
    generated_at: str
    software_version: str
    template_version: str
    source_identities: tuple[dict[str, Any], ...] = ()
    input_artifacts: tuple[dict[str, Any], ...] = ()
    environment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        project_id: str,
        project_revision: int,
        software_version: str,
        template_version: str,
        source_identities: Iterable[dict[str, Any]] = (),
        input_artifacts: Iterable[dict[str, Any]] = (),
    ) -> "ReportSnapshot":
        return cls(
            snapshot_id=f"SNAP-{uuid.uuid4().hex[:16].upper()}",
            project_id=str(project_id),
            project_revision=int(project_revision),
            generated_at=utc_now(),
            software_version=str(software_version),
            template_version=str(template_version),
            source_identities=tuple(dict(item) for item in source_identities),
            input_artifacts=tuple(dict(item) for item in input_artifacts),
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "implementation": platform.python_implementation(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportSection:
    section_id: str
    title: str
    content: dict[str, Any]


@dataclass(frozen=True)
class ReportDocument:
    title: str
    snapshot: ReportSnapshot
    sections: tuple[ReportSection, ...]
    approval: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mygpr.report_document.v1",
            "title": self.title,
            "snapshot": self.snapshot.to_dict(),
            "approval": dict(self.approval),
            "sections": [asdict(section) for section in self.sections],
        }


@dataclass(frozen=True)
class ReportSeal:
    seal_id: str
    snapshot_id: str
    created_at: str
    status: str
    manifest_sha256: str
    files: tuple[dict[str, Any], ...]
    approval: dict[str, str]
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReportSealer:
    """Create a deterministic audit seal for an already rendered package."""

    @staticmethod
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def seal(
        self,
        package_dir: str | Path,
        *,
        snapshot: ReportSnapshot,
        approval: dict[str, str],
        status: str = "sealed",
    ) -> Path:
        raw_root = Path(package_dir)
        if raw_root.is_symlink():
            raise ValueError("报告包目录不能是符号链接。")
        root = raw_root.resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        files: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"报告包不能包含符号链接：{path.relative_to(root).as_posix()}")
            if not path.is_file() or path.name == "report_seal.json":
                continue
            files.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": self.sha256_file(path),
            })
        canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        seal = ReportSeal(
            seal_id=f"SEAL-{uuid.uuid4().hex[:16].upper()}",
            snapshot_id=snapshot.snapshot_id,
            created_at=utc_now(),
            status=status,
            manifest_sha256=hashlib.sha256(canonical).hexdigest(),
            files=tuple(files),
            approval=dict(approval),
        )
        path = root / "report_seal.json"
        atomic_write_json(path, {
            "schema": "mygpr.report_seal.v1",
            **seal.to_dict(),
            "snapshot": snapshot.to_dict(),
        })
        return path

    def verify(self, seal_path: str | Path) -> tuple[bool, list[str]]:
        path = Path(seal_path).resolve()
        errors: list[str] = []
        if not path.is_file() or path.is_symlink():
            return False, ["missing:report_seal.json"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False, ["invalid:report_seal.json"]
        if payload.get("schema") != "mygpr.report_seal.v1":
            errors.append("schema:report_seal.json")
        root = path.parent.resolve()
        raw_files = payload.get("files")
        if not isinstance(raw_files, list):
            return False, [*errors, "invalid:files"]
        canonical_rows: list[dict[str, Any]] = []
        declared: set[str] = set()
        declared_casefold: set[str] = set()
        for raw_item in raw_files:
            if not isinstance(raw_item, dict):
                errors.append("invalid:file-entry")
                continue
            raw_rel = str(raw_item.get("path") or "")
            try:
                rel = safe_relative_path(raw_rel).as_posix()
                candidate = resolve_managed_path(root, rel, require_file=True)
            except (UnsafeManagedPathError, FileNotFoundError):
                errors.append(f"unsafe-or-missing:{raw_rel}")
                continue
            folded = rel.casefold()
            if rel in declared or folded in declared_casefold:
                errors.append(f"duplicate:{rel}")
                continue
            declared.add(rel)
            declared_casefold.add(folded)
            expected_size = raw_item.get("size_bytes")
            if not isinstance(expected_size, int) or candidate.stat().st_size != expected_size:
                errors.append(f"size:{rel}")
            actual_hash = self.sha256_file(candidate)
            if actual_hash != str(raw_item.get("sha256") or ""):
                errors.append(f"hash:{rel}")
            canonical_rows.append({"path": rel, "size_bytes": expected_size, "sha256": str(raw_item.get("sha256") or "")})
        canonical = json.dumps(canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != str(payload.get("manifest_sha256") or ""):
            errors.append("hash:seal-manifest")
        actual_files: set[str] = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                errors.append(f"symlink:{candidate.relative_to(root).as_posix()}")
                continue
            if candidate.is_file() and candidate != path:
                actual_files.add(candidate.relative_to(root).as_posix())
        for rel in sorted(actual_files - declared):
            errors.append(f"extra:{rel}")
        for rel in sorted(declared - actual_files):
            if f"unsafe-or-missing:{rel}" not in errors:
                errors.append(f"missing:{rel}")
        return not errors, errors


__all__ = [
    "ReportDocument",
    "ReportSeal",
    "ReportSealer",
    "ReportSection",
    "ReportSnapshot",
]

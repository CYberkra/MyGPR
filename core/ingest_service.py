#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import loose GPR data into temporary or formal MyGPR projects."""

from __future__ import annotations

import os
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from core.gpr_format_registry import get_format_spec
from core.project_models import LineRecordV1, RawFileRef
from core.project_service import ProjectService, sha256_path, utc_now


_SIDECAR_NAMES = {
    "rtk": ("rtk.csv",),
    "imu": ("imu.csv",),
    "altimeter": ("altimeter.csv",),
    "trace_timestamps": ("trace_timestamps.csv",),
}


class IngestService:
    @staticmethod
    def open_temporary(source: str | Path) -> ProjectService:
        source_path = Path(source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        root = Path(tempfile.mkdtemp(prefix="MyGPR-preview-"))
        project = ProjectService.create(root, name=f"临时检查 - {source_path.stem}", temporary=True)
        line = IngestService._build_line_record(source_path, temporary=True)
        project.add_line(line)
        return project

    @staticmethod
    def import_into_project(
        project: ProjectService,
        source: str | Path,
        *,
        verify_hashes: bool = True,
    ) -> LineRecordV1:
        source_path = Path(source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        line = IngestService._build_line_record(source_path, temporary=project.manifest.temporary)
        if not project.manifest.temporary:
            line = IngestService._copy_line_into_project(project, line, verify_hashes=verify_hashes)
        project.add_line(line)
        return line

    @staticmethod
    def formalize(
        temporary: ProjectService,
        destination: str | Path,
        *,
        name: str,
        verify_hashes: bool = True,
    ) -> ProjectService:
        formal = ProjectService.create(destination, name=name, temporary=False)
        try:
            for line in temporary.list_lines():
                formal.add_line(
                    IngestService._copy_line_into_project(
                        formal,
                        line,
                        verify_hashes=verify_hashes,
                    )
                )
        except Exception:
            formal.close()
            raise
        return formal

    @staticmethod
    def verify_project_integrity(
        project: ProjectService,
        line_ids: list[str] | None = None,
    ) -> list[str]:
        targets = line_ids or list(project.manifest.line_ids)
        verified: list[str] = []
        for line_id in targets:
            line = project.get_line(line_id)
            mismatch = False
            for ref in line.raw_files:
                path = project.resolve_relative_path(ref.path)
                current_hash = sha256_path(path)
                if ref.sha256:
                    if current_hash == ref.sha256:
                        ref.integrity_status = "verified"
                    else:
                        ref.integrity_status = "mismatch"
                        mismatch = True
                    continue
                ref.sha256 = current_hash
                ref.integrity_status = "verified"
                ref.size_bytes = IngestService._path_size(path)
            line.status = "integrity_error" if mismatch else "verified"
            line.updated_at = utc_now()
            project.add_line(line)
            verified.append(line_id)
        return verified

    @staticmethod
    def assign_sidecar(project: ProjectService, line_id: str, kind: str, path: str | Path) -> LineRecordV1:
        if kind not in {"rtk", "imu", "altimeter", "trace_timestamps"}:
            raise ValueError(f"Unsupported sidecar kind: {kind}")
        sidecar = Path(path).resolve()
        if not sidecar.exists() or not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        line = project.get_line(line_id)
        if project.manifest.temporary:
            line.sidecars[kind] = str(sidecar)
        else:
            destination = project.resolve_relative_path(
                f"raw/{line_id}/sidecars/{kind}_{sidecar.name}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sidecar, destination)
            IngestService._make_read_only(destination)
            line.sidecars[kind] = destination.relative_to(project.root).as_posix()
            line.raw_files = [
                ref for ref in line.raw_files if ref.role != f"sidecar:{kind}"
            ]
            line.raw_files.append(
                RawFileRef(
                    path=destination.relative_to(project.root).as_posix(),
                    role=f"sidecar:{kind}",
                    size_bytes=IngestService._path_size(destination),
                    sha256=None,
                    integrity_status="pending",
                    source_path=str(sidecar),
                )
            )
            line.status = "pending_integrity"
        project.add_line(line)
        return line

    @staticmethod
    def _build_line_record(source: Path, *, temporary: bool) -> LineRecordV1:
        line_id = f"L-{uuid.uuid4().hex[:10]}"
        spec = get_format_spec(source) if source.is_file() else None
        sidecars = IngestService._discover_sidecars(source)
        raw = RawFileRef(
            path=str(source),
            role="primary",
            size_bytes=IngestService._path_size(source),
            integrity_status="external_preview" if temporary else "pending",
            source_path=str(source),
        )
        return LineRecordV1(
            line_id=line_id,
            name=source.stem if source.is_file() else source.name,
            raw_files=[raw],
            sidecars=sidecars,
            source_format=spec.key if spec else ("ascan_folder" if source.is_dir() else "unknown"),
            created_at=utc_now(),
            updated_at=utc_now(),
        )

    @staticmethod
    def _copy_line_into_project(
        project: ProjectService,
        line: LineRecordV1,
        *,
        verify_hashes: bool = True,
    ) -> LineRecordV1:
        raw_dir = project.resolve_relative_path(f"raw/{line.line_id}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        copied_refs: list[RawFileRef] = []
        for ref in line.raw_files:
            source = Path(ref.source_path or ref.path).resolve()
            destination = raw_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            IngestService._make_read_only(destination)
            copied_refs.append(
                RawFileRef(
                    path=destination.relative_to(project.root).as_posix(),
                    role=ref.role,
                    size_bytes=IngestService._path_size(destination),
                    sha256=sha256_path(destination) if verify_hashes else None,
                    integrity_status="verified" if verify_hashes else "pending",
                    source_path=str(source),
                )
            )
        copied_sidecars: dict[str, str] = {}
        copied_sidecar_refs: list[RawFileRef] = []
        for kind, path in line.sidecars.items():
            source = Path(path).resolve()
            destination = raw_dir / "sidecars" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            IngestService._make_read_only(destination)
            copied_sidecars[kind] = destination.relative_to(project.root).as_posix()
            copied_sidecar_refs.append(
                RawFileRef(
                    path=destination.relative_to(project.root).as_posix(),
                    role=f"sidecar:{kind}",
                    size_bytes=IngestService._path_size(destination),
                    sha256=sha256_path(destination) if verify_hashes else None,
                    integrity_status="verified" if verify_hashes else "pending",
                    source_path=str(source),
                )
            )
        line.raw_files = copied_refs + copied_sidecar_refs
        line.sidecars = copied_sidecars
        line.status = "verified" if verify_hashes else "pending_integrity"
        line.updated_at = utc_now()
        return line

    @staticmethod
    def _discover_sidecars(source: Path) -> dict[str, str]:
        base = source if source.is_dir() else source.parent
        discovered: dict[str, str] = {}
        manifest_path = base / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                primary = manifest.get("primary_data_file") or manifest.get("data_file") or manifest.get("main_csv")
                primary_matches = not primary or (base / str(primary)).resolve() == source.resolve()
                sidecars = manifest.get("sidecars") if isinstance(manifest.get("sidecars"), dict) else {}
                if primary_matches:
                    for kind in _SIDECAR_NAMES:
                        value = sidecars.get(kind) or sidecars.get(f"{kind}_file") or manifest.get(f"{kind}_file")
                        if isinstance(value, str):
                            candidate = (base / value).resolve()
                            if candidate.exists():
                                discovered[kind] = str(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        for kind, names in _SIDECAR_NAMES.items():
            if kind in discovered:
                continue
            for name in names:
                candidate = base / name
                if candidate.exists() and candidate.resolve() != source:
                    discovered[kind] = str(candidate.resolve())
                    break
        return discovered

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_dir():
            return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        return path.stat().st_size

    @staticmethod
    def _make_read_only(path: Path) -> None:
        if path.is_dir():
            for item in path.rglob("*"):
                if item.is_file():
                    os.chmod(item, 0o444)
            return
        os.chmod(path, 0o444)


__all__ = ["IngestService"]

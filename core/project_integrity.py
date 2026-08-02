#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-module integrity audit and conservative linkage repair.

The five workbench pages exchange concrete engineering objects through project
files.  This service checks that those object references still resolve after a
crash, manual file movement, backup restore or partial external copy.  Repairs
are deliberately conservative: only workspace pointers and current-version
indexes are changed; immutable processing, annotation, spatial and report
artifacts are never rewritten or deleted.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python <3.11 fallback
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass
from pathlib import Path
from typing import Any, Iterable

from core.field_project_models import local_now
from core.processing_artifact_index import index_processing_artifacts
from core.project_state_tracker import ProjectStateTracker
from core.report_package_versions import ReportPackageVersionService
from core.spatial_result_versions import SpatialResultVersionService
from core.storage_primitives import atomic_write_json
from core.workspace_context import WorkspaceContextStore
from core.storage_uri import is_h5_uri, resolve_h5_uri
from core.hdf5_line_container import (
    RAW_MATRIX_PATH,
    compute_dataset_sha256,
    list_processing_artifact_ids,
    read_raw_metadata,
)

PROJECT_INTEGRITY_SCHEMA = "mygpr.project_integrity_report.v1"
PROJECT_AUDIT_ERRORS = (
    OSError, UnicodeError, json.JSONDecodeError, sqlite3.Error,
    RuntimeError, TypeError, ValueError, KeyError,
)


class IntegritySeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    severity: str
    module: str
    message: str
    object_id: str = ""
    path: str = ""
    repairable: bool = False
    repaired: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectIntegrityReport:
    project_root: str
    generated_at: str
    issues: tuple[IntegrityIssue, ...]
    repairs: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
    schema: str = PROJECT_INTEGRITY_SCHEMA

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == IntegritySeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == IntegritySeverity.WARNING)

    @property
    def healthy(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "project_root": self.project_root,
            "generated_at": self.generated_at,
            "healthy": self.healthy,
            "summary": {
                "issue_count": len(self.issues),
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "repair_count": len(self.repairs),
                "elapsed_ms": round(float(self.elapsed_ms), 3),
            },
            "issues": [item.to_dict() for item in self.issues],
            "repairs": list(self.repairs),
        }


class ProjectIntegrityAuditor:
    """Audit and repair cross-page project references."""

    def __init__(self, project_store: Any) -> None:
        self.project = project_store
        self.root = Path(project_store.root).resolve()
        self.report_path = self.root / "metadata" / "project_integrity.json"

    def audit(
        self,
        *,
        repair_context: bool = False,
        clean_staging: bool = False,
        staging_min_age_s: float = 3600.0,
        persist: bool = True,
        deep_hash: bool = False,
    ) -> ProjectIntegrityReport:
        started = time.perf_counter()
        issues: list[IntegrityIssue] = []
        repairs: list[str] = []

        line_ids = {str(line.line_id) for line in self.project.list_lines()}
        self._audit_hybrid_storage(issues, line_ids, deep_hash=deep_hash)
        processing_records = index_processing_artifacts(self.root)
        processing_ids = {record.artifact_id for record in processing_records}
        processing_by_line: dict[str, list[str]] = {}
        for record in processing_records:
            processing_by_line.setdefault(record.line_id, []).append(record.artifact_id)
            self._check_registered_file(issues, "processing", record.artifact_id, record.data_path, required=True)
            self._check_registered_file(issues, "processing", record.artifact_id, record.params_path, required=False)
            self._check_registered_file(issues, "processing", record.artifact_id, record.manifest_path, required=False)
            if record.line_id not in line_ids:
                issues.append(self._issue(
                    "processing.orphan_line", IntegritySeverity.ERROR, "processing",
                    f"处理版本 {record.artifact_id} 引用了不存在的测线 {record.line_id}",
                    object_id=record.artifact_id,
                ))

        self._audit_lines(issues, line_ids)
        annotation_info = self._audit_annotations(issues, line_ids, processing_ids)
        spatial_ids = self._audit_spatial(issues, line_ids, processing_ids, annotation_info)
        report_ids = self._audit_reports(issues, spatial_ids)
        context, context_changed = self._audit_workspace_context(
            issues,
            repairs,
            line_ids=line_ids,
            processing_by_line=processing_by_line,
            processing_ids=processing_ids,
            annotation_info=annotation_info,
            spatial_ids=spatial_ids,
            report_ids=report_ids,
            repair=repair_context,
        )
        if context_changed:
            WorkspaceContextStore(self.root).save(context)

        self._audit_indexes(issues, repairs, spatial_ids, report_ids, repair=repair_context)
        self._audit_project_state(issues)
        self._audit_staging(
            issues,
            repairs,
            clean=clean_staging,
            min_age_s=max(0.0, float(staging_min_age_s)),
        )

        report = ProjectIntegrityReport(
            project_root=str(self.root),
            generated_at=local_now(),
            issues=tuple(issues),
            repairs=tuple(repairs),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        if persist and not bool(getattr(self.project, "read_only", False)):
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.report_path, report.to_dict())
        return report

    def load_last_report(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.report_path.read_text(encoding="utf-8"))
        except PROJECT_AUDIT_ERRORS:
            return None

    # ------------------------------------------------------------------
    # Audits
    # ------------------------------------------------------------------
    def _audit_hybrid_storage(
        self,
        issues: list[IntegrityIssue],
        line_ids: set[str],
        *,
        deep_hash: bool,
    ) -> None:
        storage = getattr(self.project, "storage", None)
        if not getattr(storage, "is_hybrid", False):
            return
        try:
            ok, message = storage.catalog.integrity_check()
        except PROJECT_AUDIT_ERRORS as exc:
            issues.append(self._issue(
                "storage.catalog_unreadable", IntegritySeverity.ERROR, "storage",
                f"SQLite 项目目录不可读取：{exc}", path="catalog.sqlite",
            ))
            return
        if not ok:
            issues.append(self._issue(
                "storage.catalog_integrity", IntegritySeverity.ERROR, "storage",
                f"SQLite 项目目录完整性检查失败：{message}", path="catalog.sqlite",
            ))
        try:
            catalog_line_ids = {str(row["line_id"]) for row in storage.catalog.list_lines()}
        except PROJECT_AUDIT_ERRORS as exc:
            issues.append(self._issue(
                "storage.catalog_lines_unreadable", IntegritySeverity.ERROR, "storage",
                f"SQLite 测线目录不可读取：{exc}", path="catalog.sqlite",
            ))
            return
        for line_id in sorted(line_ids - catalog_line_ids):
            issues.append(self._issue(
                "storage.catalog_line_missing", IntegritySeverity.ERROR, "storage",
                f"测线 {line_id} 未登记到 SQLite 目录。", object_id=line_id, path="catalog.sqlite",
            ))
        for line_id in sorted(catalog_line_ids - line_ids):
            issues.append(self._issue(
                "storage.catalog_orphan_line", IntegritySeverity.ERROR, "storage",
                f"SQLite 目录包含项目清单中不存在的测线 {line_id}。", object_id=line_id, path="catalog.sqlite",
            ))
        transaction_journal = getattr(storage, "transaction_journal", None)
        if transaction_journal is not None:
            for journal_path in transaction_journal.pending_paths():
                issues.append(self._issue(
                    "storage.pending_hybrid_transaction", IntegritySeverity.ERROR, "storage",
                    "检测到未完成的 SQLite/HDF5 提交事务；请以可写模式重新打开项目执行自动恢复。",
                    path=self._rel(journal_path), repairable=True,
                ))
        for line_id in sorted(line_ids):
            container = storage.line_container_path(line_id)
            if not container.exists():
                line = self.project.get_line(line_id)
                if getattr(line, "gpr_dataset_path", "") or storage.catalog.list_artifacts(line_id=line_id):
                    issues.append(self._issue(
                        "storage.line_container_missing", IntegritySeverity.ERROR, "storage",
                        f"测线 {line_id} 的 HDF5 容器缺失。", object_id=line_id,
                        path=self._rel(container),
                    ))
                continue
            for message in storage.validate_line(line_id):
                issues.append(self._issue(
                    "storage.line_container_invalid", IntegritySeverity.ERROR, "storage",
                    f"测线 {line_id} HDF5 校验失败：{message}", object_id=line_id,
                    path=self._rel(container),
                ))
            try:
                h5_artifact_ids = set(list_processing_artifact_ids(container))
                catalog_rows = storage.catalog.list_artifacts(
                    line_id=line_id, artifact_kind="processing"
                )
                catalog_artifact_ids = {
                    str(row.get("artifact_id") or "")
                    for row in catalog_rows
                    if str(row.get("artifact_id") or "")
                }
            except PROJECT_AUDIT_ERRORS as exc:
                issues.append(self._issue(
                    "storage.artifact_index_unreadable", IntegritySeverity.ERROR, "storage",
                    f"测线 {line_id} 的处理产物索引无法核对：{exc}", object_id=line_id,
                    path=self._rel(container),
                ))
                continue
            for artifact_id in sorted(h5_artifact_ids - catalog_artifact_ids):
                issues.append(self._issue(
                    "storage.unindexed_hdf5_artifact", IntegritySeverity.WARNING, "storage",
                    f"HDF5 中存在未登记到 SQLite 的处理产物 {artifact_id}。",
                    object_id=artifact_id, path=self._rel(container),
                    details={"line_id": line_id, "recovery_action": "reindex_or_quarantine"},
                ))
            for artifact_id in sorted(catalog_artifact_ids - h5_artifact_ids):
                issues.append(self._issue(
                    "storage.catalog_artifact_missing", IntegritySeverity.ERROR, "storage",
                    f"SQLite 登记的处理产物 {artifact_id} 在 HDF5 中不存在。",
                    object_id=artifact_id, path=self._rel(container),
                    details={"line_id": line_id},
                ))
            if deep_hash:
                self._audit_hdf5_hashes(issues, line_id, container, catalog_rows)

    def _audit_hdf5_hashes(
        self,
        issues: list[IntegrityIssue],
        line_id: str,
        container: Path,
        catalog_rows: list[dict[str, Any]],
    ) -> None:
        try:
            raw_metadata = read_raw_metadata(container)
            expected_raw = str(raw_metadata.get("data_sha256") or "")
            actual_raw = compute_dataset_sha256(container, RAW_MATRIX_PATH)
            if not expected_raw:
                issues.append(self._issue(
                    "storage.raw_hash_missing", IntegritySeverity.WARNING, "storage",
                    f"测线 {line_id} 原始矩阵未记录 SHA-256。", object_id=line_id,
                    path=self._rel(container),
                ))
            elif actual_raw != expected_raw:
                issues.append(self._issue(
                    "storage.raw_hash_mismatch", IntegritySeverity.ERROR, "storage",
                    f"测线 {line_id} 原始矩阵 SHA-256 不匹配。", object_id=line_id,
                    path=self._rel(container),
                    details={"expected": expected_raw, "actual": actual_raw},
                ))
        except PROJECT_AUDIT_ERRORS as exc:
            issues.append(self._issue(
                "storage.raw_hash_unreadable", IntegritySeverity.ERROR, "storage",
                f"测线 {line_id} 原始矩阵深度校验失败：{exc}", object_id=line_id,
                path=self._rel(container),
            ))

        for row in catalog_rows:
            artifact_id = str(row.get("artifact_id") or "")
            dataset_path = str(row.get("dataset_path") or "")
            expected = str(row.get("sha256") or "")
            if not artifact_id or not dataset_path:
                continue
            try:
                actual = compute_dataset_sha256(container, dataset_path)
            except PROJECT_AUDIT_ERRORS as exc:
                issues.append(self._issue(
                    "storage.artifact_hash_unreadable", IntegritySeverity.ERROR, "storage",
                    f"处理产物 {artifact_id} 深度校验失败：{exc}", object_id=artifact_id,
                    path=self._rel(container), details={"line_id": line_id},
                ))
                continue
            if not expected:
                issues.append(self._issue(
                    "storage.artifact_hash_missing", IntegritySeverity.WARNING, "storage",
                    f"处理产物 {artifact_id} 未记录 SHA-256。", object_id=artifact_id,
                    path=self._rel(container), details={"line_id": line_id},
                ))
            elif actual != expected:
                issues.append(self._issue(
                    "storage.artifact_hash_mismatch", IntegritySeverity.ERROR, "storage",
                    f"处理产物 {artifact_id} SHA-256 不匹配。", object_id=artifact_id,
                    path=self._rel(container),
                    details={"line_id": line_id, "expected": expected, "actual": actual},
                ))

    def _audit_lines(self, issues: list[IntegrityIssue], line_ids: set[str]) -> None:
        for line in self.project.list_lines():
            line_id = str(line.line_id)
            if not line_id:
                issues.append(self._issue(
                    "line.missing_id", IntegritySeverity.ERROR, "project", "项目中存在没有标识的测线。"
                ))
                continue
            for attr, label, required in (
                ("gpr_dataset_path", "标准化雷达数据", False),
                ("raw_path", "原始雷达数据", False),
                ("trajectory_path", "轨迹数据", False),
            ):
                rel = str(getattr(line, attr, "") or "")
                if not rel:
                    continue
                path = self.root / rel
                if not path.exists():
                    issues.append(self._issue(
                        f"line.missing_{attr}", IntegritySeverity.ERROR, "project",
                        f"{line_id} 的{label}不存在：{rel}", object_id=line_id, path=rel,
                    ))
            if not (getattr(line, "gpr_dataset_path", "") or getattr(line, "raw_path", "")):
                issues.append(self._issue(
                    "line.no_radar_source", IntegritySeverity.WARNING, "project",
                    f"{line_id} 尚未绑定雷达数据。", object_id=line_id,
                ))
        if not line_ids:
            issues.append(self._issue(
                "project.no_lines", IntegritySeverity.INFO, "project", "项目尚未创建或导入测线。"
            ))

    def _audit_annotations(
        self,
        issues: list[IntegrityIssue],
        line_ids: set[str],
        processing_ids: set[str],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for line_id in sorted(line_ids):
            path = self.project.interface_annotation_path(line_id)
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except PROJECT_AUDIT_ERRORS as exc:
                issues.append(self._issue(
                    "annotation.unreadable", IntegritySeverity.ERROR, "annotation",
                    f"{line_id} 标注文件不可读取：{exc}", object_id=line_id, path=self._rel(path),
                ))
                continue
            source = str(payload.get("source_result_id") or f"{line_id}_raw")
            version = str(payload.get("version") or "")
            status = str(payload.get("status") or "draft")
            result[line_id] = {"source_result_id": source, "version": version, "status": status, "path": self._rel(path)}
            if source not in {f"{line_id}_raw", "raw", ""} and source not in processing_ids:
                issues.append(self._issue(
                    "annotation.missing_processing_source", IntegritySeverity.ERROR, "annotation",
                    f"{line_id} 标注绑定的处理版本 {source} 已缺失。",
                    object_id=line_id, path=self._rel(path),
                ))
            if str(payload.get("line_id") or line_id) != line_id:
                issues.append(self._issue(
                    "annotation.line_mismatch", IntegritySeverity.ERROR, "annotation",
                    f"标注文件中的测线标识与文件名不一致：{line_id}", object_id=line_id,
                ))
        return result

    def _audit_spatial(
        self,
        issues: list[IntegrityIssue],
        line_ids: set[str],
        processing_ids: set[str],
        annotation_info: dict[str, dict[str, Any]],
    ) -> set[str]:
        service = SpatialResultVersionService(self.project)
        records = service.list_results()
        ids = {record.result_id for record in records}
        for record in records:
            manifest_dir = self.root / "spatial" / "results" / record.result_id
            if record.stale:
                issues.append(self._issue(
                    "spatial.stale", IntegritySeverity.WARNING, "spatial",
                    f"空间成果 {record.result_id} 的上游来源已变化。", object_id=record.result_id,
                ))
            for line_id in record.line_ids:
                if line_id not in line_ids:
                    issues.append(self._issue(
                        "spatial.missing_line", IntegritySeverity.ERROR, "spatial",
                        f"空间成果 {record.result_id} 引用了不存在的测线 {line_id}。",
                        object_id=record.result_id,
                    ))
            for label, rel in record.files.items():
                path = manifest_dir / str(rel)
                if not path.exists():
                    issues.append(self._issue(
                        "spatial.missing_file", IntegritySeverity.ERROR, "spatial",
                        f"空间成果 {record.result_id} 缺少文件 {label}: {rel}",
                        object_id=record.result_id, path=self._rel(path),
                    ))
            source_lines = dict(record.sources.get("lines") or {})
            for line_id, source in source_lines.items():
                if line_id not in line_ids:
                    continue
                proc = str((source or {}).get("processing_result") or "")
                if proc and proc not in processing_ids and proc not in {f"{line_id}_raw", "raw"}:
                    issues.append(self._issue(
                        "spatial.missing_processing_source", IntegritySeverity.ERROR, "spatial",
                        f"空间成果 {record.result_id} 引用了缺失处理版本 {proc}。",
                        object_id=record.result_id,
                    ))
                ann = annotation_info.get(line_id)
                expected_version = str((source or {}).get("annotation_version") or "")
                if expected_version and ann is None:
                    issues.append(self._issue(
                        "spatial.missing_annotation", IntegritySeverity.ERROR, "spatial",
                        f"空间成果 {record.result_id} 的测线 {line_id} 缺少当前标注文件。",
                        object_id=record.result_id,
                    ))
        return ids

    def _audit_reports(self, issues: list[IntegrityIssue], spatial_ids: set[str]) -> set[str]:
        service = ReportPackageVersionService(self.project)
        records = service.list_reports()
        ids = {record.report_id for record in records}
        for record in records:
            if record.stale:
                issues.append(self._issue(
                    "report.stale", IntegritySeverity.WARNING, "report",
                    f"报告 {record.report_id} 的来源已变化。", object_id=record.report_id,
                    details={"reasons": list(record.stale_reasons)},
                ))
            if record.spatial_result_id and record.spatial_result_id not in spatial_ids:
                issues.append(self._issue(
                    "report.missing_spatial_source", IntegritySeverity.ERROR, "report",
                    f"报告 {record.report_id} 引用了缺失空间成果 {record.spatial_result_id}。",
                    object_id=record.report_id,
                ))
            package_dir = Path(str(record.result.get("package_dir") or ""))
            if package_dir and not package_dir.is_absolute():
                package_dir = self.root / package_dir
            if not package_dir or not package_dir.exists():
                issues.append(self._issue(
                    "report.missing_package", IntegritySeverity.ERROR, "report",
                    f"报告 {record.report_id} 的正式报告目录不存在。",
                    object_id=record.report_id, path=self._rel(package_dir) if package_dir else "",
                ))
                continue
            ok, verify_issues = service.verify(record.report_id)
            if not ok:
                issues.append(self._issue(
                    "report.seal_invalid", IntegritySeverity.ERROR, "report",
                    f"报告 {record.report_id} 封印校验失败。", object_id=record.report_id,
                    details={"verify_issues": list(verify_issues)},
                ))
        return ids

    def _audit_workspace_context(
        self,
        issues: list[IntegrityIssue],
        repairs: list[str],
        *,
        line_ids: set[str],
        processing_by_line: dict[str, list[str]],
        processing_ids: set[str],
        annotation_info: dict[str, dict[str, Any]],
        spatial_ids: set[str],
        report_ids: set[str],
        repair: bool,
    ) -> tuple[dict[str, Any], bool]:
        store = WorkspaceContextStore(self.root)
        try:
            context = store.load()
        except PROJECT_AUDIT_ERRORS as exc:
            issues.append(self._issue(
                "workspace.unreadable", IntegritySeverity.ERROR, "workspace",
                f"工作区上下文不可读取：{exc}", repairable=repair,
            ))
            context = {
                "active_workspace": "data_management",
                "selected_line_id": None,
                "processing_source_by_line": {},
                "annotation_by_line": {},
                "selected_spatial_result_id": "",
                "selected_report_id": "",
                "last_handoff": {},
            }
            changed = bool(repair)
            if repair:
                repairs.append("重建工作区上下文")
            return context, changed

        changed = False
        selected_line = str(context.get("selected_line_id") or "")
        if selected_line and selected_line not in line_ids:
            issues.append(self._issue(
                "workspace.missing_selected_line", IntegritySeverity.WARNING, "workspace",
                f"工作区选择的测线 {selected_line} 已不存在。", object_id=selected_line,
                repairable=True, repaired=repair,
            ))
            if repair:
                context["selected_line_id"] = sorted(line_ids)[0] if line_ids else None
                repairs.append(f"重置当前测线：{selected_line}")
                changed = True

        processing_map = dict(context.get("processing_source_by_line") or {})
        clean_processing: dict[str, str] = {}
        for line_id, artifact_id in processing_map.items():
            line_id = str(line_id)
            artifact_id = str(artifact_id)
            valid = line_id in line_ids and artifact_id in processing_ids
            if valid:
                clean_processing[line_id] = artifact_id
                continue
            issues.append(self._issue(
                "workspace.invalid_processing_binding", IntegritySeverity.WARNING, "workspace",
                f"工作区处理来源绑定无效：{line_id} → {artifact_id}", object_id=line_id,
                repairable=True, repaired=repair,
            ))
            if repair and line_id in line_ids and processing_by_line.get(line_id):
                replacement = processing_by_line[line_id][0]
                clean_processing[line_id] = replacement
                repairs.append(f"处理来源重绑定：{line_id} → {replacement}")
            elif repair:
                repairs.append(f"移除无效处理来源绑定：{line_id} → {artifact_id}")
            if repair:
                changed = True
        if repair:
            context["processing_source_by_line"] = clean_processing

        annotation_map = dict(context.get("annotation_by_line") or {})
        clean_annotations: dict[str, Any] = {}
        for line_id, binding in annotation_map.items():
            line_id = str(line_id)
            binding = dict(binding or {})
            actual = annotation_info.get(line_id)
            if line_id in line_ids and actual is not None:
                source = str(binding.get("source_result_id") or actual.get("source_result_id") or "")
                if source and source not in processing_ids and source not in {f"{line_id}_raw", "raw"}:
                    source = str(actual.get("source_result_id") or f"{line_id}_raw")
                    if repair:
                        repairs.append(f"标注来源重绑定：{line_id} → {source}")
                        changed = True
                clean_annotations[line_id] = {
                    "version": str(actual.get("version") or binding.get("version") or ""),
                    "status": str(actual.get("status") or binding.get("status") or ""),
                    "source_result_id": source,
                    "updated_at": str(binding.get("updated_at") or local_now()),
                }
                continue
            issues.append(self._issue(
                "workspace.invalid_annotation_binding", IntegritySeverity.WARNING, "workspace",
                f"工作区标注绑定无效：{line_id}", object_id=line_id,
                repairable=True, repaired=repair,
            ))
            if repair:
                repairs.append(f"移除无效标注绑定：{line_id}")
                changed = True
        if repair:
            context["annotation_by_line"] = clean_annotations

        selected_spatial = str(context.get("selected_spatial_result_id") or "")
        if selected_spatial and selected_spatial not in spatial_ids:
            issues.append(self._issue(
                "workspace.invalid_spatial_selection", IntegritySeverity.WARNING, "workspace",
                f"工作区空间成果 {selected_spatial} 已不存在。", object_id=selected_spatial,
                repairable=True, repaired=repair,
            ))
            if repair:
                replacement = self._latest_spatial_id()
                context["selected_spatial_result_id"] = replacement
                repairs.append(f"重置空间成果：{selected_spatial} → {replacement or '空'}")
                changed = True

        selected_report = str(context.get("selected_report_id") or "")
        if selected_report and selected_report not in report_ids:
            issues.append(self._issue(
                "workspace.invalid_report_selection", IntegritySeverity.WARNING, "workspace",
                f"工作区报告 {selected_report} 已不存在。", object_id=selected_report,
                repairable=True, repaired=repair,
            ))
            if repair:
                replacement = self._latest_report_id()
                context["selected_report_id"] = replacement
                repairs.append(f"重置报告：{selected_report} → {replacement or '空'}")
                changed = True
        return context, changed

    def _audit_indexes(
        self,
        issues: list[IntegrityIssue],
        repairs: list[str],
        spatial_ids: set[str],
        report_ids: set[str],
        *,
        repair: bool,
    ) -> None:
        spatial_service = SpatialResultVersionService(self.project)
        current_spatial = spatial_service.current_result_id()
        if current_spatial and current_spatial not in spatial_ids:
            issues.append(self._issue(
                "spatial.invalid_current_index", IntegritySeverity.WARNING, "spatial",
                f"空间成果索引指向缺失版本 {current_spatial}。", object_id=current_spatial,
                repairable=True, repaired=repair,
            ))
            if repair:
                replacement = self._latest_spatial_id()
                if replacement:
                    spatial_service.set_current(replacement)
                else:
                    self._clear_index(spatial_service.index_path, "mygpr.spatial_result_index.v1", "current_result_id")
                repairs.append(f"修复空间成果当前索引：{replacement or '空'}")

        report_service = ReportPackageVersionService(self.project)
        current_report = report_service.current_report_id()
        if current_report and current_report not in report_ids:
            issues.append(self._issue(
                "report.invalid_current_index", IntegritySeverity.WARNING, "report",
                f"报告索引指向缺失版本 {current_report}。", object_id=current_report,
                repairable=True, repaired=repair,
            ))
            if repair:
                replacement = self._latest_report_id()
                if replacement:
                    report_service.set_current(replacement)
                else:
                    self._clear_index(report_service.index_path, "mygpr.report_version_index.v1", "current_report_id")
                repairs.append(f"修复报告当前索引：{replacement or '空'}")

    def _audit_project_state(self, issues: list[IntegrityIssue]) -> None:
        try:
            state = ProjectStateTracker(self.root).load()
        except PROJECT_AUDIT_ERRORS as exc:
            issues.append(self._issue(
                "state.unreadable", IntegritySeverity.ERROR, "project",
                f"项目依赖状态不可读取：{exc}",
            ))
            return
        dirty = dict(state.get("dirty") or {})
        known = {"project", "processing", "targets", "spatial", "report"}
        unknown = sorted(set(dirty) - known)
        if unknown:
            issues.append(self._issue(
                "state.unknown_modules", IntegritySeverity.WARNING, "project",
                f"项目状态包含未知模块：{', '.join(unknown)}",
            ))

    def _audit_staging(
        self,
        issues: list[IntegrityIssue],
        repairs: list[str],
        *,
        clean: bool,
        min_age_s: float,
    ) -> None:
        now = time.time()
        candidates: list[Path] = []
        for path in self.root.rglob("*"):
            name = path.name.lower()
            if not (name.startswith(".") and ("staging" in name or name.endswith(".tmp"))):
                continue
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age < min_age_s:
                continue
            candidates.append(path)
        for path in sorted(candidates, key=lambda item: len(item.parts), reverse=True):
            rel = self._rel(path)
            repaired = False
            if clean and not bool(getattr(self.project, "read_only", False)):
                try:
                    if path.is_dir():
                        import shutil
                        shutil.rmtree(path)
                    else:
                        path.unlink(missing_ok=True)
                    repairs.append(f"清理遗留临时项：{rel}")
                    repaired = True
                except OSError:
                    repaired = False
            issues.append(self._issue(
                "storage.orphan_staging", IntegritySeverity.WARNING, "storage",
                f"发现遗留临时项：{rel}", path=rel,
                repairable=True, repaired=repaired,
            ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_registered_file(
        self,
        issues: list[IntegrityIssue],
        module: str,
        object_id: str,
        rel: str,
        *,
        required: bool,
    ) -> None:
        if not rel:
            if required:
                issues.append(self._issue(
                    f"{module}.missing_registered_path", IntegritySeverity.ERROR, module,
                    f"{object_id} 没有登记必需文件路径。", object_id=object_id,
                ))
            return
        if is_h5_uri(rel):
            try:
                import h5py
                path, dataset_path = resolve_h5_uri(self.root, rel)
                if not path.exists():
                    raise FileNotFoundError(path)
                with h5py.File(path, "r", libver="latest", swmr=True) as handle:
                    if dataset_path not in handle:
                        raise KeyError(dataset_path)
                return
            except PROJECT_AUDIT_ERRORS:
                issues.append(self._issue(
                    f"{module}.missing_registered_dataset", IntegritySeverity.ERROR, module,
                    f"{object_id} 登记的 HDF5 数据集不存在：{rel}", object_id=object_id, path=rel,
                ))
                return
        path = self.root / rel
        if not path.exists():
            issues.append(self._issue(
                f"{module}.missing_registered_file", IntegritySeverity.ERROR, module,
                f"{object_id} 登记的文件不存在：{rel}", object_id=object_id, path=rel,
            ))

    def _latest_spatial_id(self) -> str:
        rows = SpatialResultVersionService(self.project).list_results()
        return rows[0].result_id if rows else ""

    def _latest_report_id(self) -> str:
        rows = ReportPackageVersionService(self.project).list_reports()
        return rows[0].report_id if rows else ""

    @staticmethod
    def _clear_index(path: Path, schema: str, key: str) -> None:
        atomic_write_json(path, {"schema": schema, key: "", "updated_at": local_now()})

    def _rel(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except PROJECT_AUDIT_ERRORS:
            return str(path)

    @staticmethod
    def _issue(
        code: str,
        severity: IntegritySeverity | str,
        module: str,
        message: str,
        *,
        object_id: str = "",
        path: str = "",
        repairable: bool = False,
        repaired: bool = False,
        details: dict[str, Any] | None = None,
    ) -> IntegrityIssue:
        return IntegrityIssue(
            code=str(code),
            severity=str(severity),
            module=str(module),
            message=str(message),
            object_id=str(object_id or ""),
            path=str(path or ""),
            repairable=bool(repairable),
            repaired=bool(repaired),
            details=dict(details or {}),
        )


__all__ = [
    "PROJECT_INTEGRITY_SCHEMA",
    "IntegrityIssue",
    "IntegritySeverity",
    "ProjectIntegrityAuditor",
    "ProjectIntegrityReport",
]

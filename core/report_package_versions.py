#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable report-package versions and delivery-bundle lifecycle.

The report page may show live project information, but a formal report must bind
one explicit project revision and, when available, one immutable spatial result.
This service assigns stable ``report_vNNN`` identities, delegates rendering to
the report exporter, exposes staleness, and verifies the sealed delivery bundle.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.field_project_models import local_now
from core.field_report_export import generate_project_report_package
from core.project_state_tracker import ProjectStateTracker, load_project_state
from core.reporting_model import ReportSealer
from core.security_paths import ensure_direct_child, resolve_managed_path
from core.spatial_result_versions import SpatialResultVersionService

REPORT_VERSION_INDEX_SCHEMA = "mygpr.report_version_index.v1"
REPORT_VERSION_SCHEMA = "mygpr.report_version.v1"


@dataclass(frozen=True)
class ReportVersionRecord:
    report_id: str
    report_revision: str
    created_at: str
    title: str
    template_version: str
    project_revision: int
    snapshot_id: str
    spatial_result_id: str
    approval: dict[str, str] = field(default_factory=dict)
    lifecycle: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    stale: bool = False
    stale_reasons: tuple[str, ...] = ()

    @classmethod
    def from_manifest(
        cls,
        payload: dict[str, Any],
        *,
        stale: bool = False,
        stale_reasons: list[str] | tuple[str, ...] = (),
    ) -> "ReportVersionRecord":
        summary = dict(payload.get("summary") or {})
        snapshot = dict(summary.get("snapshot") or {})
        binding = dict(summary.get("source_binding") or {})
        project = dict(summary.get("project") or {})
        return cls(
            report_id=str(payload.get("report_id") or Path(str((payload.get("result") or {}).get("package_dir") or "report")).name),
            report_revision=str(payload.get("report_revision") or summary.get("report_revision") or "R1"),
            created_at=str(payload.get("generated_at") or summary.get("generated_at") or ""),
            title=str(project.get("name") or "基覆界面识别成果报告"),
            template_version=str(snapshot.get("template_version") or summary.get("template_version") or ""),
            project_revision=int(snapshot.get("project_revision") or 0),
            snapshot_id=str(snapshot.get("snapshot_id") or ""),
            spatial_result_id=str(binding.get("spatial_result_id") or ""),
            approval=dict(summary.get("approval") or {}),
            lifecycle=dict(summary.get("lifecycle") or {}),
            result=dict(payload.get("result") or {}),
            stale=bool(stale),
            stale_reasons=tuple(str(item) for item in stale_reasons),
        )


class ReportPackageVersionService:
    """Create, list, verify and query formal report-package versions."""

    def __init__(self, project_store: Any):
        self.project = project_store
        self.root = Path(project_store.root) / "reports" / "versions"
        self.index_path = self.root / "index.json"

    def list_reports(self) -> list[ReportVersionRecord]:
        records: list[ReportVersionRecord] = []
        if not self.root.exists():
            return records
        for manifest_path in self.root.glob("*/report_manifest.json"):
            try:
                records.append(self.load_report(manifest_path.parent.name))
            except (OSError, UnicodeError, ValueError, TypeError, KeyError):
                continue
        records.sort(key=lambda item: (item.created_at, item.report_id), reverse=True)
        return records

    def current_report_id(self) -> str:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            return str(payload.get("current_report_id") or "")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return ""

    def set_current(self, report_id: str) -> None:
        record = self.load_report(report_id)
        self._atomic_json(
            self.index_path,
            {
                "schema": REPORT_VERSION_INDEX_SCHEMA,
                "current_report_id": record.report_id,
                "updated_at": local_now(),
            },
        )

    def load_report(self, report_id: str) -> ReportVersionRecord:
        safe = _safe_report_id(report_id)
        manifest_path = self.root / safe / "report_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        report_dir = ensure_direct_child(self.root, manifest_path.parent)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(payload.get("report_id") or "") != safe:
            raise ValueError(f"报告清单编号与目录不一致：{safe}")
        result = dict(payload.get("result") or {})
        package_rel = str(result.get("package_dir") or "")
        if package_rel and resolve_managed_path(self.project.root, package_rel, require_dir=True) != report_dir:
            raise ValueError(f"报告包路径与版本目录不一致：{package_rel}")
        stale, reasons = self._stale_state(payload)
        return ReportVersionRecord.from_manifest(payload, stale=stale, stale_reasons=reasons)

    def preflight(self, *, spatial_result_id: str | None = None) -> dict[str, Any]:
        warnings: list[str] = []
        errors: list[str] = []
        spatial_service = SpatialResultVersionService(self.project)
        selected = str(spatial_result_id or spatial_service.current_result_id() or "")
        if selected:
            try:
                spatial = spatial_service.load_result(selected)
                if spatial.stale:
                    warnings.append(f"空间成果 {selected} 的来源已变化；报告将明确标记风险。")
            except Exception as exc:
                errors.append(f"空间成果版本不可读取：{selected}（{exc}）")
        else:
            warnings.append("当前没有正式空间成果版本；报告只能引用逐测线空间曲线。")
        state = load_project_state(self.project.root)
        if bool((state.get("dirty") or {}).get("spatial")):
            warnings.append("项目空间成果状态为需刷新。")
        return {
            "schema": "mygpr.report_preflight.v1",
            "spatial_result_id": selected,
            "warnings": warnings,
            "errors": errors,
            "passed": not errors,
        }

    def create_report(
        self,
        *,
        title: str = "基覆界面识别成果报告",
        report_revision: str = "R1",
        template_version: str = "field-industry-v2",
        compiler: str = "",
        reviewer: str = "",
        approver: str = "",
        spatial_result_id: str | None = None,
        cancel_checker=None,
        progress_callback=None,
    ) -> ReportVersionRecord:
        self.project.assert_writable()
        preflight = self.preflight(spatial_result_id=spatial_result_id)
        if not preflight["passed"]:
            raise ValueError("；".join(preflight["errors"]))
        report_id = f"report_v{self._next_revision():03d}"
        generate_project_report_package(
            self.project,
            package_name=f"versions/{report_id}",
            report_profile={
                "report_id": report_id,
                "title": title,
                "revision": report_revision,
                "template_version": template_version,
                "compiler": compiler,
                "reviewer": reviewer,
                "approver": approver,
                "spatial_result_id": preflight["spatial_result_id"],
                "preflight_warnings": list(preflight["warnings"]),
            },
            cancel_checker=cancel_checker,
            progress_callback=progress_callback,
        )
        self.set_current(report_id)
        ProjectStateTracker(self.project.root).mark_report_generated()
        record = self.load_report(report_id)
        for kind, key in (
            ("report_pdf", "pdf_path"),
            ("report_delivery_zip", "delivery_zip_path"),
            ("report_seal", "seal_path"),
        ):
            rel = str(record.result.get(key) or "")
            if not rel:
                continue
            try:
                self.project.register_project_export(
                    rel,
                    export_kind=kind,
                    metadata={
                        "report_id": record.report_id,
                        "snapshot_id": record.snapshot_id,
                        "spatial_result_id": record.spatial_result_id,
                    },
                )
            except Exception as exc:
                self.project.append_log(f"报告成果目录登记失败 {rel}: {exc}")
        return record

    def verify(self, report_id: str) -> tuple[bool, list[str]]:
        record = self.load_report(report_id)
        seal_rel = str(record.result.get("seal_path") or "")
        if not seal_rel:
            return False, ["missing:report_seal.json"]
        report_dir = ensure_direct_child(self.root, self.root / record.report_id)
        seal_path = resolve_managed_path(self.project.root, seal_rel, require_file=True)
        if seal_path.parent != report_dir:
            return False, ["unsafe:seal_path"]
        return ReportSealer().verify(seal_path)

    def delivery_zip_path(self, report_id: str) -> Path:
        record = self.load_report(report_id)
        rel = str(record.result.get("delivery_zip_path") or "")
        if not rel:
            raise FileNotFoundError("delivery zip is not registered")
        path = resolve_managed_path(self.project.root, rel, require_file=True)
        delivery_root = (Path(self.project.root) / "reports" / "delivery").resolve()
        try:
            path.relative_to(delivery_root)
        except ValueError as exc:
            raise ValueError(f"交付包路径越界：{rel}") from exc
        return path

    def _next_revision(self) -> int:
        pattern = re.compile(r"^report_v(\d+)$")
        revisions = []
        for record in self.list_reports():
            match = pattern.fullmatch(record.report_id)
            if match:
                revisions.append(int(match.group(1)))
        return max(revisions, default=0) + 1

    def _stale_state(self, payload: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        summary = dict(payload.get("summary") or {})
        snapshot = dict(summary.get("snapshot") or {})
        binding = dict(summary.get("source_binding") or {})
        state = load_project_state(self.project.root)
        current_revision = int(state.get("data_revision") or 0)
        frozen_revision = int(snapshot.get("project_revision") or 0)
        if current_revision != frozen_revision:
            reasons.append(f"项目数据修订已由 {frozen_revision} 变为 {current_revision}")
        spatial_result_id = str(binding.get("spatial_result_id") or "")
        if spatial_result_id:
            try:
                spatial = SpatialResultVersionService(self.project).load_result(spatial_result_id)
                if spatial.stale:
                    reasons.append(f"空间成果 {spatial_result_id} 的来源已变化")
                expected = str(binding.get("spatial_manifest_sha256") or "")
                manifest = Path(self.project.root) / "spatial" / "results" / spatial_result_id / "manifest.json"
                if expected and _sha256_file(manifest) != expected:
                    reasons.append(f"空间成果 {spatial_result_id} 清单校验不一致")
            except Exception:
                reasons.append(f"空间成果 {spatial_result_id} 已缺失或不可读取")
        return bool(reasons), reasons

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def _safe_report_id(value: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"report_v\d{3,}", text):
        raise ValueError(f"invalid report id: {value!r}")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "REPORT_VERSION_INDEX_SCHEMA",
    "REPORT_VERSION_SCHEMA",
    "ReportPackageVersionService",
    "ReportVersionRecord",
]

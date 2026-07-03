#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Field delivery checks, report, file index, and checksum package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from core.interpretation_service import InterpretationService
from core.project_service import ProjectService, atomic_write_json, utc_now
from core.qc_service import QcService
from core.spatial_synthesis_service import SpatialSynthesisService
from core.user_labels import delivery_role_label, qc_code_label, severity_label


class DeliveryService:
    def __init__(self, project: ProjectService):
        self.project = project

    def run_checks(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        interpretation_count = 0
        for line in self.project.list_lines():
            report = QcService(self.project).run_line_qc(line.line_id)
            for item in report.items:
                if item.severity == "error":
                    items.append(
                        {
                            "code": item.code,
                            "severity": "error",
                            "line_id": line.line_id,
                            "message": item.message,
                        }
                    )
                elif item.severity == "warning" and not item.acknowledged:
                    items.append(
                        {
                            "code": item.code,
                            "severity": "warning",
                            "line_id": line.line_id,
                            "message": item.message,
                        }
                    )
            interpretation_count += len(
                InterpretationService(self.project).list_features(line.line_id)
            )
        result_count = len(self.project.list_processing_results())
        if result_count == 0:
            items.append(
                {
                    "code": "no_processing_result",
                    "severity": "warning",
                    "line_id": None,
                    "message": "项目尚无保存的处理结果。",
                }
            )
        if interpretation_count == 0:
            items.append(
                {
                    "code": "no_interpretation",
                    "severity": "warning",
                    "line_id": None,
                    "message": "项目尚无目标标注。",
                }
            )
        return {
            "schema": "mygpr.delivery_checks.v1",
            "created_at": utc_now(),
            "items": items,
            "summary": {
                "line_count": len(self.project.manifest.line_ids),
                "result_count": result_count,
                "interpretation_count": interpretation_count,
                "error_count": sum(item["severity"] == "error" for item in items),
                "warning_count": sum(item["severity"] == "warning" for item in items),
            },
        }

    def build_package(self, name: str) -> Path:
        checks = self.run_checks()
        if checks["summary"]["error_count"]:
            raise RuntimeError("成果检查存在阻断错误，无法生成交付包")
        package_name = _safe_name(name)
        package = self.project.resolve_relative_path(f"exports/{package_name}")
        package.mkdir(parents=True, exist_ok=True)
        spatial_path = package / "spatial_synthesis.json"
        atomic_write_json(spatial_path, SpatialSynthesisService(self.project).build())
        evidence = self._build_evidence_index()
        manifest = {
            "schema": "mygpr.delivery.v1",
            "created_at": utc_now(),
            "project": self.project.manifest.to_dict(),
            "checks": checks,
            "evidence": evidence,
            "spatial_synthesis": spatial_path.relative_to(self.project.root).as_posix(),
        }
        atomic_write_json(package / "manifest.json", manifest)
        (package / "report.md").write_text(
            self._build_report(manifest),
            encoding="utf-8",
        )
        checksums = self._build_checksums(package)
        (package / "checksums.sha256").write_text(checksums, encoding="utf-8")
        return package

    def _build_evidence_index(self) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for line in self.project.list_lines():
            evidence.append(
                {
                    "role": "line_record",
                    "line_id": line.line_id,
                    "path": f"lines/{line.line_id}/line.json",
                }
            )
            qc_path = self.project.resolve_relative_path(f"qc/{line.line_id}/latest.json")
            if qc_path.exists():
                evidence.append(
                    {
                        "role": "qc_report",
                        "line_id": line.line_id,
                        "path": qc_path.relative_to(self.project.root).as_posix(),
                    }
                )
            interpretation = self.project.resolve_relative_path(
                f"interpretations/{line.line_id}.geojson"
            )
            if interpretation.exists():
                evidence.append(
                    {
                        "role": "interpretation",
                        "line_id": line.line_id,
                        "path": interpretation.relative_to(self.project.root).as_posix(),
                    }
                )
        for result in self.project.list_processing_results():
            evidence.append(
                {
                    "role": "processing_result",
                    "line_id": result.line_id,
                    "result_id": result.result_id,
                    "path": f"results/{result.line_id}/{result.result_id}/result.json",
                    "data_path": result.data_path,
                }
            )
        return evidence

    def _build_report(self, manifest: dict[str, Any]) -> str:
        summary = manifest["checks"]["summary"]
        lines = [
            f"# {self.project.manifest.name} 成果交付报告",
            "",
            f"- 生成时间：{manifest['created_at']}",
            f"- 测线数量：{summary['line_count']}",
            f"- 处理结果：{summary['result_count']}",
            f"- 目标标注：{summary['interpretation_count']}",
            f"- 阻断错误：{summary['error_count']}",
            f"- 待复核警告：{summary['warning_count']}",
            "",
            "## 成果检查",
            "",
        ]
        if not manifest["checks"]["items"]:
            lines.append("- 未发现检查项。")
        else:
            for item in manifest["checks"]["items"]:
                lines.append(
                    f"- [{severity_label(item['severity'])}] {qc_code_label(item['code'])}：{item['message']}"
                )
        lines.extend(["", "## 交付文件索引", ""])
        for item in manifest["evidence"]:
            lines.append(f"- {delivery_role_label(item['role'])}：`{item['path']}`")
        lines.extend(
            [
                "",
                "## 声明边界",
                "",
                "- 本报告索引工程内正式记录，不修改原始数据。",
                "- 无空间定位测线不会被赋予推测坐标。",
                "- 目标标注的置信度和来源处理结果保留在 GeoJSON 属性中。",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _build_checksums(package: Path) -> str:
        rows: list[str] = []
        for path in sorted(item for item in package.rglob("*") if item.is_file()):
            if path.name == "checksums.sha256":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(package).as_posix()}")
        return "\n".join(rows) + "\n"


def _safe_name(value: str) -> str:
    text = re.sub(r"[^\w._-]+", "_", str(value).strip(), flags=re.UNICODE)
    return text.strip("._") or "项目成果"


__all__ = ["DeliveryService"]

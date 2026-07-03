#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-line ingest quality control with graded findings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.gpr_io import auto_load_data
from core.project_models import QcItem, QcReportV1
from core.project_service import ProjectService, atomic_write_json, sha256_path, utc_now
from core.user_labels import sidecar_label
from core.sidecar_parsers import parse_sidecar_csv


class QcService:
    def __init__(self, project: ProjectService):
        self.project = project

    def run_line_qc(self, line_id: str) -> QcReportV1:
        line = self.project.get_line(line_id)
        items: list[QcItem] = []
        primary = line.raw_files[0] if line.raw_files else None
        if primary is None:
            items.append(QcItem("missing_primary", "error", "测线没有主雷达数据。"))
            return self._save(line_id, items)

        path = self._resolve_ref(primary.path)
        if not path.exists():
            items.append(QcItem("missing_raw", "error", "原始数据文件不存在。", {"path": str(path)}))
            return self._save(line_id, items)

        for raw_ref in line.raw_files:
            raw_path = self._resolve_ref(raw_ref.path)
            if not raw_path.exists():
                items.append(
                    QcItem(
                        "missing_raw",
                        "error",
                        "原始数据文件不存在。",
                        {"path": str(raw_path), "role": raw_ref.role},
                    )
                )
                continue
            if raw_ref.sha256:
                current_hash = sha256_path(raw_path)
                if current_hash != raw_ref.sha256:
                    items.append(
                        QcItem(
                            "raw_integrity_mismatch",
                            "error",
                            "原始数据完整性校验失败，文件可能已被修改。",
                            {
                                "expected": raw_ref.sha256,
                                "actual": current_hash,
                                "role": raw_ref.role,
                                "path": str(raw_path),
                            },
                        )
                    )
            else:
                items.append(
                    QcItem(
                        "raw_integrity_pending",
                        "warning",
                        "原始数据尚未完成完整哈希验证。",
                        {"role": raw_ref.role, "path": str(raw_path)},
                    )
                )

        try:
            loaded = auto_load_data(str(path))
            data = np.asarray(loaded.get("data"))
            if data.ndim != 2 or not data.size:
                items.append(QcItem("invalid_matrix", "error", "主数据不是有效的二维 B-scan 矩阵。"))
            else:
                finite_ratio = float(np.isfinite(data).sum() / data.size)
                if finite_ratio < 1.0:
                    severity = "error" if finite_ratio < 0.95 else "warning"
                    items.append(
                        QcItem(
                            "non_finite_samples",
                            severity,
                            "数据包含非有限采样值。",
                            {"finite_ratio": finite_ratio},
                        )
                    )
                items.append(
                    QcItem(
                        "matrix_shape",
                        "info",
                        "主数据结构可读取。",
                        {"samples": int(data.shape[0]), "traces": int(data.shape[1])},
                    )
                )
                metadata = loaded.get("trace_metadata") or {}
                if line.source_format == "mygpr_csv" and not metadata:
                    items.append(
                        QcItem(
                            "airborne_metadata_missing",
                            "warning",
                            "未发现可用于空间关联的逐道实测元数据。",
                        )
                    )
        except Exception as exc:
            items.append(QcItem("unreadable_raw", "error", f"主数据无法读取：{exc}"))

        if line.source_format in {"mygpr_csv", "ascan_folder"}:
            for kind in ("rtk", "imu", "altimeter"):
                sidecar_value = line.sidecars.get(kind)
                if not sidecar_value:
                    items.append(QcItem(f"{kind}_missing", "warning", f"未发现 {sidecar_label(kind)} 辅助文件。"))
                    continue
                sidecar_path = self._resolve_ref(sidecar_value)
                if not sidecar_path.exists():
                    items.append(QcItem(f"{kind}_missing", "warning", f"{sidecar_label(kind)} 辅助文件不存在。"))
                    continue
                try:
                    payload = parse_sidecar_csv(sidecar_path, kind=kind)
                    count = int(np.asarray(payload.get("timestamp_s", [])).size)
                    items.append(
                        QcItem(
                            f"{kind}_valid",
                            "info",
                            f"{sidecar_label(kind)} 辅助文件结构有效。",
                            {"records": count, "path": str(sidecar_path)},
                        )
                    )
                except Exception as exc:
                    items.append(
                        QcItem(
                            f"{kind}_invalid",
                            "warning",
                            f"{sidecar_label(kind)} 辅助文件无法规范化：{exc}",
                            {"path": str(sidecar_path)},
                        )
                    )
        return self._save(line_id, items)

    def acknowledge_warning(self, line_id: str, code: str, note: str) -> QcReportV1:
        note = str(note).strip()
        if not note:
            raise ValueError("请填写现场确认说明。")
        path = self.project.resolve_relative_path(f"qc/{line_id}/latest.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = [
            QcItem(
                code=item["code"],
                severity=item["severity"],
                message=item["message"],
                evidence=dict(item.get("evidence") or {}),
                acknowledged=bool(item.get("acknowledged", False)),
                acknowledgement_note=str(item.get("acknowledgement_note") or ""),
            )
            for item in payload.get("items", [])
        ]
        matched = False
        for item in items:
            if item.code == code and item.severity == "warning":
                item.acknowledged = True
                item.acknowledgement_note = note
                matched = True
        if not matched:
            raise KeyError(f"未找到可确认的质控警告：{code}")
        return self._save(line_id, items)

    def _resolve_ref(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project.resolve_relative_path(path)

    def _save(self, line_id: str, items: list[QcItem]) -> QcReportV1:
        self._restore_acknowledgements(line_id, items)
        report = QcReportV1(line_id=line_id, items=items, created_at=utc_now())
        atomic_write_json(
            self.project.resolve_relative_path(f"qc/{line_id}/latest.json"),
            report.to_dict(),
        )
        return report

    def _restore_acknowledgements(self, line_id: str, items: list[QcItem]) -> None:
        path = self.project.resolve_relative_path(f"qc/{line_id}/latest.json")
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        previous = {
            self._warning_signature(item): str(item.get("acknowledgement_note") or "")
            for item in payload.get("items", [])
            if isinstance(item, dict)
            and item.get("severity") == "warning"
            and bool(item.get("acknowledged"))
            and str(item.get("acknowledgement_note") or "").strip()
        }
        for item in items:
            if item.severity != "warning" or item.acknowledged:
                continue
            note = previous.get(self._warning_signature(item.to_dict()))
            if note:
                item.acknowledged = True
                item.acknowledgement_note = note

    @staticmethod
    def _warning_signature(item: dict[str, object]) -> str:
        payload = {
            "code": item.get("code"),
            "message": item.get("message"),
            "evidence": item.get("evidence") or {},
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


__all__ = ["QcService"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target annotation CSV persistence mixin for field projects."""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any

from core.field_project_models import TARGET_FIELDS, local_now, validate_line_id

_TARGET_COLORS = ["#25B26B", "#7C4DFF", "#F04444", "#2B86F6", "#F5A623", "#0D91B2", "#E879F9", "#22C55E"]


class FieldTargetStoreMixin:
    """Manage target annotation files under ``targets/``."""

    def targets_path(self, line_id: str) -> Path:
        safe_line_id = validate_line_id(line_id)
        return self.root / "targets" / f"{safe_line_id}_targets.csv"

    def default_targets(self, line_id: str) -> list[dict[str, Any]]:
        now = local_now()
        source = f"{line_id}_raw"
        return [
            self._target_payload("T-01", line_id, 18.62, 1.35, "疑似管线", "★★★★☆", "已确认", "疑似电缆管线，走向近似垂直", now, source),
            self._target_payload("T-02", line_id, 62.47, 2.02, "疑似空洞", "★★★☆☆", "待复核", "振幅弱，双曲线特征不连续", now, source),
            self._target_payload("T-03", line_id, 96.83, 1.60, "疑似排水管", "★★★★☆", "已确认", "双曲线征清晰，尺寸较大", now, source),
            self._target_payload("T-04", line_id, 142.18, 1.25, "疑似管线", "★★★☆☆", "待复核", "信号中等，建议开挖验证", now, source),
            self._target_payload("T-05", line_id, 179.41, 1.90, "疑似结构物", "★★★☆☆", "待确认", "可能为检查井或井室结构", now, source),
        ]

    @staticmethod
    def _target_payload(target_id: str, line_id: str, distance_m: float, depth_m: float, target_type: str, confidence: str, status: str, note: str, ts: str, source: str) -> dict[str, Any]:
        x = 345516.0 + float(distance_m)
        y = 3845030.0 + float(distance_m) * 0.92
        return {
            "target_id": target_id,
            "line_id": line_id,
            "distance_m": round(float(distance_m), 3),
            "depth_m": round(float(depth_m), 3),
            "x": round(x, 3),
            "y": round(y, 3),
            "type": target_type,
            "confidence": confidence,
            "status": status,
            "note": note,
            "created_at": ts,
            "updated_at": ts,
            "source_result_id": source,
        }

    @staticmethod
    def _json_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)

    def save_targets(self, line_id: str, targets: list[dict[str, Any]]) -> Path:
        safe_line_id = validate_line_id(line_id)
        path = self.targets_path(safe_line_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self._normalize_target_for_csv(safe_line_id, target) for target in targets]
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=TARGET_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
        try:
            line = self.get_line(safe_line_id)
            line.target_count = len(rows)
            line.updated_at = local_now()
            self.upsert_line(line)
        except KeyError:
            pass
        self.export_spatial_targets_xy(safe_line_id)
        self.append_log(f"保存目标标注 {safe_line_id}: {len(rows)} rows")
        return path

    def _normalize_target_for_csv(self, line_id: str, target: dict[str, Any]) -> dict[str, Any]:
        now = local_now()
        target_id = str(target.get("target_id") or target.get("name") or f"T-{uuid.uuid4().hex[:4]}")
        distance_m = float(target.get("distance_m", target.get("mileage", 0.0)))
        depth_m = float(target.get("depth_m", target.get("depth", 0.0)))
        if target.get("x") not in (None, "") and target.get("y") not in (None, ""):
            x: float | str = float(target.get("x"))
            y: float | str = float(target.get("y"))
        else:
            try:
                point = self.load_trajectory(line_id).interpolate(distance_m)
                x = float(point.x)
                y = float(point.y)
            except Exception:
                x = ""
                y = ""
        return {
            "target_id": target_id,
            "line_id": line_id,
            "distance_m": f"{distance_m:.3f}",
            "depth_m": f"{depth_m:.3f}",
            "x": f"{x:.3f}" if isinstance(x, float) else "",
            "y": f"{y:.3f}" if isinstance(y, float) else "",
            "type": str(target.get("type", "疑似管线")),
            "confidence": str(target.get("confidence", "★★★☆☆")),
            "status": str(target.get("status", "待确认")),
            "note": str(target.get("note", "")),
            "created_at": str(target.get("created_at", now)),
            "updated_at": now,
            "source_result_id": str(target.get("source_result_id", f"{line_id}_raw")),
            "source_mode": str(target.get("source_mode", "raw")),
            "source_data_path": str(target.get("source_data_path", "")),
            "source_manifest_path": str(target.get("source_manifest_path", "")),
            "source_method_id": str(target.get("source_method_id", "raw")),
            "source_method_name": str(target.get("source_method_name", "原始 B-scan")),
            "source_artifact_role": str(target.get("source_artifact_role", "raw_data")),
            "source_axis_transform": self._json_text(target.get("source_axis_transform", "")),
            "source_input_shape": self._json_text(target.get("source_input_shape", "")),
            "source_output_shape": self._json_text(target.get("source_output_shape", "")),
        }

    def load_targets(self, line_id: str) -> list[dict[str, Any]]:
        safe_line_id = validate_line_id(line_id)
        path = self.targets_path(safe_line_id)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        targets: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            target_id = row.get("target_id", f"T-{idx+1:02d}")
            targets.append(
                {
                    "name": target_id,
                    "target_id": target_id,
                    "line_id": row.get("line_id", safe_line_id),
                    "type": row.get("type", "疑似管线"),
                    "mileage": float(row.get("distance_m") or 0.0),
                    "depth": float(row.get("depth_m") or 0.0),
                    "x": float(row.get("x")) if row.get("x") not in (None, "") else "",
                    "y": float(row.get("y")) if row.get("y") not in (None, "") else "",
                    "confidence": row.get("confidence", "★★★☆☆"),
                    "status": row.get("status", "待确认"),
                    "note": row.get("note", ""),
                    "created_at": row.get("created_at", ""),
                    "updated_at": row.get("updated_at", ""),
                    "source_result_id": row.get("source_result_id", f"{safe_line_id}_raw"),
                    "source_mode": row.get("source_mode", "raw"),
                    "source_data_path": row.get("source_data_path", ""),
                    "source_manifest_path": row.get("source_manifest_path", ""),
                    "source_method_id": row.get("source_method_id", "raw"),
                    "source_method_name": row.get("source_method_name", "原始 B-scan"),
                    "source_artifact_role": row.get("source_artifact_role", "raw_data"),
                    "source_axis_transform": row.get("source_axis_transform", ""),
                    "source_input_shape": row.get("source_input_shape", ""),
                    "source_output_shape": row.get("source_output_shape", ""),
                    "color": _TARGET_COLORS[idx % len(_TARGET_COLORS)],
                    "width": 100 if target_id == "T-03" else 52,
                    "height": 78 if target_id == "T-03" else 58,
                }
            )
        return targets


__all__ = ["FieldTargetStoreMixin"]

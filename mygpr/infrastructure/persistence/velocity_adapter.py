#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Velocity model persistence mixin.

双曲线拟合结果写回测线：ε 与深度轴由 GPRDataSet 从 metadata 中的
velocity_analysis 证据自动恢复（gpr_data_model.__post_init__ 链路），
因此保存走完整 GPRDataSet 原子重写（transpose 同款、低频操作可接受）。
"""
from __future__ import annotations

from typing import Any, Mapping

from core.gpr_data_model import GPRDataSet

__all__ = ["VelocityPersistenceMixin"]


class VelocityPersistenceMixin:
    """ProjectSessionPort velocity methods backed by FieldProjectStore."""

    def save_velocity_model(
        self,
        line_id: str,
        v_m_ns: float,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        from core.field_project_models import validate_line_id
        from mygpr.domain.velocity.errors import VelocityAnalysisError

        safe = validate_line_id(line_id)
        try:
            velocity = float(v_m_ns)
        except (TypeError, ValueError) as exc:
            raise VelocityAnalysisError(
                f"速度值无效：{v_m_ns!r}", hint="速度必须为正实数（m/ns）。") from exc
        if not velocity > 0 or velocity >= 0.299792458:
            raise VelocityAnalysisError(
                f"速度超出物理范围：{velocity:.4g} m/ns（应在 (0, c) 内）。",
                hint="土壤相对介电常数 ε>1，对应 v<c≈0.2998 m/ns。")

        if evidence is None:
            from mygpr.domain.velocity.models import VELOCITY_ANALYSIS_EVIDENCE_SCHEMA
            from mygpr.application.velocity.evidence import compute_velocity_body_digest

            epsilon = (0.299792458 / velocity) ** 2
            body = {"v_m_ns": velocity, "dielectric_constant": float(epsilon)}
            evidence = {
                "schema": VELOCITY_ANALYSIS_EVIDENCE_SCHEMA,
                "line_id": safe,
                "created_at": None,
                "body": body,
                "body_sha256": compute_velocity_body_digest(body),
            }
        payload = dict(evidence)
        from core.gpr_data_model import time_to_depth_axis

        with self._lock:
            dataset = self._store.load_gpr_dataset(safe)
            epsilon = float(payload["body"]["dielectric_constant"])
            if not epsilon > 1.0:
                raise VelocityAnalysisError(
                    f"介电常数必须 > 1，收到 {epsilon:.4g}。",
                    hint="ε 由拟合速度换算：ε=(c/v)²，速度过快说明拾取有误。")
            metadata = dict(dataset.metadata or {})
            metadata["velocity_analysis"] = payload
            rebuilt = GPRDataSet(
                line_id=dataset.line_id,
                matrix=dataset.matrix,
                distance_axis_m=dataset.distance_axis_m,
                time_axis_ns=dataset.time_axis_ns,
                depth_axis_m=time_to_depth_axis(dataset.time_axis_ns, epsilon),
                time_window_ns=dataset.time_window_ns,
                dielectric_constant=epsilon,
                metadata=metadata,
            )
            self._store.save_gpr_dataset(safe, rebuilt)

    def load_velocity_model(self, line_id: str) -> Mapping[str, Any] | None:
        from core.field_project_models import validate_line_id

        safe = validate_line_id(line_id)
        with self._lock:
            dataset = self._store.load_gpr_dataset(safe)
            metadata = dict(getattr(dataset, "metadata", None) or {})
            # to_metadata()（asdict 路径）把用户 metadata 嵌套在 "metadata" 键下：
            # 经存储往返后 velocity_analysis 可能位于顶层或嵌套层，两处都读。
            payload = metadata.get("velocity_analysis")
            if not isinstance(payload, Mapping):
                nested = metadata.get("metadata")
                payload = nested.get("velocity_analysis") if isinstance(nested, Mapping) else None
            if isinstance(payload, Mapping):
                return dict(payload)
            return None

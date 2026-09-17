#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Velocity analysis application service.

编排：拾取点 → domain 双曲线拟合 → 证据链构建 → 经 ProjectSessionPort
写回测线速度模型（ε 与深度轴）。无 Qt 依赖；持久化细节留在 infrastructure。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.project.service import ProjectService
from mygpr.domain.velocity.errors import VelocityAnalysisError
from mygpr.domain.velocity.fitting import fit_hyperbola
from mygpr.domain.velocity.models import VelocityPick

from mygpr.application.velocity.evidence import build_velocity_evidence

__all__ = ["VelocityAnalysisService"]


class VelocityAnalysisService:
    """Project-scoped hyperbola velocity analysis use cases."""

    def __init__(self, projects: ProjectService) -> None:
        self._projects = projects

    def analyze(
        self,
        project_id: str,
        line_id: str,
        picks: Sequence[Mapping[str, Any] | VelocityPick],
        *,
        apply: bool = True,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """拟合双曲线并（默认）写回测线速度模型。

        picks 项支持 VelocityPick 或 {"trace_index", "sample_index"} 映射
        （x/t 由 header_info 轴解析；显式给 x_m/t_ns 时直接采用）。

        Returns:
            {"fit": HyperbolaFit, "evidence": dict, "applied": bool,
             "dielectric_constant": float | None}

        Raises:
            VelocityAnalysisError: 拾取无效、轴缺失或拟合非物理。
        """
        if context is not None:
            context.raise_if_cancelled()
        normalized = [self._normalize_pick(item) for item in (picks or ())]
        if not normalized:
            raise VelocityAnalysisError(
                "未提供任何拾取点。", hint="在 B-scan 上沿绕射双曲线至少拾取 3 个点。")

        dataset = self._projects.read_dataset(project_id, line_id)
        header = dataset.header_info or {}
        distance_axis = self._axis(header, "distance_axis_m")
        time_axis = self._axis(header, "time_axis_ns")
        resolved: list[VelocityPick] = []
        for pick in normalized:
            x_m, t_ns = pick.x_m, pick.t_ns
            if x_m is None:
                x_m = self._axis_value(distance_axis, pick.trace_index, "distance_axis_m")
            if t_ns is None:
                t_ns = self._axis_value(time_axis, pick.sample_index, "time_axis_ns")
            resolved.append(VelocityPick(
                trace_index=pick.trace_index, sample_index=pick.sample_index,
                x_m=float(x_m), t_ns=float(t_ns)))

        if context is not None:
            context.raise_if_cancelled()
        fit = fit_hyperbola(resolved)
        evidence = build_velocity_evidence(
            line_id=line_id, fit=fit, picks=resolved, dataset_shape=dataset.data.shape)

        applied = False
        dielectric: float | None = None
        if apply:
            self._projects.save_velocity_model(
                project_id, line_id, fit.v_m_ns, evidence=evidence)
            dielectric = (0.299792458 / fit.v_m_ns) ** 2
            applied = True
        return {
            "fit": fit,
            "evidence": evidence,
            "applied": applied,
            "dielectric_constant": dielectric,
        }

    @staticmethod
    def _normalize_pick(item: Mapping[str, Any] | VelocityPick) -> VelocityPick:
        if isinstance(item, VelocityPick):
            return item
        if not isinstance(item, Mapping):
            raise VelocityAnalysisError(
                f"拾取点格式无效：{type(item).__name__}。",
                hint="拾取点应为 {trace_index, sample_index} 字典或 VelocityPick。")
        try:
            return VelocityPick(
                trace_index=int(item.get("trace_index", item.get("trace", 0))),
                sample_index=int(item.get("sample_index", item.get("sample", 0))),
                x_m=item.get("x_m"),
                t_ns=item.get("t_ns"),
            )
        except (TypeError, ValueError) as exc:
            raise VelocityAnalysisError(
                f"拾取点字段无法解析：{exc}",
                hint="确认 trace_index/sample_index 为整数。") from exc

    @staticmethod
    def _axis(header: Mapping[str, Any], key: str) -> Any:
        axis = header.get(key)
        if axis is None:
            raise VelocityAnalysisError(
                f"测线 header_info 缺少 {key}，无法把拾取索引解析为物理坐标。",
                hint="先完成数据导入（导入流程会生成里程轴与走时轴）。")
        return axis

    @staticmethod
    def _axis_value(axis: Any, index: int, name: str) -> float:
        try:
            return float(axis[int(index)])
        except (IndexError, TypeError, ValueError) as exc:
            raise VelocityAnalysisError(
                f"{name}[{index}] 越界或不可解析。",
                hint="拾取索引必须落在当前测线范围内。") from exc

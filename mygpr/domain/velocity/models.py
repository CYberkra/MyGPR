#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure domain models for hyperbola velocity analysis."""
from __future__ import annotations

from dataclasses import dataclass

# 证据链 schema 版本（登记 config/schema_catalog.json，immutable-evidence）。
# 结构变更必须升版本号（v1 → v2），不得原地修改字段语义。
VELOCITY_ANALYSIS_EVIDENCE_SCHEMA = "mygpr.velocity_analysis_evidence.v1"


@dataclass(frozen=True, slots=True)
class VelocityPick:
    """一次双曲线拾取：原始数据索引 + 物理坐标。

    x_m 来自 header_info["distance_axis_m"][trace_index]，
    t_ns 来自 header_info["time_axis_ns"][sample_index]。
    """

    trace_index: int
    sample_index: int
    x_m: float
    t_ns: float


@dataclass(frozen=True, slots=True)
class HyperbolaFit:
    """双曲线拟合结果（物理参数化）。

    模型：t(x) = (2/v)·sqrt((x - x0)² + z0²)，v 单位 m/ns。
    与介电常数的关系：ε = (c/v)²，c = 0.299792458 m/ns。
    """

    v_m_ns: float
    x0_m: float
    z0_m: float
    rmse_ns: float
    r_squared: float
    pick_count: int


__all__ = [
    "VELOCITY_ANALYSIS_EVIDENCE_SCHEMA",
    "HyperbolaFit",
    "VelocityPick",
]

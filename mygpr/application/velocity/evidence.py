#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Velocity analysis evidence payload.

与 AutoTune 证据链同构：白名单字段 + body SHA-256 摘要，防止载荷漂移。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Sequence

from mygpr.domain.velocity.models import (
    VELOCITY_ANALYSIS_EVIDENCE_SCHEMA,
    HyperbolaFit,
    VelocityPick,
)

__all__ = ["build_velocity_evidence", "compute_velocity_body_digest"]

_LIGHT_SPEED_M_PER_NS = 0.299792458


def compute_velocity_body_digest(body: Any) -> str:
    """证据摘要与 AutoTune 同源：sorted-keys JSON 的 SHA-256。"""
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_velocity_evidence(
    *,
    line_id: str,
    fit: HyperbolaFit,
    picks: Sequence[VelocityPick],
    dataset_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """构建速度分析证据载荷（JSON-safe，可直接经 JobSnapshot 落盘）。"""
    body = {
        "v_m_ns": float(fit.v_m_ns),
        "x0_m": float(fit.x0_m),
        "z0_m": float(fit.z0_m),
        "rmse_ns": float(fit.rmse_ns),
        "r_squared": float(fit.r_squared),
        "pick_count": int(fit.pick_count),
        "picks": [
            {
                "trace_index": int(p.trace_index),
                "sample_index": int(p.sample_index),
                "x_m": float(p.x_m),
                "t_ns": float(p.t_ns),
            }
            for p in picks
        ],
        "dielectric_constant": float((_LIGHT_SPEED_M_PER_NS / fit.v_m_ns) ** 2),
    }
    return {
        "schema": VELOCITY_ANALYSIS_EVIDENCE_SCHEMA,
        "line_id": str(line_id),
        "data_shape": [int(v) for v in (dataset_shape or ())],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "body": body,
        "body_sha256": compute_velocity_body_digest(body),
    }

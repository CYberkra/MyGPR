#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune 证据导出：把一次参数推荐会话落盘为可审计 JSON。

证据文件包含全部 trials（含无效试验与淘汰原因）、最终推荐、偏好排序审计，
以及结果体 SHA-256——满足"谁在什么数据上推荐了什么参数、为何推荐"的溯源需求。
纯函数、无 Qt 依赖（P1-4）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

AUTOTUNE_EVIDENCE_SCHEMA = "mygpr.autotune_evidence.v1"


def build_autotune_evidence(
    result: dict[str, Any],
    *,
    method_key: str,
    data_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """把 AutoTune 结果体组装成证据载荷（不含落盘）。"""
    payload = dict(result or {})
    public_keys = (
        "method",
        "method_name",
        "recommended_params",
        "best_params",
        "best_score",
        "recommended_profile",
        "top_candidates",
        "preference_audit",
        "trial_count",
        "valid_trial_count",
        "all_trials",
        "runtime_warnings",
        "constraints",
        "stage_summary",
    )
    body = {key: payload.get(key) for key in public_keys if payload.get(key) is not None}
    digest = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    evidence = {
        "schema": AUTOTUNE_EVIDENCE_SCHEMA,
        "method_key": str(method_key),
        "data_shape": [int(v) for v in (data_shape or ())],
        "body": body,
        "body_sha256": digest,
    }
    return evidence


def export_autotune_evidence(
    result: dict[str, Any],
    output_path: str | Path,
    *,
    method_key: str,
    data_shape: tuple[int, int] | None = None,
) -> Path:
    """落盘 AutoTune 证据 JSON；路径父目录自动创建。

    经 ``DurableWritePort`` 原子发布：证据 JSON 是报告资产，中断留下的
    半截文件会被后续读回当成完整证据，故必须走 fsync + 原子替换。

    序列化保留本模块原有的 ``default=str`` 与尾随换行——AutoTune 结果里
    可能有 numpy 标量/Path 等非 JSON 原生类型，而端口的 ``write_json``
    走的是 core 的 ``json.dumps`` 默认参数（无 ``default=str``）。
    """
    evidence = build_autotune_evidence(result, method_key=method_key, data_shape=data_shape)
    out = Path(output_path).expanduser().resolve()
    from mygpr.infrastructure.persistence.durable_write import default_durable_write
    return default_durable_write().write_text(
        out,
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n",
    )


__all__ = [
    "AUTOTUNE_EVIDENCE_SCHEMA",
    "build_autotune_evidence",
    "export_autotune_evidence",
]

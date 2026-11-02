"""Validated normalized-position nodes shared by processing and its curve editor."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def normalize_gain_nodes(value: Any = None) -> list[dict[str, float]]:
    """Return independent, strictly ordered 0..1 / dB nodes; never repair a curve."""
    if value is None:
        return [{"tau": tau, "db": 0.0} for tau in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if isinstance(value, str):
        if len(value) > 16384:
            raise ValueError("增益曲线参数过长")
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("增益曲线必须是节点 JSON 数组") from exc
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= 32:
        raise ValueError("增益曲线必须包含 2～32 个节点")
    nodes = []
    for node in value:
        if not isinstance(node, dict) or set(node) != {"tau", "db"}:
            raise ValueError("增益节点必须包含 tau 和 db")
        try:
            if any(isinstance(node[key], bool) for key in ("tau", "db")):
                raise ValueError
            tau, db = float(node["tau"]), float(node["db"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("增益节点必须是有限数值") from exc
        if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
            raise ValueError("增益节点位置必须在 0%～100% 内")
        if not math.isfinite(db) or not -60.0 <= db <= 60.0:
            raise ValueError("增益必须在 -60～+60 dB 内")
        nodes.append({"tau": tau, "db": db})
    if nodes[0]["tau"] != 0.0 or nodes[-1]["tau"] != 1.0:
        raise ValueError("增益曲线首尾节点必须位于 0% 和 100%")
    if any(right["tau"] <= left["tau"] for left, right in zip(nodes, nodes[1:])):
        raise ValueError("增益节点位置必须严格递增，不能重复")
    return nodes


def encode_gain_nodes(value: Any = None) -> str:
    """Canonical JSON is embedded in the pipeline, never a mutable file reference."""
    return json.dumps(normalize_gain_nodes(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def apply_manual_gain_curve(data: Any, value: Any = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Shared normalization and amplitude rule for processing and live preview."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("手动曲线增益需要非空二维输入")
    warnings: list[dict[str, Any]] = []
    finite = np.isfinite(array)
    if not finite.all():
        fill = float(np.mean(array[finite])) if finite.any() else 0.0
        array = np.nan_to_num(array, nan=fill, posinf=fill, neginf=fill)
        warnings.append({"code": "data_sanitized", "message": "输入数据包含 NaN/Inf，已使用均值填充。",
                         "method_id": "manual_gain", "fill_value": fill})
    nodes = normalize_gain_nodes(value)
    positions = np.linspace(0.0, 1.0, array.shape[0])
    db = np.interp(positions, [node["tau"] for node in nodes], [node["db"] for node in nodes])
    result = array * np.power(10.0, db / 20.0)[:, None]
    if not np.isfinite(result).all() or np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError("手动曲线增益结果超出 float32 范围，请降低增益")
    return result.astype(np.float32), db, warnings

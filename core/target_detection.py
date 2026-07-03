#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replaceable target-candidate detector for GPR B-scans.

The implementation here is deliberately transparent and deterministic.  It is
not the final PGDA-CSNet model; it supplies a stable interface that the model can
replace without changing the UI/project contract.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from core.gpr_data_model import GPRDataSet


@dataclass
class TargetCandidate:
    target_id: str
    line_id: str
    distance_m: float
    depth_m: float
    confidence_score: float
    target_type: str = "疑似管线"
    status: str = "待复核"
    note: str = "算法候选，建议人工复核"

    def to_target_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        score = max(0.0, min(1.0, float(self.confidence_score)))
        if score >= 0.82:
            confidence = "★★★★★"
        elif score >= 0.68:
            confidence = "★★★★☆"
        elif score >= 0.52:
            confidence = "★★★☆☆"
        else:
            confidence = "★★☆☆☆"
        return {
            "target_id": self.target_id,
            "name": self.target_id,
            "line_id": self.line_id,
            "mileage": round(float(self.distance_m), 3),
            "distance_m": round(float(self.distance_m), 3),
            "depth": round(float(self.depth_m), 3),
            "depth_m": round(float(self.depth_m), 3),
            "type": self.target_type,
            "confidence": confidence,
            "status": self.status,
            "note": self.note,
            "source_result_id": "target_detection_v1",
            "confidence_score": score,
        }


def detect_targets(dataset: GPRDataSet, *, max_targets: int = 7, start_index: int = 1) -> list[TargetCandidate]:
    """Detect high-energy hyperbola-like candidate locations.

    MVP heuristic: compute a robust per-trace energy curve over the mid-depth
    band, select local maxima with spacing, then estimate the peak depth.  This
    gives stable candidates on real matrices and deterministic demo data.
    """
    data = dataset.normalized_matrix
    if data.size == 0:
        return []
    rows, cols = data.shape
    r0 = max(8, int(rows * 0.18))
    r1 = max(r0 + 8, int(rows * 0.78))
    band = np.abs(data[r0:r1, :])
    energy = np.percentile(band, 90, axis=0) + 0.35 * np.mean(band, axis=0)
    # Smooth along distance.
    win = max(7, cols // 55)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    smooth = np.convolve(energy, kernel, mode="same")
    threshold = float(np.percentile(smooth, 72))
    candidate_cols: list[int] = []
    min_spacing = max(16, cols // 11)
    order = np.argsort(smooth)[::-1]
    for col in order:
        c = int(col)
        if smooth[c] < threshold:
            break
        if all(abs(c - existing) >= min_spacing for existing in candidate_cols):
            candidate_cols.append(c)
        if len(candidate_cols) >= max_targets:
            break
    candidate_cols.sort()
    out: list[TargetCandidate] = []
    e_min = float(np.min(smooth))
    e_ptp = float(np.ptp(smooth)) or 1.0
    for idx, col in enumerate(candidate_cols, start=start_index):
        col_band = np.abs(data[r0:r1, max(0, col - 2) : min(cols, col + 3)]).mean(axis=1)
        row = r0 + int(np.argmax(col_band))
        distance = float(dataset.distance_axis_m[col]) if col < len(dataset.distance_axis_m) else float(col)
        depth = float(dataset.depth_axis_m[row]) if row < len(dataset.depth_axis_m) else float(row)
        score = (float(smooth[col]) - e_min) / e_ptp
        out.append(
            TargetCandidate(
                target_id=f"A-{idx:02d}",
                line_id=dataset.line_id,
                distance_m=distance,
                depth_m=max(0.35, depth),
                confidence_score=score,
            )
        )
    return out


__all__ = ["TargetCandidate", "detect_targets"]

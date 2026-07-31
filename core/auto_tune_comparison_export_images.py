#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Locked-scale B-scan image rendering for comparison evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.scalar_utils import to_float

def _locked_display_spec(
    manual_arr: np.ndarray,
    auto_arr: np.ndarray,
    source_spec: dict[str, Any] | None,
    *,
    cmap: str,
) -> dict[str, Any]:
    source = dict(source_spec or {})
    clip = source.get("percentile_clip")
    finite_abs = _finite_abs_values(manual_arr, auto_arr)
    if finite_abs.size == 0:
        limit = 1.0
    elif clip is not None:
        percentile = max(0.0, min(to_float(clip, default=100.0), 100.0))
        limit = float(np.percentile(finite_abs, percentile))
    else:
        limit = float(np.max(finite_abs))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    return {
        "locked_scale": True,
        "lock_color_scale": True,
        "normalize": False,
        "percentile_clip": clip,
        "cmap": str(cmap or "gray"),
        "vmin": -limit,
        "vmax": limit,
    }

def _finite_abs_values(*arrays: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for arr in arrays:
        values = np.ravel(np.asarray(arr, dtype=np.float32))
        if values.size == 0:
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(np.abs(finite.astype(np.float64, copy=False)))
    if not chunks:
        return np.asarray([], dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)

def _save_single_bscan(
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    display_spec: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=150)
    try:
        image = ax.imshow(
            np.asarray(data, dtype=np.float32),
            cmap=str(display_spec["cmap"]),
            aspect="auto",
            vmin=float(display_spec["vmin"]),
            vmax=float(display_spec["vmax"]),
        )
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        fig.colorbar(image, ax=ax, shrink=0.82)
        fig.tight_layout()
        fig.savefig(out_path)
    finally:
        plt.close(fig)

def _save_side_by_side(
    manual_data: np.ndarray,
    auto_data: np.ndarray,
    out_path: Path,
    *,
    display_spec: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=150, constrained_layout=True)
    try:
        for ax, arr, title in [
            (axes[0], manual_data, "Manual baseline"),
            (axes[1], auto_data, "Auto-tuned"),
        ]:
            image = ax.imshow(
                np.asarray(arr, dtype=np.float32),
                cmap=str(display_spec["cmap"]),
                aspect="auto",
                vmin=float(display_spec["vmin"]),
                vmax=float(display_spec["vmax"]),
            )
            ax.set_title(title)
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
        fig.savefig(out_path)
    finally:
        plt.close(fig)

__all__ = ['_locked_display_spec', '_finite_abs_values', '_save_single_bscan', '_save_side_by_side']

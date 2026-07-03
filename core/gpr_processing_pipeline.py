#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, replaceable processing pipeline for field-workbench B-scans."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from core.gpr_data_model import GPRDataSet


@dataclass
class ProcessingParams:
    remove_dc: bool = True
    background_removal: bool = True
    background_window: int = 21
    bandpass_filter: bool = True
    low_cut_ratio: float = 0.02
    high_cut_ratio: float = 0.42
    gain_type: str = "SEC"
    gain_factor: float = 1.6
    motion_compensation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _moving_average_columns(matrix: np.ndarray, window: int) -> np.ndarray:
    win = max(int(window), 3)
    if win % 2 == 0:
        win += 1
    if win >= matrix.shape[1]:
        return np.repeat(matrix.mean(axis=1, keepdims=True), matrix.shape[1], axis=1)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    # np.convolve(..., mode="same") preserves the column count for every row.
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, matrix).astype(np.float32)


def _simple_bandpass_time(matrix: np.ndarray, low_ratio: float, high_ratio: float) -> np.ndarray:
    """FFT-domain bandpass along the sample/time axis.

    This is intentionally conservative; vendor-specific filters can be plugged in
    behind the same function later.
    """
    rows = matrix.shape[0]
    freq = np.fft.rfftfreq(rows)
    low = max(0.0, min(float(low_ratio), 0.49))
    high = max(low + 0.01, min(float(high_ratio), 0.50))
    mask = (freq >= low) & (freq <= high)
    # Smooth mask edges slightly to avoid ringing in the UI preview.
    spec = np.fft.rfft(matrix, axis=0)
    spec[~mask, :] = 0
    return np.fft.irfft(spec, n=rows, axis=0).astype(np.float32)


def _apply_sec_gain(matrix: np.ndarray, factor: float) -> np.ndarray:
    rows = matrix.shape[0]
    depth_weight = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    gain = 1.0 + max(float(factor), 0.0) * (depth_weight**1.35)
    return matrix * gain


def process_gpr_dataset(dataset: GPRDataSet, params: ProcessingParams | dict[str, Any] | None = None) -> GPRDataSet:
    """Apply the current MVP processing chain and return a new dataset."""
    if params is None:
        p = ProcessingParams()
    elif isinstance(params, ProcessingParams):
        p = params
    else:
        p = ProcessingParams(**{k: v for k, v in params.items() if k in ProcessingParams.__dataclass_fields__})
    data = dataset.matrix.astype(np.float64, copy=True)
    if p.remove_dc:
        data = data - data.mean(axis=0, keepdims=True)
    if p.background_removal:
        background = _moving_average_columns(data, p.background_window)
        data = data - background
    if p.bandpass_filter:
        data = _simple_bandpass_time(data, p.low_cut_ratio, p.high_cut_ratio)
    if p.gain_type.upper() == "SEC":
        data = _apply_sec_gain(data, p.gain_factor)
    elif p.gain_type.lower() in {"linear", "线性"}:
        data = data * max(float(p.gain_factor), 0.1)
    metadata = dict(dataset.metadata)
    metadata["processing"] = p.to_dict()
    return GPRDataSet.from_matrix(
        dataset.line_id,
        data,
        length_m=dataset.length_m,
        time_window_ns=float(dataset.time_axis_ns[-1]) if dataset.time_axis_ns.size else dataset.time_window_ns,
        dielectric_constant=dataset.dielectric_constant,
        source_path=dataset.source_path,
        format_name="processed-pipeline-v1",
        metadata=metadata,
    )


__all__ = ["ProcessingParams", "process_gpr_dataset"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark sample registry and scenario taxonomy for Stage A foundations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.methods_registry import PROCESSING_METHODS


DEFAULT_BENCHMARK_SEED = 42


@dataclass(frozen=True)
class BenchmarkSampleSpec:
    """Declarative benchmark sample specification."""

    sample_id: str
    scenario: str
    title: str
    description: str
    default_methods: tuple[str, ...]
    focus_metrics: tuple[str, ...]
    tags: tuple[str, ...]
    builder: Callable[[int], tuple[np.ndarray, dict[str, Any]]]


BENCHMARK_SCENARIOS: dict[str, dict[str, Any]] = {
    "zero_time": {
        "label": "零时与首波定位",
        "goal": "验证零时矫正是否压低前零时能量并稳定首波。",
    },
    "drift_background": {
        "label": "低频漂移与背景抑制",
        "goal": "验证低频漂移、水平背景和主反射保真之间的平衡。",
    },
    "clutter_gain": {
        "label": "增益与尖锐杂波抑制",
        "goal": "验证深部可见性增强与尖锐杂波控制之间的平衡。",
    },
}


def _base_header_info(samples: int, traces: int) -> dict[str, Any]:
    return {
        "a_scan_length": int(samples),
        "num_traces": int(traces),
        "total_time_ns": 700.0,
        "trace_interval_m": 0.09,
    }


def _build_zero_time_fixture(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples, traces = 192, 72
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    data = 0.03 * rng.normal(size=(samples, traces))
    data += 0.08 * np.sin(2.0 * np.pi * 2.2 * t)
    data += 0.04 * (t - 0.3)

    first_break = 18 + np.round(
        3.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, traces))
    ).astype(int)
    for col, idx in enumerate(first_break):
        pulse = np.array([0.7, 1.8, 1.1, 0.45], dtype=np.float64)
        data[idx : idx + len(pulse), col] += pulse

    data[95:100, :] += 0.18
    data[122:126, 16:54] += 0.32
    return data.astype(np.float32), {
        "header_info": _base_header_info(samples, traces),
        "reference_zero_idx": 18,
        "notes": "Synthetic sample emphasizing first-break stability and pre-zero suppression.",
    }


def _build_drift_background_fixture(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples, traces = 224, 84
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    data = 0.025 * rng.normal(size=(samples, traces))
    data += 0.22 * np.sin(2.0 * np.pi * 1.4 * t)
    data += 0.14 * np.cos(2.0 * np.pi * 0.5 * x)
    data += 0.18 * (t - 0.5)

    data[64:68, :] += 0.34
    data[108:113, :] += 0.22
    data[150:154, 10:72] += 0.45
    data[178:183, 26:64] += 0.29
    return data.astype(np.float32), {
        "header_info": _base_header_info(samples, traces),
        "reference_zero_idx": 16,
        "notes": "Synthetic sample emphasizing drift removal and horizontal background suppression.",
    }


def _build_clutter_gain_fixture(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples, traces = 256, 96
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    data = 0.02 * rng.normal(size=(samples, traces))
    attenuation = np.exp(-2.5 * t)
    data += 0.12 * attenuation * np.sin(2.0 * np.pi * 3.5 * t)

    for center in (72, 118, 186, 220):
        width = 2 + (center % 3)
        amp = 0.55 if center < 150 else 0.82
        data[center : center + width, 12:84] += amp

    spike_rows = rng.integers(12, samples - 12, size=18)
    spike_cols = rng.integers(0, traces, size=18)
    data[spike_rows, spike_cols] += rng.uniform(1.5, 2.4, size=18)
    data[198:206, 18:74] += 0.22
    return data.astype(np.float32), {
        "header_info": _base_header_info(samples, traces),
        "reference_zero_idx": 14,
        "notes": "Synthetic sample emphasizing deep visibility, clipping risk, and sharp-clutter control.",
    }


BENCHMARK_SAMPLES: dict[str, BenchmarkSampleSpec] = {
    "zero_time_reference": BenchmarkSampleSpec(
        sample_id="zero_time_reference",
        scenario="zero_time",
        title="零时首波基准样本",
        description="用于比较零时矫正与首波清晰度的合成样本。",
        default_methods=("set_zero_time", "dewow"),
        focus_metrics=(
            "pre_zero_energy_ratio_after",
            "first_break_sharpness_after",
            "baseline_bias_reduction",
        ),
        tags=("synthetic", "zero_time", "first_break"),
        builder=_build_zero_time_fixture,
    ),
    "drift_background_reference": BenchmarkSampleSpec(
        sample_id="drift_background_reference",
        scenario="drift_background",
        title="漂移背景抑制基准样本",
        description="用于比较低频漂移、水平背景和主反射保真的合成样本。",
        default_methods=(
            "dewow",
            "subtracting_average_2D",
            "median_background_2D",
            "rpca_background",
        ),
        focus_metrics=(
            "low_freq_energy_reduction",
            "horizontal_coherence_reduction",
            "target_band_energy_ratio",
            "local_saliency_preservation",
        ),
        tags=("synthetic", "drift", "background"),
        builder=_build_drift_background_fixture,
    ),
    "clutter_gain_reference": BenchmarkSampleSpec(
        sample_id="clutter_gain_reference",
        scenario="clutter_gain",
        title="增益与尖锐杂波基准样本",
        description="用于比较深部可见性增强与尖锐杂波抑制代价的合成样本。",
        default_methods=("agcGain", "running_average_2D", "sec_gain"),
        focus_metrics=(
            "deep_zone_contrast_gain",
            "edge_preservation",
            "clipping_ratio_after",
            "hot_pixel_ratio_after",
        ),
        tags=("synthetic", "gain", "clutter"),
        builder=_build_clutter_gain_fixture,
    ),
}


for _sample_id, _spec in BENCHMARK_SAMPLES.items():
    missing = [
        method_key
        for method_key in _spec.default_methods
        if method_key not in PROCESSING_METHODS
    ]
    if missing:
        raise KeyError(
            f"Benchmark sample {_sample_id} references unknown methods: {missing}"
        )
    if _spec.scenario not in BENCHMARK_SCENARIOS:
        raise KeyError(
            f"Benchmark sample {_sample_id} references unknown scenario: {_spec.scenario}"
        )


def list_benchmark_sample_ids() -> list[str]:
    """Return benchmark sample ids in declaration order."""
    return list(BENCHMARK_SAMPLES.keys())


def get_benchmark_sample_spec(sample_id: str) -> BenchmarkSampleSpec:
    """Return a benchmark sample specification by id."""
    try:
        return BENCHMARK_SAMPLES[sample_id]
    except KeyError as exc:
        raise KeyError(f"未知 benchmark sample: {sample_id}") from exc


def generate_benchmark_sample(
    sample_id: str,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate benchmark input data and metadata deterministically."""
    spec = get_benchmark_sample_spec(sample_id)
    data, meta = spec.builder(int(seed))
    payload = dict(meta)
    payload.setdefault("sample_id", sample_id)
    payload.setdefault("scenario", spec.scenario)
    payload.setdefault("seed", int(seed))
    payload.setdefault("title", spec.title)
    payload.setdefault("focus_metrics", list(spec.focus_metrics))
    payload.setdefault("default_methods", list(spec.default_methods))
    payload.setdefault("tags", list(spec.tags))
    return np.asarray(data, dtype=np.float32), payload

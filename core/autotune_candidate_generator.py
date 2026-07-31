#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune V1 bounded candidate parameter generator.

This module turns the V1 final-candidate profile/recipe contract into a small,
auditable candidate space.  It is deliberately *not* a workflow search engine and
it does not execute any processing algorithm.  Its output is intended for trial
planning, manifest/export records and later backend integration.

Design rules:
- fixed candidates provide reproducible lower/upper bounds;
- lightweight data diagnostics add a few adaptive candidates;
- profile caps keep interface/landslide/deep targets conservative;
- display-only transforms are explicitly marked and excluded from synthetic
  full-reference scoring;
- every generated space has a deterministic candidate_space_hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from core.autotune_v1_config import AutoTuneV1Config, load_autotune_v1_config


@dataclass(frozen=True)
class CandidateFeatureSummary:
    """Lightweight data descriptors used to generate bounded candidates."""

    n_samples: int | None = None
    n_traces: int | None = None
    dt_seconds: float | None = None
    sample_rate_hz: float | None = None
    nyquist_hz: float | None = None
    center_frequency_hz: float | None = None
    dominant_frequency_hz: float | None = None
    lateral_correlation_length_traces: int | None = None
    singular_elbow_rank: int | None = None
    attenuation_ratio: float | None = None
    spikiness: float | None = None
    hot_pixel_ratio: float | None = None
    trace_spacing_m: float | None = None
    target_lateral_scale_m: float | None = None
    source: str = "metadata_and_data_diagnostics"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedCandidate:
    """Single auditable processing-parameter candidate."""

    category: str
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    candidate_group: str = ""
    source: str = "fixed"
    metric_safe: bool = True
    display_only: bool = False
    profile_limited: bool = False
    rationale: str = ""
    warnings: tuple[str, ...] = ()

    def stable_key(self) -> str:
        payload = {
            "category": self.category,
            "method": self.method,
            "parameters": self.parameters,
            "candidate_group": self.candidate_group,
            "source": self.source,
            "metric_safe": self.metric_safe,
            "display_only": self.display_only,
            "profile_limited": self.profile_limited,
            "warnings": list(self.warnings),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["warnings"] = list(self.warnings)
        row["candidate_id"] = _short_hash(self.stable_key())
        return row


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Complete bounded candidate space for one profile/data context."""

    profile_id: str
    recipe_ids: tuple[str, ...]
    features: CandidateFeatureSummary
    candidates: tuple[GeneratedCandidate, ...]
    candidate_space_hash: str
    config_version: str
    warnings: tuple[str, ...] = ()

    def by_category(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.category, []).append(candidate.to_dict())
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "recipe_ids": list(self.recipe_ids),
            "features": self.features.to_dict(),
            "candidate_space_hash": self.candidate_space_hash,
            "config_version": self.config_version,
            "warnings": list(self.warnings),
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidates_by_category": self.by_category(),
        }


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stable_hash_payload(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite_2d(data: Any | None) -> np.ndarray | None:
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"AutoTune candidate generator expects 2D non-empty data, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
    return arr


def _metadata_float(metadata: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            try:
                value = float(metadata[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                return value
    return None


def _normalize_frequency(value: float | None) -> float | None:
    """Return frequency in Hz, accepting common MHz-like metadata values."""
    if value is None or value <= 0.0 or not math.isfinite(value):
        return None
    # GPR headers often store center_frequency as MHz.  Values below 100 kHz are
    # treated as MHz for convenience, e.g. 250 -> 250 MHz.
    if value < 1.0e5:
        return value * 1.0e6
    return value


def _extract_dt_seconds(metadata: Mapping[str, Any]) -> float | None:
    dt = _metadata_float(metadata, "dt_seconds", "sample_interval_seconds", "time_step_seconds")
    if dt:
        return dt
    dt_ns = _metadata_float(metadata, "dt_ns", "sample_interval_ns", "time_step_ns")
    if dt_ns:
        return dt_ns * 1.0e-9
    total_time_ns = _metadata_float(metadata, "total_time_ns", "time_window_ns")
    n_samples = _metadata_float(metadata, "n_samples", "samples")
    if total_time_ns and n_samples and n_samples > 1:
        return (total_time_ns * 1.0e-9) / (n_samples - 1.0)
    return None


def _dominant_frequency_hz(arr: np.ndarray | None, dt_seconds: float | None) -> float | None:
    if arr is None or not dt_seconds or dt_seconds <= 0.0:
        return None
    if arr.shape[0] < 8:
        return None
    trace = np.nanmedian(arr - np.nanmedian(arr, axis=0, keepdims=True), axis=1)
    spectrum = np.abs(np.fft.rfft(trace))
    freqs = np.fft.rfftfreq(trace.size, d=dt_seconds)
    if spectrum.size <= 1:
        return None
    spectrum[0] = 0.0
    idx = int(np.argmax(spectrum))
    if idx <= 0 or not math.isfinite(float(freqs[idx])):
        return None
    return float(freqs[idx])


def _lateral_correlation_length(arr: np.ndarray | None) -> int | None:
    if arr is None or arr.shape[1] < 5:
        return None
    z = arr - np.nanmedian(arr)
    scale = float(np.nanpercentile(np.abs(z), 95)) + 1e-12
    z = z / scale
    max_lag = min(51, max(2, arr.shape[1] // 3))
    corrs: list[float] = []
    for lag in range(1, max_lag + 1):
        a = z[:, :-lag].ravel()
        b = z[:, lag:].ravel()
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        corrs.append(float(np.dot(a, b) / denom))
    for lag, corr in enumerate(corrs, start=1):
        if corr < 0.5:
            return max(3, lag)
    return max(3, min(max_lag, arr.shape[1] // 6 or 3))


def _singular_elbow_rank(arr: np.ndarray | None) -> int | None:
    if arr is None:
        return None
    max_rank = min(arr.shape)
    if max_rank < 3:
        return 1 if max_rank >= 1 else None
    try:
        centered = arr - np.nanmedian(arr, axis=1, keepdims=True)
        singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None
    singular = np.asarray(singular, dtype=np.float64)
    singular = singular[np.isfinite(singular) & (singular > 0.0)]
    if singular.size < 3:
        return 1
    log_s = np.log(singular[: min(16, singular.size)] + 1e-12)
    # Knee proxy: largest discrete curvature in log singular spectrum.
    curvature = np.abs(np.diff(log_s, n=2))
    if curvature.size == 0:
        return 1
    return int(np.argmax(curvature) + 1)


def _attenuation_ratio(arr: np.ndarray | None) -> float | None:
    if arr is None or arr.shape[0] < 6:
        return None
    amp = np.nanmedian(np.abs(arr), axis=1)
    n = amp.size
    shallow = float(np.nanmedian(amp[: max(1, n // 3)])) + 1e-12
    deep = float(np.nanmedian(amp[max(0, 2 * n // 3) :])) + 1e-12
    return float(deep / shallow)


def _spikiness(arr: np.ndarray | None) -> tuple[float | None, float | None]:
    if arr is None:
        return None, None
    centered = arr - float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(centered))) + 1e-12
    z = np.abs(centered) / (1.4826 * mad)
    hot = float(np.mean(z >= 6.0))
    spike = float(np.percentile(z, 99.5))
    return spike, hot


def summarize_candidate_features(
    data: Any | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateFeatureSummary:
    """Build deterministic lightweight features for candidate generation."""

    metadata = dict(metadata or {})
    arr = _finite_2d(data)
    n_samples = int(arr.shape[0]) if arr is not None else int(_metadata_float(metadata, "n_samples", "samples") or 0) or None
    n_traces = int(arr.shape[1]) if arr is not None else int(_metadata_float(metadata, "n_traces", "traces") or 0) or None

    dt_seconds = _extract_dt_seconds({**metadata, **({"n_samples": n_samples} if n_samples else {})})
    sample_rate_hz = (1.0 / dt_seconds) if dt_seconds and dt_seconds > 0.0 else None
    nyquist_hz = (0.5 * sample_rate_hz) if sample_rate_hz else None
    center_frequency_hz = _normalize_frequency(_metadata_float(metadata, "center_frequency_hz", "antenna_center_frequency_hz", "center_frequency", "antenna_frequency_mhz", "antenna_center_frequency_mhz"))
    dominant_frequency_hz = _normalize_frequency(_metadata_float(metadata, "dominant_frequency_hz", "dominant_frequency"))
    if dominant_frequency_hz is None:
        dominant_frequency_hz = _dominant_frequency_hz(arr, dt_seconds)

    spikiness, hot_pixel_ratio = _spikiness(arr)
    return CandidateFeatureSummary(
        n_samples=n_samples,
        n_traces=n_traces,
        dt_seconds=dt_seconds,
        sample_rate_hz=sample_rate_hz,
        nyquist_hz=nyquist_hz,
        center_frequency_hz=center_frequency_hz,
        dominant_frequency_hz=dominant_frequency_hz,
        lateral_correlation_length_traces=_lateral_correlation_length(arr),
        singular_elbow_rank=_singular_elbow_rank(arr),
        attenuation_ratio=_attenuation_ratio(arr),
        spikiness=spikiness,
        hot_pixel_ratio=hot_pixel_ratio,
        trace_spacing_m=_metadata_float(metadata, "trace_spacing_m", "dx_m", "trace_interval_m"),
        target_lateral_scale_m=_metadata_float(metadata, "target_lateral_scale_m", "target_width_m", "expected_target_width_m"),
    )


def _odd_clamp(value: int | float, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        maximum = minimum
    v = int(round(float(value)))
    v = max(minimum, min(maximum, v))
    if v % 2 == 0:
        if v + 1 <= maximum:
            v += 1
        elif v - 1 >= minimum:
            v -= 1
    return int(v)


def _unique_ints(values: Sequence[int | float | None]) -> tuple[int, ...]:
    clean = sorted({int(v) for v in values if v is not None and math.isfinite(float(v)) and int(v) > 0})
    return tuple(clean)


def _profile_caps(profile_id: str, features: CandidateFeatureSummary) -> dict[str, Any]:
    """Conservative caps derived from the V1 design contract."""

    elbow = features.singular_elbow_rank or 1
    if profile_id in {"interface_layer_preservation", "landslide_bedrock_sliding_surface"}:
        return {
            "svd_max_rank": 1,
            "min_dewow_samples": 32,
            "aggressive_highpass": False,
            "background_strength": "weak_or_rank1_only",
        }
    if profile_id in {"wet_weak_zone", "deep_weak_reflector"}:
        return {
            "svd_max_rank": max(1, min(2, elbow)),
            "min_dewow_samples": 32,
            "aggressive_highpass": False,
            "background_strength": "weak",
        }
    if profile_id == "object_like_anomaly":
        return {
            "svd_max_rank": 8,
            "min_dewow_samples": 16,
            "aggressive_highpass": True,
            "background_strength": "moderate_to_strong",
        }
    return {
        "svd_max_rank": max(1, min(3, elbow + 1)),
        "min_dewow_samples": 16,
        "aggressive_highpass": False,
        "background_strength": "balanced",
    }


def _candidate_add_unique(candidates: list[GeneratedCandidate], candidate: GeneratedCandidate) -> None:
    key = candidate.stable_key()
    if key not in {item.stable_key() for item in candidates}:
        candidates.append(candidate)


def _max_odd_window(features: CandidateFeatureSummary) -> int:
    if features.n_traces and features.n_traces >= 7:
        return _odd_clamp(min(101, max(7, features.n_traces - 1)), 7, max(7, min(101, features.n_traces - 1)))
    return 101


def _generate_background_candidates(
    config: AutoTuneV1Config,
    profile_id: str,
    features: CandidateFeatureSummary,
) -> list[GeneratedCandidate]:
    spec = config.candidate_spec.get("background_suppression", {}) or {}
    fixed = spec.get("fixed_candidates", {}) or {}
    caps = _profile_caps(profile_id, features)
    candidates: list[GeneratedCandidate] = []

    for method in ("mean_background", "median_background"):
        _candidate_add_unique(
            candidates,
            GeneratedCandidate(
                category="background_suppression",
                method=method,
                parameters={"mode": "global", "axis": "trace"},
                candidate_group="global_background",
                source="fixed_v1_default",
                rationale="Global background estimate is reproducible but only safe when target responses are sparse.",
            ),
        )

    max_window = _max_odd_window(features)
    window_values = list(fixed.get("sliding_window_ntraces", []) or [])
    if features.lateral_correlation_length_traces:
        corr = features.lateral_correlation_length_traces
        window_values.extend([corr, 1.5 * corr, 2.0 * corr, 3.0 * corr])
    if features.trace_spacing_m and features.target_lateral_scale_m:
        traces = max(3.0, features.target_lateral_scale_m / features.trace_spacing_m)
        window_values.extend([traces, 1.5 * traces, 2.0 * traces])
    windows = _unique_ints(_odd_clamp(value, 7, max_window) for value in window_values)
    for window in windows:
        for method, group in (("sliding_mean_background", "sliding_background"), ("sliding_median_background", "sliding_background")):
            _candidate_add_unique(
                candidates,
                GeneratedCandidate(
                    category="background_suppression",
                    method=method,
                    parameters={"window_ntraces": int(window), "axis": "trace"},
                    candidate_group=group,
                    source="fixed_plus_adaptive" if features.lateral_correlation_length_traces else "fixed_v1_default",
                    rationale="Sliding window background follows slowly varying clutter while keeping candidate space bounded.",
                ),
            )

    rank_values = list(fixed.get("svd_rank", []) or [])
    if features.singular_elbow_rank:
        rank_values.extend([features.singular_elbow_rank - 1, features.singular_elbow_rank, features.singular_elbow_rank + 1])
    max_rank_by_shape = max(1, min(features.n_samples or 8, features.n_traces or 8, 8))
    max_rank = max(1, min(int(caps["svd_max_rank"]), max_rank_by_shape))
    for rank in _unique_ints(rank_values):
        if rank > max_rank:
            continue
        warnings: list[str] = []
        if profile_id in {"interface_layer_preservation", "landslide_bedrock_sliding_surface"} and rank >= 1:
            warnings.append("svd_rank_may_remove_interface")
        _candidate_add_unique(
            candidates,
            GeneratedCandidate(
                category="background_suppression",
                method="svd_rank_sweep",
                parameters={"remove_rank": int(rank)},
                candidate_group="low_rank_background",
                source="fixed_plus_singular_elbow" if features.singular_elbow_rank else "fixed_v1_default",
                profile_limited=max_rank < max(rank_values or [max_rank]),
                rationale="SVD rank is capped by profile to avoid deleting true horizontal/interface-like responses.",
                warnings=tuple(warnings),
            ),
        )
    return candidates


def _generate_dewow_candidates(
    config: AutoTuneV1Config,
    profile_id: str,
    features: CandidateFeatureSummary,
) -> list[GeneratedCandidate]:
    spec = config.candidate_spec.get("dewow", {}) or {}
    caps = _profile_caps(profile_id, features)
    min_window = int(caps["min_dewow_samples"])
    max_window = max(min_window, min(512, int(features.n_samples or 512)))
    values = list(spec.get("fixed_candidates_samples", []) or [])
    source = "fixed_v1_default"
    if features.sample_rate_hz and features.center_frequency_hz:
        period_samples = features.sample_rate_hz / features.center_frequency_hz
        values.extend([period_samples, 2.0 * period_samples])
        source = "fixed_plus_frequency_period"
    elif features.sample_rate_hz and features.dominant_frequency_hz:
        period_samples = features.sample_rate_hz / features.dominant_frequency_hz
        values.extend([period_samples, 2.0 * period_samples])
        source = "fixed_plus_dominant_period"
    windows = _unique_ints(_odd_clamp(v, min_window, max_window) for v in values)
    return [
        GeneratedCandidate(
            category="dewow",
            method="moving_average_dewow",
            parameters={"window_samples": int(window)},
            candidate_group="low_frequency_drift",
            source=source,
            metric_safe=True,
            rationale="Dewow window is bounded and optionally tied to 1–2 dominant/center-frequency periods.",
            warnings=("highpass_may_remove_deep_or_wet_response",) if window <= 16 and profile_id in {"wet_weak_zone", "deep_weak_reflector", "interface_layer_preservation", "landslide_bedrock_sliding_surface"} else (),
        )
        for window in windows
    ]


def _clip_freq(value: float, nyquist: float) -> float:
    return float(max(1.0, min(value, nyquist * 0.95)))


def _generate_bandpass_candidates(
    config: AutoTuneV1Config,
    profile_id: str,
    features: CandidateFeatureSummary,
) -> list[GeneratedCandidate]:
    if not features.nyquist_hz:
        return []
    nyq = float(features.nyquist_hz)
    ref = features.dominant_frequency_hz or features.center_frequency_hz or nyq / 3.0
    ref = max(1.0, min(float(ref), nyq * 0.8))
    if profile_id in {"wet_weak_zone", "deep_weak_reflector", "interface_layer_preservation", "landslide_bedrock_sliding_surface"}:
        presets = [
            (0.08 * ref, 1.45 * ref, "low_cut_guarded"),
            (0.12 * ref, 1.65 * ref, "low_frequency_preserving"),
            (0.18 * ref, 1.80 * ref, "balanced_conservative"),
        ]
    elif profile_id == "object_like_anomaly":
        presets = [
            (0.20 * ref, 1.80 * ref, "wide_object"),
            (0.30 * ref, 2.20 * ref, "contrast_object"),
            (0.12 * ref, 2.00 * ref, "wide_safe"),
        ]
    else:
        presets = [
            (0.12 * ref, 1.60 * ref, "balanced_low"),
            (0.20 * ref, 1.90 * ref, "balanced_mid"),
            (0.08 * ref, 1.40 * ref, "conservative_low_cut"),
        ]
    candidates: list[GeneratedCandidate] = []
    for low, high, label in presets:
        low_hz = _clip_freq(low, nyq)
        high_hz = _clip_freq(max(high, low_hz * 1.5), nyq)
        if high_hz <= low_hz:
            continue
        warnings: list[str] = []
        if low_hz > 0.25 * ref and profile_id in {"wet_weak_zone", "deep_weak_reflector", "interface_layer_preservation", "landslide_bedrock_sliding_surface"}:
            warnings.append("highpass_may_remove_deep_or_wet_response")
        _candidate_add_unique(
            candidates,
            GeneratedCandidate(
                category="bandpass",
                method="butterworth_bandpass",
                parameters={"low_cut_hz": round(low_hz, 3), "high_cut_hz": round(high_hz, 3), "order": 4},
                candidate_group=label,
                source="adaptive_spectrum_metadata",
                metric_safe=True,
                rationale="Bandpass is derived from Nyquist and dominant/center frequency; deep/wet/interface profiles keep low cut conservative.",
                warnings=tuple(warnings),
            ),
        )
    return candidates


def _generate_gain_candidates(
    config: AutoTuneV1Config,
    profile_id: str,
    features: CandidateFeatureSummary,
) -> list[GeneratedCandidate]:
    ratio = features.attenuation_ratio
    if ratio is None:
        ratio = 0.35
    deep_profile = profile_id in {"wet_weak_zone", "deep_weak_reflector", "landslide_bedrock_sliding_surface", "interface_layer_preservation"}
    sec_values = [2.5, 3.5, 4.5] if not deep_profile else [3.5, 5.5, 7.0]
    if ratio < 0.20 and deep_profile:
        sec_values.append(9.0)
    candidates = [
        GeneratedCandidate(
            category="gain",
            method="sec_gain",
            parameters={"gain_max": float(value)},
            candidate_group="metric_safe_depth_compensation",
            source="fixed_plus_attenuation_ratio",
            metric_safe=True,
            rationale="SEC gain is recorded as metric-safe depth compensation when used as processing, unlike AGC display flattening.",
        )
        for value in sorted(set(sec_values))
    ]
    candidates.extend(
        GeneratedCandidate(
            category="gain",
            method="exponential_gain",
            parameters={"alpha": float(value)},
            candidate_group="metric_safe_depth_compensation",
            source="fixed_v1_default",
            metric_safe=True,
            rationale="Exponential gain provides bounded depth compensation candidates.",
        )
        for value in ([0.005, 0.010, 0.020] if deep_profile else [0.003, 0.007, 0.012])
    )
    agc_windows = [31, 61, 101]
    candidates.extend(
        GeneratedCandidate(
            category="gain",
            method="agc",
            parameters={"window_samples": int(window)},
            candidate_group="display_only_gain",
            source="fixed_v1_display_only",
            metric_safe=False,
            display_only=True,
            rationale="AGC changes relative amplitudes and is excluded from synthetic full-reference scoring.",
            warnings=("agc_excluded_from_scoring", "display_only_transform_used"),
        )
        for window in agc_windows
    )
    return candidates


def _generate_denoise_candidates(
    config: AutoTuneV1Config,
    profile_id: str,
    features: CandidateFeatureSummary,
) -> list[GeneratedCandidate]:
    conservative = profile_id in {"interface_layer_preservation", "landslide_bedrock_sliding_surface", "wet_weak_zone", "deep_weak_reflector"}
    max_window = 5 if conservative else 9
    candidates: list[GeneratedCandidate] = [
        GeneratedCandidate(
            category="denoise",
            method="trace_median_light",
            parameters={"window_traces": 3 if conservative else 5},
            candidate_group="metric_safe_light_filter",
            source="fixed_v1_default",
            metric_safe=True,
            rationale="Light trace median filter can suppress isolated trace noise while preserving broad structure.",
        ),
        GeneratedCandidate(
            category="denoise",
            method="savitzky_golay_light",
            parameters={"window_samples": max_window, "polyorder": 2},
            candidate_group="metric_safe_light_filter",
            source="fixed_v1_default",
            metric_safe=True,
            rationale="Light Savitzky-Golay smoothing is capped for interface-like profiles to avoid erasing local breaks.",
        ),
    ]
    if (features.hot_pixel_ratio or 0.0) >= 0.001 or (features.spikiness or 0.0) >= 8.0:
        candidates.append(
            GeneratedCandidate(
                category="denoise",
                method="hampel_spike_removal",
                parameters={"window_samples": 7 if conservative else 11, "n_sigma": 3.0},
                candidate_group="spike_filter",
                source="adaptive_spikiness",
                metric_safe=True,
                rationale="Hampel spike removal is enabled only when hot-pixel/spikiness diagnostics justify it.",
            )
        )
    return candidates


def _generate_migration_candidates(
    config: AutoTuneV1Config,
    profile_id: str,
    features: CandidateFeatureSummary,
    metadata: Mapping[str, Any],
) -> list[GeneratedCandidate]:
    if profile_id != "object_like_anomaly":
        return [
            GeneratedCandidate(
                category="migration",
                method="migration_disabled",
                parameters={"enabled": False},
                candidate_group="v1_production_default",
                source="profile_guardrail",
                metric_safe=True,
                rationale="Migration is disabled for production V1 scoring outside object-like advanced recipes because it is velocity-sensitive.",
            )
        ]
    velocity = _metadata_float(metadata, "velocity_m_per_ns", "migration_velocity_m_per_ns")
    candidates = [
        GeneratedCandidate(
            category="migration",
            method="migration_disabled",
            parameters={"enabled": False},
            candidate_group="v1_production_default",
            source="profile_guardrail",
            metric_safe=True,
            rationale="No-migration baseline is kept for auditability.",
        )
    ]
    if velocity:
        for scale in (0.9, 1.0, 1.1):
            candidates.append(
                GeneratedCandidate(
                    category="migration",
                    method="kirchhoff_or_fk_migration_velocity_sweep",
                    parameters={"enabled": True, "velocity_m_per_ns": round(velocity * scale, 6)},
                    candidate_group="experimental_object_focus",
                    source="metadata_velocity_experimental",
                    metric_safe=False,
                    display_only=False,
                    rationale="Experimental object-like candidate; requires explicit velocity and remains outside default production scoring.",
                    warnings=("migration_velocity_sensitive_experimental",),
                )
            )
    return candidates


def generate_autotune_v1_candidates(
    data: Any | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    target_goal: str | None = None,
    config: AutoTuneV1Config | None = None,
    include_display_only: bool = True,
    include_experimental: bool = False,
) -> CandidateGenerationResult:
    """Generate bounded AutoTune V1 candidates for one data/profile context."""

    config = config or load_autotune_v1_config()
    metadata = dict(metadata or {})
    profile_id = config.resolve_profile_id(target_goal)
    features = summarize_candidate_features(data, metadata=metadata)
    recipe_ids = tuple(recipe.recipe_id for recipe in config.recipes_for_profile(profile_id))

    candidates: list[GeneratedCandidate] = []
    candidates.extend(_generate_background_candidates(config, profile_id, features))
    candidates.extend(_generate_dewow_candidates(config, profile_id, features))
    candidates.extend(_generate_bandpass_candidates(config, profile_id, features))
    candidates.extend(_generate_gain_candidates(config, profile_id, features))
    candidates.extend(_generate_denoise_candidates(config, profile_id, features))
    candidates.extend(_generate_migration_candidates(config, profile_id, features, metadata))

    if not include_display_only:
        candidates = [candidate for candidate in candidates if not candidate.display_only]
    if not include_experimental:
        candidates = [
            candidate
            for candidate in candidates
            if "experimental" not in candidate.candidate_group and "experimental" not in candidate.source
        ]

    unique: list[GeneratedCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.stable_key()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    candidates = sorted(unique, key=lambda item: (item.category, item.method, item.stable_key()))

    warnings: list[str] = []
    if profile_id in {"interface_layer_preservation", "landslide_bedrock_sliding_surface"}:
        warnings.append("interface_profiles_use_svd_rank_cap")
    if any(candidate.display_only for candidate in candidates):
        warnings.append("display_only_candidates_excluded_from_full_reference_scoring")
    if not features.dt_seconds:
        warnings.append("missing_dt_seconds_bandpass_limited")
    if not features.center_frequency_hz and not features.dominant_frequency_hz:
        warnings.append("missing_frequency_metadata_dewow_bandpass_less_adaptive")

    hash_payload = {
        "config_version": config.version,
        "profile_id": profile_id,
        "recipe_ids": recipe_ids,
        "features": features.to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "include_display_only": include_display_only,
        "include_experimental": include_experimental,
    }
    candidate_space_hash = _stable_hash_payload(hash_payload)
    return CandidateGenerationResult(
        profile_id=profile_id,
        recipe_ids=recipe_ids,
        features=features,
        candidates=tuple(candidates),
        candidate_space_hash=candidate_space_hash,
        config_version=config.version,
        warnings=tuple(warnings),
    )


def candidate_space_hash(result_or_payload: CandidateGenerationResult | Mapping[str, Any]) -> str:
    """Return a deterministic hash for a generation result or serializable payload."""

    if isinstance(result_or_payload, CandidateGenerationResult):
        return result_or_payload.candidate_space_hash
    return _stable_hash_payload(result_or_payload)


__all__ = [
    "CandidateFeatureSummary",
    "CandidateGenerationResult",
    "GeneratedCandidate",
    "candidate_space_hash",
    "generate_autotune_v1_candidates",
    "summarize_candidate_features",
]

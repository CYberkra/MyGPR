#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract loader for external gprMax validation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.gpr_io import read_gprmax_out


MANIFEST_CANDIDATES = (
    "*_manifest.json",
    "manifest.json",
    "dataset_manifest.json",
)
GROUND_TRUTH_SCHEMA = "gprmax_ground_truth_v1"
MYGPR_GROUND_TRUTH_SCHEMA = "mygpr_gprmax_ground_truth_v1"


@dataclass(frozen=True)
class GprMaxDatasetPackage:
    """Loaded gprMax dataset package ready for MyGPR validation."""

    scenario_id: str
    manifest_path: Path
    dataset_dir: Path
    primary_out_file: Path
    metadata_file: Path | None
    ground_truth_file: Path
    data: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray] | None
    manifest: dict[str, Any]
    metadata: dict[str, Any]
    ground_truth_raw: dict[str, Any]
    ground_truth: dict[str, Any]
    ground_truth_paths: dict[str, Path]


def load_gprmax_dataset_contract(path: str | Path) -> GprMaxDatasetPackage:
    """Load a gprMax dataset folder or manifest for AutoTune validation."""
    manifest_path = resolve_manifest_path(path)
    dataset_dir = manifest_path.parent
    manifest = _read_json(manifest_path)

    primary_out_file = _resolve_required_path(
        dataset_dir,
        manifest,
        keys=("primary_out_file", "primary_out_path", "out_file", "merged_out_file"),
        label="primary_out_file",
    )
    metadata_file = _resolve_optional_path(
        dataset_dir,
        manifest,
        keys=("metadata_file", "metadata_path"),
    )
    ground_truth_file = _resolve_required_path(
        dataset_dir,
        manifest,
        keys=("ground_truth_file", "ground_truth_path", "ground_truth_yaml"),
        label="ground_truth_file",
    )

    out_payload = read_gprmax_out(str(primary_out_file))
    data = np.asarray(out_payload["data"], dtype=np.float32)
    metadata = _read_json(metadata_file) if metadata_file else {}
    ground_truth_raw = load_ground_truth_yaml(ground_truth_file)
    scenario_id = str(
        manifest.get("scenario_id")
        or metadata.get("scenario_id")
        or ground_truth_raw.get("scenario_id")
        or primary_out_file.stem
    )
    ground_truth = adapt_gprmax_ground_truth(
        ground_truth_raw,
        data_shape=data.shape,
        scenario_id=scenario_id,
    )

    return GprMaxDatasetPackage(
        scenario_id=scenario_id,
        manifest_path=manifest_path,
        dataset_dir=dataset_dir,
        primary_out_file=primary_out_file,
        metadata_file=metadata_file,
        ground_truth_file=ground_truth_file,
        data=data,
        header_info=dict(out_payload.get("header_info") or {}),
        trace_metadata=out_payload.get("trace_metadata"),
        manifest=manifest,
        metadata=metadata,
        ground_truth_raw=ground_truth_raw,
        ground_truth=ground_truth,
        ground_truth_paths={
            "manifest_file": manifest_path,
            "primary_out_file": primary_out_file,
            "ground_truth_file": ground_truth_file,
            **({"metadata_file": metadata_file} if metadata_file else {}),
        },
    )


def resolve_manifest_path(path: str | Path) -> Path:
    """Resolve a dataset directory or manifest file to a concrete manifest path."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"gprMax dataset path not found: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"gprMax dataset path must be a file or directory: {candidate}")

    matches: list[Path] = []
    for pattern in MANIFEST_CANDIDATES:
        matches.extend(sorted(candidate.glob(pattern)))
    unique = list(dict.fromkeys(matches))
    if not unique:
        raise FileNotFoundError(f"No gprMax manifest JSON found in {candidate}")
    if len(unique) > 1:
        names = ", ".join(path.name for path in unique)
        raise ValueError(f"Multiple gprMax manifest JSON files found: {names}")
    return unique[0]


def load_ground_truth_yaml(path: str | Path) -> dict[str, Any]:
    """Read gprMax ground-truth YAML sidecar."""
    ground_truth_path = Path(path).expanduser().resolve()
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"gprMax ground_truth.yaml not found: {ground_truth_path}")
    with ground_truth_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ground_truth.yaml must contain a mapping: {ground_truth_path}")
    return payload


def adapt_gprmax_ground_truth(
    ground_truth: dict[str, Any],
    *,
    data_shape: tuple[int, int] | None = None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Convert gprMax closed-interval ROI YAML into MyGPR truth manifest."""
    source = dict(ground_truth or {})
    source_schema = str(source.get("schema") or "")
    if source_schema and source_schema != GROUND_TRUTH_SCHEMA:
        if source_schema == MYGPR_GROUND_TRUTH_SCHEMA:
            return source
        raise ValueError(f"Unsupported gprMax ground-truth schema: {source_schema}")

    resolved_scenario_id = str(
        scenario_id or source.get("scenario_id") or source.get("name") or "gprmax_dataset"
    )
    targets = _adapt_targets(source)
    adapted: dict[str, Any] = {
        "schema": MYGPR_GROUND_TRUTH_SCHEMA,
        "source_schema": source_schema or GROUND_TRUTH_SCHEMA,
        "scenario_id": resolved_scenario_id,
        "targets": targets,
    }

    analysis_roi = _adapt_optional_roi(source.get("analysis_roi"))
    if analysis_roi is None and data_shape is not None:
        analysis_roi = {
            "time_start_idx": 0,
            "time_end_idx": int(data_shape[0]),
            "dist_start_idx": 0,
            "dist_end_idx": int(data_shape[1]),
        }
    if analysis_roi is not None:
        adapted["analysis_roi"] = _clamp_roi(analysis_roi, data_shape)

    wavefield_rois = _adapt_wavefield_rois(source.get("wavefield_rois"), data_shape)
    if wavefield_rois:
        adapted["wavefield_rois"] = wavefield_rois

    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        adapted["metadata"] = dict(metadata)

    return adapted


def _adapt_targets(source: dict[str, Any]) -> list[dict[str, Any]]:
    raw_targets = source.get("targets")
    if isinstance(raw_targets, list):
        return [
            _adapt_target(item, index)
            for index, item in enumerate(raw_targets, start=1)
            if isinstance(item, dict)
        ]
    target_roi = source.get("target_roi")
    if target_roi is None:
        return []
    return [
        _adapt_target(
            {
                "target_id": source.get("target_id") or source.get("id"),
                "type": source.get("target_type") or source.get("type"),
                "target_roi": target_roi,
                "must_preserve": source.get("must_preserve", True),
            },
            1,
        )
    ]


def _adapt_target(raw: dict[str, Any], index: int) -> dict[str, Any]:
    roi = _adapt_optional_roi(raw.get("target_roi") or raw.get("roi"))
    if roi is None:
        raise ValueError(f"gprMax target #{index} is missing target_roi")
    target: dict[str, Any] = {
        "target_id": str(raw.get("target_id") or raw.get("id") or f"target_{index:02d}"),
        "type": str(raw.get("target_type") or raw.get("type") or "target"),
        "roi": roi,
        "must_preserve": bool(raw.get("must_preserve", True)),
    }
    for key in ("label", "material", "notes"):
        if key in raw:
            target[key] = raw[key]
    return target


def _adapt_wavefield_rois(
    raw_rois: Any,
    data_shape: tuple[int, int] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_rois, dict):
        return {}
    adapted: dict[str, dict[str, Any]] = {}
    for key, value in raw_rois.items():
        if isinstance(value, dict) and ("sample_range" in value or "trace_range" in value):
            roi = _adapt_optional_roi(value)
            payload: dict[str, Any] = {}
        elif isinstance(value, dict):
            roi = _adapt_optional_roi(value.get("roi"))
            payload = {k: v for k, v in value.items() if k != "roi"}
        else:
            continue
        if roi is None:
            continue
        payload["roi"] = _clamp_roi(roi, data_shape)
        adapted[str(key)] = payload
    return adapted


def _adapt_optional_roi(raw_roi: Any) -> dict[str, int] | None:
    if raw_roi is None:
        return None
    if not isinstance(raw_roi, dict):
        raise ValueError(f"ROI must be a mapping, got {type(raw_roi).__name__}")
    if {"time_start_idx", "time_end_idx", "dist_start_idx", "dist_end_idx"}.issubset(
        raw_roi.keys()
    ):
        return {
            "time_start_idx": int(raw_roi["time_start_idx"]),
            "time_end_idx": int(raw_roi["time_end_idx"]),
            "dist_start_idx": int(raw_roi["dist_start_idx"]),
            "dist_end_idx": int(raw_roi["dist_end_idx"]),
        }
    sample_range = raw_roi.get("sample_range")
    trace_range = raw_roi.get("trace_range")
    if sample_range is None or trace_range is None:
        raise ValueError(f"ROI requires sample_range and trace_range: {raw_roi}")
    s0, s1 = _closed_range(sample_range, label="sample_range")
    t0, t1 = _closed_range(trace_range, label="trace_range")
    return {
        "time_start_idx": s0,
        "time_end_idx": s1 + 1,
        "dist_start_idx": t0,
        "dist_end_idx": t1 + 1,
    }


def _closed_range(value: Any, *, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-item closed interval")
    start = int(value[0])
    end = int(value[1])
    if start < 0 or end < start:
        raise ValueError(f"{label} must satisfy 0 <= start <= end: {value}")
    return start, end


def _clamp_roi(
    roi: dict[str, int],
    data_shape: tuple[int, int] | None,
) -> dict[str, int]:
    if data_shape is None:
        return dict(roi)
    samples, traces = int(data_shape[0]), int(data_shape[1])
    time_start = max(0, min(int(roi["time_start_idx"]), max(samples - 1, 0)))
    time_end = max(time_start + 1, min(int(roi["time_end_idx"]), samples))
    dist_start = max(0, min(int(roi["dist_start_idx"]), max(traces - 1, 0)))
    dist_end = max(dist_start + 1, min(int(roi["dist_end_idx"]), traces))
    return {
        "time_start_idx": time_start,
        "time_end_idx": time_end,
        "dist_start_idx": dist_start,
        "dist_end_idx": dist_end,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return payload


def _resolve_required_path(
    dataset_dir: Path,
    manifest: dict[str, Any],
    *,
    keys: tuple[str, ...],
    label: str,
) -> Path:
    path = _resolve_optional_path(dataset_dir, manifest, keys=keys)
    if path is None:
        raise ValueError(f"gprMax manifest missing required {label}")
    if not path.exists():
        raise FileNotFoundError(f"gprMax manifest {label} not found: {path}")
    return path


def _resolve_optional_path(
    dataset_dir: Path,
    manifest: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> Path | None:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value)
            return path.expanduser().resolve() if path.is_absolute() else (dataset_dir / path).resolve()
    return None


__all__ = [
    "GprMaxDatasetPackage",
    "adapt_gprmax_ground_truth",
    "load_gprmax_dataset_contract",
    "load_ground_truth_yaml",
    "resolve_manifest_path",
]

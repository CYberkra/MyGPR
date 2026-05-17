#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract loader for external gprMax validation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.gprmax_ground_truth import (
    convert_gprmax_ground_truth_to_mygpr,
    load_gprmax_ground_truth,
)
from core.gpr_io import read_gprmax_out


MANIFEST_CANDIDATES = (
    "*_manifest.json",
    "manifest.json",
    "dataset_manifest.json",
)


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
    header_info = dict(out_payload.get("header_info") or {})
    header_info["ground_truth"] = ground_truth

    return GprMaxDatasetPackage(
        scenario_id=scenario_id,
        manifest_path=manifest_path,
        dataset_dir=dataset_dir,
        primary_out_file=primary_out_file,
        metadata_file=metadata_file,
        ground_truth_file=ground_truth_file,
        data=data,
        header_info=header_info,
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
    return load_gprmax_ground_truth(str(path))


def adapt_gprmax_ground_truth(
    ground_truth: dict[str, Any],
    *,
    data_shape: tuple[int, int] | None = None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Convert gprMax closed-interval ROI YAML into MyGPR truth manifest."""
    source = dict(ground_truth or {})
    if scenario_id and not source.get("dataset_id"):
        source["dataset_id"] = scenario_id
    return convert_gprmax_ground_truth_to_mygpr(source, data_shape=data_shape)


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
    groups = [
        manifest,
        manifest.get("paths_relative_to_output_dir"),
        manifest.get("paths"),
        manifest.get("files"),
    ]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for key in keys:
            value = group.get(key)
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

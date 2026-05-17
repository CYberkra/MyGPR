#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load and adapt gprMax ground-truth sidecars for MyGPR AutoTune."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import yaml

from core.scalar_utils import to_int_or_none


GPRMAX_GROUND_TRUTH_SCHEMA = "gprmax_ground_truth_v1"
MYGPR_GROUND_TRUTH_SCHEMA = "mygpr_gprmax_ground_truth_v1"


def load_gprmax_ground_truth(path: str) -> dict:
    """Read a gprMax ``ground_truth.yaml`` sidecar."""
    sidecar_path = Path(path).expanduser().resolve()
    if not sidecar_path.exists():
        raise FileNotFoundError(f"gprMax ground_truth.yaml not found: {sidecar_path}")
    with sidecar_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ground_truth.yaml must contain a mapping: {sidecar_path}")
    return payload


def convert_gprmax_ground_truth_to_mygpr(
    sidecar: dict,
    data_shape: tuple[int, int] | None = None,
) -> dict:
    """Convert gprMax closed ROI intervals into MyGPR half-open ROI format."""
    if not isinstance(sidecar, dict):
        raise ValueError("gprMax ground truth sidecar must be a mapping")
    source = copy.deepcopy(sidecar)
    schema = str(source.get("schema") or GPRMAX_GROUND_TRUTH_SCHEMA)
    if schema == MYGPR_GROUND_TRUTH_SCHEMA:
        converted = copy.deepcopy(source)
        converted.setdefault("raw_sidecar", copy.deepcopy(source))
        return converted
    if schema != GPRMAX_GROUND_TRUTH_SCHEMA:
        raise ValueError(f"Unsupported gprMax ground-truth schema: {schema}")

    targets = _convert_targets(source, data_shape)
    analysis_roi = _convert_analysis_roi(source, targets, data_shape)
    background_rois = _convert_background_rois(source, data_shape)
    conversion_warnings = _roi_conversion_warnings(targets, background_rois)

    converted: dict[str, Any] = {
        "schema": MYGPR_GROUND_TRUTH_SCHEMA,
        "source_schema": schema,
        "scenario_id": str(
            source.get("dataset_id")
            or source.get("scenario_id")
            or source.get("name")
            or "gprmax_dataset"
        ),
        "analysis_roi": analysis_roi,
        "targets": targets,
        "background_rois": background_rois,
        "raw_sidecar": source,
    }
    for key in ("metrics_contract", "wavefield_rois", "metadata"):
        value = source.get(key)
        if isinstance(value, dict):
            if key == "wavefield_rois":
                converted[key] = _convert_wavefield_rois(value, data_shape)
            else:
                converted[key] = copy.deepcopy(value)
    warnings_list = source.get("_conversion_warnings")
    if isinstance(warnings_list, list):
        conversion_warnings.extend(str(item) for item in warnings_list if str(item))
    if conversion_warnings:
        converted["conversion_warnings"] = conversion_warnings
    return converted


def load_ground_truth_from_manifest(
    manifest_path: str,
    data_shape: tuple[int, int] | None = None,
) -> dict | None:
    """Load and adapt gprMax ground truth referenced by a manifest JSON file."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"gprMax manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"gprMax manifest must contain a mapping: {path}")

    ground_truth_file = _manifest_path_value(
        manifest,
        "ground_truth_file",
        "ground_truth_path",
        "ground_truth_yaml",
    )
    if not ground_truth_file:
        return None

    primary_out_file = _manifest_path_value(
        manifest,
        "primary_out_file",
        "primary_out_path",
        "out_file",
        "merged_out_file",
    )
    sidecar_path = _resolve_manifest_path(path.parent, ground_truth_file)
    sidecar = load_gprmax_ground_truth(str(sidecar_path))
    warnings_list: list[str] = []
    if primary_out_file:
        _check_output_file_consistency(sidecar, primary_out_file, warnings_list)
    if warnings_list:
        sidecar = copy.deepcopy(sidecar)
        sidecar["_conversion_warnings"] = warnings_list
    converted = convert_gprmax_ground_truth_to_mygpr(sidecar, data_shape=data_shape)
    converted.setdefault("source_paths", {})
    converted["source_paths"].update(
        {
            "manifest_file": str(path),
            "ground_truth_file": str(sidecar_path),
        }
    )
    if primary_out_file:
        converted["source_paths"]["primary_out_file"] = str(
            _resolve_manifest_path(path.parent, primary_out_file)
        )
    return converted


def _convert_targets(
    source: dict[str, Any],
    data_shape: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    raw_targets = source.get("targets")
    if isinstance(raw_targets, list):
        return [
            _convert_target(item, index, data_shape)
            for index, item in enumerate(raw_targets)
            if isinstance(item, dict)
        ]
    target_roi = source.get("target_roi")
    if target_roi is None:
        return []
    nested_target = source.get("target")
    if not isinstance(nested_target, dict):
        nested_target = {}
    target_payload = {
        "id": source.get("target_id") or source.get("id") or "target_0",
        "target_id": source.get("target_id") or source.get("id") or "target_0",
        "type": _first_present(nested_target, "type", source, "target_type", "type"),
        "material": _first_present(nested_target, "material", source, "material"),
        "depth_m": _first_present(nested_target, "depth_m", source, "depth_m"),
        "center_x_m": _first_present(nested_target, "center_x_m", source, "center_x_m"),
        "center_y_m": _first_present(nested_target, "center_y_m", source, "center_y_m"),
        "radius_m": _first_present(nested_target, "radius_m", source, "radius_m"),
        "must_preserve": source.get("must_preserve", True),
        "target_roi": target_roi,
    }
    return [_convert_target(target_payload, 0, data_shape)]


def _first_present(
    primary: dict[str, Any],
    primary_key: str,
    fallback: dict[str, Any],
    *fallback_keys: str,
) -> Any:
    if primary.get(primary_key) is not None:
        return primary[primary_key]
    for key in fallback_keys:
        if fallback.get(key) is not None:
            return fallback[key]
    return None


def _convert_target(
    raw: dict[str, Any],
    index: int,
    data_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    nested_target = raw.get("target")
    if not isinstance(nested_target, dict):
        nested_target = {}
    target_id = str(raw.get("id") or raw.get("target_id") or f"target_{index}")
    roi = _convert_roi(raw.get("target_roi") or raw.get("roi"), data_shape)
    target: dict[str, Any] = {
        "id": target_id,
        "target_id": str(raw.get("target_id") or target_id),
        "type": str(
            raw.get("target_type")
            or _first_present(nested_target, "type", raw, "type")
            or "target"
        ),
        "must_preserve": bool(raw.get("must_preserve", True)),
        "roi": roi,
    }
    for key in (
        "material",
        "depth_m",
        "center_x_m",
        "center_y_m",
        "radius_m",
        "label",
        "notes",
    ):
        value = _first_present(nested_target, key, raw, key)
        if value is not None:
            target[key] = copy.deepcopy(value)
    return target


def _convert_analysis_roi(
    source: dict[str, Any],
    targets: list[dict[str, Any]],
    data_shape: tuple[int, int] | None,
) -> dict[str, int]:
    if isinstance(source.get("analysis_roi"), dict):
        return _convert_roi(source["analysis_roi"], data_shape)
    if targets:
        return dict(targets[0]["roi"])
    if data_shape is not None:
        return {
            "time_start_idx": 0,
            "time_end_idx": int(data_shape[0]),
            "dist_start_idx": 0,
            "dist_end_idx": int(data_shape[1]),
        }
    return {
        "time_start_idx": 0,
        "time_end_idx": 1,
        "dist_start_idx": 0,
        "dist_end_idx": 1,
    }


def _convert_background_rois(
    source: dict[str, Any],
    data_shape: tuple[int, int] | None,
) -> list[dict[str, int]]:
    raw = source.get("background_rois")
    if raw is None:
        raw = source.get("background_roi")
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [_convert_roi(raw, data_shape)]
    if isinstance(raw, list):
        return [
            _convert_roi(item, data_shape)
            for item in raw
            if isinstance(item, dict)
        ]
    raise ValueError("background_roi/background_rois must be a mapping or list")


def _convert_wavefield_rois(
    raw_rois: dict[str, Any],
    data_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in raw_rois.items():
        if not isinstance(value, dict):
            continue
        if "roi" in value and isinstance(value["roi"], dict):
            roi_source = value["roi"]
            payload = {k: copy.deepcopy(v) for k, v in value.items() if k != "roi"}
        else:
            roi_source = value
            payload = {}
        try:
            payload["roi"] = _convert_roi(roi_source, data_shape)
        except ValueError:
            continue
        converted[str(key)] = payload
    return converted


def _roi_conversion_warnings(
    targets: list[dict[str, Any]],
    background_rois: list[dict[str, int]],
) -> list[str]:
    warnings_list: list[str] = []
    if not targets or not background_rois:
        return warnings_list
    target_rois = [
        target.get("roi")
        for target in targets
        if isinstance(target.get("roi"), dict)
    ]
    for background_index, background_roi in enumerate(background_rois):
        background_area = _roi_area(background_roi)
        if background_area <= 0:
            continue
        for target_index, target_roi in enumerate(target_rois):
            if not isinstance(target_roi, dict):
                continue
            overlap = _roi_intersection_area(background_roi, target_roi)
            if overlap <= 0:
                continue
            fraction = overlap / max(background_area, 1)
            if fraction >= 0.95:
                warnings_list.append(
                    "background_roi overlaps target ROI almost completely; "
                    f"background_index={background_index}, target_index={target_index}, "
                    f"overlap_fraction={fraction:.3f}"
                )
            else:
                warnings_list.append(
                    "background_roi overlaps target ROI; "
                    f"background_index={background_index}, target_index={target_index}, "
                    f"overlap_fraction={fraction:.3f}"
                )
    return warnings_list


def _roi_area(roi: dict[str, int]) -> int:
    return max(0, int(roi["time_end_idx"]) - int(roi["time_start_idx"])) * max(
        0,
        int(roi["dist_end_idx"]) - int(roi["dist_start_idx"]),
    )


def _roi_intersection_area(left: dict[str, int], right: dict[str, int]) -> int:
    time_start = max(int(left["time_start_idx"]), int(right["time_start_idx"]))
    time_end = min(int(left["time_end_idx"]), int(right["time_end_idx"]))
    dist_start = max(int(left["dist_start_idx"]), int(right["dist_start_idx"]))
    dist_end = min(int(left["dist_end_idx"]), int(right["dist_end_idx"]))
    return max(0, time_end - time_start) * max(0, dist_end - dist_start)


def _convert_roi(
    raw_roi: Any,
    data_shape: tuple[int, int] | None,
) -> dict[str, int]:
    if not isinstance(raw_roi, dict):
        raise ValueError(f"ROI must be a mapping, got {type(raw_roi).__name__}")
    internal_keys = {
        "time_start_idx",
        "time_end_idx",
        "dist_start_idx",
        "dist_end_idx",
    }
    if internal_keys.issubset(raw_roi.keys()):
        roi = {
            "time_start_idx": _required_int(
                raw_roi["time_start_idx"], "time_start_idx"
            ),
            "time_end_idx": _required_int(raw_roi["time_end_idx"], "time_end_idx"),
            "dist_start_idx": _required_int(
                raw_roi["dist_start_idx"], "dist_start_idx"
            ),
            "dist_end_idx": _required_int(raw_roi["dist_end_idx"], "dist_end_idx"),
        }
    else:
        sample_range = raw_roi.get("sample_range")
        trace_range = raw_roi.get("trace_range")
        if sample_range is None or trace_range is None:
            raise ValueError(f"ROI requires sample_range and trace_range: {raw_roi}")
        s0, s1 = _closed_range(sample_range, "sample_range")
        t0, t1 = _closed_range(trace_range, "trace_range")
        roi = {
            "time_start_idx": s0,
            "time_end_idx": s1 + 1,
            "dist_start_idx": t0,
            "dist_end_idx": t1 + 1,
        }
    return _clamp_roi(roi, data_shape)


def _closed_range(value: Any, label: str) -> tuple[int, int]:
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a two-item closed interval") from exc
    if len(values) != 2:
        raise ValueError(f"{label} must be a two-item closed interval")
    start = to_int_or_none(values[0])
    end = to_int_or_none(values[1])
    if start is None or end is None:
        raise ValueError(f"{label} must contain finite numeric indices: {value}")
    if start < 0 or end < start:
        raise ValueError(f"{label} must satisfy 0 <= start <= end: {value}")
    return start, end


def _required_int(value: Any, label: str) -> int:
    parsed = to_int_or_none(value)
    if parsed is None:
        raise ValueError(f"ROI {label} must contain a finite numeric index: {value!r}")
    return parsed


def _clamp_roi(
    roi: dict[str, int],
    data_shape: tuple[int, int] | None,
) -> dict[str, int]:
    if data_shape is None:
        return {key: _required_int(value, key) for key, value in roi.items()}
    samples = max(1, to_int_or_none(data_shape[0]) or 1)
    traces = max(1, to_int_or_none(data_shape[1]) or 1)
    time_start = max(
        0,
        min(
            _required_int(roi["time_start_idx"], "time_start_idx"),
            max(samples - 1, 0),
        ),
    )
    time_end = max(
        time_start + 1,
        min(_required_int(roi["time_end_idx"], "time_end_idx"), samples),
    )
    dist_start = max(
        0,
        min(
            _required_int(roi["dist_start_idx"], "dist_start_idx"),
            max(traces - 1, 0),
        ),
    )
    dist_end = max(
        dist_start + 1,
        min(_required_int(roi["dist_end_idx"], "dist_end_idx"), traces),
    )
    return {
        "time_start_idx": time_start,
        "time_end_idx": time_end,
        "dist_start_idx": dist_start,
        "dist_end_idx": dist_end,
    }


def _manifest_path_value(manifest: dict[str, Any], *keys: str) -> str | None:
    path_groups = [
        manifest,
        manifest.get("paths_relative_to_output_dir"),
        manifest.get("paths"),
        manifest.get("files"),
    ]
    for group in path_groups:
        if not isinstance(group, dict):
            continue
        for key in keys:
            value = group.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _resolve_manifest_path(base_dir: Path, value: str) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path.expanduser().resolve()
    return (base_dir / raw_path).resolve()


def _check_output_file_consistency(
    sidecar: dict[str, Any],
    primary_out_file: str,
    warnings_list: list[str],
) -> None:
    output_file = sidecar.get("output_file") or sidecar.get("primary_out_file")
    if not isinstance(output_file, str) or not output_file.strip():
        return
    sidecar_name = Path(output_file).name
    manifest_name = Path(primary_out_file).name
    if sidecar_name != manifest_name:
        message = (
            "ground_truth.yaml output_file does not match manifest primary_out_file: "
            f"{sidecar_name!r} != {manifest_name!r}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        warnings_list.append(message)


__all__ = [
    "GPRMAX_GROUND_TRUTH_SCHEMA",
    "MYGPR_GROUND_TRUTH_SCHEMA",
    "convert_gprmax_ground_truth_to_mygpr",
    "load_gprmax_ground_truth",
    "load_ground_truth_from_manifest",
]

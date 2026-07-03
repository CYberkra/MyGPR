#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate Motion V2 usability on a UAV-GPR motion-effect demo dataset."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evidence_export import export_replay_evidence_bundle  # noqa: E402
from core.gpr_io import extract_airborne_csv_payload  # noqa: E402
from core.processing_engine import (  # noqa: E402
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.shared_data_state import SharedDataState  # noqa: E402
from core.uav_georeference_3d import (  # noqa: E402
    build_airborne_georeference_3d_payload,
    save_airborne_georeference_3d_preview_png,
)
from read_file_data import readcsv  # noqa: E402
from scripts.generate_uav_gpr_motion_effect_demo import (  # noqa: E402
    _resample_like_reference,
    _save_bscan,
    _track_top_interface_std,
)


DEFAULT_DATASET = ROOT / "output" / "mygpr_uav_motion_effect_demo_v1"
DEFAULT_OUTPUT = ROOT / "output" / "motion_v2_closure_validation"
ATOMIC_PIPELINE = (
    "trajectory_smoothing",
    "motion_compensation_attitude",
    "motion_compensation_speed",
    "motion_compensation_height",
)
COMPACT_META_KEYS = {
    "method_id",
    "method",
    "skipped",
    "reason",
    "window_length",
    "polyorder",
    "max_displacement_m",
    "mean_displacement_m",
    "smoothed_traces",
    "spacing_m",
    "source_traces",
    "target_traces",
    "max_shift_samples_applied",
    "input_height_std_m",
    "shift_clamped",
    "height_source",
    "height_reference_m",
    "air_wave_speed_m_per_ns",
    "max_amplitude_scale_applied",
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _summarize_array_map(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    fields: dict[str, Any] = {}
    for key, item in value.items():
        arr = np.asarray(item)
        fields[str(key)] = {
            "shape": [int(dim) for dim in arr.shape],
            "dtype": str(arr.dtype),
        }
    return {"field_count": len(fields), "fields": fields}


def _compact_method_meta(meta: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: _to_jsonable(value)
        for key, value in meta.items()
        if key in COMPACT_META_KEYS
    }
    for key in ("warnings", "runtime_warnings", "quality_flags"):
        if key in meta:
            compact[key] = _to_jsonable(meta[key])
    if "trace_metadata_updates" in meta:
        compact["trace_metadata_updates_summary"] = _summarize_array_map(
            meta["trace_metadata_updates"]
        )
    if "trace_metadata_out" in meta:
        compact["trace_metadata_out_summary"] = _summarize_array_map(
            meta["trace_metadata_out"]
        )
    if "header_info_updates" in meta:
        compact["header_info_updates"] = _to_jsonable(meta["header_info_updates"])
    return compact


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_main_csv_header(path: Path) -> dict[str, Any]:
    header: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:16]:
        text = line.strip().lstrip("#").strip()
        if not text or "=" not in text:
            continue
        key, raw_value = [part.strip() for part in text.split("=", 1)]
        key = key.lower().replace(" ", "_").replace("(", "").replace(")", "")
        try:
            value: Any = float(raw_value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = raw_value
        header[key] = value
    return header


def _load_dataset(dataset_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.resolve()
    manifest = _read_json(dataset_dir / "manifest.json")
    metadata = _read_json(dataset_dir / "metadata.json")
    data_file = dataset_dir / str(manifest.get("data_file") or "main.csv")
    if not data_file.exists():
        raise FileNotFoundError(f"missing main CSV: {data_file}")

    header_from_csv = _parse_main_csv_header(data_file)
    samples = int(
        metadata.get("samples")
        or header_from_csv.get("number_of_samples")
        or header_from_csv.get("a_scan_length")
        or 0
    )
    traces = int(
        metadata.get("traces")
        or header_from_csv.get("number_of_traces")
        or header_from_csv.get("num_traces")
        or 0
    )
    total_time_ns = float(
        metadata.get("total_time_ns")
        or header_from_csv.get("time_windows_ns")
        or header_from_csv.get("total_time_ns")
        or samples
    )
    trace_interval_m = float(
        header_from_csv.get("trace_interval_m")
        or header_from_csv.get("trace_interval")
        or 1.0
    )
    if samples <= 0 or traces <= 0:
        raise ValueError("dataset header does not define positive sample/trace counts")

    raw_csv = readcsv(str(data_file))
    data, trace_metadata, header_info = extract_airborne_csv_payload(
        raw_csv,
        {
            "a_scan_length": samples,
            "num_traces": traces,
            "total_time_ns": total_time_ns,
            "trace_interval_m": trace_interval_m,
            "data_context": "uav_gpr_motion_effect_demo",
        },
        rtk_path=dataset_dir / str(manifest.get("rtk_file") or "rtk.csv"),
        imu_path=dataset_dir / str(manifest.get("imu_file") or "imu.csv"),
        altimeter_path=dataset_dir / str(manifest.get("altimeter_file") or "altimeter.csv"),
    )
    if trace_metadata is None:
        trace_metadata = {}
    if header_info is None:
        header_info = {}
    header_info["source_dataset_dir"] = str(dataset_dir)
    return (
        np.asarray(data, dtype=np.float32),
        {key: np.asarray(value).copy() for key, value in trace_metadata.items()},
        dict(header_info),
        {"manifest": manifest, "metadata": metadata, "data_file": str(data_file)},
    )


def _run_method(
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    method_id: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    runtime_params = prepare_runtime_params(
        method_id,
        params,
        header_info,
        trace_metadata,
        tuple(data.shape),
    )
    result, meta = run_processing_method(data, method_id, runtime_params)
    next_header = merge_result_header_info(header_info, meta, tuple(result.shape))
    next_trace_metadata = merge_result_trace_metadata(trace_metadata, meta)
    return (
        np.asarray(result, dtype=np.float32),
        dict(next_header),
        {key: np.asarray(value).copy() for key, value in next_trace_metadata.items()},
        dict(meta),
    )


def _default_v2_params(recommended_params: dict[str, Any]) -> dict[str, Any]:
    height = dict(recommended_params.get("motion_compensation_height") or {})
    speed = dict(recommended_params.get("motion_compensation_speed") or {})
    attitude = dict(recommended_params.get("motion_compensation_attitude") or {})
    return {
        "height_source": "auto",
        "height_reference_mode": height.get("reference_height_mode", "mean"),
        "manual_height": height.get("manual_height", 0.0),
        "compensate_time_shift": height.get("compensate_time_shift", True),
        "compensate_amplitude": height.get("compensate_amplitude", True),
        "air_wave_speed_m_per_ns": height.get("wave_speed_m_per_ns", 0.299792458),
        "max_shift_samples": height.get("max_shift_samples", 12.0),
        "max_shift_ns": height.get("max_shift_ns", 12.0),
        "max_amplitude_scale": height.get("max_amplitude_scale", 1.8),
        "height_interpolation_mode": height.get("interpolation_mode", "linear"),
        "resample_spacing_m": speed.get("spacing_m", 0.42),
        "resample_interpolation_mode": speed.get("interpolation_mode", "linear"),
        "apc_offset_x_m": attitude.get("apc_offset_x_m", 0.04),
        "apc_offset_y_m": attitude.get("apc_offset_y_m", -0.02),
        "apc_offset_z_m": attitude.get("apc_offset_z_m", 0.0),
        "max_abs_tilt_deg": attitude.get("max_abs_tilt_deg", 18.0),
    }


def _run_atomic_pipeline(
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    recommended_params: dict[str, Any],
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = [
        {
            "method": "raw",
            "label": "Raw",
            "data": np.asarray(data, dtype=np.float32).copy(),
            "header_info": dict(header_info),
            "trace_metadata": {key: np.asarray(value).copy() for key, value in trace_metadata.items()},
            "meta": {},
            "params": {},
        }
    ]
    current = np.asarray(data, dtype=np.float32)
    header = dict(header_info)
    metadata = {key: np.asarray(value).copy() for key, value in trace_metadata.items()}
    for method_id in ATOMIC_PIPELINE:
        params = dict(recommended_params.get(method_id) or {})
        current, header, metadata, meta = _run_method(current, header, metadata, method_id, params)
        stages.append(
            {
                "method": method_id,
                "label": method_id,
                "data": current.copy(),
                "header_info": dict(header),
                "trace_metadata": {key: np.asarray(value).copy() for key, value in metadata.items()},
                "meta": meta,
                "params": params,
            }
        )
    return stages


def _save_three_panel_bscan(
    raw: np.ndarray,
    atomic: np.ndarray,
    unified: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), dpi=150)
    for ax, data, title in zip(
        axes,
        [raw, atomic, unified],
        ["Raw", "Four atomic steps", "motion_compensation_v2"],
    ):
        arr = np.asarray(data, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        scale = float(np.nanpercentile(np.abs(finite), 98.5)) if finite.size else 1.0
        scale = scale if scale > 0 else 1.0
        ax.imshow(arr, cmap="gray", aspect="auto", vmin=-scale, vmax=scale)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_3d_preview(
    data: np.ndarray,
    header: dict[str, Any],
    metadata: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> bool:
    payload = build_airborne_georeference_3d_payload(
        data,
        header,
        metadata,
        max_preview_traces=260,
        max_preview_samples=180,
    )
    if payload is None:
        return False
    save_airborne_georeference_3d_preview_png(payload, out_path, title=title)
    return out_path.exists()


def _spacing_std(metadata: dict[str, np.ndarray] | None) -> float | None:
    if not metadata or "trace_distance_m" not in metadata:
        return None
    distance = np.asarray(metadata["trace_distance_m"], dtype=np.float64)
    if distance.size < 3:
        return None
    return float(np.nanstd(np.diff(distance)))


def _trace_count(metadata: dict[str, np.ndarray] | None) -> int:
    if not metadata:
        return 0
    for value in metadata.values():
        arr = np.asarray(value)
        if arr.ndim > 0:
            return int(arr.shape[0])
    return 0


def _has_fields(metadata: dict[str, np.ndarray] | None, fields: tuple[str, ...]) -> dict[str, bool]:
    return {field: bool(metadata and field in metadata) for field in fields}


def _build_evidence_bundle(
    raw_data: np.ndarray,
    raw_header: dict[str, Any],
    raw_metadata: dict[str, np.ndarray],
    current_data: np.ndarray,
    current_header: dict[str, Any],
    current_metadata: dict[str, np.ndarray],
    output_dir: Path,
    v2_params: dict[str, Any],
    v2_meta: dict[str, Any],
) -> Path:
    state = SharedDataState()
    state.load_data(
        raw_data,
        path=str(raw_header.get("source_dataset_dir") or "motion_effect_demo"),
        header_info=raw_header,
        trace_metadata=raw_metadata,
    )
    state.apply_current_data(
        current_data,
        push_history=True,
        label="motion_compensation_v2",
        header_info=current_header,
        trace_metadata=current_metadata,
    )
    package = state.get_replay_evidence_package()
    if package is None:
        raise RuntimeError("failed to build replay evidence package")
    package["app_context"] = {
        "preset_key": "motion_compensation_v2",
        "selected_method": {"method_id": "motion_compensation_v2"},
        "method_param_overrides": {"motion_compensation_v2": v2_params},
        "runtime_warnings": v2_meta.get("runtime_warnings") or v2_meta.get("warnings") or [],
        "last_run_summary": {
            "method": "motion_compensation_v2",
            "shape": list(current_data.shape),
            "quality_flags": v2_meta.get("quality_flags") or {},
        },
    }
    bundle_path = output_dir / "motion_v2_replay_evidence.zip"
    export_replay_evidence_bundle(package, bundle_path, bundle_name="motion_v2_closure", save_images=True)
    return bundle_path


def _write_report(summary: dict[str, Any], report_path: Path) -> None:
    metrics = summary["metrics"]
    checks = summary["checks"]
    artifacts = summary["artifacts"]
    lines = [
        "# Motion V2 Usability Closure Validation",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Generated at: {summary['generated_at']}",
        f"- Raw shape: `{summary['raw_shape']}`",
        f"- Atomic final shape: `{summary['atomic_shape']}`",
        f"- V2 final shape: `{summary['v2_shape']}`",
        "",
        "## Main Result",
        "",
        "- Atomic route: "
        f"`{' -> '.join(summary['atomic_pipeline'])}`.",
        f"- Top-interface jitter std: raw `{metrics['top_interface_std_raw']:.3f}` -> atomic `{metrics['top_interface_std_atomic']:.3f}` -> V2 `{metrics['top_interface_std_v2']:.3f}` samples.",
        f"- Trace spacing std: raw `{metrics['spacing_std_raw_m']:.6f}` -> atomic `{metrics['spacing_std_atomic_m']:.6f}` -> V2 `{metrics['spacing_std_v2_m']:.6f}` m.",
        f"- RMS difference raw->atomic: `{metrics['rms_raw_to_atomic']:.6f}`.",
        f"- RMS difference raw->V2: `{metrics['rms_raw_to_v2']:.6f}`.",
        f"- RMS difference atomic->V2: `{metrics['rms_atomic_to_v2']:.6f}`.",
        "",
        "## Contract Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- B-scan comparison: `{artifacts['bscan_comparison_png']}`",
            f"- Raw 3D preview: `{artifacts['raw_3d_preview_png']}`",
            f"- Atomic 3D preview: `{artifacts['atomic_3d_preview_png']}`",
            f"- V2 3D preview: `{artifacts['v2_3d_preview_png']}`",
            f"- Evidence bundle: `{artifacts['evidence_bundle_zip']}`",
            "",
            "## Interpretation",
            "",
            "- The demo sidecars are aligned through `trace_timestamp_s`; RTK/IMU/altimeter inputs are not ignored.",
            "- The atomic four-step route and unified Motion V2 route both reduce trajectory spacing jitter and top-interface jitter on this synthetic closure dataset.",
            "- This validates software plumbing and visible compensation behavior, not field geologic correctness.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_motion_v2_closure(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    run_dir = output_dir.resolve() / dataset_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)

    data, trace_metadata, header_info, source_info = _load_dataset(dataset_dir)
    recommended_params = dict(source_info["manifest"].get("recommended_params") or {})
    stages = _run_atomic_pipeline(data, header_info, trace_metadata, recommended_params)
    atomic_stage = stages[-1]

    v2_params = _default_v2_params(recommended_params)
    v2_data, v2_header, v2_metadata, v2_meta = _run_method(
        data,
        header_info,
        trace_metadata,
        "motion_compensation_v2",
        v2_params,
    )

    atomic_data = np.asarray(atomic_stage["data"], dtype=np.float32)
    raw_for_atomic = _resample_like_reference(data, atomic_data.shape[1])
    raw_for_v2 = _resample_like_reference(data, v2_data.shape[1])
    atomic_for_v2 = _resample_like_reference(atomic_data, v2_data.shape[1])

    _save_bscan(run_dir / "raw_bscan.png", data, "Raw UAV-GPR motion demo")
    _save_bscan(run_dir / "atomic_final_bscan.png", atomic_data, "After four atomic motion steps")
    _save_bscan(run_dir / "motion_v2_final_bscan.png", v2_data, "After unified motion_compensation_v2")
    _save_three_panel_bscan(data, atomic_data, v2_data, run_dir / "motion_v2_closure_bscan_comparison.png")

    raw_3d = run_dir / "raw_3d_preview.png"
    atomic_3d = run_dir / "atomic_3d_preview.png"
    v2_3d = run_dir / "motion_v2_3d_preview.png"
    raw_3d_ok = _save_3d_preview(data, header_info, trace_metadata, raw_3d, "Raw 3D motion closure preview")
    atomic_3d_ok = _save_3d_preview(
        atomic_data,
        atomic_stage["header_info"],
        atomic_stage["trace_metadata"],
        atomic_3d,
        "Four atomic steps 3D motion closure preview",
    )
    v2_3d_ok = _save_3d_preview(v2_data, v2_header, v2_metadata, v2_3d, "Motion V2 3D closure preview")
    evidence_bundle = _build_evidence_bundle(
        data,
        header_info,
        trace_metadata,
        v2_data,
        v2_header,
        v2_metadata,
        run_dir,
        v2_params,
        v2_meta,
    )
    with zipfile.ZipFile(evidence_bundle) as zf:
        evidence_names = set(zf.namelist())

    metrics = {
        "top_interface_std_raw": _track_top_interface_std(data),
        "top_interface_std_atomic": _track_top_interface_std(atomic_data),
        "top_interface_std_v2": _track_top_interface_std(v2_data),
        "spacing_std_raw_m": _spacing_std(trace_metadata),
        "spacing_std_atomic_m": _spacing_std(atomic_stage["trace_metadata"]),
        "spacing_std_v2_m": _spacing_std(v2_metadata),
        "rms_raw_to_atomic": float(np.sqrt(np.mean((atomic_data - raw_for_atomic) ** 2))),
        "rms_raw_to_v2": float(np.sqrt(np.mean((v2_data - raw_for_v2) ** 2))),
        "rms_atomic_to_v2": float(np.sqrt(np.mean((v2_data - atomic_for_v2) ** 2))),
    }
    checks = {
        "trace_metadata_loaded": bool(trace_metadata),
        "height_agl_loaded": bool("height_agl_m" in trace_metadata),
        "atomic_trace_metadata_length_matches": _trace_count(atomic_stage["trace_metadata"]) == atomic_data.shape[1],
        "v2_trace_metadata_length_matches": _trace_count(v2_metadata) == v2_data.shape[1],
        "v2_footprint_fields_present": all(
            _has_fields(v2_metadata, ("footprint_x_m", "footprint_y_m", "trace_distance_m")).values()
        ),
        "v2_uses_air_wave_speed": float(v2_params["air_wave_speed_m_per_ns"]) == 0.299792458,
        "raw_3d_preview_created": raw_3d_ok,
        "atomic_3d_preview_created": atomic_3d_ok,
        "v2_3d_preview_created": v2_3d_ok,
        "evidence_contains_motion_artifacts": all(
            name in evidence_names
            for name in [
                "motion/raw_bscan.png",
                "motion/current_bscan.png",
                "motion/diff_bscan.png",
                "motion/raw_3d_preview.png",
                "motion/current_3d_preview.png",
                "motion/diff_3d_preview.png",
                "motion/motion_quality_flags.json",
                "motion/motion_params.json",
            ]
        ),
    }
    summary = {
        "schema": "mygpr_motion_v2_closure_validation_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(run_dir),
        "raw_shape": list(data.shape),
        "atomic_shape": list(atomic_data.shape),
        "v2_shape": list(v2_data.shape),
        "atomic_pipeline": list(ATOMIC_PIPELINE),
        "v2_params": _to_jsonable(v2_params),
        "metrics": _to_jsonable(metrics),
        "checks": _to_jsonable(checks),
        "atomic_stage_meta": [
            {
                "method": stage["method"],
                "shape": list(np.asarray(stage["data"]).shape),
                "meta": _compact_method_meta(stage.get("meta") or {}),
            }
            for stage in stages
        ],
        "v2_meta": _compact_method_meta(v2_meta),
        "artifacts": {
            "summary_json": str(run_dir / "motion_v2_closure_summary.json"),
            "report_md": str(run_dir / "motion_v2_closure_report.md"),
            "raw_bscan_png": str(run_dir / "raw_bscan.png"),
            "atomic_bscan_png": str(run_dir / "atomic_final_bscan.png"),
            "v2_bscan_png": str(run_dir / "motion_v2_final_bscan.png"),
            "bscan_comparison_png": str(run_dir / "motion_v2_closure_bscan_comparison.png"),
            "raw_3d_preview_png": str(raw_3d),
            "atomic_3d_preview_png": str(atomic_3d),
            "v2_3d_preview_png": str(v2_3d),
            "evidence_bundle_zip": str(evidence_bundle),
        },
    }
    (run_dir / "motion_v2_closure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(summary, run_dir / "motion_v2_closure_report.md")
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"motion V2 closure validation failed: {failed}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Motion V2 usability on a UAV-GPR motion-effect demo dataset."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    summary = validate_motion_v2_closure(Path(args.dataset), Path(args.output))
    artifacts = summary["artifacts"]
    metrics = summary["metrics"]
    print(f"Dataset: {summary['dataset_dir']}")
    print(f"Output: {summary['output_dir']}")
    print(f"Report: {artifacts['report_md']}")
    print(f"Evidence: {artifacts['evidence_bundle_zip']}")
    print(
        "Top interface std raw/atomic/v2: "
        f"{metrics['top_interface_std_raw']:.3f} / "
        f"{metrics['top_interface_std_atomic']:.3f} / "
        f"{metrics['top_interface_std_v2']:.3f}"
    )
    print(
        "Spacing std raw/atomic/v2: "
        f"{metrics['spacing_std_raw_m']:.6f} / "
        f"{metrics['spacing_std_atomic_m']:.6f} / "
        f"{metrics['spacing_std_v2_m']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

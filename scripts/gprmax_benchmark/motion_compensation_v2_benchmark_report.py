#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a UAV-GPR motion compensation V2 benchmark HTML report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PythonModule.motion_compensation_v2 import (  # noqa: E402
    AIR_WAVE_SPEED_M_PER_NS,
    method_motion_compensation_v2,
)
from PythonModule.motion_compensation_core import resample_bscan_columns  # noqa: E402
from core.sidecar_integration import load_and_integrate_optional_sidecars  # noqa: E402
from scripts.gprmax_benchmark.gprmax_multi_scenario_report import (  # noqa: E402
    build_scenario_definitions,
    build_synthetic_trace_metadata,
    write_synthetic_sidecars,
)


DEFAULT_OUTPUT_ROOT = ROOT / "output" / "motion_compensation_v2_benchmark_reports"
DEFAULT_SCENARIO = "airborne_height_variation_cylinder_v1"


def run_motion_v2_benchmark_report(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_scenario_dir: Path | None = None,
    scenario: str = DEFAULT_SCENARIO,
    runs: int = 48,
    sample_count: int = 256,
    total_time_ns: float = 18.0,
    max_shift_samples: float = 0.0,
    max_shift_ns: float = 20.0,
    save_images_flag: bool = True,
) -> dict[str, Any]:
    """Run motion compensation V2 and write a timestamped report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_root) / f"motion_v2_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    if source_scenario_dir is None:
        data, metadata, header_info, source = _build_synthetic_case(
            report_dir,
            scenario_id=scenario,
            runs=runs,
            sample_count=sample_count,
            total_time_ns=total_time_ns,
        )
    else:
        data, metadata, header_info, source = _load_source_scenario_dir(
            Path(source_scenario_dir)
        )

    params = {
        "height_reference_mode": "mean",
        "height_source": "auto",
        "compensate_time_shift": True,
        "compensate_amplitude": True,
        "max_shift_samples": float(max_shift_samples),
        "max_shift_ns": float(max_shift_ns),
        "max_amplitude_scale": 2.0,
        "resample_spacing_m": 0.0,
    }
    corrected, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        header_info=header_info,
        time_window_ns=header_info.get("total_time_ns"),
        **params,
    )
    corrected = np.asarray(corrected, dtype=np.float32)

    payload = _build_payload(
        report_dir=report_dir,
        source=source,
        params=params,
        raw=data,
        corrected=corrected,
        trace_metadata=metadata,
        motion_meta=meta,
        save_images_flag=save_images_flag,
    )
    summary_json = report_dir / "summary.json"
    summary_json.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path = report_dir / "index.html"
    html_path.write_text(_render_html(payload), encoding="utf-8")
    payload["artifacts"]["html"] = str(html_path.resolve())
    payload["artifacts"]["summary_json"] = str(summary_json.resolve())
    summary_json.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _jsonable(payload)


def _build_synthetic_case(
    report_dir: Path,
    *,
    scenario_id: str,
    runs: int,
    sample_count: int,
    total_time_ns: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    scenarios = build_scenario_definitions("airborne")
    if scenario_id not in scenarios:
        raise ValueError(f"Unsupported airborne scenario: {scenario_id}")
    definition = scenarios[scenario_id]
    trace_count = max(2, int(runs))
    metadata = build_synthetic_trace_metadata(definition, trace_count=trace_count)
    sidecars_dir = report_dir / "synthetic_sidecars"
    sidecars_dir.mkdir(parents=True, exist_ok=True)
    sidecars = write_synthetic_sidecars(definition, sidecars_dir, runs=trace_count)
    data = _build_motion_affected_bscan(
        metadata,
        sample_count=max(16, int(sample_count)),
        total_time_ns=float(total_time_ns),
    )
    bscan_csv = report_dir / "synthetic_bscan.csv"
    np.savetxt(bscan_csv, data, delimiter=",", fmt="%.8e")
    header_info = {
        "total_time_ns": float(total_time_ns),
        "data_context": "gprmax_airborne_motion_v2_synthetic",
    }
    source = {
        "kind": "synthetic_airborne_motion_case",
        "scenario_id": scenario_id,
        "scenario_label": definition.label,
        "bscan_csv": str(bscan_csv.resolve()),
        "sidecars": {key: str(path.resolve()) for key, path in sidecars.items()},
        "notes": "Synthetic motion-affected B-scan built from airborne height profile; no gprMax run was executed.",
    }
    return data, metadata, header_info, source


def _load_source_scenario_dir(
    scenario_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    source_dir = Path(scenario_dir)
    bscan_csv = source_dir / "mygpr_bscan.csv"
    if not bscan_csv.exists():
        raise FileNotFoundError(f"missing MyGPR B-scan CSV: {bscan_csv}")
    data = _load_bscan_csv(bscan_csv)
    trace_count = int(data.shape[1])
    scenario_json = _read_json_if_exists(source_dir / "scenario.json")
    simulation = scenario_json.get("simulation", {}) if scenario_json else {}
    total_time_ns = simulation.get("total_time_ns") or scenario_json.get("total_time_ns")
    header_info = {
        "total_time_ns": float(total_time_ns) if total_time_ns else 0.0,
        "data_context": "gprmax_airborne_motion_v2_source_dir",
    }
    base_metadata: dict[str, np.ndarray] = {
        "trace_index": np.arange(trace_count, dtype=np.int32),
    }
    trace_step = simulation.get("trace_step_m") or scenario_json.get("trace_step_m")
    if trace_step is not None:
        base_metadata["trace_distance_m"] = (
            np.arange(trace_count, dtype=np.float64) * float(trace_step)
        )
    trace_timestamps = _read_trace_timestamps(source_dir / "trace_timestamps.csv")
    if trace_timestamps is None:
        trace_timestamps = np.arange(trace_count, dtype=np.float64)
    metadata = load_and_integrate_optional_sidecars(
        base_metadata,
        trace_timestamps_s=trace_timestamps,
        rtk_path=_existing_path(source_dir / "rtk.csv"),
        imu_path=_existing_path(source_dir / "imu.csv"),
        altimeter_path=_existing_path(source_dir / "altimeter.csv"),
    )
    source = {
        "kind": "gprmax_airborne_scenario_dir",
        "scenario_dir": str(source_dir.resolve()),
        "scenario_id": scenario_json.get("scenario_id", source_dir.name),
        "scenario_label": scenario_json.get("label", source_dir.name),
        "bscan_csv": str(bscan_csv.resolve()),
        "sidecars": {
            name: str((source_dir / filename).resolve())
            for name, filename in {
                "trace_timestamps": "trace_timestamps.csv",
                "rtk": "rtk.csv",
                "imu": "imu.csv",
                "altimeter": "altimeter.csv",
            }.items()
            if (source_dir / filename).exists()
        },
    }
    return data, metadata, header_info, source


def _build_motion_affected_bscan(
    metadata: dict[str, np.ndarray],
    *,
    sample_count: int,
    total_time_ns: float,
) -> np.ndarray:
    trace_count = int(np.asarray(metadata["trace_timestamp_s"]).size)
    traces = np.arange(trace_count, dtype=np.float64)
    samples = np.arange(sample_count, dtype=np.float64)[:, None]
    phase = traces / max(trace_count - 1, 1)
    base = 0.006 * np.sin(2.0 * np.pi * (0.025 * samples + 1.7 * phase[None, :]))
    base += _pulse_grid(sample_count, trace_count, 18.0, 2.4, amplitude=0.55)
    base += _pulse_grid(
        sample_count,
        trace_count,
        58.0 + 9.0 * (phase - 0.5) ** 2,
        3.2,
        amplitude=1.0,
    )
    base += _pulse_grid(
        sample_count,
        trace_count,
        112.0 + 8.0 * np.sin(2.0 * np.pi * phase),
        4.0,
        amplitude=0.45,
    )
    height = np.asarray(metadata.get("flight_height_m"), dtype=np.float64)
    ref = float(np.mean(height)) if height.size else 1.0
    dt_ns = float(total_time_ns) / max(sample_count - 1, 1)
    shift = 2.0 * (height - ref) / AIR_WAVE_SPEED_M_PER_NS / max(dt_ns, 1.0e-9)
    scale = np.clip((ref / np.maximum(height, 1.0e-6)) ** 2, 0.5, 2.0)
    out = np.empty_like(base, dtype=np.float32)
    sample_axis = np.arange(sample_count, dtype=np.float64)
    for col in range(trace_count):
        out[:, col] = np.interp(
            sample_axis - float(shift[col]),
            sample_axis,
            base[:, col],
            left=0.0,
            right=0.0,
        ).astype(np.float32)
        out[:, col] *= np.float32(scale[col])
    return out


def _pulse_grid(
    sample_count: int,
    trace_count: int,
    center: float | np.ndarray,
    width: float,
    *,
    amplitude: float,
) -> np.ndarray:
    rows = np.arange(sample_count, dtype=np.float64)[:, None]
    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.ndim == 0:
        center_arr = np.full(trace_count, float(center_arr), dtype=np.float64)
    return float(amplitude) * np.exp(-0.5 * ((rows - center_arr[None, :]) / float(width)) ** 2)


def _build_payload(
    *,
    report_dir: Path,
    source: dict[str, Any],
    params: dict[str, Any],
    raw: np.ndarray,
    corrected: np.ndarray,
    trace_metadata: dict[str, np.ndarray],
    motion_meta: dict[str, Any],
    save_images_flag: bool,
) -> dict[str, Any]:
    comparison_raw, comparison_corrected, comparison_alignment = _comparison_arrays(
        raw,
        corrected,
        trace_metadata=trace_metadata,
        motion_meta=motion_meta,
    )
    delta = comparison_corrected - comparison_raw
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, str] = {}
    if save_images_flag:
        vlim = _robust_vlim(raw, corrected)
        image_specs = {
            "raw_bscan": (raw, "补偿前 B-scan", vlim),
            "corrected_bscan": (corrected, "补偿后 B-scan", vlim),
            "delta_bscan": (delta, "补偿差异", _robust_vlim(delta)),
        }
        for key, (arr, title, limit) in image_specs.items():
            path = assets_dir / f"{key}.png"
            _save_bscan_image(arr, path, title, limit)
            images[key] = _relpath(path, report_dir)
        curves = _save_curve_images(
            assets_dir,
            report_dir,
            trace_metadata=trace_metadata,
            motion_meta=motion_meta,
        )
        images.update(curves)

    warnings = list(motion_meta.get("runtime_warnings", []) or [])
    warning_codes = [str(item.get("code", "")) for item in warnings]
    motion_curves = _extract_motion_curves(trace_metadata, motion_meta)
    return {
        "report_type": "motion_compensation_v2_benchmark",
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "params": params,
        "input_summary": _array_summary(raw),
        "output_summary": _array_summary(corrected),
        "comparison_reference_summary": _array_summary(comparison_raw),
        "comparison_alignment": comparison_alignment,
        "correction_metrics": _correction_metrics(comparison_raw, comparison_corrected),
        "motion_meta": _jsonable(motion_meta),
        "input_quality": _jsonable(motion_meta.get("input_quality", {})),
        "quality_flags": list(motion_meta.get("quality_flags", []) or []),
        "warning_codes": warning_codes,
        "runtime_warnings": _jsonable(warnings),
        "motion_curves": _jsonable(motion_curves),
        "artifacts": {
            "report_dir": str(report_dir.resolve()),
            "assets_dir": str(assets_dir.resolve()),
            "images": images,
        },
    }


def _comparison_arrays(
    raw: np.ndarray,
    corrected: np.ndarray,
    *,
    trace_metadata: dict[str, np.ndarray],
    motion_meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return raw B-scan aligned to corrected trace axis for report deltas."""
    raw_arr = np.asarray(raw, dtype=np.float32)
    corrected_arr = np.asarray(corrected, dtype=np.float32)
    if raw_arr.shape == corrected_arr.shape:
        return raw_arr, corrected_arr, {"mode": "same_shape", "shape": list(raw_arr.shape)}

    trace_metadata_out = motion_meta.get("trace_metadata_out")
    if isinstance(trace_metadata_out, dict):
        source_distance = np.asarray(trace_metadata.get("trace_distance_m"), dtype=np.float64)
        target_distance = np.asarray(trace_metadata_out.get("trace_distance_m"), dtype=np.float64)
        if (
            source_distance.ndim == 1
            and target_distance.ndim == 1
            and source_distance.size == raw_arr.shape[1]
            and target_distance.size == corrected_arr.shape[1]
            and np.isfinite(source_distance).all()
            and np.isfinite(target_distance).all()
            and not np.any(np.diff(source_distance) < 0.0)
        ):
            aligned = resample_bscan_columns(raw_arr, source_distance, target_distance)
            return aligned.astype(np.float32, copy=False), corrected_arr, {
                "mode": "resampled_raw_to_corrected_trace_axis",
                "source_traces": int(source_distance.size),
                "target_traces": int(target_distance.size),
                "source_start_m": float(source_distance[0]),
                "source_end_m": float(source_distance[-1]),
                "target_start_m": float(target_distance[0]),
                "target_end_m": float(target_distance[-1]),
            }

    common_traces = min(raw_arr.shape[1], corrected_arr.shape[1])
    common_samples = min(raw_arr.shape[0], corrected_arr.shape[0])
    return raw_arr[:common_samples, :common_traces], corrected_arr[:common_samples, :common_traces], {
        "mode": "cropped_fallback",
        "raw_shape": list(raw_arr.shape),
        "corrected_shape": list(corrected_arr.shape),
        "comparison_shape": [int(common_samples), int(common_traces)],
    }


def _save_bscan_image(data: np.ndarray, path: Path, title: str, vlim: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    ax.imshow(np.asarray(data, dtype=np.float32), cmap="gray", aspect="auto", vmin=-vlim, vmax=vlim)
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_curve_images(
    assets_dir: Path,
    report_dir: Path,
    *,
    trace_metadata: dict[str, np.ndarray],
    motion_meta: dict[str, Any],
) -> dict[str, str]:
    curves = _extract_motion_curves(trace_metadata, motion_meta)
    images: dict[str, str] = {}
    trace_idx = np.arange(int(curves["trace_count"]), dtype=np.int32)
    if curves.get("height_agl_m"):
        path = assets_dir / "height_profile.png"
        _save_line_plot(path, trace_idx, curves["height_agl_m"], "AGL 高度曲线", "Trace", "Height (m)")
        images["height_profile"] = _relpath(path, report_dir)
    if curves.get("time_shift_samples"):
        path = assets_dir / "time_shift_samples.png"
        _save_line_plot(
            path,
            trace_idx,
            curves["time_shift_samples"],
            "高度时移曲线",
            "Trace",
            "Shift (samples)",
        )
        images["time_shift_samples"] = _relpath(path, report_dir)
    if curves.get("trace_distance_m"):
        path = assets_dir / "trace_distance.png"
        _save_line_plot(
            path,
            trace_idx,
            curves["trace_distance_m"],
            "轨迹累计距离",
            "Trace",
            "Distance (m)",
        )
        images["trace_distance"] = _relpath(path, report_dir)
    return images


def _save_line_plot(
    path: Path,
    x: np.ndarray,
    y_values: Any,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    y = np.asarray(y_values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=140)
    ax.plot(x[: y.size], y, color="#1864ab", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _extract_motion_curves(
    trace_metadata: dict[str, np.ndarray],
    motion_meta: dict[str, Any],
) -> dict[str, Any]:
    updates = motion_meta.get("trace_metadata_updates", {}) or {}
    trace_count = int(motion_meta.get("source_traces", 0) or _metadata_trace_count(trace_metadata))
    curves = {"trace_count": trace_count}
    for key in (
        "height_agl_m",
        "height_confidence",
        "trace_distance_m",
        "trace_timestamp_s",
        "time_shift_ns",
        "time_shift_samples",
        "height_amplitude_scale",
    ):
        value = motion_meta.get(key)
        if value is None and isinstance(updates, dict):
            value = updates.get(key)
        if value is None:
            value = trace_metadata.get(key)
        if value is not None:
            arr = np.asarray(value)
            if arr.ndim == 1 and arr.size:
                curves[key] = arr.astype(float, copy=False).tolist()
    return curves


def _render_html(payload: dict[str, Any]) -> str:
    source = payload.get("source", {})
    images = payload.get("artifacts", {}).get("images", {})
    warning_codes = payload.get("warning_codes", [])
    cards = [
        ("输入", payload.get("input_summary", {})),
        ("输出", payload.get("output_summary", {})),
        ("补偿指标", payload.get("correction_metrics", {})),
        ("输入质量", payload.get("input_quality", {})),
    ]
    figures = "".join(
        f'<figure><img src="{_esc(path)}" alt="{_esc(key)}"><figcaption>{_esc(_image_label(key))}</figcaption></figure>'
        for key, path in images.items()
    )
    warnings_html = (
        "<ul>"
        + "".join(f"<li><code>{_esc(code)}</code></li>" for code in warning_codes)
        + "</ul>"
        if warning_codes
        else "<p>未记录运行告警。</p>"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>motion_compensation_v2 Benchmark</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #17202a; background: #f6f8fb; }}
    header {{ background: #102a43; color: white; padding: 24px 32px; }}
    main {{ padding: 24px 32px 40px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ background: #fff; border: 1px solid #d8e0ea; border-radius: 8px; padding: 18px; margin: 0 0 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #d8e0ea; border-radius: 8px; padding: 12px; background: #fbfcfe; }}
    figure {{ margin: 0; border: 1px solid #d8e0ea; border-radius: 8px; background: #fff; padding: 8px; }}
    figure img {{ width: 100%; display: block; }}
    figcaption {{ font-size: 13px; color: #334e68; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border-bottom: 1px solid #edf2f7; padding: 7px 8px; text-align: left; vertical-align: top; }}
    code {{ background: #edf2f7; padding: 1px 4px; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; background: #0b1720; color: #d9f99d; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <header>
    <h1>motion_compensation_v2 Benchmark</h1>
    <p>补偿前后 B-scan、时移/高度曲线、输入质量与运行告警。</p>
  </header>
  <main>
    <section>
      <h2>数据来源</h2>
      <table>
        <tr><th>字段</th><th>值</th></tr>
        <tr><td>来源类型</td><td>{_esc(source.get("kind"))}</td></tr>
        <tr><td>场景</td><td>{_esc(source.get("scenario_id", ""))} {_esc(source.get("scenario_label", ""))}</td></tr>
        <tr><td>B-scan</td><td><code>{_esc(source.get("bscan_csv", ""))}</code></td></tr>
      </table>
    </section>
    <section>
      <h2>补偿前后 B-scan</h2>
      <div class="grid">{figures}</div>
    </section>
    <section>
      <h2>质量摘要</h2>
      <div class="grid">{''.join(_summary_card(title, data) for title, data in cards)}</div>
    </section>
    <section>
      <h2>质量告警</h2>
      {warnings_html}
    </section>
    <section>
      <h2>运行参数</h2>
      <pre>{_esc(json.dumps(_jsonable(payload.get("params", {})), ensure_ascii=False, indent=2))}</pre>
    </section>
    <section>
      <h2>motion_compensation_v2 元数据</h2>
      <pre>{_esc(json.dumps(_compact_motion_meta(payload.get("motion_meta", {})), ensure_ascii=False, indent=2))}</pre>
    </section>
  </main>
</body>
</html>
"""


def _summary_card(title: str, data: Any) -> str:
    if not isinstance(data, dict):
        data = {"value": data}
    rows = "".join(
        f"<tr><td>{_esc(key)}</td><td>{_esc(_format_value(value))}</td></tr>"
        for key, value in data.items()
    )
    return f'<div class="card"><h3>{_esc(title)}</h3><table>{rows}</table></div>'


def _compact_motion_meta(meta: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in dict(meta).items()
        if key not in {"trace_metadata_updates", "trace_metadata_out"}
    }
    return _jsonable(compact)


def _image_label(key: str) -> str:
    return {
        "raw_bscan": "补偿前 B-scan",
        "corrected_bscan": "补偿后 B-scan",
        "delta_bscan": "补偿差异",
        "height_profile": "AGL 高度曲线",
        "time_shift_samples": "高度时移曲线",
        "trace_distance": "轨迹累计距离",
    }.get(key, key)


def _array_summary(data: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"shape": list(arr.shape), "finite_ratio": 0.0}
    return {
        "shape": [int(item) for item in arr.shape],
        "finite_ratio": float(finite.size / arr.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite**2))),
    }


def _correction_metrics(raw: np.ndarray, corrected: np.ndarray) -> dict[str, float]:
    raw_arr = np.asarray(raw, dtype=np.float64)
    corrected_arr = np.asarray(corrected, dtype=np.float64)
    diff = corrected_arr - raw_arr
    finite = np.isfinite(diff) & np.isfinite(raw_arr) & np.isfinite(corrected_arr)
    if not finite.any():
        return {}
    diff_f = diff[finite]
    raw_f = raw_arr[finite]
    raw_rms = float(np.sqrt(np.mean(raw_f**2)))
    return {
        "mean_abs_delta": float(np.mean(np.abs(diff_f))),
        "rms_delta": float(np.sqrt(np.mean(diff_f**2))),
        "relative_rms_delta": float(np.sqrt(np.mean(diff_f**2)) / max(raw_rms, 1.0e-12)),
        "max_abs_delta": float(np.max(np.abs(diff_f))),
    }


def _robust_vlim(*arrays: np.ndarray) -> float:
    finite_abs = _finite_abs_values(*arrays)
    if finite_abs.size == 0:
        return 1.0
    return max(float(np.percentile(finite_abs, 98.5)), 1.0e-6)


def _finite_abs_values(*arrays: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float64).reshape(-1)
        if values.size == 0:
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(np.abs(finite))
    if not chunks:
        return np.asarray([], dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _load_bscan_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"expected non-empty 2D B-scan CSV: {path}")
    return arr.astype(np.float32, copy=False)


def _read_trace_timestamps(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "timestamp_s" not in reader.fieldnames:
            raise ValueError(f"trace timestamp CSV must contain timestamp_s: {path}")
        for row in reader:
            values.append(float(row["timestamp_s"]))
    return np.asarray(values, dtype=np.float64)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _metadata_trace_count(metadata: dict[str, np.ndarray]) -> int:
    for value in metadata.values():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size > 1:
            return int(arr.size)
    return 0


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list) and len(value) > 8:
        return f"[{', '.join(_format_value(item) for item in value[:4])}, ...] ({len(value)} items)"
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a motion_compensation_v2 benchmark HTML report."
    )
    parser.add_argument("--source-scenario-dir", type=Path, default=None)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--runs", type=int, default=48)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--total-time-ns", type=float, default=18.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-shift-samples", type=float, default=0.0)
    parser.add_argument("--max-shift-ns", type=float, default=20.0)
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_motion_v2_benchmark_report(
        output_root=args.output_root,
        source_scenario_dir=args.source_scenario_dir,
        scenario=args.scenario,
        runs=args.runs,
        sample_count=args.sample_count,
        total_time_ns=args.total_time_ns,
        max_shift_samples=args.max_shift_samples,
        max_shift_ns=args.max_shift_ns,
        save_images_flag=not args.no_images,
    )
    print(f"HTML report: {payload['artifacts']['html']}")
    print(f"Summary JSON: {payload['artifacts']['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

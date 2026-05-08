#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare gain-method choices on existing gprMax validation scenes."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from dataclasses import dataclass
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

from core.auto_tune import auto_tune_method  # noqa: E402
from core.gprmax_truth_metrics import compute_ground_truth_metrics  # noqa: E402
from core.methods_registry import PROCESSING_METHODS  # noqa: E402
from core.preset_profiles import RECOMMENDED_RUN_PROFILES  # noqa: E402
from core.processing_engine import (  # noqa: E402
    clone_header_info,
    clone_trace_metadata,
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.quality_metrics import (  # noqa: E402
    clipping_ratio,
    compute_benchmark_metrics,
    hot_pixel_ratio,
    ratio_fidelity,
)
from scripts.gprmax_benchmark import gprmax_multi_scenario_report as base_report  # noqa: E402


DEFAULT_SOURCE_ROOT = ROOT / "output" / "gprmax_full_effect_reports"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "gprmax_gain_method_reports"
DEFAULT_BASELINE_PROFILE = "uav_gpr_experience_baseline_v1"
PRE_GAIN_METHODS = ["set_zero_time", "dewow", "subtracting_average_2D"]
POST_GAIN_METHODS = ["svd_subspace"]
GAIN_METHODS = ["sec_gain", "agcGain", "compensatingGain", "no_gain"]

MANUAL_GAIN_PARAMS = {
    "sec_gain": {"gain_min": 1.0, "gain_max": 4.2, "power": 1.2},
    "agcGain": {"window": 121},
    "compensatingGain": {"gain_min": 1.0, "gain_max": 6.0},
    "no_gain": {},
}

GAIN_METHOD_NOTES = {
    "sec_gain": {
        "label": "SEC 深度补偿",
        "principle": "同一深度样点使用同一条随时间增加的增益曲线，倾向保留横向相对振幅。",
        "best_for": "幅值解释、层状界面、材料差异比较、gprMax 真值论证，以及噪声不希望被局部均衡放大的数据。",
        "risks": "指数/幂次过大时会抬升晚时窗噪声，必须限制最大增益并检查深部过曝。",
    },
    "agcGain": {
        "label": "AGC 自动增益",
        "principle": "每道按局部 RMS 归一化，优先让弱事件可见，但会破坏相对振幅。",
        "best_for": "只关心结构连续性或弱双曲线可见性的快速解释图，不用于判断反射强弱。",
        "risks": "窗口过短或低能量区域过多时会放大空白区、噪声和条带，且不能保留强弱反射关系。",
    },
    "compensatingGain": {
        "label": "线性/手动 TGC",
        "principle": "从浅到深施加固定深度曲线，比 AGC 更保守，比 SEC 更弱。",
        "best_for": "衰减较轻、介质未知或需要低风险预览的数据。",
        "risks": "深部衰减明显时补偿不足，难以显著拉出深部弱目标。",
    },
    "no_gain": {
        "label": "不施加增益",
        "principle": "保留前序处理后的原始幅值尺度，作为报告中的安全参照。",
        "best_for": "无目标背景、强噪声或需要确认增益是否制造假异常的 QA 场景。",
        "risks": "深部目标可能不可见，不适合作为最终解释图。",
    },
}

RESEARCH_SOURCES = [
    {
        "title": "Noviello et al. 2022, An Overview on Down-Looking UAV-Based GPR Systems",
        "url": "https://www.mdpi.com/2072-4292/14/14/3245",
        "note": "UAV-GPR 标准处理通常包含 time gating/background removal/gain；UAV 场景额外受高度、 clutter、穿透能力限制。",
    },
    {
        "title": "GPR Processing Basics, GPR Consortium, 2021",
        "url": "https://gpr-consortium.com/wp-content/uploads/2021/03/GPR-Processing-Basics.pdf",
        "note": "AGC 提升可解释性但会丢失相对振幅；能量补偿/SEC 更适合保留反射强弱关系。",
    },
    {
        "title": "Sensors & Software Processing Module User Guide, SEC2/AGC",
        "url": "https://www.sensoft.ca/wp-content/uploads/2024/04/Processing-Module-User-Guide.pdf",
        "note": "SEC2 补偿球面扩散和欧姆耗散，通常更接近物理现实；AGC 适合反射连续性但不保留相对振幅。",
    },
    {
        "title": "RGPR basic data processing tutorial",
        "url": "https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/",
        "note": "常见增益包括 power、exponential 和 AGC；指数参数可从振幅包络的对数衰减斜率估计。",
    },
    {
        "title": "Geolitix GPR processing documentation",
        "url": "https://docs.geolitix.com/layers/gpr-processing.html",
        "note": "没有单一增益函数适合所有场景；energy decay 通常最通用，AGC 会丢失相对振幅。",
    },
    {
        "title": "Sensors 2018, attenuation compensation method",
        "url": "https://www.mdpi.com/1424-8220/18/5/1366",
        "note": "GPR 振幅随频率和距离衰减；理论上应估计反衰减函数，但介质参数未知时只能做近似补偿。",
    },
    {
        "title": "Remote Sensing 2026, drone-mounted GPR signal factors",
        "url": "https://www.mdpi.com/1424-8220/26/6/1873",
        "note": "飞行高度、地形、植被/含水率会影响空载 GPR 的振幅、时延和质量，应进入自动增益选择的风险项。",
    },
]


@dataclass(frozen=True)
class ScenarioPackage:
    """Input artifacts for one already generated gprMax scene."""

    scenario_id: str
    label: str
    description: str
    structure_notes: list[str]
    bscan: np.ndarray
    scenario: dict[str, Any]
    ground_truth: dict[str, Any]
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    source_dir: Path


@dataclass(frozen=True)
class StageRun:
    """Result of running a fixed preprocessing stage."""

    data: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    roi: dict[str, int]
    records: list[dict[str, Any]]


def run_gain_method_report(
    *,
    source_report: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    scenario_ids: list[str] | None = None,
    search_mode: str = "standard",
    baseline_profile_key: str = DEFAULT_BASELINE_PROFILE,
    include_no_gain: bool = True,
) -> dict[str, Any]:
    """Build a static HTML report comparing gain algorithms on existing gprMax data."""
    source_dir = resolve_source_report(source_report)
    source_payload = _read_json(source_dir / "summary.json")
    selected_ids = set(scenario_ids or [])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_gain_methods")
    report_dir = Path(output_root) / timestamp
    assets_dir = report_dir / "assets"
    scenario_out_dir = report_dir / "scenarios"
    assets_dir.mkdir(parents=True, exist_ok=True)
    scenario_out_dir.mkdir(parents=True, exist_ok=True)

    method_keys = list(GAIN_METHODS)
    if not include_no_gain:
        method_keys = [key for key in method_keys if key != "no_gain"]

    scenario_records: list[dict[str, Any]] = []
    for source_record in source_payload.get("scenarios", []):
        scenario_id = str(source_record.get("scenario_id") or "")
        if selected_ids and scenario_id not in selected_ids:
            continue
        package = load_scenario_package(source_dir, source_record)
        scenario_dir = scenario_out_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        _write_scenario_inputs(package, scenario_dir)

        pre_gain = run_pre_gain_pipeline(
            package,
            baseline_profile_key=baseline_profile_key,
            search_mode=search_mode,
        )
        variants = [
            evaluate_gain_method(
                package,
                pre_gain,
                method_key=method_key,
                search_mode=search_mode,
                baseline_profile_key=baseline_profile_key,
            )
            for method_key in method_keys
        ]
        best_choice = choose_gain_choice(variants)
        image_records = save_gain_report_images(
            scenario_id=scenario_id,
            pre_gain=pre_gain,
            variants=variants,
            assets_dir=assets_dir,
            report_dir=report_dir,
        )
        structure_preview = assets_dir / f"{scenario_id}_structure.png"
        _render_or_copy_structure(package, structure_preview)
        scenario_records.append(
            {
                "scenario_id": package.scenario_id,
                "label": package.label,
                "description": package.description,
                "structure_notes": package.structure_notes,
                "shape": list(package.bscan.shape),
                "source_report": str(source_dir),
                "scenario_json": _relpath(scenario_dir / "scenario.json", report_dir),
                "ground_truth_json": _relpath(scenario_dir / "ground_truth.json", report_dir),
                "bscan_csv": _relpath(scenario_dir / "mygpr_bscan.csv", report_dir),
                "structure_preview": _relpath(structure_preview, report_dir),
                "pre_gain_pipeline": _json_safe(pre_gain.records),
                "pre_gain_roi": _json_safe(pre_gain.roi),
                "gain_methods": _json_safe(_strip_variant_arrays(variants)),
                "images": image_records,
                "best_gain_choice": best_choice,
            }
        )

    payload = {
        "schema": "mygpr_gprmax_gain_method_report_v1",
        "generated_at": timestamp,
        "source_report": str(source_dir),
        "baseline_profile_key": baseline_profile_key,
        "search_mode": search_mode,
        "gain_methods": method_keys,
        "research_sources": RESEARCH_SOURCES,
        "method_notes": GAIN_METHOD_NOTES,
        "scenarios": scenario_records,
        "selection_rule": _selection_rule_summary(),
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path = render_gain_method_html(report_dir, payload)
    payload["summary_json"] = str(summary_path)
    payload["html_report"] = str(html_path)
    summary_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def resolve_source_report(source_report: Path | None) -> Path:
    """Resolve a report directory that contains summary.json and scenario CSVs."""
    if source_report is not None:
        path = Path(source_report)
        if path.is_file():
            path = path.parent
        if not (path / "summary.json").exists():
            raise FileNotFoundError(f"source report summary not found: {path / 'summary.json'}")
        return path

    candidates = [
        path.parent
        for path in DEFAULT_SOURCE_ROOT.rglob("summary.json")
        if "scenarios" in str(path)
        or (path.parent / "scenarios").exists()
    ]
    candidates = [path for path in candidates if (path / "summary.json").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"no gprMax source reports found under {DEFAULT_SOURCE_ROOT}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_scenario_package(
    source_dir: Path,
    source_record: dict[str, Any],
) -> ScenarioPackage:
    """Load BScan, scenario, and ground-truth files referenced by a source report."""
    scenario_id = str(source_record["scenario_id"])
    scenario_path = source_dir / str(source_record["scenario_json"])
    ground_truth_path = source_dir / str(source_record["ground_truth_json"])
    bscan_path = source_dir / str(source_record["bscan_csv"])
    bscan = np.loadtxt(bscan_path, delimiter=",", dtype=np.float32)
    scenario = _read_json(scenario_path)
    ground_truth = _read_json(ground_truth_path)
    simulation = dict(source_record.get("simulation") or scenario.get("simulation") or {})
    header_info = {
        "a_scan_length": int(bscan.shape[0]),
        "num_traces": int(bscan.shape[1]),
        "total_time_ns": float(
            simulation.get("total_time_ns")
            or scenario.get("simulation", {}).get("total_time_ns")
            or 1.0
        ),
        "trace_interval_m": float(
            simulation.get("trace_step_m")
            or scenario.get("simulation", {}).get("trace_step_m")
            or 1.0
        ),
    }
    header_info["track_length_m"] = header_info["trace_interval_m"] * max(
        int(bscan.shape[1]) - 1, 1
    )
    return ScenarioPackage(
        scenario_id=scenario_id,
        label=str(source_record.get("label") or scenario_id),
        description=str(source_record.get("description") or ""),
        structure_notes=[str(item) for item in source_record.get("structure_notes", [])],
        bscan=bscan,
        scenario=scenario,
        ground_truth=ground_truth,
        header_info=header_info,
        trace_metadata={},
        source_dir=source_dir / "scenarios" / scenario_id,
    )


def run_pre_gain_pipeline(
    package: ScenarioPackage,
    *,
    baseline_profile_key: str,
    search_mode: str,
) -> StageRun:
    """Run the fixed part of the report pipeline before gain."""
    data = np.asarray(package.bscan, dtype=np.float32)
    header_info = clone_header_info(package.header_info)
    trace_metadata = clone_trace_metadata(package.trace_metadata)
    roi = _analysis_roi(package.ground_truth, data.shape)
    params_by_method = _baseline_params(baseline_profile_key)
    records: list[dict[str, Any]] = []

    for method_key in PRE_GAIN_METHODS:
        params = dict(params_by_method.get(method_key, {}))
        if method_key == "set_zero_time":
            params = _aligned_zero_time_params(
                data,
                header_info=header_info,
                trace_metadata=trace_metadata,
                roi=roi,
                ground_truth=package.ground_truth,
                search_mode=search_mode,
                base_params=params,
            )
        before_shape = tuple(data.shape)
        data, meta = _run_method(data, method_key, params, header_info, trace_metadata)
        header_info = merge_result_header_info(header_info, meta, data.shape)
        trace_metadata = merge_result_trace_metadata(trace_metadata, meta)
        roi = base_report._roi_after_method(roi, method_key, meta, data.shape)
        records.append(
            {
                "method_key": method_key,
                "method_name": _method_name(method_key),
                "params": _json_safe(params),
                "before_shape": list(before_shape),
                "after_shape": list(data.shape),
                "warnings": _extract_warning_messages(meta),
            }
        )
    return StageRun(
        data=np.asarray(data, dtype=np.float32),
        header_info=header_info,
        trace_metadata=trace_metadata,
        roi=roi,
        records=records,
    )


def evaluate_gain_method(
    package: ScenarioPackage,
    pre_gain: StageRun,
    *,
    method_key: str,
    search_mode: str,
    baseline_profile_key: str,
) -> dict[str, Any]:
    """Evaluate manual and auto-tuned parameters for one gain algorithm."""
    method_label = GAIN_METHOD_NOTES[method_key]["label"]
    manual_params = _manual_gain_params(method_key, baseline_profile_key)
    roi_spec = {
        "mode": "manual",
        "bounds": dict(pre_gain.roi),
        "label": f"{package.scenario_id} gain truth ROI",
    }
    auto_tune_summary: dict[str, Any] = {}
    auto_params = dict(manual_params)
    if method_key != "no_gain":
        try:
            tune_result = auto_tune_method(
                pre_gain.data,
                method_key,
                header_info=pre_gain.header_info,
                trace_metadata=pre_gain.trace_metadata,
                base_params=manual_params,
                roi_spec=roi_spec,
                search_mode=search_mode,
            )
            auto_tune_summary = base_report._compact_auto_tune_result(tune_result)
            auto_params = dict(
                tune_result.get("recommended_params")
                or tune_result.get("best_params")
                or manual_params
            )
        except Exception as exc:
            auto_tune_summary = {"error": str(exc), "recommended_params": manual_params}
            auto_params = dict(manual_params)

    manual_eval = run_gain_branch(
        package,
        pre_gain,
        method_key=method_key,
        params=manual_params,
        branch_label="manual",
    )
    auto_eval = run_gain_branch(
        package,
        pre_gain,
        method_key=method_key,
        params=auto_params,
        branch_label="auto",
    )
    return {
        "method_key": method_key,
        "method_label": method_label,
        "manual": _json_safe(manual_eval),
        "auto": _json_safe(auto_eval),
        "auto_tune_summary": _json_safe(auto_tune_summary),
        "manual_vs_auto": _compare_branch_scores(manual_eval, auto_eval),
        "method_note": GAIN_METHOD_NOTES[method_key],
    }


def choose_gain_choice(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the best gain method/parameter branch for the scenario."""
    candidates: list[dict[str, Any]] = []
    target_count = 0.0
    for variant in variants:
        for branch_key in ["manual", "auto"]:
            branch = variant.get(branch_key, {})
            metrics = branch.get("final_metrics", {}) or {}
            target_count = max(target_count, float(metrics.get("target_count", 0.0)))
            candidates.append(
                {
                    "method_key": variant["method_key"],
                    "method_label": variant["method_label"],
                    "branch": branch_key,
                    "score": float(branch.get("selection_score", -1.0e9)),
                    "params": _json_safe(branch.get("params", {})),
                    "reason": branch.get("selection_reason", ""),
                }
            )
    if target_count > 0.0:
        gain_candidates = [item for item in candidates if item["method_key"] != "no_gain"]
        if gain_candidates:
            candidates = gain_candidates
    best = max(candidates, key=lambda item: float(item["score"]))
    return _json_safe(best)


def _strip_variant_arrays(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove ndarray payloads before writing JSON summary."""
    cleaned: list[dict[str, Any]] = []
    for variant in variants:
        item = dict(variant)
        for branch_key in ["manual", "auto"]:
            branch = dict(item.get(branch_key, {}))
            branch.pop("gain_output", None)
            branch.pop("final_output", None)
            item[branch_key] = branch
        cleaned.append(item)
    return cleaned


def run_gain_branch(
    package: ScenarioPackage,
    pre_gain: StageRun,
    *,
    method_key: str,
    params: dict[str, Any],
    branch_label: str,
) -> dict[str, Any]:
    """Run one gain branch plus fixed post-gain steps."""
    header_info = clone_header_info(pre_gain.header_info)
    trace_metadata = clone_trace_metadata(pre_gain.trace_metadata)
    roi = dict(pre_gain.roi)
    if method_key == "no_gain":
        gain_output = np.array(pre_gain.data, copy=True)
        gain_meta = {"method": "no_gain", "skipped": True, "reason": "QA reference"}
    else:
        gain_output, gain_meta = _run_method(
            pre_gain.data,
            method_key,
            params,
            header_info,
            trace_metadata,
        )
        header_info = merge_result_header_info(header_info, gain_meta, gain_output.shape)
        trace_metadata = merge_result_trace_metadata(trace_metadata, gain_meta)
        roi = base_report._roi_after_method(roi, method_key, gain_meta, gain_output.shape)

    final = np.asarray(gain_output, dtype=np.float32)
    post_records: list[dict[str, Any]] = []
    post_params_by_method = _baseline_params(DEFAULT_BASELINE_PROFILE)
    for post_method in POST_GAIN_METHODS:
        post_params = dict(post_params_by_method.get(post_method, {}))
        before_shape = tuple(final.shape)
        final, post_meta = _run_method(
            final,
            post_method,
            post_params,
            header_info,
            trace_metadata,
        )
        header_info = merge_result_header_info(header_info, post_meta, final.shape)
        trace_metadata = merge_result_trace_metadata(trace_metadata, post_meta)
        roi = base_report._roi_after_method(roi, post_method, post_meta, final.shape)
        post_records.append(
            {
                "method_key": post_method,
                "method_name": _method_name(post_method),
                "params": _json_safe(post_params),
                "before_shape": list(before_shape),
                "after_shape": list(final.shape),
                "warnings": _extract_warning_messages(post_meta),
            }
        )

    gain_metrics = compute_gain_metrics(
        pre_gain.data,
        gain_output,
        package.ground_truth,
        pre_gain_roi=pre_gain.roi,
        output_roi=base_report._roi_after_method(
            dict(pre_gain.roi), method_key, gain_meta, gain_output.shape
        ),
        method_key=method_key,
    )
    final_metrics = compute_gain_metrics(
        pre_gain.data,
        final,
        package.ground_truth,
        pre_gain_roi=pre_gain.roi,
        output_roi=roi,
        method_key=method_key,
    )
    selection_score = gain_method_selection_score(final_metrics, method_key)
    return {
        "branch": branch_label,
        "params": _json_safe(params),
        "gain_warnings": _extract_warning_messages(gain_meta),
        "post_pipeline": post_records,
        "gain_metrics": _json_safe(gain_metrics),
        "final_metrics": _json_safe(final_metrics),
        "selection_score": float(selection_score),
        "selection_reason": _selection_reason(final_metrics, method_key, selection_score),
        "visual_analysis": _visual_gain_analysis(final_metrics, method_key),
        "metric_analysis": _metric_gain_analysis(final_metrics, method_key),
        "gain_output": gain_output,
        "final_output": final,
        "final_roi": roi,
    }


def compute_gain_metrics(
    before: np.ndarray,
    after: np.ndarray,
    ground_truth: dict[str, Any],
    *,
    pre_gain_roi: dict[str, int],
    output_roi: dict[str, int],
    method_key: str,
) -> dict[str, float]:
    """Compute method-selection metrics specialized for gain comparison."""
    before_slice, after_slice = base_report._slice_pair_rois(
        before,
        after,
        pre_gain_roi,
        output_roi,
    )
    metrics = compute_benchmark_metrics(before_slice, after_slice)
    metrics.update(
        compute_ground_truth_metrics(
            before,
            after,
            ground_truth,
            reference_roi=pre_gain_roi,
            processed_roi=output_roi,
        )
    )
    metrics["lateral_profile_corr"] = _lateral_profile_corr(before, after, output_roi)
    metrics["depth_balance_score"] = _depth_balance_score(after)
    metrics["relative_amplitude_preservation_score"] = _relative_amplitude_score(
        method_key,
        metrics["lateral_profile_corr"],
    )
    metrics["clipping_ratio_after"] = clipping_ratio(after)
    metrics["hot_pixel_ratio_after"] = hot_pixel_ratio(after)
    metrics["target_count"] = float(len(ground_truth.get("targets") or []))
    return {key: float(value) for key, value in metrics.items() if _is_number(value)}


def gain_method_selection_score(metrics: dict[str, float], method_key: str) -> float:
    """Score a gain result using truth ROI, artifact risk, and literature priors."""
    target_count = float(metrics.get("target_count", 0.0))
    truth = float(metrics.get("truth_score", 0.0))
    target_energy = float(metrics.get("truth_target_energy_preservation", 1.0))
    target_band = float(metrics.get("target_band_energy_ratio", 1.0))
    edge = float(metrics.get("edge_preservation", 1.0))
    saliency_gain = float(metrics.get("truth_target_saliency_gain", 1.0))
    contrast = float(metrics.get("truth_target_contrast_after", 0.0))
    false_positive = float(metrics.get("truth_false_positive_ratio", 1.0))
    background_reduction = float(metrics.get("truth_background_energy_reduction", 0.0))
    lateral = float(metrics.get("lateral_profile_corr", 0.0))
    amp_preserve = float(metrics.get("relative_amplitude_preservation_score", 0.0))
    depth_balance = float(metrics.get("depth_balance_score", 0.0))
    clip = float(metrics.get("clipping_ratio_after", 0.0))
    hot = float(metrics.get("hot_pixel_ratio_after", 0.0))

    method_prior = {
        "sec_gain": 0.18,
        "compensatingGain": 0.08,
        "agcGain": -0.10,
        "no_gain": 0.04 if target_count <= 0 else -0.12,
    }.get(method_key, 0.0)

    if target_count <= 0:
        return float(
            1.4 * truth
            + 1.1 * np.clip(background_reduction, -1.0, 1.0)
            + 0.5 * amp_preserve
            - 1.6 * max(0.0, false_positive - 1.0)
            - 8.0 * clip
            - 5.0 * hot
            + method_prior
        )

    saliency_term = float(np.clip(np.log1p(max(0.0, saliency_gain)) / np.log(4.0), 0.0, 1.6))
    contrast_term = float(np.clip(np.log1p(max(0.0, contrast)) / np.log(6.0), 0.0, 1.4))
    false_positive_penalty = max(0.0, false_positive - 0.85)
    background_penalty = max(0.0, -background_reduction)
    target_vanish_penalty = (
        5.0 * max(0.0, 0.35 - target_energy)
        + 3.5 * max(0.0, 0.35 - target_band)
        + 2.5 * max(0.0, 0.35 - edge)
        + 1.4 * max(0.0, 0.90 - saliency_gain)
    )
    return float(
        1.25 * truth
        + 0.85 * saliency_term
        + 0.45 * contrast_term
        + 0.55 * amp_preserve
        + 0.35 * depth_balance
        - 1.25 * false_positive_penalty
        - 0.40 * background_penalty
        - target_vanish_penalty
        - 8.0 * clip
        - 5.0 * hot
        + method_prior
    )


def save_gain_report_images(
    *,
    scenario_id: str,
    pre_gain: StageRun,
    variants: list[dict[str, Any]],
    assets_dir: Path,
    report_dir: Path,
) -> list[dict[str, Any]]:
    """Render gain-stage and final-stage images for each variant."""
    pre_path = assets_dir / f"{scenario_id}_pre_gain.png"
    all_outputs = [pre_gain.data]
    for variant in variants:
        all_outputs.extend(
            [
                variant["manual"]["gain_output"],
                variant["auto"]["gain_output"],
                variant["manual"]["final_output"],
                variant["auto"]["final_output"],
            ]
        )
    vlim = base_report._locked_vlim(all_outputs)
    base_report.save_bscan_image(pre_gain.data, pre_path, "before gain", vlim, pre_gain.roi)

    image_records: list[dict[str, Any]] = []
    for variant in variants:
        method_key = variant["method_key"]
        prefix = f"{scenario_id}_{method_key}"
        paths = {
            "manual_gain": assets_dir / f"{prefix}_manual_gain.png",
            "auto_gain": assets_dir / f"{prefix}_auto_gain.png",
            "manual_final": assets_dir / f"{prefix}_manual_final.png",
            "auto_final": assets_dir / f"{prefix}_auto_final.png",
        }
        base_report.save_bscan_image(
            variant["manual"]["gain_output"],
            paths["manual_gain"],
            f"{method_key} manual gain",
            vlim,
            pre_gain.roi,
        )
        base_report.save_bscan_image(
            variant["auto"]["gain_output"],
            paths["auto_gain"],
            f"{method_key} auto gain",
            vlim,
            pre_gain.roi,
        )
        base_report.save_bscan_image(
            variant["manual"]["final_output"],
            paths["manual_final"],
            f"{method_key} manual final",
            vlim,
            variant["manual"]["final_roi"],
        )
        base_report.save_bscan_image(
            variant["auto"]["final_output"],
            paths["auto_final"],
            f"{method_key} auto final",
            vlim,
            variant["auto"]["final_roi"],
        )
        image_records.append(
            {
                "method_key": method_key,
                "method_label": variant["method_label"],
                "pre_gain": _relpath(pre_path, report_dir),
                "manual_gain": _relpath(paths["manual_gain"], report_dir),
                "auto_gain": _relpath(paths["auto_gain"], report_dir),
                "manual_final": _relpath(paths["manual_final"], report_dir),
                "auto_final": _relpath(paths["auto_final"], report_dir),
            }
        )
    return image_records


def render_gain_method_html(report_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the gain comparison HTML report."""
    html_path = report_dir / "index.html"
    scenario_sections = "\n".join(_render_scenario_section(item) for item in payload["scenarios"])
    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MyGPR gprMax 增益方法对比报告</title>
  <style>{_html_css()}</style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="eyebrow">MyGPR / gprMax gain selection</p>
      <h1>UAV-GPR 增益方法选择与 gprMax 对比报告</h1>
      <p>本报告复用已有 gprMax 正演数据，在相同前序处理和相同后续 SVD 步骤下，仅替换增益方法，比较 SEC、AGC、线性 TGC 与不施加增益的人工参数和自动参数效果。</p>
    </section>
    {_render_research_section(payload)}
    {_render_selection_section(payload)}
    {_render_overall_summary(payload)}
    {scenario_sections}
  </main>
</body>
</html>
"""
    html_path.write_text(body, encoding="utf-8")
    return html_path


def _render_research_section(payload: dict[str, Any]) -> str:
    source_items = "\n".join(
        f'<li><a href="{_esc(item["url"])}">{_esc(item["title"])}</a>: {_esc(item["note"])}</li>'
        for item in payload.get("research_sources", [])
    )
    method_cards = "\n".join(
        f"""
        <article class="method-card">
          <h3>{_esc(note["label"])} <code>{_esc(key)}</code></h3>
          <p><strong>原理：</strong>{_esc(note["principle"])}</p>
          <p><strong>适用：</strong>{_esc(note["best_for"])}</p>
          <p><strong>风险：</strong>{_esc(note["risks"])}</p>
        </article>
"""
        for key, note in payload.get("method_notes", {}).items()
    )
    return f"""
    <section>
      <h2>调研结论</h2>
      <p>UAV-GPR 文献通常把 gain 放在标准处理链中，但并没有给出一个适用于所有场景的单一最优增益。普通 GPR 软件和处理手册的共识更明确：SEC/energy-decay 更接近物理补偿、能保留相对振幅；AGC 更像视觉均衡，适合看连续性，但不能用于反射强弱解释。</p>
      <div class="method-grid">{method_cards}</div>
      <h3>依据来源</h3>
      <ul>{source_items}</ul>
    </section>
"""


def _render_selection_section(payload: dict[str, Any]) -> str:
    rule = payload.get("selection_rule", {})
    items = "".join(f"<li>{_esc(item)}</li>" for item in rule.get("rules", []))
    return f"""
    <section>
      <h2>自动选择方法</h2>
      <p>{_esc(rule.get("summary"))}</p>
      <ul>{items}</ul>
      <p class="note">对 gprMax，评分使用真实结构 ROI；对未来真实 UAV 数据，应把真实 ROI 替换为结构候选 ROI、背景 ROI、深部噪声 ROI，并把分数差距小的结果标记为需要人工复核。</p>
    </section>
"""


def _render_overall_summary(payload: dict[str, Any]) -> str:
    rows = "\n".join(_render_overall_row(item) for item in payload.get("scenarios", []))
    return f"""
    <section>
      <h2>总体结果</h2>
      <table class="params-table">
        <thead><tr><th>场景</th><th>自动推荐增益</th><th>score</th><th>推荐参数</th><th>理由</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
"""


def _render_overall_row(record: dict[str, Any]) -> str:
    best = record.get("best_gain_choice", {})
    return f"""
          <tr>
            <td>{_esc(record.get("label"))} <code>{_esc(record.get("scenario_id"))}</code></td>
            <td>{_esc(best.get("method_label"))} <code>{_esc(best.get("method_key"))}</code> / {_esc(best.get("branch"))}</td>
            <td>{_fmt(best.get("score"))}</td>
            <td><code>{_esc(json.dumps(best.get("params", {}), ensure_ascii=False))}</code></td>
            <td>{_esc(best.get("reason"))}</td>
          </tr>
"""


def _render_scenario_section(record: dict[str, Any]) -> str:
    notes = "".join(f"<li>{_esc(note)}</li>" for note in record.get("structure_notes", []))
    variant_sections = "\n".join(
        _render_gain_variant(record, variant)
        for variant in record.get("gain_methods", [])
    )
    pre_steps = "\n".join(
        f"<li>{_esc(step['method_name'])} <code>{_esc(step['method_key'])}</code>: "
        f"<code>{_esc(json.dumps(step.get('params', {}), ensure_ascii=False))}</code></li>"
        for step in record.get("pre_gain_pipeline", [])
    )
    best = record.get("best_gain_choice", {})
    return f"""
    <section class="scenario">
      <div class="scenario-heading">
        <div>
          <p class="eyebrow">{_esc(record.get("scenario_id"))}</p>
          <h2>{_esc(record.get("label"))}</h2>
        </div>
        <span class="verdict">推荐 {_esc(best.get("method_label"))} / {_esc(best.get("branch"))}</span>
      </div>
      <p>{_esc(record.get("description"))}</p>
      <h3>真实地质结构</h3>
      <div class="structure-grid">
        <img src="{_esc(record.get("structure_preview"))}" alt="{_esc(record.get("label"))} true structure">
        <div>
          <ul>{notes}</ul>
          <p class="note">增益选择评分使用该场景的真实结构 ROI，因此可以区分“目标被增强”和“背景/空白区被误增强”。</p>
        </div>
      </div>
      <h3>固定前序流程</h3>
      <ul>{pre_steps}</ul>
      {variant_sections}
    </section>
"""


def _render_gain_variant(record: dict[str, Any], variant: dict[str, Any]) -> str:
    images = _images_for_method(record, str(variant.get("method_key")))
    manual = variant.get("manual", {})
    auto = variant.get("auto", {})
    auto_summary = variant.get("auto_tune_summary", {})
    return f"""
      <article class="step-card">
        <h3>{_esc(variant.get("method_label"))} <code>{_esc(variant.get("method_key"))}</code></h3>
        <div class="metric-grid">
          <div><span>人工 score</span><strong>{_fmt(manual.get("selection_score"))}</strong></div>
          <div><span>自动 score</span><strong>{_fmt(auto.get("selection_score"))}</strong></div>
          <div><span>人工参数</span><strong><code>{_esc(json.dumps(manual.get("params", {}), ensure_ascii=False))}</code></strong></div>
          <div><span>自动参数</span><strong><code>{_esc(json.dumps(auto.get("params", {}), ensure_ascii=False))}</code></strong></div>
        </div>
        <p><strong>自动选参说明：</strong>{_esc(auto_summary.get("best_reason") or auto_summary.get("error") or "无")}</p>
        <p><strong>视觉评价：</strong>{_esc(auto.get("visual_analysis"))}</p>
        <p><strong>指标评价：</strong>{_esc(auto.get("metric_analysis"))}</p>
        <div class="image-grid five">
          <figure><img src="{_esc(images.get("pre_gain"))}" alt="before gain"><figcaption>增益前</figcaption></figure>
          <figure><img src="{_esc(images.get("manual_gain"))}" alt="manual gain"><figcaption>人工增益后</figcaption></figure>
          <figure><img src="{_esc(images.get("auto_gain"))}" alt="auto gain"><figcaption>自动增益后</figcaption></figure>
          <figure><img src="{_esc(images.get("manual_final"))}" alt="manual final"><figcaption>人工完整流程后</figcaption></figure>
          <figure><img src="{_esc(images.get("auto_final"))}" alt="auto final"><figcaption>自动完整流程后</figcaption></figure>
        </div>
        {_render_metrics_table(auto.get("final_metrics", {}))}
      </article>
"""


def _render_metrics_table(metrics: dict[str, Any]) -> str:
    keys = [
        ("truth_score", "真值评分"),
        ("truth_target_saliency_gain", "目标显著性增益"),
        ("truth_target_contrast_after", "目标/背景对比"),
        ("truth_background_energy_reduction", "背景能量变化"),
        ("truth_false_positive_ratio", "假异常比例"),
        ("lateral_profile_corr", "横向幅值形态相关"),
        ("relative_amplitude_preservation_score", "相对幅值保真"),
        ("depth_balance_score", "深浅能量均衡"),
        ("clipping_ratio_after", "过曝比例"),
        ("hot_pixel_ratio_after", "热点比例"),
    ]
    rows = "\n".join(
        f"<tr><td>{_esc(label)}</td><td>{_fmt(metrics.get(key))}</td></tr>"
        for key, label in keys
    )
    return f"""
        <table class="params-table small">
          <tbody>{rows}</tbody>
        </table>
"""


def _images_for_method(record: dict[str, Any], method_key: str) -> dict[str, str]:
    for item in record.get("images", []):
        if item.get("method_key") == method_key:
            return dict(item)
    return {}


def _run_method(
    data: np.ndarray,
    method_key: str,
    params: dict[str, Any],
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime_params = prepare_runtime_params(
        method_key,
        params,
        header_info,
        trace_metadata,
        tuple(np.asarray(data).shape),
    )
    return run_processing_method(data, method_key, runtime_params)


def _aligned_zero_time_params(
    data: np.ndarray,
    *,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    roi: dict[str, int],
    ground_truth: dict[str, Any],
    search_mode: str,
    base_params: dict[str, Any],
) -> dict[str, Any]:
    roi_spec = {
        "mode": "manual",
        "bounds": roi,
        "label": f"{ground_truth.get('scenario_id', 'scenario')} ground-truth ROI",
    }
    params, _summary, _notes = base_report._derive_aligned_zero_time_params(
        data,
        header_info=header_info,
        trace_metadata=trace_metadata,
        roi_spec=roi_spec,
        search_mode=search_mode,
        base_params=base_params,
    )
    return dict(params)


def _baseline_params(profile_key: str) -> dict[str, dict[str, Any]]:
    profile = RECOMMENDED_RUN_PROFILES.get(profile_key, {})
    params = {
        str(key): dict(value)
        for key, value in profile.get("method_params", {}).items()
    }
    return params


def _manual_gain_params(method_key: str, baseline_profile_key: str) -> dict[str, Any]:
    params = dict(MANUAL_GAIN_PARAMS.get(method_key, {}))
    profile_params = _baseline_params(baseline_profile_key)
    if method_key in profile_params:
        params.update(profile_params[method_key])
    return params


def _analysis_roi(ground_truth: dict[str, Any], shape: tuple[int, int]) -> dict[str, int]:
    roi = ground_truth.get("analysis_roi")
    if not isinstance(roi, dict):
        roi = {
            "time_start_idx": 0,
            "time_end_idx": int(shape[0]),
            "dist_start_idx": 0,
            "dist_end_idx": int(shape[1]),
        }
    return base_report._clamp_roi(roi, shape)


def _lateral_profile_corr(
    before: np.ndarray,
    after: np.ndarray,
    roi: dict[str, int],
) -> float:
    before_roi = base_report._slice_roi(before, roi)
    after_roi = base_report._slice_roi(after, roi)
    rows = min(before_roi.shape[0], after_roi.shape[0])
    cols = min(before_roi.shape[1], after_roi.shape[1])
    if rows <= 0 or cols <= 1:
        return 1.0
    before_profile = np.mean(np.abs(before_roi[:rows, :cols]), axis=0)
    after_profile = np.mean(np.abs(after_roi[:rows, :cols]), axis=0)
    if float(np.std(before_profile)) < 1.0e-12 or float(np.std(after_profile)) < 1.0e-12:
        return 1.0
    corr = float(np.corrcoef(before_profile, after_profile)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _depth_balance_score(data: np.ndarray) -> float:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(arr**2, axis=1))
    mean = float(np.mean(rms))
    if mean <= 1.0e-12:
        return 0.0
    cv = float(np.std(rms) / mean)
    return float(np.exp(-cv))


def _relative_amplitude_score(method_key: str, lateral_corr: float) -> float:
    corr_score = (float(lateral_corr) + 1.0) / 2.0
    if method_key == "agcGain":
        return float(0.45 * corr_score)
    if method_key == "no_gain":
        return float(0.95 * corr_score)
    return float(corr_score)


def _selection_reason(metrics: dict[str, float], method_key: str, score: float) -> str:
    return (
        f"{GAIN_METHOD_NOTES[method_key]['label']} selection_score={score:.3f}；"
        f"真值评分={metrics.get('truth_score', 0.0):.3f}，"
        f"目标显著性增益={metrics.get('truth_target_saliency_gain', 0.0):.3f}，"
        f"假异常比例={metrics.get('truth_false_positive_ratio', 0.0):.3f}，"
        f"相对幅值保真={metrics.get('relative_amplitude_preservation_score', 0.0):.3f}。"
    )


def _visual_gain_analysis(metrics: dict[str, float], method_key: str) -> str:
    saliency = float(metrics.get("truth_target_saliency_gain", 1.0))
    false_positive = float(metrics.get("truth_false_positive_ratio", 1.0))
    amp = float(metrics.get("relative_amplitude_preservation_score", 0.0))
    if method_key == "agcGain":
        base = "AGC 的视觉目标是拉平深浅能量并增强弱事件连续性。"
    elif method_key == "sec_gain":
        base = "SEC 的视觉目标是在增强深部反射的同时维持横向相对强弱。"
    elif method_key == "compensatingGain":
        base = "线性 TGC 的视觉目标是温和提升深部可见性，避免强均衡。"
    else:
        base = "不施加增益用于判断前序处理本身是否已经足够。"
    return (
        f"{base} 本场景目标显著性增益为 {saliency:.3f}，"
        f"假异常比例为 {false_positive:.3f}，相对幅值保真为 {amp:.3f}。"
    )


def _metric_gain_analysis(metrics: dict[str, float], method_key: str) -> str:
    return (
        f"{GAIN_METHOD_NOTES[method_key]['label']} 的真值评分为 "
        f"{metrics.get('truth_score', 0.0):.3f}，目标/背景对比为 "
        f"{metrics.get('truth_target_contrast_after', 0.0):.3f}，背景能量变化为 "
        f"{metrics.get('truth_background_energy_reduction', 0.0):.3f}，"
        f"过曝比例为 {metrics.get('clipping_ratio_after', 0.0):.4f}。"
    )


def _compare_branch_scores(
    manual_eval: dict[str, Any],
    auto_eval: dict[str, Any],
) -> dict[str, Any]:
    delta = float(auto_eval["selection_score"]) - float(manual_eval["selection_score"])
    if delta > 0.02:
        verdict = "auto_params_better"
    elif delta < -0.02:
        verdict = "manual_params_better"
    else:
        verdict = "near_tie"
    return {"delta_auto_minus_manual": delta, "verdict": verdict}


def _selection_rule_summary() -> dict[str, Any]:
    return {
        "summary": (
            "可靠自动选择不应只比较图像亮不亮，而应把目标 ROI、背景 ROI、深部噪声、"
            "相对振幅保真和过曝风险一起评分；gprMax 场景可用真值 ROI，真实数据需用候选结构 ROI 近似。"
        ),
        "rules": [
            "若任务需要反射强弱、材料差异或层状结构解释，默认优先 SEC/energy-decay，并用最大增益约束控制深部噪声。",
            "若任务只要求弱双曲线/连续事件可见，且 AGC 没有显著提高假异常比例，可选择 AGC；报告必须提示相对振幅不可解释。",
            "若深部衰减不明显或背景噪声较高，优先线性 TGC 或不施加增益，避免把空白区放大成假目标。",
            "若最佳与次优 selection_score 差距很小，应同时展示两种增益，不自动下结论。",
            "UAV-GPR 真实数据还应把飞行高度变化、地形起伏、姿态误差和植被/含水率造成的振幅变化作为风险项。",
        ],
    }


def _render_or_copy_structure(package: ScenarioPackage, out_path: Path) -> None:
    scenario_defs = base_report.build_scenario_definitions()
    definition = scenario_defs.get(package.scenario_id)
    if definition is not None:
        base_report.save_structure_preview(definition, out_path)
        return
    source_image = package.source_dir / "structure.png"
    if source_image.exists():
        shutil.copy2(source_image, out_path)
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    ax.text(0.5, 0.5, package.label, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_scenario_inputs(package: ScenarioPackage, scenario_dir: Path) -> None:
    np.savetxt(scenario_dir / "mygpr_bscan.csv", package.bscan, delimiter=",", fmt="%.8e")
    (scenario_dir / "scenario.json").write_text(
        json.dumps(_json_safe(package.scenario), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (scenario_dir / "ground_truth.json").write_text(
        json.dumps(_json_safe(package.ground_truth), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _extract_warning_messages(meta: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    for warning in meta.get("runtime_warnings", []) or []:
        if isinstance(warning, dict):
            messages.append(str(warning.get("message") or warning.get("code") or warning))
        else:
            messages.append(str(warning))
    for warning in meta.get("warnings", []) or []:
        messages.append(str(warning))
    if meta.get("skipped"):
        messages.append(str(meta.get("reason") or "method skipped"))
    return messages


def _method_name(method_key: str) -> str:
    if method_key == "no_gain":
        return "No gain"
    return str(PROCESSING_METHODS.get(method_key, {}).get("name") or method_key)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _relpath(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _is_number(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.4f}"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _html_css() -> str:
    return base_report._html_css() + """
.method-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}
.method-card {
  border: 1px solid #d9dee7;
  border-radius: 8px;
  padding: 14px;
  background: #fbfcfe;
}
.image-grid.five {
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.params-table.small td:first-child {
  width: 45%;
}
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gain methods on existing gprMax validation data."
    )
    parser.add_argument(
        "--source-report",
        default=None,
        help="Existing gprMax report directory or summary.json. Defaults to latest full-effect report.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where the gain-method report will be written.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Scenario id to include. Repeat for multiple scenarios.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["fast", "standard", "thorough"],
        default="standard",
        help="Auto-tune search mode for per-method parameters.",
    )
    parser.add_argument(
        "--baseline-profile",
        default=DEFAULT_BASELINE_PROFILE,
        help="Recommended profile used for fixed non-gain parameters.",
    )
    parser.add_argument(
        "--without-no-gain",
        action="store_true",
        help="Do not include the no-gain QA reference branch.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_gain_method_report(
        source_report=Path(args.source_report) if args.source_report else None,
        output_root=Path(args.output_root),
        scenario_ids=list(args.scenario or []) or None,
        search_mode=str(args.search_mode),
        baseline_profile_key=str(args.baseline_profile),
        include_no_gain=not bool(args.without_no_gain),
    )
    print(f"HTML report: {payload['html_report']}")
    print(f"Summary JSON: {payload['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

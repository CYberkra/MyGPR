#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-AT-SCORE-001 synthetic paired AutoTune smoke utilities."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.linalg import svd


CLAIM_BOUNDARY_LINES = [
    "synthetic paired scoring MVP only",
    "not AutoTune superiority evidence",
    "not field validation",
    "not real-world detection correctness evidence",
]

REQUIRED_TASK_SUBDIRS = {"1_模型输入", "2_gprMax原始输出", "3_MyGPR读取文件", "4_日志与报告"}
RUNNING_WINDOW_MINUTES = 20


@dataclass
class TaskInventory:
    """Inventory entry for one Output V5 run/task directory."""

    scene_id_guess: str
    task_dir: Path
    root_kind: str
    expected_trace_count: int | None
    raw_count: int
    background_count: int
    raw_indices: list[int]
    background_indices: list[int]
    component: str | None
    raw_shape: list[int] | None
    background_shape: list[int] | None
    target_response_shape: list[int] | None
    target_response_exists: bool
    mygpr_target_response_out_exists: bool
    status: str
    pair_status: str
    latest_mtime_iso: str
    notes: list[str]
    paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id_guess": self.scene_id_guess,
            "task_dir": str(self.task_dir),
            "root_kind": self.root_kind,
            "expected_trace_count": self.expected_trace_count,
            "raw_count": self.raw_count,
            "background_count": self.background_count,
            "raw_indices": self.raw_indices,
            "background_indices": self.background_indices,
            "component": self.component,
            "raw_shape": self.raw_shape,
            "background_shape": self.background_shape,
            "target_response_shape": self.target_response_shape,
            "target_response_exists": self.target_response_exists,
            "mygpr_target_response_out_exists": self.mygpr_target_response_out_exists,
            "status": self.status,
            "pair_status": self.pair_status,
            "latest_mtime": self.latest_mtime_iso,
            "notes": self.notes,
            "paths": self.paths,
        }


@dataclass
class CandidateScore:
    """Candidate score row for one scene."""

    scene_id: str
    task_dir: str
    component: str
    candidate: str
    parameters: dict[str, Any]
    roi_mode: str
    roi_sample_range: list[int]
    roi_trace_range: list[int]
    mae: float
    mse: float
    rmse: float
    psnr: float | None
    ssim: float | None
    roi_energy_retention: float | None
    outside_roi_residual_energy: float | None
    cnr_proxy: float | None
    cnr_gain_vs_baseline: float | None
    selected: bool
    status: str
    warnings: list[str]
    claim_boundary: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "task_dir": self.task_dir,
            "component": self.component,
            "candidate": self.candidate,
            "parameters": self.parameters,
            "roi_mode": self.roi_mode,
            "roi_sample_range": self.roi_sample_range,
            "roi_trace_range": self.roi_trace_range,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "roi_energy_retention": self.roi_energy_retention,
            "outside_roi_residual_energy": self.outside_roi_residual_energy,
            "cnr_proxy": self.cnr_proxy,
            "cnr_gain_vs_baseline": self.cnr_gain_vs_baseline,
            "selected": self.selected,
            "status": self.status,
            "warnings": self.warnings,
            "claim_boundary": self.claim_boundary,
        }


def discover_output_v5_task_dirs(roots: list[Path]) -> list[Path]:
    """Recursively discover Output V5 task directories."""
    task_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for current, dirnames, _ in os.walk(root):
            current_path = Path(current)
            if REQUIRED_TASK_SUBDIRS.issubset(set(dirnames)):
                task_dirs.append(current_path)
                # Skip descending into task internals to keep walk light.
                dirnames[:] = [d for d in dirnames if d not in REQUIRED_TASK_SUBDIRS]
    return sorted(task_dirs)


def build_inventory(roots: list[Path], component_preference: str = "Ey") -> list[TaskInventory]:
    """Build task-level paired inventory from configured roots."""
    inventories: list[TaskInventory] = []
    now = datetime.now(timezone.utc)
    running_threshold = now - timedelta(minutes=RUNNING_WINDOW_MINUTES)
    for task_dir in discover_output_v5_task_dirs(roots):
        entry = _inventory_one_task(task_dir, component_preference, running_threshold)
        inventories.append(entry)
    return inventories


def select_stable_pairs(inventories: list[TaskInventory]) -> list[TaskInventory]:
    """Select stable completed pairs only."""
    selected: list[TaskInventory] = []
    for item in inventories:
        if item.status != "stable_completed":
            continue
        if item.expected_trace_count is None:
            continue
        if item.raw_count != item.expected_trace_count or item.background_count != item.expected_trace_count:
            continue
        if item.raw_count != item.background_count:
            continue
        if not item.target_response_exists:
            continue
        selected.append(item)
    return selected


def ensure_scene_arrays(
    item: TaskInventory,
    *,
    component: str = "Ey",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load or prepare raw/background/target_response arrays for scoring."""
    read_dir = Path(item.paths["mygpr_read_dir"])
    read_dir.mkdir(parents=True, exist_ok=True)
    raw_npy = read_dir / f"raw_{component}.npy"
    bg_npy = read_dir / f"background_{component}.npy"
    tr_npy = read_dir / f"target_response_{component}.npy"
    summary_path = read_dir / "conversion_summary.json"

    summary: dict[str, Any] = {
        "component": component,
        "task_dir": str(item.task_dir),
        "raw_source": None,
        "background_source": None,
        "target_response_source": None,
        "generated_files": [],
        "notes": [],
    }

    raw = _load_or_build_array(
        raw_npy,
        item.paths.get("raw_dir"),
        component,
        summary,
        source_key="raw_source",
        label="raw",
    )
    bg = _load_or_build_array(
        bg_npy,
        item.paths.get("background_dir"),
        component,
        summary,
        source_key="background_source",
        label="background",
    )

    if tr_npy.exists():
        target = np.load(tr_npy)
        summary["target_response_source"] = str(tr_npy)
    else:
        merged_out = read_dir / "MyGPR_target_response.out"
        if merged_out.exists():
            target = _read_component_from_out(merged_out, component)
            np.save(tr_npy, target)
            summary["generated_files"].append(str(tr_npy))
            summary["target_response_source"] = str(merged_out)
            summary["notes"].append("target_response loaded from MyGPR_target_response.out")
        else:
            target = raw - bg
            np.save(tr_npy, target)
            summary["generated_files"].append(str(tr_npy))
            summary["target_response_source"] = "computed_raw_minus_background"
            summary["notes"].append("target_response computed from raw/background")

    if raw.shape != bg.shape or raw.shape != target.shape:
        raise ValueError(
            f"shape mismatch for scene={item.scene_id_guess}: raw={raw.shape}, background={bg.shape}, target={target.shape}"
        )

    summary["raw_shape"] = list(raw.shape)
    summary["background_shape"] = list(bg.shape)
    summary["target_response_shape"] = list(target.shape)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw, bg, target, summary


def score_scene_candidates(
    *,
    scene_id: str,
    task_dir: Path,
    raw: np.ndarray,
    target_response: np.ndarray,
    roi: dict[str, Any] | None,
    component: str = "Ey",
) -> list[CandidateScore]:
    """Run baseline/mean/median/SVD candidate scoring for one scene."""
    candidates: list[tuple[str, dict[str, Any], np.ndarray]] = []
    candidates.append(("baseline", {}, raw.copy()))

    mean_bg = np.mean(raw, axis=1, keepdims=True)
    candidates.append(("mean_background", {"mode": "global_mean"}, raw - mean_bg))

    median_bg = np.median(raw, axis=1, keepdims=True)
    candidates.append(("median_background", {"mode": "global_median"}, raw - median_bg))

    for rank in (1, 2, 3, 5, 8, 10):
        r = int(max(1, min(rank, min(raw.shape))))
        u, s, vt = svd(raw, full_matrices=False, check_finite=False)
        s_bg = np.zeros_like(s)
        s_bg[:r] = s[:r]
        bg = (u * s_bg) @ vt
        candidates.append((f"svd_rank_{rank}", {"remove_rank": rank}, raw - bg))

    roi_mode, sample_range, trace_range = _resolve_roi(raw.shape, roi)
    baseline_cnr: float | None = None
    rows: list[CandidateScore] = []

    for name, params, processed in candidates:
        warnings: list[str] = []
        mae = float(np.mean(np.abs(processed - target_response)))
        mse = float(np.mean(np.square(processed - target_response)))
        rmse = float(np.sqrt(mse))
        psnr = _compute_psnr(processed, target_response, mse)
        ssim_value = _compute_global_ssim(processed, target_response)

        roi_retention = _roi_energy_retention(processed, target_response, sample_range, trace_range)
        outside_residual = _outside_roi_residual(processed, target_response, sample_range, trace_range)
        cnr_proxy = _cnr_proxy(processed, sample_range, trace_range)
        if name == "baseline":
            baseline_cnr = cnr_proxy
        cnr_gain = None
        if baseline_cnr is not None and cnr_proxy is not None:
            cnr_gain = float(cnr_proxy - baseline_cnr)

        if roi_mode == "auto_proxy":
            warnings.append("roi_mode=auto_proxy")
        if psnr is None:
            warnings.append("psnr_none")
        if ssim_value is None:
            warnings.append("ssim_none")

        rows.append(
            CandidateScore(
                scene_id=scene_id,
                task_dir=str(task_dir),
                component=component,
                candidate=name,
                parameters=params,
                roi_mode=roi_mode,
                roi_sample_range=[sample_range[0], sample_range[1]],
                roi_trace_range=[trace_range[0], trace_range[1]],
                mae=mae,
                mse=mse,
                rmse=rmse,
                psnr=psnr,
                ssim=ssim_value,
                roi_energy_retention=roi_retention,
                outside_roi_residual_energy=outside_residual,
                cnr_proxy=cnr_proxy,
                cnr_gain_vs_baseline=cnr_gain,
                selected=False,
                status="ok",
                warnings=warnings,
                claim_boundary=CLAIM_BOUNDARY_LINES,
            )
        )

    rows.sort(key=lambda r: (r.mae, r.rmse, -(r.psnr if r.psnr is not None else -math.inf)))
    if rows:
        rows[0].selected = True
    return rows


def write_inventory_outputs(
    inventories: list[TaskInventory],
    *,
    output_md: Path,
    output_json: Path,
) -> None:
    """Write inventory markdown/json outputs."""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root_scan_time": datetime.now(timezone.utc).isoformat(),
        "inventory_count": len(inventories),
        "items": [item.to_dict() for item in inventories],
        "claim_boundary": CLAIM_BOUNDARY_LINES,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# gprMax Paired Inventory (GX-AT-SCORE-001)",
        "",
        f"- scan_time_utc: `{payload['root_scan_time']}`",
        f"- total_tasks: `{len(inventories)}`",
        "- claim_boundary:",
    ]
    lines.extend([f"  - {line}" for line in CLAIM_BOUNDARY_LINES])
    lines.extend(["", "## Tasks", ""])
    for idx, item in enumerate(inventories, start=1):
        lines.extend(
            [
                f"### {idx}. {item.scene_id_guess}",
                f"- task_dir: `{item.task_dir}`",
                f"- status: `{item.status}`",
                f"- pair_status: `{item.pair_status}`",
                f"- expected_trace_count: `{item.expected_trace_count}`",
                f"- raw_count/background_count: `{item.raw_count}/{item.background_count}`",
                f"- component: `{item.component}`",
                f"- raw_shape/background_shape/target_shape: `{item.raw_shape}` / `{item.background_shape}` / `{item.target_response_shape}`",
                f"- target_response_exists: `{item.target_response_exists}`",
                f"- MyGPR_target_response.out exists: `{item.mygpr_target_response_out_exists}`",
                f"- latest_mtime: `{item.latest_mtime_iso}`",
            ]
        )
        if item.notes:
            lines.append("- notes:")
            lines.extend([f"  - {note}" for note in item.notes])
        lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def write_scoring_outputs(
    trial_rows: list[CandidateScore],
    *,
    report_md: Path,
    trial_csv: Path,
    metrics_json: Path,
    selected_json: Path,
    claim_md: Path,
) -> None:
    """Write scoring report artifacts."""
    report_md.parent.mkdir(parents=True, exist_ok=True)
    rows_dict = [row.to_dict() for row in trial_rows]

    fieldnames = [
        "scene_id",
        "task_dir",
        "component",
        "candidate",
        "parameters",
        "roi_mode",
        "roi_sample_range",
        "roi_trace_range",
        "mae",
        "mse",
        "rmse",
        "psnr",
        "ssim",
        "roi_energy_retention",
        "outside_roi_residual_energy",
        "cnr_proxy",
        "cnr_gain_vs_baseline",
        "selected",
        "status",
        "warnings",
    ]
    with trial_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_dict:
            flat = row.copy()
            flat["parameters"] = json.dumps(flat["parameters"], ensure_ascii=False)
            flat["roi_sample_range"] = json.dumps(flat["roi_sample_range"], ensure_ascii=False)
            flat["roi_trace_range"] = json.dumps(flat["roi_trace_range"], ensure_ascii=False)
            flat["warnings"] = json.dumps(flat["warnings"], ensure_ascii=False)
            flat.pop("claim_boundary", None)
            writer.writerow(flat)

    per_scene_best: dict[str, dict[str, Any]] = {}
    for row in rows_dict:
        if not row["selected"]:
            continue
        per_scene_best[row["scene_id"]] = row

    summary = {
        "task_id": "GX-AT-SCORE-001-PAIRED-AUTOTUNE-SMOKE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trial_count": len(rows_dict),
        "scene_count": len({row["scene_id"] for row in rows_dict}),
        "selected_per_scene": per_scene_best,
        "claim_boundary": CLAIM_BOUNDARY_LINES,
    }
    metrics_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_payload = {
        "task_id": "GX-AT-SCORE-001-PAIRED-AUTOTUNE-SMOKE",
        "selected_per_scene": per_scene_best,
        "selection_rule": "min MAE, then min RMSE, then max PSNR",
        "claim_boundary": CLAIM_BOUNDARY_LINES,
    }
    selected_json.write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    claim_lines = [
        "# GX-AT-SCORE-001 Claim Boundary",
        "",
    ]
    claim_lines.extend([f"- {line}" for line in CLAIM_BOUNDARY_LINES])
    claim_lines.extend(
        [
            "",
            "- ROI auto_proxy may be used when explicit ROI is missing.",
            "- ROI metrics are diagnostic proxies, not detection accuracy metrics.",
        ]
    )
    claim_md.write_text("\n".join(claim_lines), encoding="utf-8")

    report_lines = [
        "# GX-AT-SCORE-001 Paired AutoTune Smoke Report",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- trial_count: `{summary['trial_count']}`",
        f"- scene_count: `{summary['scene_count']}`",
        "",
        "## Selected Candidate Per Scene",
        "",
    ]
    if not per_scene_best:
        report_lines.append("- No stable scene entered scoring.")
    else:
        for scene_id, row in per_scene_best.items():
            report_lines.extend(
                [
                    f"### {scene_id}",
                    f"- candidate: `{row['candidate']}`",
                    f"- mae/rmse/psnr/ssim: `{row['mae']}` / `{row['rmse']}` / `{row['psnr']}` / `{row['ssim']}`",
                    f"- roi_mode: `{row['roi_mode']}`",
                    f"- claim_boundary: `{'; '.join(CLAIM_BOUNDARY_LINES)}`",
                    "",
                ]
            )
    report_md.write_text("\n".join(report_lines), encoding="utf-8")


def _inventory_one_task(task_dir: Path, component_preference: str, running_threshold: datetime) -> TaskInventory:
    model_dir = task_dir / "1_模型输入"
    output_dir = task_dir / "2_gprMax原始输出"
    read_dir = task_dir / "3_MyGPR读取文件"
    log_dir = task_dir / "4_日志与报告"
    manifest_path = task_dir / "run_manifest.json"
    conversion_summary_path = read_dir / "conversion_summary.json"

    root_kind = "single" if "01_单次仿真" in str(task_dir) else ("batch" if "02_批量仿真" in str(task_dir) else "unknown")
    scene_id_guess = _scene_id_guess(task_dir, manifest_path)
    manifest = _load_json_if_exists(manifest_path)
    conv = _load_json_if_exists(conversion_summary_path)

    raw_dir = _pick_output_subdir(output_dir, want="raw")
    bg_dir = _pick_output_subdir(output_dir, want="background")
    raw_indices = _collect_out_indices(raw_dir)
    bg_indices = _collect_out_indices(bg_dir)
    raw_count = len(raw_indices)
    background_count = len(bg_indices)
    expected = _expected_trace_count(manifest, conv)

    component = _detect_component(read_dir, conv, component_preference)
    raw_shape = _safe_shape_from_npy(read_dir / f"raw_{component_preference}.npy")
    background_shape = _safe_shape_from_npy(read_dir / f"background_{component_preference}.npy")
    target_shape = _safe_shape_from_npy(read_dir / f"target_response_{component_preference}.npy")

    target_response_exists = (read_dir / f"target_response_{component_preference}.npy").exists()
    mygpr_target_out_exists = (read_dir / "MyGPR_target_response.out").exists()

    latest_mtime = _latest_mtime(task_dir)
    latest_iso = latest_mtime.isoformat()
    notes: list[str] = []

    if expected is None:
        notes.append("expected_trace_count_unknown")
    if raw_count != background_count:
        notes.append("raw_background_count_mismatch")
    if expected is not None:
        if raw_count != expected:
            notes.append(f"raw_count_not_expected({raw_count}!={expected})")
        if background_count != expected:
            notes.append(f"background_count_not_expected({background_count}!={expected})")
    if latest_mtime >= running_threshold:
        notes.append("recent_mtime_within_running_window")

    status = _classify_status(
        expected=expected,
        raw_count=raw_count,
        background_count=background_count,
        target_response_exists=target_response_exists,
        latest_mtime=latest_mtime,
        running_threshold=running_threshold,
        manifest=manifest,
    )
    pair_status = _classify_pair_status(expected, raw_count, background_count, target_response_exists)

    return TaskInventory(
        scene_id_guess=scene_id_guess,
        task_dir=task_dir,
        root_kind=root_kind,
        expected_trace_count=expected,
        raw_count=raw_count,
        background_count=background_count,
        raw_indices=raw_indices,
        background_indices=bg_indices,
        component=component,
        raw_shape=raw_shape,
        background_shape=background_shape,
        target_response_shape=target_shape,
        target_response_exists=target_response_exists,
        mygpr_target_response_out_exists=mygpr_target_out_exists,
        status=status,
        pair_status=pair_status,
        latest_mtime_iso=latest_iso,
        notes=notes,
        paths={
            "model_input_dir": str(model_dir),
            "raw_dir": str(raw_dir) if raw_dir else "",
            "background_dir": str(bg_dir) if bg_dir else "",
            "mygpr_read_dir": str(read_dir),
            "log_dir": str(log_dir),
            "run_manifest": str(manifest_path),
            "conversion_summary": str(conversion_summary_path),
        },
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _scene_id_guess(task_dir: Path, manifest_path: Path) -> str:
    manifest = _load_json_if_exists(manifest_path)
    if manifest:
        value = manifest.get("scene_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return task_dir.name


def _pick_output_subdir(output_dir: Path, want: str) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [p for p in output_dir.iterdir() if p.is_dir()]
    lower_map = {p: p.name.lower() for p in candidates}
    if want == "raw":
        keys = ["raw", "含目标", "target"]
    else:
        keys = ["background", "纯背景", "bg", "no_target", "empty", "clutter"]
    for p, name in lower_map.items():
        if any(k in name for k in keys):
            return p
    return candidates[0] if candidates else None


def _collect_out_indices(folder: Path | None) -> list[int]:
    if folder is None or not folder.exists():
        return []
    indices: list[int] = []
    for file in folder.glob("*.out"):
        match = re.search(r"(\d+)$", file.stem)
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def _expected_trace_count(manifest: dict[str, Any] | None, conv: dict[str, Any] | None) -> int | None:
    if manifest:
        stages = manifest.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                if stage.get("stage") == "run_plan":
                    value = stage.get("trace_count")
                    if isinstance(value, int) and value > 0:
                        return value
    if conv:
        preflight = conv.get("preflight_analysis")
        if isinstance(preflight, dict):
            value = preflight.get("expected_count")
            if isinstance(value, int) and value > 0:
                return value
    return None


def _detect_component(read_dir: Path, conv: dict[str, Any] | None, fallback: str) -> str | None:
    if conv:
        sel = conv.get("selected_component")
        if isinstance(sel, str) and sel.strip():
            return sel.strip()
    for file in read_dir.glob("raw_*.npy"):
        suffix = file.stem.removeprefix("raw_")
        if suffix:
            return suffix
    return fallback


def _safe_shape_from_npy(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r")
        return [int(arr.shape[0]), int(arr.shape[1])] if arr.ndim == 2 else list(arr.shape)
    except Exception:
        return None


def _latest_mtime(path: Path) -> datetime:
    latest = path.stat().st_mtime
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest:
                latest = mtime
    return datetime.fromtimestamp(latest, tz=timezone.utc)


def _classify_status(
    *,
    expected: int | None,
    raw_count: int,
    background_count: int,
    target_response_exists: bool,
    latest_mtime: datetime,
    running_threshold: datetime,
    manifest: dict[str, Any] | None,
) -> str:
    if expected is not None and raw_count == expected and background_count == expected and target_response_exists:
        return "stable_completed"
    if latest_mtime >= running_threshold:
        return "running_or_unstable"
    if expected is None:
        return "unknown"
    if raw_count == 0 and background_count == 0:
        return "unknown"
    if manifest and _manifest_has_failure(manifest):
        return "incomplete"
    return "incomplete"


def _manifest_has_failure(manifest: dict[str, Any]) -> bool:
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        status = str(stage.get("status", "")).lower()
        if status in {"failed", "error", "interrupted", "timeout"}:
            return True
    return False


def _classify_pair_status(
    expected: int | None,
    raw_count: int,
    background_count: int,
    target_response_exists: bool,
) -> str:
    if expected is None:
        return "unknown"
    if raw_count == 0 and background_count > 0:
        return "missing_raw"
    if background_count == 0 and raw_count > 0:
        return "missing_background"
    if raw_count != background_count:
        return "ambiguous"
    if raw_count == expected and target_response_exists:
        return "complete_candidate"
    if raw_count < expected:
        return "incomplete"
    return "unknown"


def _load_or_build_array(
    npy_path: Path,
    out_dir_path: str | None,
    component: str,
    summary: dict[str, Any],
    *,
    source_key: str,
    label: str,
) -> np.ndarray:
    if npy_path.exists():
        summary[source_key] = str(npy_path)
        return np.load(npy_path)
    if not out_dir_path:
        raise FileNotFoundError(f"{label} npy missing and no out dir found: {npy_path}")
    out_dir = Path(out_dir_path)
    arr = _build_array_from_out_dir(out_dir, component)
    np.save(npy_path, arr)
    summary["generated_files"].append(str(npy_path))
    summary[source_key] = str(out_dir)
    summary["notes"].append(f"{label} built from raw .out files")
    return arr


def _build_array_from_out_dir(out_dir: Path, component: str) -> np.ndarray:
    out_files = sorted(out_dir.glob("*.out"), key=_out_file_sort_key)
    if not out_files:
        raise FileNotFoundError(f"no .out files found in {out_dir}")
    traces = []
    for file in out_files:
        arr = _read_component_from_out(file, component)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(f"unsupported trace shape in {file}: {arr.shape}")
        traces.append(arr.astype(np.float64, copy=False))
    n_samples = traces[0].shape[0]
    if any(t.shape[0] != n_samples for t in traces):
        raise ValueError(f"inconsistent sample length across out files in {out_dir}")
    stacked = np.column_stack(traces)
    return stacked


def _out_file_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    match = re.search(r"(\d+)$", stem)
    if not match:
        return stem, 0
    prefix = stem[: match.start(1)]
    idx = int(match.group(1))
    return prefix, idx


def _read_component_from_out(out_path: Path, component: str) -> np.ndarray:
    dataset = f"rxs/rx1/{component}"
    with h5py.File(out_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"dataset missing in {out_path}: {dataset}")
        arr = np.asarray(f[dataset], dtype=np.float64)
    return arr


def _resolve_roi(shape: tuple[int, int], roi: dict[str, Any] | None) -> tuple[str, tuple[int, int], tuple[int, int]]:
    n_samples, n_traces = shape
    if roi and isinstance(roi, dict):
        s = roi.get("sample_range")
        t = roi.get("trace_range")
        if _valid_range(s, n_samples) and _valid_range(t, n_traces):
            return "manifest", (int(s[0]), int(s[1])), (int(t[0]), int(t[1]))
    s0 = int(n_samples * 0.35)
    s1 = int(n_samples * 0.75)
    half_w = max(3, n_traces // 6)
    center = n_traces // 2
    t0 = max(0, center - half_w)
    t1 = min(n_traces, center + half_w + 1)
    return "auto_proxy", (s0, s1), (t0, t1)


def _valid_range(value: Any, upper: int) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    a, b = value
    if not isinstance(a, int) or not isinstance(b, int):
        return False
    return 0 <= a < b <= upper


def _compute_psnr(x: np.ndarray, y: np.ndarray, mse: float) -> float | None:
    if mse <= 0.0:
        return None
    peak = float(max(np.max(np.abs(x)), np.max(np.abs(y))))
    if peak <= 0.0:
        return None
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))


def _compute_global_ssim(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.shape != y.shape:
        return None
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    sigma_x = float(np.var(x))
    sigma_y = float(np.var(y))
    cov = float(np.mean((x - mu_x) * (y - mu_y)))
    dynamic = float(max(np.max(np.abs(x)), np.max(np.abs(y))))
    if dynamic <= 0.0:
        dynamic = 1.0
    c1 = (0.01 * dynamic) ** 2
    c2 = (0.03 * dynamic) ** 2
    denom = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    if denom == 0.0:
        return None
    return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / denom)


def _roi_energy_retention(
    processed: np.ndarray, target: np.ndarray, sample_range: tuple[int, int], trace_range: tuple[int, int]
) -> float | None:
    s0, s1 = sample_range
    t0, t1 = trace_range
    roi_p = float(np.sum(np.square(processed[s0:s1, t0:t1])))
    roi_t = float(np.sum(np.square(target[s0:s1, t0:t1])))
    if roi_t == 0.0:
        return None
    return float(roi_p / roi_t)


def _outside_roi_residual(
    processed: np.ndarray, target: np.ndarray, sample_range: tuple[int, int], trace_range: tuple[int, int]
) -> float:
    s0, s1 = sample_range
    t0, t1 = trace_range
    diff = processed - target
    mask = np.ones(diff.shape, dtype=bool)
    mask[s0:s1, t0:t1] = False
    vals = diff[mask]
    if vals.size == 0:
        return 0.0
    return float(np.mean(np.square(vals)))


def _cnr_proxy(processed: np.ndarray, sample_range: tuple[int, int], trace_range: tuple[int, int]) -> float | None:
    s0, s1 = sample_range
    t0, t1 = trace_range
    roi_vals = processed[s0:s1, t0:t1]
    if roi_vals.size == 0:
        return None
    mask = np.ones(processed.shape, dtype=bool)
    mask[s0:s1, t0:t1] = False
    bg_vals = processed[mask]
    if bg_vals.size == 0:
        return None
    signal = float(np.mean(np.abs(roi_vals)))
    bg_mean = float(np.mean(np.abs(bg_vals)))
    bg_std = float(np.std(bg_vals))
    denom = bg_std if bg_std > 1e-12 else 1e-12
    return float((signal - bg_mean) / denom)

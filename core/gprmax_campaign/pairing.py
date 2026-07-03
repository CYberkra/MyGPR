#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired output validation and target_response generation (backend-only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.gprmax_campaign.metrics import compute_paired_metrics


@dataclass(frozen=True)
class PairedOutputSpec:
    """Specification for validating one raw/background output pair."""

    campaign_id: str
    scene_id: str
    raw_output_path: Path
    background_output_path: Path
    output_dir: Path
    target_roi: dict[str, Any] | str | None = None
    source_format: str = "auto"


@dataclass(frozen=True)
class PairedOutputValidationResult:
    """Validation result for one raw/background output pair."""

    campaign_id: str
    scene_id: str
    status: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    raw_shape: tuple[int, int] | None = None
    background_shape: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "scene_id": self.scene_id,
            "status": self.status,
            "issues": list(self.issues),
            "raw_shape": list(self.raw_shape) if self.raw_shape else None,
            "background_shape": list(self.background_shape)
            if self.background_shape
            else None,
        }


@dataclass(frozen=True)
class TargetResponseResult:
    """Target-response generation result."""

    campaign_id: str
    scene_id: str
    status: str
    output_dir: Path
    validation_summary_path: Path
    metrics_path: Path | None = None
    target_response_npy_path: Path | None = None
    target_response_csv_path: Path | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "scene_id": self.scene_id,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "validation_summary_path": str(self.validation_summary_path),
            "metrics_path": str(self.metrics_path) if self.metrics_path else None,
            "target_response_npy_path": str(self.target_response_npy_path)
            if self.target_response_npy_path
            else None,
            "target_response_csv_path": str(self.target_response_csv_path)
            if self.target_response_csv_path
            else None,
            "issues": list(self.issues),
            "metrics": self.metrics,
        }


def discover_converted_pair_paths(
    scene_root: str | Path,
    *,
    prefer_format: str = "npy",
    raw_path: str | Path | None = None,
    background_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Discover converted raw/background array paths under a scene root.

    Expected layout:
    - <scene_root>/raw_with_target/converted/raw_bscan.{npy,csv}
    - <scene_root>/background_only/converted/background_bscan.{npy,csv}
    """
    if raw_path and background_path:
        return (
            Path(raw_path).expanduser().resolve(),
            Path(background_path).expanduser().resolve(),
        )
    root = Path(scene_root).expanduser().resolve()
    raw_converted = root / "raw_with_target" / "converted"
    bg_converted = root / "background_only" / "converted"
    raw = (
        _pick_converted_file(raw_converted, "raw_bscan", prefer_format)
        if not raw_path
        else Path(raw_path).expanduser().resolve()
    )
    bg = (
        _pick_converted_file(bg_converted, "background_bscan", prefer_format)
        if not background_path
        else Path(background_path).expanduser().resolve()
    )
    return raw, bg


def validate_paired_outputs(
    spec: PairedOutputSpec,
) -> tuple[PairedOutputValidationResult, np.ndarray | None, np.ndarray | None]:
    """Validate one pair and return parsed arrays when ready."""
    issues: list[dict[str, Any]] = []
    raw_path = Path(spec.raw_output_path).expanduser().resolve()
    background_path = Path(spec.background_output_path).expanduser().resolve()

    if not raw_path.exists():
        issues.append(
            _issue("error", "raw_missing", "raw output file does not exist", raw_path)
        )
    if not background_path.exists():
        issues.append(
            _issue(
                "error",
                "background_missing",
                "background output file does not exist",
                background_path,
            )
        )
    if issues:
        return (
            PairedOutputValidationResult(
                campaign_id=spec.campaign_id,
                scene_id=spec.scene_id,
                status="invalid",
                issues=issues,
            ),
            None,
            None,
        )

    raw = _load_numeric_array(raw_path, spec.source_format, issues, "raw")
    background = _load_numeric_array(
        background_path, spec.source_format, issues, "background"
    )
    if raw is None or background is None:
        return (
            PairedOutputValidationResult(
                campaign_id=spec.campaign_id,
                scene_id=spec.scene_id,
                status="invalid",
                issues=issues,
            ),
            None,
            None,
        )

    if raw.ndim != 2:
        issues.append(
            _issue("error", "raw_not_2d", "raw output array must be 2D", raw_path)
        )
    if background.ndim != 2:
        issues.append(
            _issue(
                "error",
                "background_not_2d",
                "background output array must be 2D",
                background_path,
            )
        )
    if raw.size == 0:
        issues.append(_issue("error", "raw_empty", "raw output array is empty", raw_path))
    if background.size == 0:
        issues.append(
            _issue("error", "background_empty", "background output array is empty", background_path)
        )
    if raw.shape != background.shape:
        issues.append(
            _issue(
                "error",
                "shape_mismatch",
                "shape mismatch: "
                f"raw={raw.shape} path={raw_path}; "
                f"background={background.shape} path={background_path}",
            )
        )
    if not np.all(np.isfinite(raw)):
        issues.append(
            _issue(
                "error",
                "raw_nan_or_inf",
                "raw output contains NaN or Inf",
                raw_path,
            )
        )
    if not np.all(np.isfinite(background)):
        issues.append(
            _issue(
                "error",
                "background_nan_or_inf",
                "background output contains NaN or Inf",
                background_path,
            )
        )

    status = "ready" if not issues else "invalid"
    result = PairedOutputValidationResult(
        campaign_id=spec.campaign_id,
        scene_id=spec.scene_id,
        status=status,
        issues=issues,
        raw_shape=tuple(raw.shape) if raw.ndim == 2 else None,
        background_shape=tuple(background.shape) if background.ndim == 2 else None,
    )
    return (result, raw if status == "ready" else None, background if status == "ready" else None)


def generate_target_response(spec: PairedOutputSpec) -> TargetResponseResult:
    """Validate pair and generate target_response artifacts when ready."""
    output_dir = Path(spec.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_summary_path = output_dir / "paired_validation_summary.json"
    metrics_path = output_dir / "paired_metrics.json"
    response_npy_path = output_dir / "target_response.npy"
    response_csv_path = output_dir / "target_response.csv"

    validation, raw, background = validate_paired_outputs(spec)
    validation_summary_path.write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if validation.status != "ready" or raw is None or background is None:
        return TargetResponseResult(
            campaign_id=spec.campaign_id,
            scene_id=spec.scene_id,
            status="invalid",
            output_dir=output_dir,
            validation_summary_path=validation_summary_path,
            issues=validation.issues,
        )

    target_response = raw - background
    np.save(response_npy_path, target_response)
    np.savetxt(response_csv_path, target_response, delimiter=",", fmt="%.10g")
    roi = _resolve_roi(spec.target_roi)
    metrics = compute_paired_metrics(raw, background, target_response, roi=roi)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return TargetResponseResult(
        campaign_id=spec.campaign_id,
        scene_id=spec.scene_id,
        status="success",
        output_dir=output_dir,
        validation_summary_path=validation_summary_path,
        metrics_path=metrics_path,
        target_response_npy_path=response_npy_path,
        target_response_csv_path=response_csv_path,
        issues=[],
        metrics=metrics,
    )


def _load_numeric_array(
    path: Path,
    source_format: str,
    issues: list[dict[str, Any]],
    role: str,
) -> np.ndarray | None:
    fmt = _resolve_source_format(path, source_format)
    if fmt is None:
        issues.append(
            _issue(
                "error",
                f"{role}_format_unsupported",
                f"unsupported source format for {role}: {path.suffix.lower()}",
                path,
            )
        )
        return None
    try:
        if fmt == "npy":
            arr = np.load(path)
        else:
            arr = np.genfromtxt(path, delimiter=",", ndmin=2)
    except Exception as exc:
        issues.append(
            _issue(
                "error",
                f"{role}_load_failed",
                f"failed to load {role}: {exc}",
                path,
            )
        )
        return None
    try:
        arr = np.asarray(arr, dtype=np.float64)
    except Exception as exc:
        issues.append(
            _issue(
                "error",
                f"{role}_dtype_conversion_failed",
                f"failed to convert {role} to float: {exc}",
                path,
            )
        )
        return None
    return arr


def _resolve_source_format(path: Path, source_format: str) -> str | None:
    fmt = (source_format or "auto").strip().lower()
    if fmt in {"csv", "npy"}:
        return fmt
    if fmt != "auto":
        return None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".csv":
        return "csv"
    return None


def _issue(level: str, code: str, message: str, path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"level": level, "code": code, "message": message}
    if path is not None:
        payload["path"] = str(path)
    return payload


def _pick_converted_file(folder: Path, stem: str, prefer_format: str) -> Path:
    suffix_priority = [".npy", ".csv"] if prefer_format.lower() != "csv" else [".csv", ".npy"]
    for suffix in suffix_priority:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    # fallback: pick first matching stem.* if present
    for candidate in folder.glob(f"{stem}.*"):
        if candidate.suffix.lower() in {".npy", ".csv"}:
            return candidate.resolve()
    return (folder / f"{stem}{suffix_priority[0]}").resolve()


def _resolve_roi(target_roi: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if target_roi is None:
        return None
    if isinstance(target_roi, dict):
        return target_roi
    roi_candidate = str(target_roi).strip()
    if not roi_candidate:
        return None
    roi_path = Path(roi_candidate).expanduser()
    if roi_path.exists():
        try:
            payload = json.loads(roi_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
    try:
        payload = json.loads(roi_candidate)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None

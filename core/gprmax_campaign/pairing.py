#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired output validation and target_response generation (backend-only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairedOutputSpec:
    """Specification for validating one raw/background output pair."""

    campaign_id: str
    scene_id: str
    raw_output_path: Path
    background_output_path: Path
    output_dir: Path
    target_roi: str | None = None
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
                f"shape mismatch: raw={raw.shape}, background={background.shape}",
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
    metrics = _build_metrics(raw, background, target_response)
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
            arr = np.genfromtxt(path, delimiter=",")
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


def _build_metrics(raw: np.ndarray, background: np.ndarray, target_response: np.ndarray) -> dict[str, Any]:
    raw_energy = float(np.sum(np.square(raw)))
    background_energy = float(np.sum(np.square(background)))
    target_energy = float(np.sum(np.square(target_response)))
    return {
        "raw_shape": list(raw.shape),
        "background_shape": list(background.shape),
        "target_response_shape": list(target_response.shape),
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "background_min": float(np.min(background)),
        "background_max": float(np.max(background)),
        "background_mean": float(np.mean(background)),
        "background_std": float(np.std(background)),
        "target_response_min": float(np.min(target_response)),
        "target_response_max": float(np.max(target_response)),
        "target_response_mean": float(np.mean(target_response)),
        "target_response_std": float(np.std(target_response)),
        "raw_energy": raw_energy,
        "background_energy": background_energy,
        "target_response_energy": target_energy,
        "target_to_background_energy_ratio": (
            float(target_energy / background_energy) if background_energy > 0 else None
        ),
        "abs_difference_mean": float(np.mean(np.abs(target_response))),
        "abs_difference_max": float(np.max(np.abs(target_response))),
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preview PNG and lightweight report generation for paired outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.gprmax_campaign.pairing import PairedOutputSpec, generate_target_response, validate_paired_outputs


@dataclass(frozen=True)
class PairPreviewReportResult:
    """Result payload for paired preview/report generation."""

    campaign_id: str
    scene_id: str
    status: str
    output_dir: Path
    raw_preview_path: Path | None
    background_preview_path: Path | None
    target_response_preview_path: Path | None
    paired_preview_panel_path: Path | None
    report_md_path: Path | None
    summary_json_path: Path | None
    issues: list[dict[str, Any]]
    metrics: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "scene_id": self.scene_id,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "raw_preview_path": str(self.raw_preview_path) if self.raw_preview_path else None,
            "background_preview_path": str(self.background_preview_path) if self.background_preview_path else None,
            "target_response_preview_path": str(self.target_response_preview_path)
            if self.target_response_preview_path
            else None,
            "paired_preview_panel_path": str(self.paired_preview_panel_path)
            if self.paired_preview_panel_path
            else None,
            "report_md_path": str(self.report_md_path) if self.report_md_path else None,
            "summary_json_path": str(self.summary_json_path) if self.summary_json_path else None,
            "issues": list(self.issues),
            "metrics": self.metrics,
        }


def generate_pair_preview_report(
    *,
    campaign_id: str,
    scene_id: str,
    raw_output_path: str | Path,
    background_output_path: str | Path,
    output_dir: str | Path,
    target_response_path: str | Path | None = None,
    source_format: str = "auto",
    target_roi: str | None = None,
    title: str | None = None,
) -> PairPreviewReportResult:
    """Generate raw/background/target preview PNGs and lightweight report stubs."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = out_dir / "paired_report_summary.json"
    report_md_path = out_dir / "paired_target_response_report.md"
    raw_preview = out_dir / "raw_preview.png"
    bg_preview = out_dir / "background_preview.png"
    tr_preview = out_dir / "target_response_preview.png"
    panel_preview = out_dir / "paired_preview_panel.png"

    pair_spec = PairedOutputSpec(
        campaign_id=campaign_id,
        scene_id=scene_id,
        raw_output_path=Path(raw_output_path),
        background_output_path=Path(background_output_path),
        output_dir=out_dir,
        target_roi=target_roi,
        source_format=source_format,
    )
    validation, raw_arr, bg_arr = validate_paired_outputs(pair_spec)
    if validation.status != "ready" or raw_arr is None or bg_arr is None:
        result = PairPreviewReportResult(
            campaign_id=campaign_id,
            scene_id=scene_id,
            status="invalid",
            output_dir=out_dir,
            raw_preview_path=None,
            background_preview_path=None,
            target_response_preview_path=None,
            paired_preview_panel_path=None,
            report_md_path=None,
            summary_json_path=summary_json_path,
            issues=list(validation.issues),
            metrics=None,
        )
        summary_json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    metrics: dict[str, Any] | None = None
    if target_response_path:
        target_path = Path(target_response_path).expanduser().resolve()
        try:
            target_arr = _load_array(target_path, "auto")
        except Exception as exc:
            result = PairPreviewReportResult(
                campaign_id=campaign_id,
                scene_id=scene_id,
                status="invalid",
                output_dir=out_dir,
                raw_preview_path=None,
                background_preview_path=None,
                target_response_preview_path=None,
                paired_preview_panel_path=None,
                report_md_path=None,
                summary_json_path=summary_json_path,
                issues=[
                    {
                        "level": "error",
                        "code": "target_response_load_failed",
                        "message": f"failed to load target_response path={target_path}: {exc}",
                    }
                ],
                metrics=None,
            )
            summary_json_path.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return result
    else:
        tr_result = generate_target_response(pair_spec)
        if tr_result.status != "success" or tr_result.target_response_npy_path is None:
            result = PairPreviewReportResult(
                campaign_id=campaign_id,
                scene_id=scene_id,
                status="invalid",
                output_dir=out_dir,
                raw_preview_path=None,
                background_preview_path=None,
                target_response_preview_path=None,
                paired_preview_panel_path=None,
                report_md_path=None,
                summary_json_path=summary_json_path,
                issues=list(tr_result.issues),
                metrics=None,
            )
            summary_json_path.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result
        target_arr = np.load(tr_result.target_response_npy_path)
        if tr_result.metrics is not None:
            metrics = dict(tr_result.metrics)

    if target_arr.shape != raw_arr.shape:
        issues = [
            {
                "level": "error",
                "code": "target_response_shape_mismatch",
                "message": (
                    "target_response shape mismatch: "
                    f"target={target_arr.shape} path="
                    f"{Path(target_response_path).expanduser().resolve() if target_response_path else out_dir / 'target_response.npy'}; "
                    f"raw={raw_arr.shape} path={Path(raw_output_path).expanduser().resolve()}"
                ),
            }
        ]
        result = PairPreviewReportResult(
            campaign_id=campaign_id,
            scene_id=scene_id,
            status="invalid",
            output_dir=out_dir,
            raw_preview_path=None,
            background_preview_path=None,
            target_response_preview_path=None,
            paired_preview_panel_path=None,
            report_md_path=None,
            summary_json_path=summary_json_path,
            issues=issues,
            metrics=metrics,
        )
        summary_json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    if metrics is None:
        metrics_path = out_dir / "paired_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    _save_single_preview(raw_arr, raw_preview, f"{title or scene_id} | raw | shape={raw_arr.shape}")
    _save_single_preview(bg_arr, bg_preview, f"{title or scene_id} | background | shape={bg_arr.shape}")
    _save_single_preview(
        target_arr, tr_preview, f"{title or scene_id} | target_response | shape={target_arr.shape}"
    )
    _save_panel_preview(raw_arr, bg_arr, target_arr, panel_preview, title or f"{scene_id} paired preview")

    _write_report_md(
        path=report_md_path,
        campaign_id=campaign_id,
        scene_id=scene_id,
        raw_output_path=Path(raw_output_path).expanduser().resolve(),
        background_output_path=Path(background_output_path).expanduser().resolve(),
        target_response_path=Path(target_response_path).expanduser().resolve()
        if target_response_path
        else (out_dir / "target_response.npy"),
        raw_shape=raw_arr.shape,
        background_shape=bg_arr.shape,
        target_shape=target_arr.shape,
        metrics=metrics,
        previews=[raw_preview, bg_preview, tr_preview, panel_preview],
    )

    result = PairPreviewReportResult(
        campaign_id=campaign_id,
        scene_id=scene_id,
        status="success",
        output_dir=out_dir,
        raw_preview_path=raw_preview,
        background_preview_path=bg_preview,
        target_response_preview_path=tr_preview,
        paired_preview_panel_path=panel_preview,
        report_md_path=report_md_path,
        summary_json_path=summary_json_path,
        issues=[],
        metrics=metrics,
    )
    summary_json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _load_array(path: Path, source_format: str) -> np.ndarray:
    fmt = source_format.lower()
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".npy":
            fmt = "npy"
        elif suffix == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"unsupported source format: {suffix or '<none>'}")
    if fmt == "npy":
        arr = np.load(path)
    else:
        arr = np.genfromtxt(path, delimiter=",", ndmin=2)
    return np.asarray(arr, dtype=np.float64)


def _save_single_preview(data: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    vmin, vmax = _robust_limits(data)
    im = ax.imshow(data, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Trace Index")
    ax.set_ylabel("Sample Index")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_panel_preview(raw: np.ndarray, bg: np.ndarray, target: np.ndarray, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=120)
    for ax, data, name in zip(axes, [raw, bg, target], ["raw", "background", "target_response"]):
        vmin, vmax = _robust_limits(data)
        im = ax.imshow(data, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\nshape={data.shape}")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _robust_limits(data: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(data[np.isfinite(data)], dtype=np.float64)
    if finite.size == 0:
        return (-1.0, 1.0)
    lo = float(np.percentile(finite, 2.0))
    hi = float(np.percentile(finite, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.mean(finite))
        spread = float(np.std(finite))
        if spread <= 0:
            spread = max(abs(center), 1.0)
        lo, hi = center - spread, center + spread
        if hi <= lo:
            lo, hi = -1.0, 1.0
    return lo, hi


def _write_report_md(
    *,
    path: Path,
    campaign_id: str,
    scene_id: str,
    raw_output_path: Path,
    background_output_path: Path,
    target_response_path: Path,
    raw_shape: tuple[int, int],
    background_shape: tuple[int, int],
    target_shape: tuple[int, int],
    metrics: dict[str, Any] | None,
    previews: list[Path],
) -> None:
    lines = [
        "# Paired Target Response Report",
        "",
        f"- campaign_id: `{campaign_id}`",
        f"- scene_id: `{scene_id}`",
        f"- raw_output_path: `{raw_output_path}`",
        f"- background_output_path: `{background_output_path}`",
        f"- target_response_path: `{target_response_path}`",
        "",
        "## Shapes",
        f"- raw_shape: `{raw_shape}`",
        f"- background_shape: `{background_shape}`",
        f"- target_response_shape: `{target_shape}`",
        "",
        "## Preview Files",
    ]
    for item in previews:
        lines.append(f"- `{item}`")
    if metrics:
        lines.extend(["", "## Metrics (paired_metrics.json)"])
        for key in [
            "raw_energy",
            "background_energy",
            "target_response_energy",
            "target_to_background_energy_ratio",
            "abs_difference_mean",
            "abs_difference_max",
        ]:
            if key in metrics:
                lines.append(f"- {key}: `{metrics[key]}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "- This output is synthetic/paired diagnostic only.",
            "- It is not a real field validation claim.",
            "- It is not an AutoTune superiority claim.",
            "- It is not a target correctness claim without ROI/ground-truth review.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

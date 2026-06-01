#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only GX-008 gprMax model draft inspector."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GX008_MODELS_DIR = ROOT / "experiments" / "gprmax" / "GX-008" / "models"
DEFAULT_GPRMAX_PYTHON = r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"


@dataclass
class ModelInspection:
    """Normalized read-only model draft inspection result."""

    scene_id: str
    scene_role: str = ""
    domain: str = ""
    dx_dy_dz: str = ""
    time_window: str = ""
    waveform: str = ""
    source: str = ""
    receiver: str = ""
    src_steps: str = ""
    rx_steps: str = ""
    expected_num_runs: int | None = None
    soil_type: str = ""
    target_material: str = ""
    target_type: str = ""
    target_depth_class: str = ""
    roi: dict[str, Any] | None = None
    pair_contract_status: str = "warning"
    pair_contract_checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    generated_gpu_command: str = ""
    raw_text: str = ""
    background_text: str = ""
    materials_text: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_role": self.scene_role,
            "domain": self.domain,
            "dx_dy_dz": self.dx_dy_dz,
            "time_window": self.time_window,
            "waveform": self.waveform,
            "source": self.source,
            "receiver": self.receiver,
            "src_steps": self.src_steps,
            "rx_steps": self.rx_steps,
            "expected_num_runs": self.expected_num_runs,
            "soil_type": self.soil_type,
            "target_material": self.target_material,
            "target_type": self.target_type,
            "target_depth_class": self.target_depth_class,
            "roi": self.roi,
            "pair_contract_status": self.pair_contract_status,
            "pair_contract_checks": self.pair_contract_checks,
            "warnings": self.warnings,
            "generated_gpu_command": self.generated_gpu_command,
            "raw_text": self.raw_text,
            "background_text": self.background_text,
            "materials_text": self.materials_text,
            "manifest": self.manifest,
        }


def _read_text(path: Path, warnings: list[str]) -> str:
    if not path.exists():
        warnings.append(f"missing file: {path}")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"failed to read file: {path}: {exc}")
        return ""


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing json: {path}")
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"malformed json: {path}: {exc}")
        return {}
    if not isinstance(loaded, dict):
        warnings.append(f"json is not an object: {path}")
        return {}
    return loaded


def _directives(text: str) -> dict[str, list[str]]:
    directives: dict[str, list[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped[1:].split(":", 1)
        directives.setdefault(key.strip(), []).append(value.strip())
    return directives


def _first(directives: dict[str, list[str]], key: str) -> str:
    values = directives.get(key) or []
    return values[0] if values else ""


def _target_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#cylinder:") or stripped.startswith("#sphere:"):
            lines.append(stripped)
        elif stripped.startswith("#box:"):
            material = stripped.split()[-1].lower() if stripped.split() else ""
            if not material.startswith(("dry_sand", "damp_sand", "air", "free_space")):
                lines.append(stripped)
    return lines


def _compare_directive(raw_directives: dict[str, list[str]], bg_directives: dict[str, list[str]], key: str) -> bool:
    return raw_directives.get(key, []) == bg_directives.get(key, [])


def inspect_pair_contract(raw_text: str, background_text: str) -> tuple[str, dict[str, bool], list[str]]:
    """Inspect the current GX-008 subset pair contract."""
    warnings: list[str] = []
    raw_directives = _directives(raw_text)
    bg_directives = _directives(background_text)
    checks = {
        "domain_same": _compare_directive(raw_directives, bg_directives, "domain"),
        "grid_same": _compare_directive(raw_directives, bg_directives, "dx_dy_dz"),
        "time_window_same": _compare_directive(raw_directives, bg_directives, "time_window"),
        "waveform_same": _compare_directive(raw_directives, bg_directives, "waveform"),
        "source_same": _compare_directive(raw_directives, bg_directives, "hertzian_dipole"),
        "receiver_same": _compare_directive(raw_directives, bg_directives, "rx"),
        "src_steps_same": _compare_directive(raw_directives, bg_directives, "src_steps"),
        "rx_steps_same": _compare_directive(raw_directives, bg_directives, "rx_steps"),
        "materials_same": _compare_directive(raw_directives, bg_directives, "material"),
        "raw_has_target": bool(_target_lines(raw_text)),
        "background_has_no_target": not bool(_target_lines(background_text)),
    }
    for key, passed in checks.items():
        if not passed:
            warnings.append(f"pair contract check failed: {key}")
    status = "pairable" if all(checks.values()) else "warning"
    return status, checks, warnings


def generate_gpu_command(
    scene_id: str,
    expected_num_runs: int | None = 41,
    gprmax_python: str = DEFAULT_GPRMAX_PYTHON,
) -> str:
    """Generate a copyable command; it does not execute anything."""
    runs = expected_num_runs or 41
    return (
        "scripts\\run_gprmax_gpu_env.bat -- python scripts\\gprmax_campaign_runner.py "
        "--campaign experiments/gprmax/GX-008/campaign_draft.yaml "
        f"--run-scene {scene_id} --variant raw_with_target --num-runs {runs} "
        f"--gpu-device 0 --gprmax-python {gprmax_python} --timeout-seconds 1800"
    )


def load_scene_model(scene_id: str, models_root: str | Path | None = None) -> ModelInspection:
    """Load a GX-008 scene draft in read-only protected mode."""
    root = Path(models_root) if models_root else GX008_MODELS_DIR
    scene_dir = root / scene_id
    warnings: list[str] = []
    raw_text = _read_text(scene_dir / "raw_with_target.in", warnings)
    background_text = _read_text(scene_dir / "background_only.in", warnings)
    materials_text = _read_text(scene_dir / "materials.txt", warnings)
    roi = _read_json(scene_dir / "roi_draft.json", warnings)
    manifest = _read_json(scene_dir / "scene_manifest_draft.json", warnings)

    raw_directives = _directives(raw_text)
    status, checks, contract_warnings = inspect_pair_contract(raw_text, background_text)
    warnings.extend(contract_warnings)
    if not roi:
        warnings.append("ROI draft missing or invalid")
    if not manifest:
        warnings.append("scene manifest draft missing or invalid")

    scan_design = manifest.get("scan_design") if isinstance(manifest.get("scan_design"), dict) else {}
    target_design = manifest.get("target_design") if isinstance(manifest.get("target_design"), dict) else {}
    target_geometry = manifest.get("target_geometry") if isinstance(manifest.get("target_geometry"), dict) else {}
    expected_num_runs = manifest.get("expected_num_runs", scan_design.get("expected_num_runs"))
    if not isinstance(expected_num_runs, int):
        expected_num_runs = None
        warnings.append("expected_num_runs missing from scene manifest")

    result = ModelInspection(
        scene_id=scene_id,
        scene_role=str(manifest.get("scene_role") or ""),
        domain=_first(raw_directives, "domain"),
        dx_dy_dz=_first(raw_directives, "dx_dy_dz"),
        time_window=_first(raw_directives, "time_window"),
        waveform=_first(raw_directives, "waveform"),
        source=_first(raw_directives, "hertzian_dipole"),
        receiver=_first(raw_directives, "rx"),
        src_steps=_first(raw_directives, "src_steps"),
        rx_steps=_first(raw_directives, "rx_steps"),
        expected_num_runs=expected_num_runs,
        soil_type=str(manifest.get("soil_type") or ""),
        target_material=str(
            manifest.get("target_material")
            or target_design.get("material")
            or target_geometry.get("material")
            or ""
        ),
        target_type=str(
            manifest.get("target_type")
            or target_design.get("type")
            or target_geometry.get("type")
            or ""
        ),
        target_depth_class=str(manifest.get("target_depth_class") or ""),
        roi=roi or None,
        pair_contract_status=status,
        pair_contract_checks=checks,
        warnings=warnings,
        generated_gpu_command=generate_gpu_command(scene_id, expected_num_runs),
        raw_text=raw_text,
        background_text=background_text,
        materials_text=materials_text,
        manifest=manifest,
    )
    return result


def summarize_scene_model(scene_id: str, models_root: str | Path | None = None) -> dict[str, Any]:
    """Return a UI-ready scene model summary."""
    return load_scene_model(scene_id, models_root).to_dict()


def validate_scene_draft(scene_id: str, models_root: str | Path | None = None) -> dict[str, Any]:
    """Validate a scene draft without executing gprMax or writing files."""
    inspection = load_scene_model(scene_id, models_root)
    return {
        "scene_id": inspection.scene_id,
        "status": inspection.pair_contract_status,
        "checks": inspection.pair_contract_checks,
        "warnings": inspection.warnings,
    }


def clone_scene_as_draft(*_args: Any, **_kwargs: Any) -> None:
    """Reserved for a future edit mode; disabled in v0."""
    raise PermissionError("gprMax model editor v0 is read-only protected mode")


def save_scene_draft(*_args: Any, **_kwargs: Any) -> None:
    """Reserved for a future edit mode; disabled in v0."""
    raise PermissionError("gprMax model editor v0 is read-only protected mode")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a native gprMax-to-CSV benchmark package for MyGPR."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gprmax_ground_truth import (
    convert_gprmax_ground_truth_to_mygpr,
    load_gprmax_ground_truth,
)
from scripts.gprmax_benchmark.audit_gprmax_package import audit_gprmax_package

try:  # pragma: no cover - exercised in environments with h5py installed
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def prepare_native_package(
    *,
    model_in: str | Path,
    out_path: str | Path,
    output_dir: str | Path,
    receiver: str,
    component: str,
    scenario_id: str,
    ground_truth: str | Path | None = None,
    copy_raw_out: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    """Convert a native gprMax receiver/component dataset into a MyGPR package."""
    model = Path(model_in).expanduser().resolve()
    raw_out = Path(out_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if h5py is None:
        return _write_pending_package(
            model=model,
            raw_out=raw_out,
            output_dir=target,
            receiver=receiver,
            component=component,
            scenario_id=scenario_id,
            reason="h5py is not installed; cannot read native gprMax .out",
            command=command,
        )
    if not model.exists() or not raw_out.exists():
        missing = []
        if not model.exists():
            missing.append(f"model.in missing: {model}")
        if not raw_out.exists():
            missing.append(f"native .out missing: {raw_out}")
        return _write_pending_package(
            model=model,
            raw_out=raw_out,
            output_dir=target,
            receiver=receiver,
            component=component,
            scenario_id=scenario_id,
            reason="; ".join(missing),
            command=command,
        )

    data, h5_metadata = _read_component(raw_out, receiver, component)
    model_target = target / f"{scenario_id}.in"
    shutil.copy2(model, model_target)
    raw_hash = _sha256(raw_out)
    csv_path = target / "mygpr_bscan.csv"
    np.savetxt(csv_path, data, delimiter=",")
    csv_hash = _sha256(csv_path)
    preview_path = target / "preview.png"
    _write_preview(data, preview_path, title=f"{scenario_id} {receiver}/{component}")

    parsed_model = _parse_model_input(model_target)
    trace_count = int(data.shape[1] if data.ndim == 2 else 1)
    sample_count = int(data.shape[0])
    total_time_ns = (
        float(h5_metadata["dt_s"]) * sample_count * 1e9
        if h5_metadata.get("dt_s")
        else parsed_model.get("time_window_ns")
    )
    scenario = {
        "schema": "mygpr_gprmax_scenario_v1",
        "scenario_id": scenario_id,
        "description": "Native gprMax .out converted to MyGPR-compatible CSV.",
        "source": {
            "kind": "native_gprmax_converted",
            "raw_out_path": str(raw_out),
            "raw_out_hash": raw_hash,
            "raw_out_committed": bool(copy_raw_out),
            "receiver": receiver,
            "component": component,
            "conversion_command": command,
        },
        "simulation": {
            "sample_count": sample_count,
            "trace_count": trace_count,
            "time_step_s": h5_metadata.get("dt_s"),
            "iterations": h5_metadata.get("iterations"),
            "total_time_ns": total_time_ns,
            "trace_step_m": _trace_step(parsed_model),
        },
        "domain_m": parsed_model.get("domain_m", []),
        "dx_dy_dz_m": parsed_model.get("dx_dy_dz_m", []),
        "antenna": {
            "source_position_m": parsed_model.get("source_position_m", []),
            "receiver_position_m": parsed_model.get("receiver_position_m", []),
            "source_step_m": parsed_model.get("source_step_m", []),
            "receiver_step_m": parsed_model.get("receiver_step_m", []),
        },
        "conversion": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "csv_file": "mygpr_bscan.csv",
            "csv_hash": csv_hash,
            "preview_file": "preview.png",
            "hdf5_dataset": f"/rxs/{receiver}/{component}",
            "hdf5_metadata": h5_metadata,
        },
    }
    (target / "scenario.json").write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "schema": "mygpr_native_gprmax_csv_manifest_v1",
        "scenario_id": scenario_id,
        "data_file": "mygpr_bscan.csv",
        "scenario_file": "scenario.json",
        "model_file": model_target.name,
        "preview_file": "preview.png",
        "raw_out_path": str(raw_out),
        "raw_out_hash": raw_hash,
        "csv_hash": csv_hash,
        "receiver": receiver,
        "component": component,
    }
    ground_truth_path = _copy_ground_truth(
        ground_truth=ground_truth,
        raw_out=raw_out,
        output_dir=target,
        data_shape=(sample_count, trace_count),
        manifest=manifest,
    )
    if copy_raw_out:
        copied = target / raw_out.name
        shutil.copy2(raw_out, copied)
        manifest["raw_out_file"] = copied.name
    if ground_truth_path:
        manifest["ground_truth_file"] = ground_truth_path.name
    (target / "native_gprmax_package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    readme = _package_readme(scenario_id, raw_out, receiver, component, data.shape, raw_hash, csv_hash)
    (target / "README.md").write_text(readme, encoding="utf-8")
    audit = audit_gprmax_package(target)
    (target / "gprmax_package_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = target / "native_gprmax_package_report.md"
    _write_native_report(
        path=report_path,
        scenario_id=scenario_id,
        raw_out=raw_out,
        receiver=receiver,
        component=component,
        shape=[sample_count, trace_count],
        raw_hash=raw_hash,
        csv_hash=csv_hash,
        audit=audit,
    )
    return {
        "status": "native_gprmax_converted",
        "package_dir": str(target),
        "csv_path": str(csv_path),
        "preview_path": str(preview_path),
        "manifest_path": str(target / "native_gprmax_package_manifest.json"),
        "audit_path": str(target / "gprmax_package_audit.json"),
        "report_path": str(report_path),
        "raw_out_hash": raw_hash,
        "csv_hash": csv_hash,
        "shape": [sample_count, trace_count],
        "receiver": receiver,
        "component": component,
        "audit": audit,
    }


def _write_pending_package(
    *,
    model: Path,
    raw_out: Path,
    output_dir: Path,
    receiver: str,
    component: str,
    scenario_id: str,
    reason: str,
    command: str | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mygpr_native_gprmax_csv_manifest_v1",
        "scenario_id": scenario_id,
        "status": "pending_native_out",
        "missing_reason": reason,
        "required_files": ["model.in", ".out or _merged.out", "ground_truth.yaml/json"],
        "model_in_requested": str(model),
        "out_requested": str(raw_out),
        "receiver": receiver,
        "component": component,
        "conversion_command": command,
    }
    path = output_dir / "native_gprmax_package_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Pending native gprMax package\n\n"
        f"Status: `pending_native_out`\n\nReason: {reason}\n\n"
        "Do not use this as native gprMax evidence until the requested `.out` exists.\n",
        encoding="utf-8",
    )
    return {
        "status": "pending_native_out",
        "package_dir": str(output_dir),
        "manifest_path": str(path),
        "missing_reason": reason,
        "receiver": receiver,
        "component": component,
    }


def _read_component(path: Path, receiver: str, component: str) -> tuple[np.ndarray, dict[str, Any]]:
    dataset = f"rxs/{receiver}/{component}"
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"native gprMax .out does not contain /{dataset}")
        data = np.asarray(handle[dataset][:], dtype=np.float32)
        attrs = dict(handle.attrs)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    metadata = {
        "iterations": _json_scalar(attrs.get("Iterations")),
        "dt_s": _json_scalar(attrs.get("dt")),
        "nx_ny_nz": _json_scalar(attrs.get("nx_ny_nz")),
        "raw_attrs": {str(key): _json_scalar(value) for key, value in attrs.items()},
    }
    return data, metadata


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_model_input(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("#") or ":" not in line:
            continue
        command, rest = line[1:].split(":", 1)
        values = rest.split()
        if command == "domain":
            parsed["domain_m"] = [float(item) for item in values[:3]]
        elif command == "dx_dy_dz":
            parsed["dx_dy_dz_m"] = [float(item) for item in values[:3]]
        elif command == "time_window" and values:
            parsed["time_window_ns"] = float(values[0]) * 1e9
        elif command == "hertzian_dipole" and len(values) >= 5:
            parsed["source_position_m"] = [float(values[1]), float(values[2]), float(values[3])]
        elif command == "rx" and len(values) >= 3:
            parsed["receiver_position_m"] = [float(values[0]), float(values[1]), float(values[2])]
        elif command == "src_steps" and len(values) >= 3:
            parsed["source_step_m"] = [float(values[0]), float(values[1]), float(values[2])]
        elif command == "rx_steps" and len(values) >= 3:
            parsed["receiver_step_m"] = [float(values[0]), float(values[1]), float(values[2])]
    return parsed


def _trace_step(parsed_model: dict[str, Any]) -> float | None:
    steps = parsed_model.get("receiver_step_m")
    if isinstance(steps, list) and steps:
        return float(steps[0])
    return None


def _copy_ground_truth(
    *,
    ground_truth: str | Path | None,
    raw_out: Path,
    output_dir: Path,
    data_shape: tuple[int, int],
    manifest: dict[str, Any],
) -> Path | None:
    candidates = []
    if ground_truth:
        candidates.append(Path(ground_truth).expanduser().resolve())
    candidates.extend([raw_out.parent / "ground_truth.yaml", raw_out.parent / "ground_truth.json"])
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() in {".yaml", ".yml"}:
            yaml_target = output_dir / "ground_truth.yaml"
            shutil.copy2(candidate, yaml_target)
            converted = convert_gprmax_ground_truth_to_mygpr(
                load_gprmax_ground_truth(str(candidate)),
                data_shape=data_shape,
            )
            json_target = output_dir / "ground_truth.json"
            json_target.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["converted_ground_truth_file"] = json_target.name
            return yaml_target
        json_target = output_dir / "ground_truth.json"
        shutil.copy2(candidate, json_target)
        return json_target
    return None


def _write_preview(data: np.ndarray, path: Path, *, title: str) -> None:
    vmax = float(np.nanpercentile(np.abs(data), 99.0)) or 1.0
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    try:
        ax.imshow(data, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_readme(
    scenario_id: str,
    raw_out: Path,
    receiver: str,
    component: str,
    shape: tuple[int, ...],
    raw_hash: str,
    csv_hash: str,
) -> str:
    return (
        f"# {scenario_id}\n\n"
        "Native gprMax `.out` converted to a MyGPR-compatible CSV benchmark package.\n\n"
        f"- raw out path: `{raw_out}`\n"
        f"- selected dataset: `/rxs/{receiver}/{component}`\n"
        f"- converted shape: `{list(shape)}`\n"
        f"- raw_out_hash: `{raw_hash}`\n"
        f"- csv_hash: `{csv_hash}`\n\n"
        "The raw `.out` may be externally referenced instead of committed. Do not make\n"
        "paper claims unless the accompanying audit reports native provenance, timing,\n"
        "PML/domain, and ROI checks as acceptable.\n"
    )


def _write_native_report(
    *,
    path: Path,
    scenario_id: str,
    raw_out: Path,
    receiver: str,
    component: str,
    shape: list[int],
    raw_hash: str,
    csv_hash: str,
    audit: dict[str, Any],
) -> None:
    lines = [
        "# GX-002 Native gprMax-to-CSV Package Report",
        "",
        "## Summary",
        f"- Scenario: `{scenario_id}`",
        f"- Status: `{audit['source']['kind']}`",
        f"- Native gprMax verified: `{audit['source']['native_gprmax_verified']}`",
        f"- Paper usable: `{audit['paper_usable']}`",
        f"- Recommendation: {audit['recommendation']}",
        "",
        "## Provenance",
        f"- Native `.out`: `{raw_out}`",
        f"- Selected receiver/component: `/rxs/{receiver}/{component}`",
        f"- Converted CSV shape: `{shape}`",
        f"- raw_out_hash: `{raw_hash}`",
        f"- csv_hash: `{csv_hash}`",
        f"- raw `.out` committed in package: `{bool(audit['files']['raw_out_files'])}`",
        "",
        "## Geometry And Timing",
        f"- domain_m: `{audit['geometry']['domain_m']}`",
        f"- dx_dy_dz_m: `{audit['geometry']['dx_dy_dz_m']}`",
        f"- time_window_ns: `{audit['geometry']['time_window_ns']}`",
        f"- time_step_s: `{audit['geometry']['time_step_s']}`",
        f"- dt source: `{audit['source']['dt_source']}`",
        f"- shape matches metadata: `{audit['shape']['shape_matches_expected']}`",
        "",
        "## PML / ROI Audit",
        f"- PML flags: `{audit['geometry']['pml_margin']['risk_flags']}`",
        f"- ROI exists: `{audit['ground_truth']['roi_exists']}`",
        f"- ROI inside B-scan: `{audit['ground_truth']['roi_inside_bscan']}`",
        "",
        "## Warnings",
    ]
    lines.extend(f"- {item}" for item in audit.get("warnings", []) or ["none"])
    lines.extend(["", "## Errors"])
    lines.extend(f"- {item}" for item in audit.get("errors", []) or ["none"])
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "This package is traceable from native gprMax `.out` to a MyGPR CSV.",
            "The raw `.out` is referenced by path and SHA-256 rather than committed.",
            "Any paper claim must preserve that raw file or regenerate it with the same",
            "model and command.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "scripts/gprmax_benchmark/prepare_native_gprmax_package.py",
        "--model-in",
        str(args.model_in),
        "--out",
        str(args.out),
        "--output-dir",
        str(args.output_dir),
        "--receiver",
        args.receiver,
        "--component",
        args.component,
        "--scenario-id",
        args.scenario_id,
    ]
    if args.ground_truth:
        parts.extend(["--ground-truth", str(args.ground_truth)])
    if args.copy_raw_out:
        parts.append("--copy-raw-out")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-in", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receiver", default="rx1")
    parser.add_argument("--component", default="Ez")
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--ground-truth")
    parser.add_argument("--copy-raw-out", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_native_package(
        model_in=args.model_in,
        out_path=args.out,
        output_dir=args.output_dir,
        receiver=args.receiver,
        component=args.component,
        scenario_id=args.scenario_id,
        ground_truth=args.ground_truth,
        copy_raw_out=args.copy_raw_out,
        command=_build_command(args),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "native_gprmax_converted" else 2


if __name__ == "__main__":
    raise SystemExit(main())

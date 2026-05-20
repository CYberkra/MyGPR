#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit physical/procedural validity of a MyGPR gprMax benchmark package."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PACKAGE = ROOT / "sample_data" / "gprmax_benchmarks" / "cylinder_single_v1"
PML_CELLS_DEFAULT = 10


def audit_gprmax_package(path: str | Path) -> dict[str, Any]:
    """Audit one benchmark package for native gprMax provenance and geometry risks."""
    package_dir = Path(path).expanduser().resolve()
    if package_dir.is_file():
        package_dir = package_dir.parent
    if not package_dir.exists():
        raise FileNotFoundError(f"gprMax benchmark package not found: {package_dir}")

    scenario_path = package_dir / "scenario.json"
    scenario = _read_json(scenario_path)
    model_path = _find_model_in(package_dir, scenario)
    model = _parse_gprmax_input(model_path) if model_path else {}
    csv_path = package_dir / "mygpr_bscan.csv"
    csv_shape = _read_csv_shape(csv_path)
    raw_out_files = sorted(str(path.name) for path in package_dir.glob("*.out"))
    ground_truth_path = package_dir / "ground_truth.json"
    ground_truth = _read_json(ground_truth_path)

    source = scenario.get("source") if isinstance(scenario.get("source"), dict) else {}
    source_kind = str(source.get("kind") or "unknown")
    simulation = scenario.get("simulation") if isinstance(scenario.get("simulation"), dict) else {}
    domain = _float_list(scenario.get("domain_m") or model.get("domain_m"))
    dx_dy_dz = _float_list(scenario.get("dx_dy_dz_m") or model.get("dx_dy_dz_m"))
    antenna = scenario.get("antenna") if isinstance(scenario.get("antenna"), dict) else {}

    expected_sample_count = _int_or_none(simulation.get("sample_count"))
    expected_trace_count = _int_or_none(simulation.get("trace_count"))
    actual_sample_count = csv_shape[0] if csv_shape else None
    actual_trace_count = csv_shape[1] if csv_shape else None
    time_window_s = _float_or_none(model.get("time_window_s"))
    if time_window_s is None and simulation.get("total_time_ns") is not None:
        time_window_s = float(simulation["total_time_ns"]) * 1e-9
    time_step_s = _float_or_none(simulation.get("time_step_s"))
    dt_source = "gprMax .out" if raw_out_files else "synthetic fallback/scenario metadata"
    dt_consistency = _dt_consistency(time_step_s, time_window_s, expected_sample_count)

    source_scan = _scan_range(
        antenna.get("source_position_m") or model.get("source_position_m"),
        antenna.get("source_step_m") or model.get("source_step_m"),
        expected_trace_count,
    )
    receiver_scan = _scan_range(
        antenna.get("receiver_position_m") or model.get("receiver_position_m"),
        antenna.get("receiver_step_m") or model.get("receiver_step_m"),
        expected_trace_count,
    )
    pml = _pml_margin_audit(
        domain_m=domain,
        dx_dy_dz_m=dx_dy_dz,
        source_scan=source_scan,
        receiver_scan=receiver_scan,
        pml_cells=PML_CELLS_DEFAULT,
    )
    roi_audit = _ground_truth_roi_audit(ground_truth, (actual_sample_count, actual_trace_count))
    warnings = []
    errors = []
    if source_kind == "synthetic_reference":
        warnings.append(
            "source.kind is synthetic_reference; this package is suitable for smoke/contract tests only."
        )
    if not raw_out_files:
        warnings.append("No native gprMax .out file found in the package.")
    if pml["risk_flags"]:
        warnings.extend(pml["risk_flags"])
    if roi_audit["warnings"]:
        warnings.extend(roi_audit["warnings"])
    if not scenario_path.exists():
        errors.append("missing scenario.json")
    if not model_path:
        warnings.append("missing model.in or scenario model_file reference")
    if ground_truth_path.exists() and not roi_audit["roi_inside_bscan"]:
        errors.append("ground_truth ROI is missing or outside the available B-scan shape")

    native_gprmax_verified = bool(raw_out_files) and source_kind not in {"synthetic_reference", "unknown"}
    paper_usable = native_gprmax_verified and not errors and not pml["risk_flags"]
    return {
        "schema": "mygpr_gprmax_package_audit_v1",
        "package_dir": str(package_dir),
        "source": {
            "kind": source_kind,
            "native_gprmax_verified": native_gprmax_verified,
            "dt_source": dt_source,
        },
        "files": {
            "scenario_json": str(scenario_path) if scenario_path.exists() else None,
            "model_in": str(model_path) if model_path else None,
            "raw_out_files": raw_out_files,
            "raw_out_exists": bool(raw_out_files),
            "mygpr_bscan_csv": str(csv_path) if csv_path.exists() else None,
            "ground_truth_json": str(ground_truth_path) if ground_truth_path.exists() else None,
        },
        "geometry": {
            "domain_m": domain,
            "dx_dy_dz_m": dx_dy_dz,
            "time_window_s": time_window_s,
            "time_window_ns": time_window_s * 1e9 if time_window_s is not None else None,
            "time_step_s": time_step_s,
            "dt_consistency": dt_consistency,
            "source_scan": source_scan,
            "receiver_scan": receiver_scan,
            "pml_margin": pml,
        },
        "shape": {
            "expected_sample_count": expected_sample_count,
            "expected_trace_count": expected_trace_count,
            "actual_sample_count": actual_sample_count,
            "actual_trace_count": actual_trace_count,
            "shape_matches_expected": (
                expected_sample_count == actual_sample_count
                and expected_trace_count == actual_trace_count
            ),
        },
        "ground_truth": roi_audit,
        "warnings": warnings,
        "errors": errors,
        "paper_usable": paper_usable,
        "recommendation": _recommendation(source_kind, native_gprmax_verified, paper_usable, errors),
    }


def write_markdown_report(audit: dict[str, Any], path: str | Path) -> Path:
    """Write a concise Markdown report for the audit payload."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GX-001 gprMax Simulation Validity Audit",
        "",
        "## Summary",
        f"- Package: `{audit['package_dir']}`",
        f"- Source kind: `{audit['source']['kind']}`",
        f"- Native gprMax .out verified: `{audit['source']['native_gprmax_verified']}`",
        f"- Paper usable as native gprMax evidence: `{audit['paper_usable']}`",
        f"- Recommendation: {audit['recommendation']}",
        "",
        "## Files",
        f"- model.in: `{audit['files']['model_in']}`",
        f"- raw .out exists: `{audit['files']['raw_out_exists']}`",
        f"- raw .out files: `{audit['files']['raw_out_files']}`",
        f"- MyGPR CSV: `{audit['files']['mygpr_bscan_csv']}`",
        f"- ground_truth.json: `{audit['files']['ground_truth_json']}`",
        "",
        "## Geometry And Timing",
        f"- domain_m: `{audit['geometry']['domain_m']}`",
        f"- dx_dy_dz_m: `{audit['geometry']['dx_dy_dz_m']}`",
        f"- time_window_ns: `{audit['geometry']['time_window_ns']}`",
        f"- time_step_s: `{audit['geometry']['time_step_s']}`",
        f"- dt source: `{audit['source']['dt_source']}`",
        f"- dt consistency: `{audit['geometry']['dt_consistency']}`",
        f"- source scan: `{audit['geometry']['source_scan']}`",
        f"- receiver scan: `{audit['geometry']['receiver_scan']}`",
        "",
        "## PML / Boundary Risk",
        f"- default PML cells assumed: `{audit['geometry']['pml_margin']['pml_cells']}`",
        f"- margin_m: `{audit['geometry']['pml_margin']['pml_margin_m']}`",
        f"- risk flags: `{audit['geometry']['pml_margin']['risk_flags']}`",
        "",
        "## Shape And Ground Truth ROI",
        f"- expected shape: `{audit['shape']['expected_sample_count']} x {audit['shape']['expected_trace_count']}`",
        f"- actual shape: `{audit['shape']['actual_sample_count']} x {audit['shape']['actual_trace_count']}`",
        f"- shape matches expected: `{audit['shape']['shape_matches_expected']}`",
        f"- ROI exists: `{audit['ground_truth']['roi_exists']}`",
        f"- ROI inside B-scan: `{audit['ground_truth']['roi_inside_bscan']}`",
        f"- ROI warnings: `{audit['ground_truth']['warnings']}`",
        "",
        "## Warnings",
    ]
    lines.extend(f"- {item}" for item in audit.get("warnings", []) or ["none"])
    lines.extend(["", "## Errors"])
    lines.extend(f"- {item}" for item in audit.get("errors", []) or ["none"])
    lines.extend(
        [
            "",
            "## Audit Conclusion",
            "The current `cylinder_single_v1` fixture can be used for deterministic smoke,",
            "contract, and stepwise-evidence plumbing tests. It must not be cited as a",
            "native gprMax forward-model result until a raw `.out` package, verified",
            "`dt`, domain/PML-safe scan geometry, and source-side `ground_truth.yaml` are",
            "present and audited.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_model_in(package_dir: Path, scenario: dict[str, Any]) -> Path | None:
    for key in ("model_file", "input_file"):
        value = scenario.get(key)
        if isinstance(value, str):
            candidate = (package_dir / value).resolve()
            if candidate.exists():
                return candidate
    models = sorted(package_dir.glob("*.in"))
    return models[0].resolve() if models else None


def _parse_gprmax_input(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("#"):
            continue
        if ":" not in line:
            continue
        command, rest = line[1:].split(":", 1)
        values = rest.split()
        if command == "domain":
            parsed["domain_m"] = [float(item) for item in values[:3]]
        elif command == "dx_dy_dz":
            parsed["dx_dy_dz_m"] = [float(item) for item in values[:3]]
        elif command == "time_window" and values:
            parsed["time_window_s"] = float(values[0])
        elif command == "hertzian_dipole" and len(values) >= 5:
            parsed["source_position_m"] = [float(values[1]), float(values[2]), float(values[3])]
        elif command == "rx" and len(values) >= 3:
            parsed["receiver_position_m"] = [float(values[0]), float(values[1]), float(values[2])]
        elif command == "src_steps" and len(values) >= 3:
            parsed["source_step_m"] = [float(values[0]), float(values[1]), float(values[2])]
        elif command == "rx_steps" and len(values) >= 3:
            parsed["receiver_step_m"] = [float(values[0]), float(values[1]), float(values[2])]
    return parsed


def _read_csv_shape(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        return (int(data.shape[0]), 1)
    return (int(data.shape[0]), int(data.shape[1]))


def _float_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return []


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scan_range(position: Any, step: Any, trace_count: int | None) -> dict[str, Any]:
    pos = _float_list(position)
    delta = _float_list(step)
    if len(pos) != 3:
        return {"available": False}
    if len(delta) != 3:
        delta = [0.0, 0.0, 0.0]
    count = max(int(trace_count or 1), 1)
    end = [pos[i] + delta[i] * (count - 1) for i in range(3)]
    return {
        "available": True,
        "start_m": pos,
        "end_m": end,
        "step_m": delta,
        "trace_count": count,
    }


def _pml_margin_audit(
    *,
    domain_m: list[float],
    dx_dy_dz_m: list[float],
    source_scan: dict[str, Any],
    receiver_scan: dict[str, Any],
    pml_cells: int,
) -> dict[str, Any]:
    if len(domain_m) != 3 or len(dx_dy_dz_m) != 3:
        return {
            "pml_cells": pml_cells,
            "pml_margin_m": [],
            "risk_flags": ["domain or dx_dy_dz unavailable; cannot assess PML margin"],
        }
    margins = [pml_cells * dx for dx in dx_dy_dz_m]
    risk_flags: list[str] = []
    skipped_axes: list[int] = []
    for axis, domain_value in enumerate(domain_m):
        if domain_value <= 2.0 * margins[axis]:
            skipped_axes.append(axis)
            risk_flags.append(
                f"axis {axis} domain {domain_value:.6g} m is thinner than two default PML margins; treating it as a thin/2D dimension for margin screening"
            )
    for label, scan in (("source", source_scan), ("receiver", receiver_scan)):
        if not scan.get("available"):
            risk_flags.append(f"{label} scan unavailable; cannot assess PML margin")
            continue
        for point_label in ("start_m", "end_m"):
            point = scan[point_label]
            for axis, value in enumerate(point):
                if axis in skipped_axes:
                    continue
                lower = margins[axis]
                upper = domain_m[axis] - margins[axis]
                if value <= lower:
                    risk_flags.append(
                        f"{label} {point_label} axis {axis} at {value:.6g} m is inside/near lower PML margin {lower:.6g} m"
                    )
                if value >= upper:
                    risk_flags.append(
                        f"{label} {point_label} axis {axis} at {value:.6g} m is inside/near upper PML margin {upper:.6g} m"
                    )
    return {"pml_cells": pml_cells, "pml_margin_m": margins, "risk_flags": risk_flags}


def _ground_truth_roi_audit(
    ground_truth: dict[str, Any],
    shape: tuple[int | None, int | None] | None,
) -> dict[str, Any]:
    targets = ground_truth.get("targets") if isinstance(ground_truth.get("targets"), list) else []
    roi = None
    if targets and isinstance(targets[0], dict):
        roi = targets[0].get("roi")
    warnings: list[str] = []
    if not isinstance(roi, dict):
        return {"roi_exists": False, "roi_inside_bscan": False, "roi": None, "warnings": ["missing target ROI"]}
    sample_count, trace_count = shape or (None, None)
    required = ("time_start_idx", "time_end_idx", "dist_start_idx", "dist_end_idx")
    if any(key not in roi for key in required):
        return {"roi_exists": True, "roi_inside_bscan": False, "roi": roi, "warnings": ["ROI missing required bounds"]}
    t0 = int(roi["time_start_idx"])
    t1 = int(roi["time_end_idx"])
    x0 = int(roi["dist_start_idx"])
    x1 = int(roi["dist_end_idx"])
    inside = True
    if sample_count is not None and not (0 <= t0 < t1 <= sample_count):
        inside = False
        warnings.append("target time ROI outside B-scan sample range")
    if trace_count is not None and not (0 <= x0 < x1 <= trace_count):
        inside = False
        warnings.append("target trace ROI outside B-scan trace range")
    return {"roi_exists": True, "roi_inside_bscan": inside, "roi": roi, "warnings": warnings}


def _dt_consistency(
    time_step_s: float | None,
    time_window_s: float | None,
    expected_sample_count: int | None,
) -> dict[str, Any]:
    if time_step_s is None or time_window_s is None or expected_sample_count is None:
        return {"available": False}
    implied = time_window_s / expected_sample_count if expected_sample_count > 0 else None
    mismatch = abs(time_step_s - implied) if implied is not None else None
    return {
        "available": True,
        "scenario_time_step_s": time_step_s,
        "time_window_over_samples_s": implied,
        "absolute_mismatch_s": mismatch,
        "relative_mismatch": mismatch / implied if implied else None,
    }


def _recommendation(
    source_kind: str,
    native_gprmax_verified: bool,
    paper_usable: bool,
    errors: list[str],
) -> str:
    if paper_usable:
        return "usable as audited native gprMax evidence for limited claims"
    if source_kind == "synthetic_reference":
        return "use for smoke/contract tests only; regenerate native gprMax package before paper claims"
    if errors:
        return "not paper-usable until audit errors are fixed"
    if not native_gprmax_verified:
        return "native gprMax provenance not verified"
    return "requires geometry/PML risk mitigation before paper claims"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE), help="Benchmark package directory")
    parser.add_argument("--output-json", help="Optional path for audit JSON")
    parser.add_argument("--output-md", help="Optional path for Markdown report")
    args = parser.parse_args(argv)

    audit = audit_gprmax_package(args.package)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown_report(audit, args.output_md)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if not audit["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit gprMax output folders against the MyGPR dataset contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gprmax_dataset_contract import MANIFEST_CANDIDATES
from core.gprmax_ground_truth import load_gprmax_ground_truth


SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def audit_gprmax_path(path: str | Path) -> dict[str, Any]:
    """Audit one gprMax dataset directory or recursively audit a root folder."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"gprMax path not found: {target}")
    if target.is_file():
        if target.name.lower() == "ground_truth.yaml":
            return audit_dataset_dir(target.parent)
        if target.suffix.lower() == ".json":
            return audit_dataset_dir(target.parent)
        raise ValueError(f"Unsupported gprMax audit file: {target}")

    if (target / "ground_truth.yaml").exists():
        return audit_dataset_dir(target)

    datasets = []
    for sidecar in _iter_ground_truth_files(target):
        datasets.append(audit_dataset_dir(sidecar.parent))
    return {
        "mode": "root",
        "root": str(target),
        "dataset_count": len(datasets),
        "ready_count": sum(1 for item in datasets if item.get("ready_for_mygpr_smoke")),
        "can_write_manifest_count": sum(1 for item in datasets if item.get("can_write_manifest")),
        "datasets": datasets,
    }


def audit_dataset_dir(dataset_dir: str | Path) -> dict[str, Any]:
    """Audit a single gprMax dataset folder."""
    root = Path(dataset_dir).expanduser().resolve()
    sidecar_path = root / "ground_truth.yaml"
    manifest_paths = _find_manifest_paths(root)
    errors: list[str] = []
    warnings: list[str] = []
    sidecar: dict[str, Any] = {}

    if not sidecar_path.exists():
        errors.append(f"missing ground_truth.yaml: {sidecar_path}")
    else:
        try:
            sidecar = load_gprmax_ground_truth(str(sidecar_path))
        except Exception as exc:  # pragma: no cover - defensive CLI surface
            errors.append(f"failed to parse ground_truth.yaml: {exc}")

    scenario_id = _scenario_id(sidecar, root)
    output_file = _string_value(sidecar, "output_file", "primary_out_file")
    model_file = _string_value(sidecar, "model_file", "input_file")
    output_path = _resolve_optional(root, output_file)
    model_path = _resolve_optional(root, model_file)

    if not output_file:
        errors.append("ground_truth.yaml missing output_file")
    elif output_path is not None and not output_path.exists():
        errors.append(f"missing referenced output_file: {output_path}")
    if model_file and model_path is not None and not model_path.exists():
        warnings.append(f"missing referenced model_file: {model_path}")
    if len(manifest_paths) > 1:
        warnings.append(
            "multiple manifest candidates found: "
            + ", ".join(path.name for path in manifest_paths)
        )
    if not manifest_paths:
        warnings.append("missing MyGPR manifest JSON")

    has_output = output_path is not None and output_path.exists()
    has_ground_truth = sidecar_path.exists() and not any(
        item.startswith("failed to parse ground_truth.yaml") for item in errors
    )
    has_manifest = bool(manifest_paths)
    return {
        "mode": "dataset",
        "dataset_dir": str(root),
        "scenario_id": scenario_id,
        "ground_truth_file": str(sidecar_path) if sidecar_path.exists() else None,
        "manifest_files": [str(path) for path in manifest_paths],
        "output_file": str(output_path) if output_path is not None else None,
        "model_file": str(model_path) if model_path is not None else None,
        "has_ground_truth": has_ground_truth,
        "has_output": has_output,
        "has_manifest": has_manifest,
        "ready_for_mygpr_smoke": has_ground_truth and has_output and has_manifest,
        "can_write_manifest": has_ground_truth and has_output and not has_manifest,
        "errors": errors,
        "warnings": warnings,
    }


def write_mygpr_manifest(audit: dict[str, Any], *, manifest_name: str | None = None) -> Path:
    """Write a minimal MyGPR-compatible manifest for a complete audited dataset."""
    if audit.get("mode") != "dataset":
        raise ValueError("Manifest can only be written for a single dataset audit")
    if not audit.get("can_write_manifest"):
        raise ValueError("Dataset is not eligible for manifest writing")
    dataset_dir = Path(str(audit["dataset_dir"]))
    scenario_id = str(audit.get("scenario_id") or dataset_dir.name)
    output_path = Path(str(audit["output_file"]))
    sidecar_path = Path(str(audit["ground_truth_file"]))
    path = dataset_dir / (manifest_name or f"{scenario_id}_manifest.json")
    payload = {
        "schema": "gprmax_dataset_manifest_v1",
        "scenario_id": scenario_id,
        "primary_out_file": _relative_or_name(output_path, dataset_dir),
        "ground_truth_file": _relative_or_name(sidecar_path, dataset_dir),
    }
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        payload["metadata_file"] = "metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _iter_ground_truth_files(root: Path) -> list[Path]:
    files: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in SKIP_DIR_NAMES:
                    stack.append(entry)
            elif entry.name.lower() == "ground_truth.yaml":
                files.append(entry)
    return sorted(files)


def _find_manifest_paths(dataset_dir: Path) -> list[Path]:
    matches: list[Path] = []
    for pattern in MANIFEST_CANDIDATES:
        matches.extend(sorted(dataset_dir.glob(pattern)))
    return list(dict.fromkeys(matches))


def _scenario_id(sidecar: dict[str, Any], dataset_dir: Path) -> str:
    value = sidecar.get("dataset_id") or sidecar.get("scenario_id")
    return str(value or dataset_dir.name)


def _string_value(source: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_optional(dataset_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path.expanduser().resolve() if path.is_absolute() else (dataset_dir / path).resolve()


def _relative_or_name(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _print_human_summary(audit: dict[str, Any]) -> None:
    if audit.get("mode") == "root":
        print(f"Root: {audit.get('root')}")
        print(f"Datasets with ground_truth.yaml: {audit.get('dataset_count')}")
        print(f"Ready for MyGPR smoke: {audit.get('ready_count')}")
        print(f"Can write manifest: {audit.get('can_write_manifest_count')}")
        for item in audit.get("datasets") or []:
            status = "READY" if item.get("ready_for_mygpr_smoke") else "NOT READY"
            print(f"- {status}: {item.get('scenario_id')} ({item.get('dataset_dir')})")
            for error in item.get("errors") or []:
                print(f"  error: {error}")
            for warning in item.get("warnings") or []:
                print(f"  warning: {warning}")
        return

    status = "READY" if audit.get("ready_for_mygpr_smoke") else "NOT READY"
    print(f"Dataset: {audit.get('dataset_dir')}")
    print(f"Scenario: {audit.get('scenario_id')}")
    print(f"Status: {status}")
    print(f"ground_truth.yaml: {audit.get('ground_truth_file')}")
    print(f"output_file: {audit.get('output_file')}")
    print(f"manifest: {', '.join(audit.get('manifest_files') or []) or None}")
    for error in audit.get("errors") or []:
        print(f"error: {error}")
    for warning in audit.get("warnings") or []:
        print(f"warning: {warning}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit gprMax output folders against MyGPR's dataset contract.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="gprMax dataset directory, ground_truth.yaml, manifest, or root folder.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the audit JSON summary.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write a minimal MyGPR manifest when the dataset has .out + ground_truth but no manifest.",
    )
    parser.add_argument(
        "--manifest-name",
        help="Manifest filename to use with --write-manifest.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when no dataset is ready for MyGPR smoke.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    audit = audit_gprmax_path(args.path)
    written_manifest: Path | None = None
    if args.write_manifest:
        written_manifest = write_mygpr_manifest(audit, manifest_name=args.manifest_name)
        audit["written_manifest"] = str(written_manifest)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_human_summary(audit)
    if written_manifest:
        print(f"written_manifest: {written_manifest}")

    if args.strict:
        if audit.get("mode") == "root":
            return 0 if int(audit.get("ready_count") or 0) > 0 else 2
        return 0 if audit.get("ready_for_mygpr_smoke") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

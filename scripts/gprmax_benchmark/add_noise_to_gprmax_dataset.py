#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a deterministic noisy gprMax validation dataset package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gprmax_dataset_contract import load_gprmax_dataset_contract


DEFAULT_OUTPUT_ROOT = ROOT / "output" / "gprmax_datasets"
DEFAULT_RECEIVER = "rx1"
DEFAULT_COMPONENT = "Ez"


def create_noisy_dataset(
    source: str | Path,
    output_root: str | Path,
    *,
    output_name: str,
    target_snr_db: float,
    seed: int = 20260517,
    receiver: str = DEFAULT_RECEIVER,
    component: str = DEFAULT_COMPONENT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy a gprMax package and inject deterministic additive Gaussian noise."""
    package = load_gprmax_dataset_contract(source)
    output_dir = Path(output_root).expanduser().resolve() / output_name
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    output_file = output_dir / f"{output_name}_merged.out"
    shutil.copy2(package.primary_out_file, output_file)

    dataset_path = f"/rxs/{receiver}/{component}"
    noise_summary = _inject_noise(
        output_file,
        dataset_path=dataset_path,
        target_snr_db=float(target_snr_db),
        seed=int(seed),
    )

    input_file = _copy_optional_model_file(package, output_dir, output_name)
    ground_truth_file = _write_ground_truth(package, output_dir, output_name, output_file, input_file)
    metadata_file = _write_metadata(
        package,
        output_dir,
        output_name,
        output_file,
        input_file,
        ground_truth_file,
        noise_summary,
    )
    manifest_file = _write_manifest(
        output_dir,
        output_name,
        output_file,
        input_file,
        ground_truth_file,
        metadata_file,
        package,
        noise_summary,
    )
    readme_file = _write_readme(
        output_dir,
        output_name,
        package,
        output_file,
        ground_truth_file,
        metadata_file,
        manifest_file,
        noise_summary,
    )

    summary = {
        "schema": "mygpr_noisy_gprmax_dataset_generation_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario_id": output_name,
        "source_dataset": str(package.dataset_dir),
        "source_manifest": str(package.manifest_path),
        "output_dir": str(output_dir),
        "primary_out_file": str(output_file),
        "manifest_file": str(manifest_file),
        "ground_truth_file": str(ground_truth_file),
        "metadata_file": str(metadata_file),
        "readme_file": str(readme_file),
        "noise": noise_summary,
    }
    summary_file = output_dir / "noise_generation_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["generation_summary_file"] = str(summary_file)
    return summary


def _inject_noise(
    out_file: Path,
    *,
    dataset_path: str,
    target_snr_db: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    with h5py.File(out_file, "r+") as handle:
        if dataset_path not in handle:
            raise KeyError(f"HDF5 dataset not found: {dataset_path}")
        dataset = handle[dataset_path]
        clean = np.asarray(dataset[()], dtype=np.float32)
        clean_rms = float(np.sqrt(np.mean(np.square(clean, dtype=np.float64))))
        target_linear = float(10.0 ** (target_snr_db / 20.0))
        noise_std = clean_rms / max(target_linear, 1e-12)
        raw_noise = rng.normal(loc=0.0, scale=1.0, size=clean.shape).astype(np.float32)
        raw_noise_rms = float(np.sqrt(np.mean(np.square(raw_noise, dtype=np.float64))))
        noise = raw_noise * (noise_std / max(raw_noise_rms, 1e-30))
        noisy = clean + noise
        dataset[...] = noisy.astype(dataset.dtype, copy=False)
        actual_noise = noisy.astype(np.float64) - clean.astype(np.float64)
        actual_noise_rms = float(np.sqrt(np.mean(np.square(actual_noise))))
        actual_snr_db = float(20.0 * np.log10(max(clean_rms, 1e-30) / max(actual_noise_rms, 1e-30)))
        handle.attrs["MyGPRNoiseAugmented"] = True
        handle.attrs["MyGPRNoiseSeed"] = int(seed)
        handle.attrs["MyGPRNoiseTargetSnrDb"] = float(target_snr_db)
        handle.attrs["MyGPRNoiseActualSnrDb"] = float(actual_snr_db)

    return {
        "type": "additive_gaussian",
        "dataset_path": dataset_path,
        "seed": int(seed),
        "target_snr_db": float(target_snr_db),
        "actual_snr_db": actual_snr_db,
        "signal_rms": clean_rms,
        "noise_rms": actual_noise_rms,
        "noise_std": noise_std,
    }


def _copy_optional_model_file(package: Any, output_dir: Path, output_name: str) -> Path | None:
    raw_sidecar = package.ground_truth_raw or {}
    model_file = raw_sidecar.get("model_file") or raw_sidecar.get("input_file")
    if not isinstance(model_file, str) or not model_file.strip():
        return None
    source_model = Path(model_file)
    if not source_model.is_absolute():
        source_model = package.dataset_dir / source_model
    if not source_model.exists():
        return None
    output_model = output_dir / f"{output_name}.in"
    shutil.copy2(source_model, output_model)
    return output_model


def _write_ground_truth(
    package: Any,
    output_dir: Path,
    output_name: str,
    output_file: Path,
    input_file: Path | None,
) -> Path:
    sidecar = dict(package.ground_truth_raw or {})
    sidecar["dataset_id"] = output_name
    sidecar["scenario_id"] = output_name
    sidecar["output_file"] = output_file.name
    if input_file is not None:
        sidecar["model_file"] = input_file.name
    metadata = dict(sidecar.get("metadata") or {})
    metadata["noise_augmented"] = True
    metadata["noise_source_dataset"] = package.scenario_id
    sidecar["metadata"] = metadata
    path = output_dir / "ground_truth.yaml"
    path.write_text(
        yaml.safe_dump(sidecar, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_metadata(
    package: Any,
    output_dir: Path,
    output_name: str,
    output_file: Path,
    input_file: Path | None,
    ground_truth_file: Path,
    noise_summary: dict[str, Any],
) -> Path:
    source_metadata = dict(package.metadata or {})
    source_metadata["scenario_id"] = output_name
    source_metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    source_metadata["source_dataset"] = str(package.dataset_dir)
    source_metadata["primary_out_file"] = str(output_file)
    source_metadata["ground_truth_file"] = str(ground_truth_file)
    if input_file is not None:
        source_metadata["input_file"] = str(input_file)
    source_metadata["noise_augmentation"] = noise_summary
    path = output_dir / f"{output_name}_metadata.json"
    path.write_text(json.dumps(source_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_manifest(
    output_dir: Path,
    output_name: str,
    output_file: Path,
    input_file: Path | None,
    ground_truth_file: Path,
    metadata_file: Path,
    package: Any,
    noise_summary: dict[str, Any],
) -> Path:
    manifest = {
        "schema": "gprmax_dataset_manifest_v1",
        "scenario_id": output_name,
        "primary_out_file": output_file.name,
        "merged_out_file": output_file.name,
        "ground_truth_file": ground_truth_file.name,
        "metadata_file": metadata_file.name,
        "source_dataset": str(package.dataset_dir),
        "source_manifest": str(package.manifest_path),
        "transform": {
            "type": "noise_augmentation",
            "noise": noise_summary,
            "truth_usage": "validation_only",
        },
        "paths_relative_to_output_dir": {
            "primary_out_file": output_file.name,
            "merged_out_file": output_file.name,
            "ground_truth_file": ground_truth_file.name,
            "metadata_file": metadata_file.name,
        },
    }
    if input_file is not None:
        manifest["input_file"] = input_file.name
        manifest["paths_relative_to_output_dir"]["input_file"] = input_file.name
    path = output_dir / f"{output_name}_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_readme(
    output_dir: Path,
    output_name: str,
    package: Any,
    output_file: Path,
    ground_truth_file: Path,
    metadata_file: Path,
    manifest_file: Path,
    noise_summary: dict[str, Any],
) -> Path:
    text = "\n".join(
        [
            f"# {output_name}",
            "",
            "Deterministic low-SNR gprMax validation dataset generated by MyGPR.",
            "",
            "## Source",
            f"- source_dataset: `{package.dataset_dir}`",
            f"- source_scenario_id: `{package.scenario_id}`",
            "",
            "## Noise",
            f"- type: `{noise_summary['type']}`",
            f"- target_snr_db: `{noise_summary['target_snr_db']:.3f}`",
            f"- actual_snr_db: `{noise_summary['actual_snr_db']:.3f}`",
            f"- seed: `{noise_summary['seed']}`",
            f"- dataset_path: `{noise_summary['dataset_path']}`",
            "",
            "## Files",
            f"- primary_out: `{output_file.name}`",
            f"- manifest: `{manifest_file.name}`",
            f"- ground_truth: `{ground_truth_file.name}`",
            f"- metadata: `{metadata_file.name}`",
            "",
            "Ground truth remains validation-only and is not used by AutoTune search.",
            "",
        ]
    )
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic noisy copy of a gprMax validation dataset.",
    )
    parser.add_argument("--source", required=True, help="Source dataset directory or manifest.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--target-snr-db", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--receiver", default=DEFAULT_RECEIVER)
    parser.add_argument("--component", default=DEFAULT_COMPONENT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = create_noisy_dataset(
        args.source,
        args.output_root,
        output_name=args.output_name,
        target_snr_db=args.target_snr_db,
        seed=args.seed,
        receiver=args.receiver,
        component=args.component,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

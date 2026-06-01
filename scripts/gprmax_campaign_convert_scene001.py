#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert gprMax .out files to pairing-ready CSV/NPY."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.gpr_io import read_gprmax_out

DEFAULT_SCENE037 = (
    "experiments/gprmax/GX-008/models/"
    "scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate"
)


def _save_array(array: np.ndarray, out_dir: Path, stem: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{stem}.npy"
    csv_path = out_dir / f"{stem}.csv"
    np.save(npy_path, array.astype(np.float32, copy=False))
    np.savetxt(csv_path, array, delimiter=",", fmt="%.10g")
    return {
        "npy_path": str(npy_path),
        "csv_path": str(csv_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _convert_one(source_out: Path, out_dir: Path, stem: str) -> dict[str, Any]:
    payload = read_gprmax_out(str(source_out))
    bscan = np.asarray(payload["data"])
    if bscan.ndim != 2:
        raise ValueError(f"Expected 2D bscan from {source_out}, got shape={bscan.shape}")
    result = _save_array(bscan, out_dir, stem)
    result["source_out"] = str(source_out)
    return result


def _convert_series(
    source_base_out: Path,
    out_dir: Path,
    stem: str,
    run_count: int,
    component: str,
) -> dict[str, Any]:
    base = source_base_out.with_suffix("")
    collected: list[np.ndarray] = []
    sources: list[str] = []
    receiver_name = ""
    available_components: list[str] = []

    for idx in range(1, run_count + 1):
        out_path = base.with_name(f"{base.name}{idx}.out")
        if not out_path.exists():
            raise FileNotFoundError(f"expected run output missing: {out_path}")
        with h5py.File(out_path, "r") as f:
            if "rxs" not in f:
                raise ValueError(f"Missing rxs group in {out_path}")
            receiver_names = sorted(str(name) for name in f["rxs"].keys())
            if not receiver_names:
                raise ValueError(f"No receiver group under rxs in {out_path}")
            receiver_name = receiver_names[0]
            receiver_group = f["rxs"][receiver_name]
            available_components = sorted(str(name) for name in receiver_group.keys())
            if component not in receiver_group:
                raise ValueError(
                    f"Missing requested component '{component}' in {out_path}; "
                    f"receiver={receiver_name}; available={available_components}"
                )
            trace = np.asarray(receiver_group[component][:], dtype=np.float64)
        if trace.ndim != 1:
            raise ValueError(
                f"Expected 1D {component} trace in {out_path}, got shape={trace.shape}"
            )
        collected.append(trace)
        sources.append(str(out_path))

    bscan = np.column_stack(collected)
    result = _save_array(np.asarray(bscan, dtype=np.float64), out_dir, stem)
    result["source_out_series"] = sources
    result["run_count"] = int(run_count)
    result["selected_component"] = component
    result["receiver_name"] = receiver_name
    result["available_components"] = available_components
    result["component_source"] = f"rxs/{receiver_name}/{component}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert gprMax .out files to CSV/NPY.")
    parser.add_argument(
        "--raw-out",
        default=f"{DEFAULT_SCENE037}/raw_with_target.out",
    )
    parser.add_argument(
        "--background-out",
        default=f"{DEFAULT_SCENE037}/background_only.out",
    )
    parser.add_argument(
        "--raw-converted-dir",
        default="output/gprmax_converted/scene_037/raw_with_target",
    )
    parser.add_argument(
        "--background-converted-dir",
        default="output/gprmax_converted/scene_037/background_only",
    )
    parser.add_argument(
        "--raw-run-count",
        type=int,
        default=1,
        help="If >1, load raw run outputs as <stem>1..N.out and merge into 2D B-scan.",
    )
    parser.add_argument(
        "--background-run-count",
        type=int,
        default=1,
        help="If >1, load background run outputs as <stem>1..N.out and merge into 2D B-scan.",
    )
    parser.add_argument("--json", default="")
    parser.add_argument(
        "--component",
        default="Ez",
        help="Receiver field component for multi-run conversion (default: Ez).",
    )
    args = parser.parse_args()

    raw_out = Path(args.raw_out).expanduser().resolve()
    bg_out = Path(args.background_out).expanduser().resolve()
    if args.raw_run_count <= 1 and not raw_out.exists():
        raise FileNotFoundError(f"raw out file not found: {raw_out}")
    if args.background_run_count <= 1 and not bg_out.exists():
        raise FileNotFoundError(f"background out file not found: {bg_out}")

    if args.raw_run_count > 1:
        raw_result = _convert_series(
            raw_out,
            Path(args.raw_converted_dir).expanduser().resolve(),
            "raw_bscan",
            args.raw_run_count,
            args.component,
        )
    else:
        raw_result = _convert_one(
            raw_out, Path(args.raw_converted_dir).expanduser().resolve(), "raw_bscan"
        )

    if args.background_run_count > 1:
        bg_result = _convert_series(
            bg_out,
            Path(args.background_converted_dir).expanduser().resolve(),
            "background_bscan",
            args.background_run_count,
            args.component,
        )
    else:
        bg_result = _convert_one(
            bg_out, Path(args.background_converted_dir).expanduser().resolve(), "background_bscan"
        )

    summary = {
        "status": "success",
        "raw": raw_result,
        "background": bg_result,
        "shape_match": raw_result["shape"] == bg_result["shape"],
        "selected_component": args.component,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).expanduser().resolve().write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

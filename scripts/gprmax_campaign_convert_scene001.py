#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert GX-007 scene_001 gprMax .out files to pairing-ready CSV/NPY."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.gpr_io import read_gprmax_out


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert GX-007 scene_001 .out to CSV/NPY.")
    parser.add_argument(
        "--raw-out",
        default="experiments/gprmax/GX-007/models/scene_001_single_shallow_pipe/raw_with_target.out",
    )
    parser.add_argument(
        "--background-out",
        default="experiments/gprmax/GX-007/models/scene_001_single_shallow_pipe/background_only.out",
    )
    parser.add_argument(
        "--raw-converted-dir",
        default="D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/raw_with_target/converted",
    )
    parser.add_argument(
        "--background-converted-dir",
        default="D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/background_only/converted",
    )
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    raw_out = Path(args.raw_out).expanduser().resolve()
    bg_out = Path(args.background_out).expanduser().resolve()
    if not raw_out.exists():
        raise FileNotFoundError(f"raw out file not found: {raw_out}")
    if not bg_out.exists():
        raise FileNotFoundError(f"background out file not found: {bg_out}")

    raw_result = _convert_one(raw_out, Path(args.raw_converted_dir).expanduser().resolve(), "raw_bscan")
    bg_result = _convert_one(bg_out, Path(args.background_converted_dir).expanduser().resolve(), "background_bscan")

    summary = {
        "status": "success",
        "raw": raw_result,
        "background": bg_result,
        "shape_match": raw_result["shape"] == bg_result["shape"],
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

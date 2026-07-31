#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect available receiver components in gprMax .out (HDF5) files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py


def inspect_out_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"file": str(path), "receivers": {}, "status": "success"}
    if not path.exists():
        return {"file": str(path), "status": "missing", "error": "file_not_found"}
    try:
        with h5py.File(path, "r") as f:
            rxs = f.get("rxs")
            if rxs is None:
                return {"file": str(path), "status": "invalid", "error": "missing_rxs_group"}
            for rx_name in rxs.keys():
                rx_group = rxs[rx_name]
                comps: dict[str, Any] = {}
                for comp_name in rx_group.keys():
                    shape = list(rx_group[comp_name].shape)
                    comps[str(comp_name)] = {"shape": shape}
                result["receivers"][str(rx_name)] = {"components": comps}
    except Exception as exc:  # pragma: no cover
        return {"file": str(path), "status": "error", "error": str(exc)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect gprMax receiver components.")
    parser.add_argument("files", nargs="+", help="One or more gprMax .out files")
    parser.add_argument("--json", default="", help="Optional JSON output path")
    args = parser.parse_args()

    summary = {"status": "success", "files": [inspect_out_file(Path(p)) for p in args.files]}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        Path(args.json).expanduser().resolve().write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parser for gprMax ``.in`` acquisition/model configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def read_gprmax_in(in_path: str) -> Dict[str, Any]:
    """Parse gprMax .in configuration file.

    Extracts key parameters like domain size, dx, time window, etc.

    Args:
        in_path: Path to .in file

    Returns:
        dict: Configuration parameters
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f".in file not found: {in_path}")

    config = {
        "title": "",
        "domain": None,
        "dx_dy_dz": None,
        "time_window": None,
        "materials": [],
        "geometry_files": [],
        "waveform": None,
        "src_position": None,
        "rx_position": None,
        "src_steps": None,
        "rx_steps": None,
    }

    with open(in_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("#title:"):
                config["title"] = line.replace("#title:", "").strip()
            elif line.startswith("#domain:"):
                parts = line.replace("#domain:", "").strip().split()
                config["domain"] = [float(p) for p in parts]
            elif line.startswith("#dx_dy_dz:"):
                parts = line.replace("#dx_dy_dz:", "").strip().split()
                config["dx_dy_dz"] = [float(p) for p in parts]
            elif line.startswith("#time_window:"):
                config["time_window"] = float(line.replace("#time_window:", "").strip())
            elif line.startswith("#material:"):
                config["materials"].append(line)
            elif line.startswith("#geometry_objects_read:"):
                parts = line.replace("#geometry_objects_read:", "").strip().split()
                if len(parts) >= 5:
                    config["geometry_files"].append(parts[3])  # h5 file
                    config["geometry_files"].append(parts[4])  # materials file
            elif line.startswith("#waveform:"):
                config["waveform"] = line
            elif line.startswith("#hertzian_dipole:"):
                parts = line.replace("#hertzian_dipole:", "").strip().split()
                if len(parts) >= 5:
                    config["src_position"] = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
            elif line.startswith("#rx:"):
                parts = line.replace("#rx:", "").strip().split()
                if len(parts) >= 3:
                    config["rx_position"] = [
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                    ]
            elif line.startswith("#src_steps:"):
                parts = line.replace("#src_steps:", "").strip().split()
                config["src_steps"] = [float(p) for p in parts]
            elif line.startswith("#rx_steps:"):
                parts = line.replace("#rx_steps:", "").strip().split()
                config["rx_steps"] = [float(p) for p in parts]

    return config


__all__ = ["read_gprmax_in"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence helpers for synchronized sensor results."""
from __future__ import annotations

from pathlib import Path

from core.sensor_sync_models import SensorSyncResult


def save_sensor_sync_result(result: SensorSyncResult, directory: str | Path, *, basename: str = "sensor_sync") -> dict[str, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    trajectory_path = result.trajectory.to_csv(root / f"{basename}_trajectory.csv")
    manifest_path = root / f"{basename}_manifest.json"
    from core.storage_primitives import atomic_write_json
    atomic_write_json(manifest_path, result.to_manifest())
    metadata_path = root / f"{basename}_trace_metadata.npz"
    from core.array_storage import atomic_save_npz_compressed
    atomic_save_npz_compressed(metadata_path, result.trace_metadata)
    return {"trajectory": trajectory_path, "manifest": manifest_path, "trace_metadata": metadata_path}


__all__ = ["save_sensor_sync_result"]

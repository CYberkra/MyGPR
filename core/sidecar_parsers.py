#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal normalized RTK/IMU CSV sidecar parsers for motion Phase 1."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any

import numpy as np


def _load_sidecar_schema() -> dict[str, Any]:
    module = importlib.import_module("core.sidecar_models")
    return {
        "RTK_REQUIRED_FIELDS": module.RTK_REQUIRED_FIELDS,
        "RTK_OPTIONAL_FIELDS": module.RTK_OPTIONAL_FIELDS,
        "IMU_REQUIRED_FIELDS": module.IMU_REQUIRED_FIELDS,
        "IMU_OPTIONAL_FIELDS": module.IMU_OPTIONAL_FIELDS,
        "RTK_COLUMN_ALIASES": module.RTK_COLUMN_ALIASES,
        "IMU_COLUMN_ALIASES": module.IMU_COLUMN_ALIASES,
    }


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Sidecar file '{csv_path}' is missing a header row")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Sidecar file '{csv_path}' contains no data rows")
    return rows


def _resolve_column(rows: list[dict[str, str]], aliases: tuple[str, ...], *, required: bool) -> str | None:
    available = {key for key in rows[0].keys() if key is not None}
    for alias in aliases:
        if alias in available:
            return alias
    if required:
        raise ValueError(f"Missing required sidecar column from aliases {aliases}")
    return None


def _coerce_float_array(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(column, "")
        if raw in (None, ""):
            raise ValueError(f"Column '{column}' contains empty values")
        values.append(float(raw))
    return np.asarray(values, dtype=np.float64)


def _coerce_int_array(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(column, "")
        if raw in (None, ""):
            raise ValueError(f"Column '{column}' contains empty values")
        values.append(int(float(raw)))
    return np.asarray(values, dtype=np.int32)


def _optional_float_array(rows: list[dict[str, str]], column: str | None) -> np.ndarray | None:
    if column is None:
        return None
    return _coerce_float_array(rows, column).astype(np.float32)


def _optional_int_array(rows: list[dict[str, str]], column: str | None) -> np.ndarray | None:
    if column is None:
        return None
    return _coerce_int_array(rows, column)


def _required_column_name(resolved: dict[str, str | None], field: str) -> str:
    column = resolved[field]
    if column is None:
        raise ValueError(f"Missing required sidecar column '{field}'")
    return column


def _sort_by_timestamp(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_s = np.asarray(payload["timestamp_s"], dtype=np.float64)
    order = np.argsort(timestamp_s, kind="stable")
    sorted_payload: dict[str, Any] = {"source_kind": payload["source_kind"]}
    for key, value in payload.items():
        if key == "source_kind":
            continue
        if isinstance(value, np.ndarray):
            sorted_payload[key] = value[order]
        else:
            sorted_payload[key] = value
    return sorted_payload


def _parse_rtk_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    schema = _load_sidecar_schema()
    resolved = {
        field: _resolve_column(
            rows,
            schema["RTK_COLUMN_ALIASES"][field],
            required=field in schema["RTK_REQUIRED_FIELDS"],
        )
        for field in schema["RTK_REQUIRED_FIELDS"] + schema["RTK_OPTIONAL_FIELDS"]
    }
    payload: dict[str, Any] = {
        "source_kind": "rtk",
        "timestamp_s": _coerce_float_array(rows, _required_column_name(resolved, "timestamp_s")),
        "longitude": _coerce_float_array(rows, _required_column_name(resolved, "longitude")),
        "latitude": _coerce_float_array(rows, _required_column_name(resolved, "latitude")),
    }
    for field in ("ground_elevation_m", "flight_height_m", "hdop"):
        value = _optional_float_array(rows, resolved[field])
        if value is not None:
            payload[field] = value
    for field in ("rtk_fix_type", "satellites"):
        value = _optional_int_array(rows, resolved[field])
        if value is not None:
            payload[field] = value
    return _sort_by_timestamp(payload)


def _parse_imu_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    schema = _load_sidecar_schema()
    resolved = {
        field: _resolve_column(
            rows,
            schema["IMU_COLUMN_ALIASES"][field],
            required=field in schema["IMU_REQUIRED_FIELDS"],
        )
        for field in schema["IMU_REQUIRED_FIELDS"] + schema["IMU_OPTIONAL_FIELDS"]
    }
    payload: dict[str, Any] = {
        "source_kind": "imu",
        "timestamp_s": _coerce_float_array(rows, _required_column_name(resolved, "timestamp_s")),
        "roll_deg": _coerce_float_array(rows, _required_column_name(resolved, "roll_deg")).astype(np.float32),
        "pitch_deg": _coerce_float_array(rows, _required_column_name(resolved, "pitch_deg")).astype(np.float32),
        "yaw_deg": _coerce_float_array(rows, _required_column_name(resolved, "yaw_deg")).astype(np.float32),
    }
    for field in schema["IMU_OPTIONAL_FIELDS"]:
        value = _optional_float_array(rows, resolved[field])
        if value is not None:
            payload[field] = value
    return _sort_by_timestamp(payload)


def parse_sidecar_csv(path: str | Path, *, kind: str) -> dict[str, Any]:
    """Parse a small RTK/IMU CSV into a normalized phase-1 schema."""
    rows = _read_csv_rows(path)
    if kind == "rtk":
        return _parse_rtk_rows(rows)
    if kind == "imu":
        return _parse_imu_rows(rows)
    raise ValueError(f"Unsupported sidecar kind: {kind}")

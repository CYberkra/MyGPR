#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-context detection and default policy helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mygpr.domain.common.scalars import first_two_floats, to_int


DATA_CONTEXT_UAV_GPR_SFCW_FIELD = "uav_gpr_sfcw_field"
DATA_CONTEXT_GPRMAX_IMPULSE = "gprmax_impulse"
DATA_CONTEXT_GPRMAX = "gprmax"
DATA_CONTEXT_GENERIC_BSCAN = "generic_bscan"

FIELD_SFCW_BAND_MHZ = (20.0, 170.0)


def infer_data_context(
    header_info: dict[str, Any] | None = None,
    *,
    trace_metadata: dict[str, Any] | None = None,
    gprmax_config: dict[str, Any] | None = None,
    source_path: str | Path | None = None,
) -> str:
    """Infer the processing context from import metadata."""
    header = dict(header_info or {})
    explicit = str(header.get("data_context") or "").strip()
    if explicit:
        return explicit

    source = str(header.get("source") or header.get("source_format") or "").lower()
    suffix = Path(source_path).suffix.lower() if source_path else ""
    cfg = dict(gprmax_config or {})
    waveform = str(
        header.get("gprmax_waveform")
        or cfg.get("waveform")
        or header.get("waveform")
        or ""
    ).lower()
    if source == "gprmax_out" or suffix == ".out" or cfg:
        if "impulse" in waveform:
            return DATA_CONTEXT_GPRMAX_IMPULSE
        return DATA_CONTEXT_GPRMAX

    metadata = trace_metadata or {}
    if header.get("has_airborne_metadata") or {
        "longitude",
        "latitude",
        "flight_height_m",
    }.issubset(set(metadata.keys())):
        return DATA_CONTEXT_UAV_GPR_SFCW_FIELD

    samples = to_int(header.get("a_scan_length"), default=0)
    if samples == 501 and source in {"airborne_csv", "csv", ""}:
        return DATA_CONTEXT_UAV_GPR_SFCW_FIELD

    return DATA_CONTEXT_GENERIC_BSCAN


def apply_data_context_defaults(
    header_info: dict[str, Any] | None,
    *,
    trace_metadata: dict[str, Any] | None = None,
    gprmax_config: dict[str, Any] | None = None,
    source_path: str | Path | None = None,
    context: str | None = None,
) -> dict[str, Any] | None:
    """Attach stable data-context fields to an existing header mapping."""
    if header_info is None:
        return None
    header = dict(header_info)
    resolved = context or infer_data_context(
        header,
        trace_metadata=trace_metadata,
        gprmax_config=gprmax_config,
        source_path=source_path,
    )
    header["data_context"] = resolved

    if resolved == DATA_CONTEXT_UAV_GPR_SFCW_FIELD:
        header.setdefault("instrument_type", "UAV-GPR SFCW")
        header.setdefault("sweep_start_mhz", FIELD_SFCW_BAND_MHZ[0])
        header.setdefault("sweep_stop_mhz", FIELD_SFCW_BAND_MHZ[1])
        header.setdefault("frequency_points", 501)
        header.setdefault("frequency_filter_band_mhz", list(FIELD_SFCW_BAND_MHZ))
        header.setdefault("frequency_filter_policy", "instrument_sweep_band")
    elif resolved in {DATA_CONTEXT_GPRMAX, DATA_CONTEXT_GPRMAX_IMPULSE}:
        header.setdefault("source", "gprmax_out")
        header.setdefault("frequency_filter_policy", "model_or_auto_tune_only")
        if gprmax_config:
            waveform = gprmax_config.get("waveform")
            if waveform:
                header.setdefault("gprmax_waveform", waveform)
            if gprmax_config.get("time_window") is not None:
                header.setdefault("gprmax_time_window_s", gprmax_config["time_window"])
            if gprmax_config.get("dx_dy_dz") is not None:
                header.setdefault("gprmax_dx_dy_dz_m", list(gprmax_config["dx_dy_dz"]))
    else:
        header.setdefault("frequency_filter_policy", "manual_or_auto_tune")

    return header


def frequency_band_from_context(
    header_info: dict[str, Any] | None,
) -> tuple[float, float] | None:
    """Return a fixed passband only when the data context justifies it."""
    header = dict(header_info or {})
    band = header.get("frequency_filter_band_mhz")
    parsed_band = first_two_floats(band)
    if parsed_band is not None:
        low, high = parsed_band
        if high > low >= 0.0:
            return low, high
    if infer_data_context(header) == DATA_CONTEXT_UAV_GPR_SFCW_FIELD:
        return FIELD_SFCW_BAND_MHZ
    return None

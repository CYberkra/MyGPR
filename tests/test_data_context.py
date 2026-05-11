#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for data-context defaults."""

from __future__ import annotations

from core.data_context import (
    apply_data_context_defaults,
    frequency_band_from_context,
    infer_data_context,
    recommended_profile_for_header,
)


def test_airborne_sfcw_context_has_fixed_instrument_band():
    header = apply_data_context_defaults(
        {
            "source": "airborne_csv",
            "a_scan_length": 501,
            "num_traces": 10,
            "has_airborne_metadata": True,
        },
        trace_metadata={
            "longitude": [100.0],
            "latitude": [30.0],
            "flight_height_m": [5.0],
        },
    )

    assert header is not None
    assert infer_data_context(header) == "uav_gpr_sfcw_field"
    assert frequency_band_from_context(header) == (20.0, 170.0)
    assert recommended_profile_for_header(header) == "high_quality_uav_gpr"


def test_gprmax_impulse_context_does_not_inherit_field_frequency_band():
    header = apply_data_context_defaults(
        {"source": "gprmax_out", "a_scan_length": 1559, "num_traces": 121},
        gprmax_config={"waveform": "#waveform: impulse 1 1.0 my_impulse"},
    )

    assert header is not None
    assert infer_data_context(header) == "gprmax_impulse"
    assert frequency_band_from_context(header) is None
    assert recommended_profile_for_header(header) == "gprmax_impulse_validation"
    assert header["frequency_filter_policy"] == "model_or_auto_tune_only"

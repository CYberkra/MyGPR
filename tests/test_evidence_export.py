#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence export regression tests for standard processing chains."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.evidence_export import (
    STANDARD_CHAIN_SPECS,
    export_chain_evidence,
    export_standard_chain_for_sample,
)


def test_export_chain_evidence_writes_summary_and_images(tmp_path: Path):
    raw = np.arange(48, dtype=np.float32).reshape(12, 4)
    summary = export_chain_evidence(
        data=raw,
        header_info={
            "a_scan_length": 12,
            "num_traces": 4,
            "total_time_ns": 120.0,
            "trace_interval_m": 0.5,
        },
        bundle_name="demo-bundle",
        chain_name="演示链",
        chain_description="dewow only",
        steps=[("dewow", {"window": 5})],
        out_dir=tmp_path,
        title_prefix="Demo",
        save_images=True,
    )

    assert summary["bundle_name"] == "demo-bundle"
    assert summary["chain_name"] == "演示链"
    assert len(summary["steps"]) == 1
    assert summary["steps"][0]["method_key"] == "dewow"
    assert (tmp_path / "demo-bundle-00-raw.png").exists()
    assert (tmp_path / "demo-bundle-01-dewow.png").exists()
    assert (tmp_path / "demo-bundle-raw-vs-final.png").exists()
    assert (tmp_path / "demo-bundle-summary.json").exists()


def test_export_standard_chain_for_sample_supports_both_stage_a_chains(tmp_path: Path):
    for chain_key in ("conservative_default", "aggressive_gain"):
        summary = export_standard_chain_for_sample(
            sample_id="drift_background_reference",
            chain_key=chain_key,
            out_dir=tmp_path / chain_key,
            save_images=False,
        )
        assert summary["sample_id"] == "drift_background_reference"
        assert summary["chain_name"] == STANDARD_CHAIN_SPECS[chain_key]["label"]
        assert len(summary["steps"]) == len(STANDARD_CHAIN_SPECS[chain_key]["steps"])
        assert (
            tmp_path
            / chain_key
            / f"drift_background_reference-{chain_key}-summary.json"
        ).exists()

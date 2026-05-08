#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax gain-method comparison report helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.gprmax_benchmark import gprmax_gain_method_report as report


def test_gain_method_set_includes_physical_and_visual_gain_families():
    assert "sec_gain" in report.GAIN_METHODS
    assert "agcGain" in report.GAIN_METHODS
    assert "compensatingGain" in report.GAIN_METHODS
    assert report.GAIN_METHOD_NOTES["sec_gain"]["best_for"]
    assert report.GAIN_METHOD_NOTES["agcGain"]["risks"]


def test_gain_selection_score_prefers_sec_when_metrics_are_otherwise_equal():
    metrics = {
        "target_count": 1.0,
        "truth_score": 1.2,
        "truth_target_saliency_gain": 2.0,
        "truth_target_contrast_after": 3.0,
        "truth_false_positive_ratio": 0.4,
        "truth_background_energy_reduction": -0.1,
        "lateral_profile_corr": 0.95,
        "relative_amplitude_preservation_score": 1.0,
        "depth_balance_score": 0.7,
        "clipping_ratio_after": 0.0,
        "hot_pixel_ratio_after": 0.0,
    }
    agc_metrics = dict(metrics)
    agc_metrics["relative_amplitude_preservation_score"] = 0.45

    sec_score = report.gain_method_selection_score(metrics, "sec_gain")
    agc_score = report.gain_method_selection_score(agc_metrics, "agcGain")

    assert sec_score > agc_score


def test_no_target_gain_selection_penalizes_false_positive_amplification():
    calm = {
        "target_count": 0.0,
        "truth_score": 0.8,
        "truth_false_positive_ratio": 0.5,
        "truth_background_energy_reduction": 0.0,
        "relative_amplitude_preservation_score": 0.9,
        "clipping_ratio_after": 0.0,
        "hot_pixel_ratio_after": 0.0,
    }
    noisy = dict(calm)
    noisy["truth_score"] = -1.0
    noisy["truth_false_positive_ratio"] = 4.0

    assert report.gain_method_selection_score(calm, "no_gain") > report.gain_method_selection_score(
        noisy, "agcGain"
    )


def test_compute_gain_metrics_exposes_truth_and_amplitude_preservation_fields():
    before = np.zeros((48, 12), dtype=np.float32)
    before[14:22, 4:8] = 1.0
    after = before.copy()
    after[14:22, 4:8] *= 1.5
    ground_truth = {
        "scenario_id": "demo",
        "analysis_roi": {
            "time_start_idx": 10,
            "time_end_idx": 28,
            "dist_start_idx": 2,
            "dist_end_idx": 10,
        },
        "targets": [
            {
                "target_id": "target",
                "must_preserve": True,
                "roi": {
                    "time_start_idx": 14,
                    "time_end_idx": 22,
                    "dist_start_idx": 4,
                    "dist_end_idx": 8,
                },
            }
        ],
    }

    metrics = report.compute_gain_metrics(
        before,
        after,
        ground_truth,
        pre_gain_roi=ground_truth["analysis_roi"],
        output_roi=ground_truth["analysis_roi"],
        method_key="sec_gain",
    )

    assert "truth_score" in metrics
    assert "truth_target_saliency_gain" in metrics
    assert "lateral_profile_corr" in metrics
    assert "relative_amplitude_preservation_score" in metrics
    assert metrics["target_count"] == 1.0


def test_choose_gain_choice_keeps_no_gain_as_reference_for_target_scenes():
    variants = [
        {
            "method_key": "no_gain",
            "method_label": "不施加增益",
            "manual": {
                "selection_score": 5.0,
                "params": {},
                "selection_reason": "reference",
                "final_metrics": {"target_count": 1.0},
            },
            "auto": {
                "selection_score": 5.0,
                "params": {},
                "selection_reason": "reference",
                "final_metrics": {"target_count": 1.0},
            },
        },
        {
            "method_key": "sec_gain",
            "method_label": "SEC 深度补偿",
            "manual": {
                "selection_score": 2.0,
                "params": {"gain_max": 4.2},
                "selection_reason": "best gain",
                "final_metrics": {"target_count": 1.0},
            },
            "auto": {
                "selection_score": 1.5,
                "params": {"gain_max": 5.0},
                "selection_reason": "auto",
                "final_metrics": {"target_count": 1.0},
            },
        },
    ]

    choice = report.choose_gain_choice(variants)

    assert choice["method_key"] == "sec_gain"
    assert choice["branch"] == "manual"


def test_strip_variant_arrays_removes_large_image_payloads():
    variants = [
        {
            "method_key": "sec_gain",
            "method_label": "SEC",
            "manual": {"gain_output": np.zeros((2, 2)), "final_output": np.ones((2, 2))},
            "auto": {"gain_output": np.zeros((2, 2)), "final_output": np.ones((2, 2))},
        }
    ]

    cleaned = report._strip_variant_arrays(variants)

    assert "gain_output" not in cleaned[0]["manual"]
    assert "final_output" not in cleaned[0]["auto"]


def test_render_gain_method_html_contains_research_and_selection_sections(tmp_path: Path):
    payload = {
        "research_sources": report.RESEARCH_SOURCES[:2],
        "method_notes": report.GAIN_METHOD_NOTES,
        "selection_rule": report._selection_rule_summary(),
        "scenarios": [],
    }

    html_path = report.render_gain_method_html(tmp_path, payload)
    html_text = html_path.read_text(encoding="utf-8")

    assert "调研结论" in html_text
    assert "SEC" in html_text
    assert "AGC" in html_text
    assert "自动选择方法" in html_text
    assert "总体结果" in html_text

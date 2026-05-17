#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax gain-method comparison report helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.gain_selection import choose_gain_candidate, gain_risk_flags, score_gain_candidate
from scripts.gprmax_benchmark import gprmax_gain_method_report as report


def test_gain_method_set_includes_physical_and_visual_gain_families():
    assert "sec_gain" in report.GAIN_METHODS
    assert "agcGain" in report.GAIN_METHODS
    assert "compensatingGain" in report.GAIN_METHODS
    assert report.GAIN_METHOD_NOTES["sec_gain"]["best_for"]
    assert report.GAIN_METHOD_NOTES["agcGain"]["risks"]
    assert report.MANUAL_GAIN_PARAMS["agcGain"]["_low_energy_guard"] is True


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


def test_core_gain_selector_prefers_sec_for_interpretation_preserving_target_scene():
    metrics = {
        "target_count": 1.0,
        "truth_score": 1.2,
        "truth_target_energy_preservation": 1.0,
        "truth_target_saliency_gain": 2.0,
        "truth_target_contrast_after": 3.0,
        "truth_false_positive_ratio": 0.4,
        "truth_background_energy_reduction": 0.1,
        "lateral_profile_corr": 0.95,
        "relative_amplitude_preservation_score": 0.95,
        "depth_balance_score": 0.7,
        "clipping_ratio_after": 0.0,
        "hot_pixel_ratio_after": 0.0,
    }
    agc_metrics = dict(metrics, relative_amplitude_preservation_score=0.35)

    decision = choose_gain_candidate(
        [
            {
                "method_key": "agcGain",
                "method_label": "AGC 自动增益",
                "branch": "auto",
                "metrics": agc_metrics,
                "params": {"window": 81},
            },
            {
                "method_key": "sec_gain",
                "method_label": "SEC 深度补偿",
                "branch": "auto",
                "metrics": metrics,
                "params": {"gain_max": 4.2},
            },
        ]
    )

    assert decision.method_key == "sec_gain"
    assert decision.confidence > 0.0
    assert "method_prior" in decision.score_terms
    assert score_gain_candidate(metrics, "sec_gain") > score_gain_candidate(
        agc_metrics,
        "agcGain",
    )
    assert "relative_amplitude_not_interpretable" in gain_risk_flags(
        agc_metrics,
        "agcGain",
    )


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


def test_gain_candidate_selection_sanitizes_non_finite_external_scores():
    decision = choose_gain_candidate(
        [
            {
                "method_key": "sec_gain",
                "score": np.nan,
                "score_terms": {"external": np.inf},
                "metrics": {"truth_score": 0.2},
            },
            {
                "method_key": "agcGain",
                "score": 0.1,
                "metrics": {"truth_score": 0.1},
            },
        ]
    )

    assert np.isfinite(decision.score)
    assert all(np.isfinite(value) for value in decision.score_terms.values())


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


def test_compute_gain_metrics_handles_single_row_roi_without_frequency_mask_error():
    before = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    after = np.array([[1.5, 1.8, 2.7]], dtype=np.float32)
    ground_truth = {
        "scenario_id": "single_row_demo",
        "analysis_roi": {
            "time_start_idx": 0,
            "time_end_idx": 1,
            "dist_start_idx": 0,
            "dist_end_idx": 3,
        },
        "targets": [],
    }

    metrics = report.compute_gain_metrics(
        before,
        after,
        ground_truth,
        pre_gain_roi=ground_truth["analysis_roi"],
        output_roi=ground_truth["analysis_roi"],
        method_key="no_gain",
    )

    assert "target_band_energy_ratio" in metrics
    assert np.isfinite(metrics["target_band_energy_ratio"])


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
                "selection_score_terms": {"truth": 5.0},
                "selection_risk_flags": [],
            },
            "auto": {
                "selection_score": 5.0,
                "params": {},
                "selection_reason": "reference",
                "final_metrics": {"target_count": 1.0},
                "selection_score_terms": {"truth": 5.0},
                "selection_risk_flags": [],
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
                "selection_score_terms": {"truth": 2.0},
                "selection_risk_flags": [],
            },
            "auto": {
                "selection_score": 1.5,
                "params": {"gain_max": 5.0},
                "selection_reason": "auto",
                "final_metrics": {"target_count": 1.0},
                "selection_score_terms": {"truth": 1.5},
                "selection_risk_flags": ["near_tie_gain_choice"],
            },
        },
    ]

    choice = report.choose_gain_choice(variants)

    assert choice["method_key"] == "sec_gain"
    assert choice["branch"] == "manual"
    assert "confidence" in choice
    assert "risk_flags" in choice
    assert "score_terms" in choice


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


def test_gain_report_images_use_variant_scale_instead_of_global_scale(tmp_path: Path, monkeypatch):
    calls = []

    def fake_save_bscan_image(data, out_path, title, vlim, roi=None):
        calls.append((title, float(vlim)))

    monkeypatch.setattr(report.base_report, "save_bscan_image", fake_save_bscan_image)
    pre_gain = report.StageRun(
        data=np.ones((8, 4), dtype=np.float32),
        header_info={},
        trace_metadata={},
        roi={
            "time_start_idx": 0,
            "time_end_idx": 8,
            "dist_start_idx": 0,
            "dist_end_idx": 4,
        },
        records=[],
    )
    variants = [
        {
            "method_key": "sec_gain",
            "method_label": "SEC",
            "manual": {
                "gain_output": np.ones((8, 4), dtype=np.float32) * 100.0,
                "final_output": np.ones((8, 4), dtype=np.float32) * 100.0,
                "final_roi": pre_gain.roi,
            },
            "auto": {
                "gain_output": np.ones((8, 4), dtype=np.float32) * 80.0,
                "final_output": np.ones((8, 4), dtype=np.float32) * 80.0,
                "final_roi": pre_gain.roi,
            },
        },
        {
            "method_key": "agcGain",
            "method_label": "AGC",
            "manual": {
                "gain_output": np.ones((8, 4), dtype=np.float32) * 1.0,
                "final_output": np.ones((8, 4), dtype=np.float32) * 1.0,
                "final_roi": pre_gain.roi,
            },
            "auto": {
                "gain_output": np.ones((8, 4), dtype=np.float32) * 0.8,
                "final_output": np.ones((8, 4), dtype=np.float32) * 0.8,
                "final_roi": pre_gain.roi,
            },
        },
    ]

    report.save_gain_report_images(
        scenario_id="demo",
        pre_gain=pre_gain,
        variants=variants,
        assets_dir=tmp_path / "assets",
        report_dir=tmp_path,
    )

    sec_vlim = next(vlim for title, vlim in calls if title == "sec_gain manual gain")
    agc_vlim = next(vlim for title, vlim in calls if title == "agcGain manual gain")
    assert sec_vlim > agc_vlim * 10.0


def test_render_gain_method_html_contains_research_and_selection_sections(tmp_path: Path):
    variant_html = report._render_gain_variant(
        {
            "scenario_id": "demo",
            "label": "测试场景",
            "images": [],
        },
        {
            "method_key": "sec_gain",
            "method_label": "SEC 深度补偿",
            "manual": {
                "selection_score": 1.0,
                "params": {},
                "final_metrics": {},
                "selection_score_terms": {"truth": 1.0},
                "selection_risk_flags": [],
            },
            "auto": {
                "selection_score": 1.1,
                "params": {},
                "final_metrics": {},
                "selection_score_terms": {"truth": 1.1},
                "selection_risk_flags": ["near_tie_gain_choice"],
            },
            "auto_tune_summary": {
                "best_reason": "评分最高",
                "risk_flags": ["constraint_adjusted"],
                "selection_recommendation": "review",
                "parameter_domain": {
                    "notes": [
                        "部分候选参数被按数据尺度收缩，实际搜索域小于原始候选列表。",
                    ]
                },
            },
            "method_note": report.GAIN_METHOD_NOTES["sec_gain"],
        },
    )
    assert "风险标记" in variant_html
    assert "参数域提示" in variant_html
    assert "选择风险" in variant_html
    assert "评分项" in variant_html

    payload = {
        "research_sources": report.RESEARCH_SOURCES[:2],
        "method_notes": report.GAIN_METHOD_NOTES,
        "selection_rule": report._selection_rule_summary(),
        "scenarios": [
            {
                "scenario_id": "demo",
                "label": "测试场景",
                "best_gain_choice": {
                    "method_key": "sec_gain",
                    "method_label": "SEC 深度补偿",
                    "branch": "auto",
                    "score": 1.2,
                    "confidence": 0.7,
                    "risk_flags": ["near_tie_gain_choice"],
                    "params": {"gain_max": 4.2},
                    "reason": "demo",
                },
            }
        ],
    }

    html_path = report.render_gain_method_html(tmp_path, payload)
    html_text = html_path.read_text(encoding="utf-8")

    assert "调研结论" in html_text
    assert "SEC" in html_text
    assert "AGC" in html_text
    assert "自动选择方法" in html_text
    assert "总体结果" in html_text
    assert "置信度" in html_text
    assert "选择风险" in html_text

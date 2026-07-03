# GprMax Truth-Aware Auto-Tune Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ground-truth-aware metrics to the gprMax auto-tune validation report so automatic parameter choices are judged by target preservation, target-outside background suppression, and false-positive risk.

**Architecture:** Create a focused `core/gprmax_truth_metrics.py` module that computes scalar truth metrics from a gprMax `ground_truth.json`-style manifest. Wire those metrics into `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py` without changing the GUI comparison backend. The report keeps existing generic metrics and adds truth-specific score terms and HTML fields.

**Tech Stack:** Python dataclasses not required, NumPy masks/ROIs, existing `core.quality_metrics.ratio_fidelity`, pytest.

---

### Task 1: Add Truth Metric Unit Tests

**Files:**
- Create: `tests/test_gprmax_truth_metrics.py`

- [x] **Step 1: Cover target preservation and background suppression**

Build a synthetic B-scan with one target ROI and a background stripe. Assert a processed result that preserves the target and suppresses the stripe scores higher than a result that suppresses the target and keeps the stripe.

- [x] **Step 2: Cover zero-time-style ROI shifts**

Use `reference_roi` and `processed_roi` with a row offset. Assert the target ROI is shifted before measuring the processed array so zero-time correction does not make truth metrics look like target loss.

### Task 2: Implement `core.gprmax_truth_metrics`

**Files:**
- Create: `core/gprmax_truth_metrics.py`

- [x] **Step 1: Resolve target and background masks**

Clamp gprMax target ROIs to the current array, shift processed target ROIs using the difference between `reference_roi` and `processed_roi`, and build a background mask from analysis ROI minus target masks.

- [x] **Step 2: Compute stable scalar metrics**

Return `truth_target_energy_preservation`, `truth_target_saliency_gain`, `truth_background_energy_reduction`, `truth_false_positive_ratio`, `truth_target_contrast_after`, `truth_score`, and `truth_target_count`.

- [x] **Step 3: Keep degenerate inputs safe**

Handle empty or nearly zero-energy masks with finite defaults instead of raising during report generation.

### Task 3: Wire gprMax Multi-Scenario Report

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Modify: `tests/test_gprmax_multi_scenario_report.py`

- [x] **Step 1: Add truth metrics to step and final summaries**

Pass `ground_truth` into `_step_metric_summary`, `_final_metric_summary`, and `_branch_metrics`; merge truth metrics with existing benchmark metrics.

- [x] **Step 2: Include truth score in verdict scoring**

Add truth-score contribution to `_comparison_score` when truth metrics are present.

- [x] **Step 3: Show truth metrics in HTML**

Render truth score delta, target energy preservation delta, background suppression delta, and false-positive delta in scenario summaries and step analysis.

### Task 4: Verify

**Files:**
- Modify: plan status only after verification

- [x] **Step 1: Run focused tests**

Run `pytest tests/test_gprmax_truth_metrics.py tests/test_gprmax_multi_scenario_report.py -q`.

- [x] **Step 2: Run related auto-tune/report tests**

Run `pytest tests/test_gprmax_benchmark_package.py tests/test_auto_tune_candidate_constraints.py tests/test_auto_tune.py -q`.

- [x] **Step 3: Run smoke and diff checks**

Run `python scripts/preflight_check.py` and `git diff --check`.

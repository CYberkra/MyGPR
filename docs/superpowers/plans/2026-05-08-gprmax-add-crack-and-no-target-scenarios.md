# GprMax Crack And No-Target Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the gprMax auto-tune validation suite with weak air-filled crack and no-target background scenarios, and make truth metrics score no-target cases correctly.

**Architecture:** Keep scenario definitions inside `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py` so the HTML runner remains the single source for report scenes. Extend `core.gprmax_truth_metrics` to treat targetless manifests as background-only tests instead of pretending the full ROI is a target. Add focused tests before implementation.

**Tech Stack:** Python, NumPy ROI masks, existing gprMax `.in` model text generation, pytest.

---

### Task 1: Add Regression Tests

**Files:**
- Modify: `tests/test_gprmax_truth_metrics.py`
- Modify: `tests/test_gprmax_multi_scenario_report.py`

- [x] **Step 1: Cover no-target truth scoring**

Add a test that uses `targets=[]`, compares a background-suppressed processed array with a processed array containing a strong false anomaly, and asserts the suppressed array has a higher `truth_score`, lower `truth_false_positive_ratio`, and `truth_target_count == 0.0`.

- [x] **Step 2: Cover crack and no-target scenario definitions**

Extend scenario-definition tests to require `crack_air_filled_v1` and `no_target_background_v1`. Assert the crack scenario produces an `air_crack` truth target and the no-target scenario produces an empty target list plus a full analysis ROI.

### Task 2: Implement No-Target Metrics

**Files:**
- Modify: `core/gprmax_truth_metrics.py`

- [x] **Step 1: Detect targetless manifests**

If `ground_truth["targets"]` has no preserveable ROI, compute metrics on the analysis mask as a background-only case and keep target preservation fields neutral.

- [x] **Step 2: Penalize false positives**

Use the processed analysis ROI high percentile divided by the reference analysis ROI high percentile as `truth_false_positive_ratio`, and include it in a no-target `truth_score`.

### Task 3: Add gprMax Scenarios

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`

- [x] **Step 1: Add `crack_air_filled_v1`**

Add a thin air-filled vertical crack represented by a low-permittivity box inside soil, plus `_crack_ground_truth()` to create a line-like ROI and expected feature notes.

- [x] **Step 2: Add `no_target_background_v1`**

Add a uniform soil scene with no target. Its ground truth should carry `targets=[]`, a full analysis ROI, and notes that no strong localized reflector should be created.

- [x] **Step 3: Render crack geometry**

Update `save_structure_preview()` to draw `air_crack` targets as narrow boxes so the HTML true-structure preview remains meaningful.

### Task 4: Documentation And Verification

**Files:**
- Modify: `docs/gprmax_auto_tune_validation_plan.md`

- [x] **Step 1: Update validation plan**

Record the two new scenario IDs and explain why no-target scoring is a false-positive guard.

- [x] **Step 2: Run focused tests**

Run `pytest tests/test_gprmax_truth_metrics.py tests/test_gprmax_multi_scenario_report.py -q`.

- [x] **Step 3: Run related smoke**

Run `python -m py_compile core/gprmax_truth_metrics.py scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`, `pytest tests/test_gprmax_benchmark_package.py tests/test_auto_tune.py -q`, `python scripts/preflight_check.py`, and `git diff --check`.

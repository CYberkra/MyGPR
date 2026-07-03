# Auto-Tune Comparison Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable backend that compares manual baseline parameters against auto-tuned parameters on the same data, ROI, and pipeline.

**Architecture:** Add a focused `core/auto_tune_comparison.py` module. It executes two candidates through the existing processing engine, computes comparable metrics against the same raw input/ROI, and returns dataclasses plus a JSON-safe summary for later GUI/export integration.

**Tech Stack:** Python 3.10+, NumPy, existing `core.auto_tune`, `core.processing_engine`, `core.quality_metrics`, pytest.

---

### Task 1: Backend Dataclasses And Runner

**Files:**
- Create: `core/auto_tune_comparison.py`
- Test: `tests/test_auto_tune_comparison.py`

- [ ] **Step 1: Write tests for manual baseline source and automatic parameters**

Create tests that build a small synthetic B-scan, run a one-step `dewow` comparison, and assert:
- manual source is `current_ui_params` when explicit params are provided
- automatic candidate records tuned params
- both candidates use the same pipeline and ROI metadata

- [ ] **Step 2: Implement minimal dataclasses and `run_auto_tune_comparison`**

The runner should:
- validate 2D non-empty data
- resolve pipeline from explicit list or recommended profile
- merge experience baseline params with user params
- execute manual and automatic candidates separately
- return `AutoTuneComparisonRun`

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest tests/test_auto_tune_comparison.py -q
```

Expected: all tests pass.

### Task 2: Metrics And JSON-Safe Summary

**Files:**
- Modify: `core/auto_tune_comparison.py`
- Test: `tests/test_auto_tune_comparison.py`

- [ ] **Step 1: Add score/metric tests**

Assert the output includes:
- `comparison_score`
- `metric_delta`
- `verdict`
- `to_summary_dict()` without raw NumPy arrays

- [ ] **Step 2: Implement metric bundle**

Use existing `compute_benchmark_metrics`, `ratio_fidelity`, and penalties for clipping/hot pixels/spikiness. Compute scores against the same raw input and ROI.

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest tests/test_auto_tune_comparison.py -q
```

Expected: all tests pass.

### Task 3: Profile Hook And Regression Gate

**Files:**
- Modify: `core/preset_profiles.py`
- Test: `tests/test_auto_tune_comparison.py`

- [ ] **Step 1: Add an experience baseline profile**

Add `uav_gpr_experience_baseline_v1` with the current research baseline methods that are already implemented. Do not include motion V2 until that backend exists.

- [ ] **Step 2: Test profile fallback**

Assert `run_auto_tune_comparison(..., baseline_profile_key="uav_gpr_experience_baseline_v1")` resolves the profile order and labels the manual source as `experience_profile` when no explicit manual params are supplied.

- [ ] **Step 3: Run focused and related tests**

Run:

```bash
python -m pytest tests/test_auto_tune_comparison.py tests/test_auto_tune.py -q
```

Expected: all tests pass.

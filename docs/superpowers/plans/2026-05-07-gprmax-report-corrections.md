# GPRMAX Report Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the gprMax HTML validation report with a fair manual-vs-auto comparison after correcting zero-time handling, gain choice, and per-step interpretation.

**Architecture:** Keep the existing multi-scenario report runner as the single reusable entry point, but add report-specific policy controls. The default report pipeline uses AGC gain, aligns manual zero-time to the auto-tuned zero-time result, and records per-step visual and metric analysis against simulation ground truth.

**Tech Stack:** Python 3.10+, gprMax 3.1.7, NumPy, Matplotlib Agg, MyGPR `core.auto_tune`, `core.processing_engine`, `core.quality_metrics`, and pytest.

---

### Task 1: Fair Report Pipeline Defaults

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Test: `tests/test_gprmax_multi_scenario_report.py`

- [ ] **Step 1: Replace SEC with AGC for this report**

Add `REPORT_PIPELINE_OVERRIDES` so the report pipeline derived from `uav_gpr_experience_baseline_v1` changes:

```python
["set_zero_time", "dewow", "subtracting_average_2D", "agcGain", "svd_subspace"]
```

Use `{"window": 31}` as a reasonable manual AGC baseline: it is a moderate smoothing window, visually plausible for a skilled operator who wants contrast without completely flattening relative amplitude.

- [ ] **Step 2: Align zero-time manual parameters to auto-tune**

In `run_stepwise_comparison()`, when `method_key == "set_zero_time"` and `zero_time_policy == "align_auto"`, run auto-tune first, copy its recommended `new_zero_time` into both branches, and mark the step note:

```text
本报告将人工分支零时参数对齐自动结果，避免经验 5.0ns 在小域正演数据中切掉有效结构。
```

The rest of the pipeline remains a real manual-vs-auto parameter comparison.

### Task 2: Per-Step Analysis

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Test: `tests/test_gprmax_multi_scenario_report.py`

- [ ] **Step 1: Compute per-step metrics**

For each method step, compute ground-truth ROI metrics for manual and auto outputs versus the correct branch input. Store:

```python
{
    "manual": compute_benchmark_metrics(manual_input_roi, manual_output_roi),
    "auto": compute_benchmark_metrics(auto_input_roi, auto_output_roi),
    "delta_auto_minus_manual": {...}
}
```

- [ ] **Step 2: Generate visual analysis text**

Use rule-based text that mentions target preservation, background suppression, edge/saliency preservation, clipping/hot-pixel artifacts, and whether the known true structure is point-target hyperbola or layer-interface reflection.

- [ ] **Step 3: Render analysis in HTML**

In each step card, add two short blocks:

- `视觉评价`
- `指标评价`

### Task 3: Regenerate and Verify

**Files:**
- Generated: `output/gprmax_multi_scenario_reports/<timestamp>/index.html`
- Generated: `output/gprmax_multi_scenario_reports/<timestamp>/summary.json`

- [ ] **Step 1: Run corrected report**

Run:

```powershell
python scripts\gprmax_benchmark\gprmax_multi_scenario_report.py --gprmax-root E:\gprMax\gprMax-v.3.1.7 --runs 36 --geometry-fixed
```

- [ ] **Step 2: Verify report content**

Confirm `summary.json` shows:

- pipeline contains `agcGain`
- pipeline does not contain `sec_gain`
- `set_zero_time` manual and auto params are equal when default policy is used
- each step has visual and metric analysis text

- [ ] **Step 3: Run verification gates**

Run:

```powershell
python -m pytest tests\test_gprmax_multi_scenario_report.py tests\test_gprmax_benchmark_smoke.py -q
python scripts\preflight_check.py
python -m pytest -q
```

Expected: all commands pass.

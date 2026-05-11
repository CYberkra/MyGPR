# Pipeline Gain Selection Closure V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gprMax validation path produce a reusable, explainable UAV-GPR gain-selection decision instead of treating AGC or SEC as a fixed report choice.

**Architecture:** Extract the gain-method scoring rule from the report script into a small core module so future GUI, CLI, and reports share one decision contract. Keep gprMax image/report rendering inside `scripts/gprmax_benchmark/`, but have it consume the core selector and print the method applicability, score components, confidence, and risk flags.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib Agg reports, pytest, existing MyGPR processing engine and gprMax truth metrics.

---

### Task 1: Core Gain Selection Contract

**Files:**
- Create: `core/gain_selection.py`
- Modify: `tests/test_gprmax_gain_method_report.py`

- [x] **Step 1: Write tests for reusable scoring**

Add tests that call `score_gain_candidate()` and `choose_gain_candidate()` from `core.gain_selection` using synthetic metric dictionaries. The tests should verify:

```python
sec_metrics = {
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
agc_metrics = dict(sec_metrics, relative_amplitude_preservation_score=0.35)
assert score_gain_candidate(sec_metrics, "sec_gain") > score_gain_candidate(agc_metrics, "agcGain")
```

For no-target scenes, verify that high false-positive amplification loses to `no_gain`.

- [x] **Step 2: Implement `core.gain_selection`**

Create:

```python
@dataclass(frozen=True)
class GainCandidateDecision:
    method_key: str
    method_label: str
    branch: str
    score: float
    params: dict[str, Any]
    reason: str
    confidence: float
    risk_flags: list[str]
    score_terms: dict[str, float]
```

Implement `score_gain_candidate(metrics, method_key)`, `gain_score_terms(metrics, method_key)`, `gain_risk_flags(metrics, method_key)`, and `choose_gain_candidate(candidates)`. The scoring should preserve the existing report behavior: SEC gets a small prior for target scenes, AGC is penalized for amplitude interpretation, no-gain is a QA reference unless the scene has no target, and clipping/hot pixels/false positives are strong penalties.

- [x] **Step 3: Verify the core tests**

Run:

```bash
pytest tests/test_gprmax_gain_method_report.py -q
```

Expected: tests pass and no report images are generated.

### Task 2: Wire Reports to Core Selector

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_gain_method_report.py`
- Modify: `tests/test_gprmax_gain_method_report.py`

- [x] **Step 1: Replace local selector math**

Import `choose_gain_candidate`, `score_gain_candidate`, `gain_score_terms`, and `gain_risk_flags` from `core.gain_selection`. Keep the public wrapper `gain_method_selection_score()` for backward compatibility, but delegate it to the core function.

- [x] **Step 2: Enrich report payload**

For every manual/auto branch in `run_gain_branch()`, add:

```python
"selection_score_terms": gain_score_terms(final_metrics, method_key),
"selection_risk_flags": gain_risk_flags(final_metrics, method_key),
```

For `choose_gain_choice()`, return the selected candidate with `confidence`, `risk_flags`, and `score_terms`.

- [x] **Step 3: Render the decision evidence**

In the HTML:
- overall summary table includes confidence and risk flags;
- each gain-method card includes branch risk flags and main score terms;
- selection section explicitly says gprMax truth ROI is used now, while real UAV data will need structure/background/noise ROIs.

- [x] **Step 4: Verify rendering tests**

Run:

```bash
pytest tests/test_gprmax_gain_method_report.py -q
```

Expected: HTML tests find `置信度`, `选择风险`, and `评分项`.

### Task 3: Align Standard Report Defaults

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Modify: `tests/test_gprmax_multi_scenario_report.py`
- Modify: `docs/uav_gpr_standard_processing_flow.md`

- [x] **Step 1: Stop hard-coding AGC as the standard interpretation chain**

Change `REPORT_PIPELINE_ORDER` back to `sec_gain` as the default explanation-chain gain, because the full gain-method report is now responsible for comparing AGC/SEC/TGC/no-gain.

- [x] **Step 2: Update the old AGC-specific test**

Replace `test_report_pipeline_uses_agc_gain_instead_of_sec_gain()` with a test asserting that the standard report uses `sec_gain`, and that the gain-method report should be used when the goal is choosing between gain families.

- [x] **Step 3: Update documentation**

Add a short paragraph to `docs/uav_gpr_standard_processing_flow.md` stating:

```text
MyGPR default interpretation chain uses SEC/energy-decay style gain, while AGC remains a display/comparison branch. For gprMax validation and future real UAV data, `core.gain_selection` should decide among SEC, AGC, compensating gain, and no gain using target preservation, background suppression, false-positive, clipping, and amplitude-preservation metrics.
```

- [x] **Step 4: Run standard-report tests**

Run:

```bash
pytest tests/test_gprmax_multi_scenario_report.py tests/test_gprmax_gain_method_report.py -q
```

Expected: both report suites pass.

### Task 4: Generate Closure Report

**Files:**
- Generate ignored output under `output/gprmax_gain_method_reports/`
- Archive stable conclusions with `scripts/archive_checkpoint.py`

- [x] **Step 1: Run py_compile**

Run:

```bash
python -m py_compile core/gain_selection.py scripts/gprmax_benchmark/gprmax_gain_method_report.py scripts/gprmax_benchmark/gprmax_multi_scenario_report.py
```

Expected: no syntax errors.

- [x] **Step 2: Run focused tests and preflight**

Run:

```bash
pytest tests/test_gprmax_gain_method_report.py tests/test_gprmax_multi_scenario_report.py tests/test_auto_tune_pipeline.py -q
python scripts/preflight_check.py
```

Expected: tests and preflight pass.

- [x] **Step 3: Generate the latest gain report**

Run:

```bash
python scripts/gprmax_benchmark/gprmax_gain_method_report.py --search-mode standard
```

Expected: command prints an `index.html` path under `output/gprmax_gain_method_reports/`.

- [x] **Step 4: Archive**

Run:

```bash
python scripts/archive_checkpoint.py --summary "流程级自动选参闭环增强：增益选择规则抽到 core，gprMax 增益报告输出置信度、风险和评分项，标准解释链回到 SEC，AGC 保留为显示/对比分支。"
```

Expected: Obsidian version snapshot and index are updated.

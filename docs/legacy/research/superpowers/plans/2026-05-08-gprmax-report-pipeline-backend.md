# gprMax Report Pipeline Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the gprMax multi-scenario HTML report through `core.auto_tune_pipeline` so report evidence uses the same flow-level auto-tune backend as the rest of MyGPR.

**Architecture:** Keep `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py` as the report renderer and scenario runner, but replace its custom stepwise manual/auto execution loop with a thin adapter around `run_auto_tune_pipeline()`. Add a small locked-parameter hook to the core backend so report-only zero-time alignment can lock `set_zero_time` to the auto-derived safe value for both branches.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib Agg, pytest, existing MyGPR core runtime.

---

### Task 1: Extend Core Backend for Locked Parameters

**Files:**
- Modify: `core/auto_tune_pipeline.py`
- Modify: `tests/test_auto_tune_pipeline.py`

- [x] **Step 1: Add a locked-parameter regression test**

Add a test that monkeypatches `auto_tune_method` to fail and verifies that a locked method uses the provided params in both branches without invoking auto-tune:

```python
def test_pipeline_locked_params_apply_to_both_branches_without_auto_tune(monkeypatch):
    raw = _build_pipeline_profile(samples=72, traces=18)

    def fail_auto_tune(*args, **kwargs):
        raise AssertionError("locked methods must not auto-tune")

    monkeypatch.setattr(auto_tune_pipeline, "auto_tune_method", fail_auto_tune)
    result = run_auto_tune_pipeline(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        locked_params_by_method={"dewow": {"window": 5}},
        search_mode="fast",
    )

    assert result.manual.params_by_method["dewow"] == {"window": 5}
    assert result.automatic.params_by_method["dewow"] == {"window": 5}
    assert result.steps[0].manual_params == {"window": 5}
    assert result.steps[0].auto_params == {"window": 5}
    assert result.steps[0].auto_tune_result is None
```

- [x] **Step 2: Add `locked_params_by_method` to `run_auto_tune_pipeline()`**

Extend the function signature with:

```python
locked_params_by_method: dict[str, dict[str, Any]] | None = None,
```

Merge locked params into the resolved manual params before building branch states.

- [x] **Step 3: Skip auto-tune for locked methods**

Pass locked params into `_resolve_auto_params()`. If the method key exists in `locked_params_by_method`, return those params and `None` as the tune result.

### Task 2: Route gprMax Report Through Pipeline Backend

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Modify: `tests/test_gprmax_multi_scenario_report.py`

- [x] **Step 1: Import `run_auto_tune_pipeline`**

Import the core backend and keep report-specific rendering helpers in the script.

- [x] **Step 2: Replace custom stepwise execution**

Rewrite `run_stepwise_comparison()` so it:

1. Resolves the report pipeline and baseline manual params.
2. Applies `zero_time_policy`:
   - `align_auto`: pre-tune `set_zero_time`, lock both branches to the derived safe zero-time params, and record the original manual params.
   - `skip`: remove `set_zero_time` from the backend pipeline.
   - `manual`: no special handling.
3. Calls `run_auto_tune_pipeline(..., ground_truth=ground_truth, rollback_on_reject=True)`.
4. Converts `AutoTunePipelineRun.steps` into the existing report step dictionary schema used by `save_step_images()` and the HTML renderer.

- [x] **Step 3: Preserve report metrics and add backend decision fields**

For compatibility, expose `comparison_score` as an alias of `pipeline_score` in step and final metrics. Add `recommendation`, `risk_flags`, and `rolled_back_to_manual` to the comparison summary and each step.

- [x] **Step 4: Update HTML snippets**

Show backend recommendation, risk flags, and rollback state in the per-step warning/decision area without changing the core layout.

- [x] **Step 5: Update tests**

Add assertions that `run_stepwise_comparison()` returns `backend == "core.auto_tune_pipeline"`, step recommendation/risk fields, and zero-time alignment still preserves original manual params while locking both branches.

### Task 3: Verify and Archive

**Files:**
- Modify through script output only if needed: Obsidian archive via `scripts/archive_checkpoint.py`

- [x] **Step 1: Run focused tests**

Run:

```bash
python -m py_compile core/auto_tune_pipeline.py scripts/gprmax_benchmark/gprmax_multi_scenario_report.py
pytest tests/test_auto_tune_pipeline.py tests/test_gprmax_multi_scenario_report.py -q
```

Expected: PASS.

- [x] **Step 2: Run related regression tests and preflight**

Run:

```bash
pytest tests/test_gprmax_truth_metrics.py tests/test_auto_tune_comparison.py tests/test_auto_tune.py -q
python scripts/preflight_check.py
git diff --check
```

Expected: PASS.

- [x] **Step 3: Archive the checkpoint**

Run:

```bash
python scripts/archive_checkpoint.py --summary "gprMax 多场景报告改接流程级自动选参后端，保留零时对齐策略并展示风险与回退结论。"
```

Expected: the Obsidian version archive index is refreshed.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add core/auto_tune_pipeline.py tests/test_auto_tune_pipeline.py scripts/gprmax_benchmark/gprmax_multi_scenario_report.py tests/test_gprmax_multi_scenario_report.py docs/superpowers/plans/2026-05-08-gprmax-report-pipeline-backend.md
git commit -m "feat: route gprmax report through pipeline auto-tune"
git push
```

Expected: branch `codex/gprmax-benchmark-minimal` is pushed.

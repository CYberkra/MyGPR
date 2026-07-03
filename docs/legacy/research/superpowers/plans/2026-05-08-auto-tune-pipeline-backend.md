# Pipeline-Level Auto-Tune Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable pipeline-level auto-tune backend that tunes each method on the current processing state, records per-step before/after B-scan arrays, applies gprMax truth-aware scoring when available, and emits risk/review/rollback decisions.

**Architecture:** Add a new `core/auto_tune_pipeline.py` module instead of changing the existing manual-vs-auto comparison API. The new module runs manual and auto branches step by step, uses `core.auto_tune.auto_tune_method` for method-local search, uses `core.processing_engine.run_processing_method` for execution, and optionally merges `core.gprmax_truth_metrics.compute_ground_truth_metrics` into step and final scoring.

**Tech Stack:** Python 3.10+, NumPy, existing MyGPR core runtime, pytest.

---

### Task 1: Lock Pipeline Backend Contract With Tests

**Files:**
- Create: `tests/test_auto_tune_pipeline.py`

- [x] **Step 1: Add synthetic B-scan fixtures and the first pipeline test**

Create `tests/test_auto_tune_pipeline.py` with deterministic synthetic data and a test that verifies the backend auto-tunes each method on the current state:

```python
def test_pipeline_auto_tunes_each_step_on_current_state():
    raw = _build_pipeline_profile()
    result = run_auto_tune_pipeline(
        raw,
        pipeline=["dewow", "subtracting_average_2D"],
        manual_params_by_method={
            "dewow": {"window": 1},
            "subtracting_average_2D": {"ntraces": 3},
        },
        roi_spec=_manual_roi(),
        search_mode="fast",
    )

    assert result.pipeline == ["dewow", "subtracting_average_2D"]
    assert [step.method_key for step in result.steps] == [
        "dewow",
        "subtracting_average_2D",
    ]
    assert result.automatic.params_by_method["dewow"]["window"] != 1
    assert result.steps[0].manual_before.shape == raw.shape
    assert result.steps[0].manual_after.shape == raw.shape
    assert result.steps[0].auto_before.shape == raw.shape
    assert result.steps[0].auto_after.shape == raw.shape
    assert np.isfinite(result.metric_delta["pipeline_score"])
    assert result.overall_recommendation in {"adopt_auto", "review", "keep_manual"}
```

- [x] **Step 2: Add truth-aware risk and rollback test**

Patch `core.auto_tune_pipeline.auto_tune_method` inside the test to force an unsafe background-removal parameter. Verify that truth-aware scoring flags the auto branch and rolls back to manual output for the next state:

```python
def test_pipeline_uses_ground_truth_metrics_and_rolls_back_unsafe_auto(
    monkeypatch,
):
    raw = _build_pipeline_profile()

    def fake_auto_tune(*args, **kwargs):
        return {
            "method_key": "subtracting_average_2D",
            "method_name": "forced",
            "recommended_params": {"ntraces": 1},
            "best_params": {"ntraces": 1},
            "selection_confidence": 0.25,
            "selection_margin": 0.0,
            "execution_stats": {"constraint_adjustment_count": 0},
            "best_reason": "forced unsafe background removal",
        }

    monkeypatch.setattr(auto_tune_pipeline, "auto_tune_method", fake_auto_tune)
    result = auto_tune_pipeline.run_auto_tune_pipeline(
        raw,
        pipeline=["subtracting_average_2D"],
        manual_params_by_method={"subtracting_average_2D": {"ntraces": 15}},
        roi_spec=_manual_roi(),
        ground_truth=_truth_manifest(),
        search_mode="fast",
    )

    step = result.steps[0]
    assert step.recommendation == "keep_manual"
    assert step.rolled_back_to_manual is True
    assert "target_truth_degraded" in step.risk_flags
    assert result.overall_recommendation == "keep_manual"
    assert result.automatic.result.shape == result.manual.result.shape
```

- [x] **Step 3: Add JSON-safe summary test**

Verify that `to_summary_dict()` keeps shapes, metrics, parameters, risks, and decisions while excluding raw arrays:

```python
def test_pipeline_summary_is_json_safe_and_excludes_arrays():
    raw = _build_pipeline_profile(samples=72, traces=18)
    result = run_auto_tune_pipeline(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )

    summary = to_summary_dict(result)

    assert summary["input_shape"] == list(raw.shape)
    assert "manual_before" not in summary["steps"][0]
    assert "auto_after" not in summary["steps"][0]
    assert isinstance(summary["automatic"]["params_by_method"]["dewow"]["window"], int)
    assert summary["overall_recommendation"] == result.overall_recommendation
```

- [x] **Step 4: Run the new test file to verify it fails**

Run:

```bash
pytest tests/test_auto_tune_pipeline.py -q
```

Expected: FAIL because `core.auto_tune_pipeline` does not exist yet.

### Task 2: Implement Pipeline Backend

**Files:**
- Create: `core/auto_tune_pipeline.py`

- [x] **Step 1: Define public dataclasses and errors**

Implement `AutoTunePipelineError`, `PipelineCandidate`, `PipelineStepRecord`, and `AutoTunePipelineRun`. `PipelineStepRecord` must keep the four arrays `manual_before`, `manual_after`, `auto_before`, and `auto_after` for HTML/report generation.

- [x] **Step 2: Implement `run_auto_tune_pipeline()`**

The function signature is:

```python
def run_auto_tune_pipeline(
    data: np.ndarray,
    *,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    pipeline: list[str] | None = None,
    manual_params_by_method: dict[str, dict[str, Any]] | None = None,
    baseline_profile_key: str | None = None,
    roi_spec: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
    search_mode: str = "standard",
    rollback_on_reject: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_checker: CancelChecker | None = None,
) -> AutoTunePipelineRun:
```

Run manual and automatic branches step by step. For each auto-enabled method, call `auto_tune_method()` with the auto branch's current data and the current ROI. Then execute both branches through `run_processing_method()`.

- [x] **Step 3: Implement scoring, risk flags, and rollback**

Compute normal benchmark metrics with `compute_benchmark_metrics()`. When `ground_truth` is provided, add `compute_ground_truth_metrics()` and include `truth_score` in `pipeline_score`. If auto loses to manual, damages truth target preservation, creates false positives in no-target scenes, has low confidence, or needed parameter constraint adjustment, add risk flags and set recommendation to `keep_manual` or `review`. When recommendation is `keep_manual` and `rollback_on_reject=True`, set the automatic branch's next state to the manual result.

- [x] **Step 4: Implement `to_summary_dict()`**

Return a JSON-safe dictionary with input shape, pipeline, ROI, ground-truth metadata, final metrics, metric deltas, step params, step metrics, risk flags, recommendations, and rollback state. Exclude all raw arrays.

- [x] **Step 5: Run the new tests**

Run:

```bash
pytest tests/test_auto_tune_pipeline.py -q
```

Expected: PASS.

### Task 3: Update Research Design Documentation

**Files:**
- Modify: `docs/auto_tune_research_comparison_design.md`

- [x] **Step 1: Add a section for the pipeline-level backend**

Document that `core.auto_tune_pipeline.run_auto_tune_pipeline()` is now the preferred backend for research-grade flow evaluation because it stores every step's before/after images, parameters, metrics, truth-aware risk flags, and rollback decision.

- [x] **Step 2: Explain how it differs from the older comparison backend**

State that `core.auto_tune_comparison.run_auto_tune_comparison()` remains useful for simple manual-vs-auto runs, while the new backend is meant for full-chain scoring, gprMax verification, and paper/patent evidence generation.

### Task 4: Verify and Archive

**Files:**
- Modify through script output only if needed: Obsidian archive via `scripts/archive_checkpoint.py`

- [x] **Step 1: Run focused verification**

Run:

```bash
python -m py_compile core/auto_tune_pipeline.py
pytest tests/test_auto_tune_pipeline.py tests/test_auto_tune_comparison.py -q
```

Expected: PASS.

- [x] **Step 2: Run related regression verification**

Run:

```bash
pytest tests/test_gprmax_truth_metrics.py tests/test_gprmax_multi_scenario_report.py tests/test_auto_tune.py -q
python scripts/preflight_check.py
git diff --check
```

Expected: PASS.

- [x] **Step 3: Archive the stable checkpoint**

Run:

```bash
python scripts/archive_checkpoint.py --summary "自动选参新增流程级后端，支持逐步选参、truth-aware 评分、风险提示和回退建议。"
```

Expected: the Obsidian version archive index is refreshed.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add core/auto_tune_pipeline.py tests/test_auto_tune_pipeline.py docs/auto_tune_research_comparison_design.md docs/superpowers/plans/2026-05-08-auto-tune-pipeline-backend.md
git status --short
git commit -m "feat: add pipeline-level auto-tune backend"
git push
```

Expected: branch `codex/gprmax-benchmark-minimal` is pushed with the new backend.

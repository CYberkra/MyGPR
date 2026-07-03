# Auto-Tune Data-Aware Parameter Domain Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Prevent MyGPR auto-tune from recommending parameters that are larger than the current B-scan data shape or physically unsafe for the selected method.

**Architecture:** Add a small constraint layer that normalizes each auto-tune trial before execution. Keep existing candidate builders and scoring functions, but make every trial carry `requested_params`, effective public `params`, and structured constraint warnings.

**Tech Stack:** Python dataclasses, NumPy shape metadata, existing `core.runtime_warnings`, pytest.

---

### Task 1: Add Regression Tests

**Files:**
- Create: `tests/test_auto_tune_candidate_constraints.py`

- [x] **Step 1: Cover oversized trace windows**

Use a 36-trace profile and explicit `candidate_params=[{"ntraces": 501}]`. Expected result: the evaluated public params and best params must not exceed the trace count, and the trial must record the requested value separately.

- [x] **Step 2: Cover oversized rank intervals**

Use a small profile and explicit `candidate_params=[{"rank_start": 1, "rank_end": 40}]`. Expected result: `rank_end` must not exceed `min(data.shape)`.

- [x] **Step 3: Cover unsafe zero-time offsets**

Use a profile with `total_time_ns=50` and explicit `new_zero_time=200`. Expected result: the effective zero-time must be constrained to the shallow search window and record a warning.

### Task 2: Implement Constraint Layer

**Files:**
- Create: `core/auto_tune_constraints.py`

- [x] **Step 1: Define result dataclass**

Return requested params, effective params, and structured warnings.

- [x] **Step 2: Implement method-aware constraints**

Clamp trace windows to trace count, sample windows to sample count, SVD ranks to rank limits, and zero-time to a conservative fraction of the time window.

- [x] **Step 3: Preserve invalid non-numeric candidates**

Do not coerce values like `"bad"` into valid numbers; they should still fail as invalid trials so failure accounting remains meaningful.

### Task 3: Wire Auto-Tune Evaluation

**Files:**
- Modify: `core/auto_tune.py`

- [x] **Step 1: Constrain each trial before runtime preparation**

Apply the constraint layer to generated and externally supplied candidates.

- [x] **Step 2: Record requested/effective params**

Use effective params for scoring, `best_params`, profiles, and recommended params. Preserve requested params for explanation and warnings.

- [x] **Step 3: Surface warning counts**

Expose top-level `constraint_warnings`, `best_constraint_warnings`, and `execution_stats.constraint_adjustment_count`.

### Task 4: Surface Risk In UI

**Files:**
- Modify: `ui/gui_auto_tune_page.py`

- [x] **Step 1: Include constraint warning summary**

Show a concise warning line in the auto-tune summary when the recommended candidate had constrained parameters.

- [x] **Step 2: Adjust risk hint**

If any constraint adjustment occurred, tell the user to inspect requested/effective parameter differences.

### Task 5: Verify

**Files:**
- Modify: plan status only after verification

- [x] **Step 1: Run focused tests**

Run `pytest tests/test_auto_tune_candidate_constraints.py tests/test_auto_tune.py -q`.

- [x] **Step 2: Run broader smoke**

Run `python scripts/preflight_check.py`.

- [x] **Step 3: Run diff checks**

Run `git diff --check`.

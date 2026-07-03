# Auto Tune Comparison Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable research-evidence export layer for manual-baseline vs auto-tune comparisons, then make the GUI call that backend.

**Architecture:** Keep comparison execution in `core/auto_tune_comparison.py` and add export-only helpers in a focused `core/auto_tune_comparison_export.py`. GUI code in `app_qt.py` should only choose the output directory, call the core helper, log the returned artifact paths, and show a completion dialog. GPRMAX forward-model data becomes the next benchmark-data source, not a blocker for this generic export layer.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib Agg, PyQt6 GUI integration, pytest.

---

### Task 1: Core Export Service

**Files:**
- Create: `core/auto_tune_comparison_export.py`
- Test: `tests/test_auto_tune_comparison_export.py`

- [ ] **Step 1: Write failing export test**

Create a test that builds a small `AutoTuneComparisonRun`, exports it, and asserts these files exist: `comparison_summary.json`, `manual_bscan.png`, `auto_bscan.png`, `side_by_side.png`, `params_table.csv`, `metrics_table.csv`, `comparison_report.md`.

- [ ] **Step 2: Run the focused test and confirm import failure**

Run: `python -m pytest tests\test_auto_tune_comparison_export.py -q`

Expected: FAIL because `core.auto_tune_comparison_export` does not exist.

- [ ] **Step 3: Implement export service**

Add `export_auto_tune_comparison_artifacts(result, out_dir, bundle_name=None, input_ref=None, notes=None)` returning a JSON-safe dict with absolute artifact paths. Use one locked symmetric display range for manual/auto images.

- [ ] **Step 4: Run export tests**

Run: `python -m pytest tests\test_auto_tune_comparison_export.py tests\test_auto_tune_comparison.py -q`

Expected: PASS.

### Task 2: GUI Reuse

**Files:**
- Modify: `app_qt.py`
- Test: existing GUI preset/import-export tests where applicable.

- [ ] **Step 1: Replace GUI-local CSV/PNG writing**

Update `export_auto_tune_comparison_artifacts()` to call `core.auto_tune_comparison_export.export_auto_tune_comparison_artifacts(...)` instead of duplicating export logic in the window class.

- [ ] **Step 2: Keep user-facing behavior stable**

The button remains `导出对比证据`; the dialog still lists exported files; logs still use repo-relative paths when possible.

- [ ] **Step 3: Run GUI-adjacent tests**

Run: `python -m pytest tests\test_gui_presets.py tests\test_import_export_report.py tests\test_auto_tune_comparison_export.py -q`

Expected: PASS.

### Task 3: GPRMAX Next-Stage Planning

**Files:**
- Modify: `docs/auto_tune_research_comparison_design.md`
- Create: `docs/gprmax_auto_tune_validation_plan.md`

- [ ] **Step 1: Document the new decision**

Record that group meeting decided to use GPRMAX forward modeling because real data cannot reliably identify which hyperbolas or structures must be preserved.

- [ ] **Step 2: Define priority**

Mark GPRMAX as the next benchmark-data source after the generic export layer is complete. The old `E:\gprMax\gprMax-v.3.1.7` tree may be inspected, but MyGPR should define a clean scenario/schema contract before depending on that experiment tree.

- [ ] **Step 3: Run docs and export tests**

Run: `python -m pytest tests\test_auto_tune_comparison_export.py tests\test_auto_tune_comparison.py -q`

Expected: PASS.

### Task 4: Verification and Archive

**Files:**
- No additional source files unless tests reveal an issue.

- [ ] **Step 1: Compile changed modules**

Run: `python -m py_compile core\auto_tune_comparison_export.py app_qt.py`

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests\test_auto_tune_comparison_export.py tests\test_auto_tune_comparison.py tests\test_gui_presets.py -q`

- [ ] **Step 3: Run broader smoke gate**

Run: `python scripts\preflight_check.py`

- [ ] **Step 4: Check diff hygiene**

Run: `git diff --check`

- [ ] **Step 5: Archive checkpoint**

Run `python scripts\archive_checkpoint.py` with a summary for auto-tune comparison export and GPRMAX validation planning.

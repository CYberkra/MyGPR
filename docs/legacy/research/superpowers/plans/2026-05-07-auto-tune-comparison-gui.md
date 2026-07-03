# Auto-Tune Comparison GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the manual-baseline vs auto-tuned backend in the GUI so users can run a research comparison and inspect two B-scan outputs.

**Architecture:** Keep computation in `core.auto_tune_comparison`; add a small Qt worker in `app_qt.py`; extend `AutoTunePage` with a comparison action and summary panel; reuse existing compare snapshots for B-scan display.

**Tech Stack:** PyQt6, existing `AutoTunePage`, existing compare snapshot system, pytest GUI offscreen tests.

---

### Task 1: AutoTunePage UI Contract

**Files:**
- Modify: `ui/gui_auto_tune_page.py`
- Modify: `tests/test_gui_presets.py`

- [ ] Add `btn_compare_manual_auto` to the quick experiment row.
- [ ] Add a result segment named `comparison` and a readonly `comparison_summary`.
- [ ] Add methods `show_comparison_running`, `show_comparison_result`, and `show_comparison_error`.
- [ ] Test button existence and comparison state transitions.

### Task 2: Main Window Worker Wiring

**Files:**
- Modify: `app_qt.py`

- [ ] Import `run_auto_tune_comparison` and `to_summary_dict`.
- [ ] Add `AutoTuneComparisonWorker`.
- [ ] Track `_auto_tune_comparison_thread`, `_auto_tune_comparison_worker`, and `_last_auto_tune_comparison_result`.
- [ ] Connect `btn_compare_manual_auto` to `start_auto_tune_comparison`.
- [ ] On finish, push manual/auto arrays into `_set_compare_snapshots` and turn on compare view.

### Task 3: Verification

**Commands:**
- `python -m pytest tests/test_gui_presets.py::test_auto_tune_page_handles_state_transitions -q`
- `python -m pytest tests/test_gui_presets.py tests/test_auto_tune_comparison.py -q`
- `python -m pytest -q`
- `python scripts/preflight_check.py`

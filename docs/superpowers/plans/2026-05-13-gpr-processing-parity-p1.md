# GPR Processing Parity P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the highest-value gaps identified from Geolitix and GPRPy comparison.

**Architecture:** Implement one independently testable processing capability at a time and checkpoint after each stable step. Keep method registration, runtime execution, GUI parameter display, and tests aligned.

**Tech Stack:** Python, NumPy, PyQt6 parameter registry, pytest, MyGPR processing engine.

---

### Task 1: Time Cut

**Files:**
- Create: `PythonModule/time_cut.py`
- Modify: `core/methods_registry.py`
- Modify: `core/processing_engine.py`
- Modify: `core/workflow_data.py`
- Test: `tests/test_round2_processing_kernels.py`
- Test: `tests/test_gui_presets.py`

- [x] Add a `time_cut` ndarray method supporting `remove_below`, `remove_above`, and `keep_range`.
- [x] Inject `time_step_s` / `time_window_ns` from header metadata in `prepare_runtime_params`.
- [x] Register the method in preprocessing so the normal page can use it.
- [x] Add kernel and GUI/order regression tests.
- [x] Verify with focused pytest and `python scripts/preflight_check.py`.
- [ ] Create a normal Git checkpoint.

### Task 2: Trace QC

**Files:**
- Create: `PythonModule/trace_qc.py`
- Modify: `core/methods_registry.py`
- Modify: `core/workflow_data.py`
- Test: `tests/test_trace_qc.py`

- [x] Add no-op-by-default trace quality controls for empty-trace and high-energy trace detection.
- [x] Return trace metadata updates describing removed/muted trace indices.
- [x] Keep destructive deletion opt-in; default mode should mute or mark, not remove.

### Task 3: Equidistant Trace Resampling

**Files:**
- Modify: `PythonModule/motion_compensation_v2.py` or create `PythonModule/equidistant_trace_resample.py`
- Modify: `core/trace_metadata_utils.py`
- Test: `tests/test_trace_metadata_utils.py`

- [x] Reuse existing trace-distance metadata helpers.
- [x] Add explicit method wrapping distance-axis resampling for GUI/CLI workflows.
- [x] Preserve sidecar metadata through resampling.

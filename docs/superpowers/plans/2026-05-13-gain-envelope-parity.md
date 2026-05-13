# Gain and Envelope Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the remaining Geolitix/GPRPy-aligned gain and envelope capabilities that MyGPR currently lacks.

**Architecture:** Add focused ndarray methods under `PythonModule/`, register them in `core/methods_registry.py`, expose them through the normal GUI parameter renderer, and cover each with deterministic pytest tests. Keep each method independently usable from GUI, CLI, workflow executor, and reports.

**Tech Stack:** Python, NumPy, SciPy, PyQt6 parameter registry, pytest, MyGPR processing engine.

---

### Task 1: Energy Decay Gain

**Files:**
- Create: `PythonModule/energy_decay_gain.py`
- Modify: `core/methods_registry.py`
- Modify: `core/workflow_data.py`
- Test: `tests/test_round2_processing_kernels.py`
- Test: `tests/test_gui_presets.py`

- [x] Add robust per-sample decay estimation using median absolute amplitude to avoid strong-reflector domination.
- [x] Smooth and clip the gain curve with `smoothing_samples`, `strength`, `min_gain`, and `max_gain`.
- [x] Register as a public gain method.
- [x] Add deterministic kernel and GUI tests.
- [x] Verify and create a normal Git checkpoint.

### Task 2: Constant Scale and Normalization

**Files:**
- Create: `PythonModule/amplitude_scale.py`
- Modify: `core/methods_registry.py`
- Modify: `core/workflow_data.py`
- Test: `tests/test_round2_processing_kernels.py`
- Test: `tests/test_gui_presets.py`

- [x] Add `constant`, `peak`, and `rms` scaling modes.
- [x] Keep constant scale explicit and normalization guarded by epsilon.
- [x] Register as a public gain method.
- [ ] Verify and create a normal Git checkpoint.

### Task 3: Hilbert Envelope

**Files:**
- Create: `PythonModule/hilbert_envelope.py`
- Modify: `core/methods_registry.py`
- Modify: `core/workflow_data.py`
- Test: `tests/test_round2_processing_kernels.py`
- Test: `tests/test_gui_presets.py`

- [ ] Compute trace-wise analytic signal envelope along the sample axis.
- [ ] Support optional normalization and log compression for display/report use.
- [ ] Register as a public attribute-analysis method instead of denoising, so denoise auto-tune contracts stay stable.
- [ ] Verify and create a normal Git checkpoint.

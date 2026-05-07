# Motion Compensation V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable `motion_compensation_v2` path for UAV-GPR CSV data with RTK/IMU/altimeter sidecars.

**Architecture:** Keep V1 methods intact and add a single V2 method that owns time alignment checks, AGL selection, air-path time shift, conservative amplitude normalization, attitude-derived APC metadata, and optional equal-distance resampling. The method returns corrected data plus `trace_metadata_out`, summaries, warnings, quality flags, and provenance so GUI/CLI/reporting can reason about it.

**Tech Stack:** Python 3.10+, NumPy, existing `core.processing_engine`, `core.sidecar_*`, `core.trace_metadata_utils`, PyQt6 GUI wiring.

---

## File Structure

- Create: `PythonModule/motion_compensation_v2.py`
  - Pure ndarray method: `method_motion_compensation_v2(data, ..., trace_metadata=None, header_info=None, time_window_ns=None, ...)`.
  - Does not mutate inputs.
  - Uses air velocity `0.299792458 m/ns` for AGL time shift.
  - Produces `trace_metadata_out` when resampling, otherwise `trace_metadata_updates`.
- Modify: `core/sidecar_models.py`
  - Add altimeter required/optional fields and aliases.
- Modify: `core/sidecar_parsers.py`
  - Add `kind="altimeter"` parsing.
- Modify: `core/trace_metadata_utils.py`
  - Integrate aligned altimeter fields into per-trace metadata.
- Modify: `core/sidecar_integration.py`
  - Accept `altimeter_path`.
- Modify: `core/gpr_io.py`
  - Forward `altimeter_path` through airborne CSV extraction.
- Modify: `cli_batch.py`
  - Accept and validate `altimeter_path`.
- Modify: `core/methods_registry.py`
  - Register `motion_compensation_v2`.
- Modify: `core/preset_profiles.py`
  - Add `motion_compensation_v2` recommended profile.
- Modify: `ui/gui_advanced_settings.py` and `app_qt.py`
  - Add minimal altimeter sidecar selection and loader forwarding.
- Test: `tests/test_motion_compensation_v2.py`
  - Unit coverage for height selection, time shift, metadata output, resampling, skip paths, no mutation.
- Test: `tests/test_cli_batch_profiles.py`
  - CLI forwards RTK/IMU/altimeter sidecars into motion runtime.

## Tasks

### Task 1: Altimeter Sidecar Contract

- [ ] **Step 1: Write failing tests**

Add tests that write `altimeter.csv` with:

```csv
timestamp_s,height_agl_m,height_source,snr,target_count,valid
0.0,1.20,nar15,18.0,1,1
0.7,1.40,nar15,20.0,1,1
```

Assert `load_gpr_csv(..., altimeter_path=...)` creates `height_agl_m`, `height_source`, `height_confidence`, and `trace_timestamp_s` arrays.

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/test_cli_batch_profiles.py::test_run_job_forwards_rtk_imu_altimeter_sidecars_into_motion_runtime -q`

- [ ] **Step 3: Implement sidecar parser and integration**

Add altimeter aliases for `height_agl_m`, `distance_m`, `flight_height_m`, and `height_m`. Interpolate numeric fields to trace timestamps. Map `distance_m` to `height_agl_m`.

- [ ] **Step 4: Run sidecar tests**

Run: `python -m pytest tests/test_cli_batch_profiles.py -q`

### Task 2: Motion Compensation V2 Backend

- [ ] **Step 1: Write backend tests**

Create `tests/test_motion_compensation_v2.py` with tests for:

- valid `height_agl_m` uses air velocity for `time_shift_ns = 2 * (height - reference) / 0.299792458`
- `flight_height_m` works as fallback with a warning
- missing height skips only height correction and still returns provenance
- non-positive height skips correction with a quality flag
- `resample_spacing_m > 0` returns `trace_metadata_out` length matching data traces
- input data and metadata are not mutated

- [ ] **Step 2: Run failing backend tests**

Run: `python -m pytest tests/test_motion_compensation_v2.py -q`

- [ ] **Step 3: Implement `PythonModule/motion_compensation_v2.py`**

Use existing helpers from `core.trace_metadata_utils` for metadata cloning/resampling where appropriate. Keep behavior deterministic and conservative.

- [ ] **Step 4: Run backend tests**

Run: `python -m pytest tests/test_motion_compensation_v2.py -q`

### Task 3: Registry, Profile, CLI

- [ ] **Step 1: Register method**

Add import and `PROCESSING_METHODS["motion_compensation_v2"]` with parameters:

- `height_reference_mode`
- `manual_height_m`
- `height_source`
- `compensate_time_shift`
- `compensate_amplitude`
- `max_shift_samples`
- `max_amplitude_scale`
- `resample_spacing_m`
- APC offsets and tilt clamp

- [ ] **Step 2: Add profile**

Add `RECOMMENDED_RUN_PROFILES["motion_compensation_v2"]` with order `["motion_compensation_v2"]`.

- [ ] **Step 3: Validate CLI run**

Run a small airborne CSV job with `recommended_profile: "motion_compensation_v2"`.

### Task 4: Minimal GUI Sidecar Entry

- [ ] **Step 1: Inspect `ui/gui_advanced_settings.py` current sidecar layout**

Add one altimeter row consistent with existing RTK/IMU controls.

- [ ] **Step 2: Wire `app_qt.py`**

Allow `_pick_sidecar_file("altimeter")`, `_set_sidecar_file("altimeter")`, `_build_sidecar_loader_kwargs()` and warning labels to include altimeter.

- [ ] **Step 3: Run GUI tests**

Run: `python -m pytest tests/test_gui_presets.py -q`

### Task 5: Verification

- [ ] **Step 1: Syntax**

Run: `python -m py_compile PythonModule/motion_compensation_v2.py core/sidecar_models.py core/sidecar_parsers.py core/trace_metadata_utils.py core/sidecar_integration.py core/gpr_io.py cli_batch.py core/methods_registry.py core/preset_profiles.py app_qt.py ui/gui_advanced_settings.py`

- [ ] **Step 2: Focused tests**

Run: `python -m pytest tests/test_motion_compensation_v2.py tests/test_cli_batch_profiles.py tests/test_motion_compensation_registry.py tests/test_motion_runtime_metadata_contract.py -q`

- [ ] **Step 3: Full test and preflight**

Run: `python -m pytest -q`

Run: `python scripts/preflight_check.py`

Run: `git diff --check`

# Airborne gprMax Model Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace MyGPR's default gprMax validation scenes with air-launched UAV-GPR geometry that includes air layer, antenna height, direct wave, air-ground reflection, subsurface targets, and synthetic sidecars.

**Architecture:** Keep the implementation inside `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py` so existing reports, gain comparisons, and tests keep using one scenario contract. Preserve old surface-coupled scenes under an explicit legacy family while making the airborne family the default.

**Tech Stack:** Python, gprMax input files, NumPy, Matplotlib, pytest.

---

### Task 1: Airborne Geometry Contract

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Test: `tests/test_gprmax_multi_scenario_report.py`

- [x] **Step 1: Add `AirborneGeometry`**

Implemented shared domain, grid, ground height, antenna height, trace step, Tx/Rx offset, default runs, PML margins, top-clearance checks, and trace-position generation.

- [x] **Step 2: Verify geometry constraints**

Covered by `test_airborne_geometry_satisfies_gprmax_safety_margins`.

### Task 2: Scenario Families

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Test: `tests/test_gprmax_multi_scenario_report.py`

- [x] **Step 1: Make airborne the default family**

`build_scenario_definitions()` now defaults to `scenario_family="airborne"` and supports `airborne|legacy|all`.

- [x] **Step 2: Rename old scenes into legacy family**

Old toy scenes are available as `legacy_surface_coupled_*` and marked `is_uav_gpr_evidence=False`.

### Task 3: gprMax Input Generation

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Test: `tests/test_gprmax_multi_scenario_report.py`

- [x] **Step 1: Fixed-height airborne `.in` files**

Constant-height scenes include `#src_steps`, `#rx_steps`, and `#geometry_view`.

- [x] **Step 2: Height-varying `.in` files**

`airborne_height_variation_cylinder_v1` writes one `.in` per trace and does not include fixed step commands.

### Task 4: Ground Truth and Sidecars

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Test: `tests/test_gprmax_multi_scenario_report.py`

- [x] **Step 1: Add wavefield ROIs**

`ground_truth.json` now includes direct air wave, air-ground reflection, background, late noise, and subsurface target ROI when applicable.

- [x] **Step 2: Add synthetic airborne sidecars**

Airborne scenes write `trace_timestamps.csv`, `rtk.csv`, `imu.csv`, and `altimeter.csv`.

### Task 5: Reports and Docs

**Files:**
- Modify: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Modify: `scripts/gprmax_benchmark/gprmax_gain_method_report.py`
- Modify: `docs/gprmax_auto_tune_validation_plan.md`
- Modify: `docs/uav_gpr_standard_processing_flow.md`
- Test: `tests/test_gprmax_multi_scenario_report.py`
- Test: `tests/test_gprmax_gain_method_report.py`

- [x] **Step 1: Render wavefield checks**

HTML reports now include direct wave, surface reflection, target/background/noise ROI checks and legacy warnings.

- [x] **Step 2: Let gain reports read airborne summaries**

The gain report resolver now searches latest multi-scenario reports before old full-effect reports.

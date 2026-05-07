# UAV GPR Motion V2 Synthetic Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic synthetic `CSV + RTK + IMU + NAR15` sample package for current MyGPR development before real field data exists.

**Architecture:** Add one repo script that generates the full package from code, so checked-in sample files and future regenerated packages stay consistent. The package uses the real airborne CSV format, real sidecar parser column names, and a CLI batch config that runs `motion_compensation_v2`.

**Tech Stack:** Python 3.10+, NumPy, JSON, CSV, existing `cli_batch.py`, existing sidecar parser/integration path.

---

### Task 1: Package Generator

**Files:**
- Create: `scripts/generate_uav_gpr_motion_v2_sample.py`
- Create by running script: `sample_data/uav_gpr_motion_v2/*`
- Create by running script: `config/uav_gpr_motion_v2_synthetic.json`

- [ ] **Step 1: Write generator tests**

Create `tests/test_uav_gpr_motion_v2_sample_package.py` and assert generated temp output contains:

```python
for name in ["main.csv", "rtk.csv", "imu.csv", "altimeter.csv", "batch_motion_v2.json", "README.md"]:
    assert (package_dir / name).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uav_gpr_motion_v2_sample_package.py -q`

- [ ] **Step 3: Implement generator**

Generate:

```text
main.csv: longitude, latitude, ground_elevation_m, amplitude, flight_height_m, trace_timestamp_s
rtk.csv: timestamp_s, longitude, latitude, ground_elevation_m, flight_height_m, rtk_fix_type, satellites, hdop
imu.csv: timestamp_s, roll_deg, pitch_deg, yaw_deg, angular_rate_x, angular_rate_y, angular_rate_z
altimeter.csv: timestamp_s, height_agl_m, height_source, snr, target_count, valid
batch_motion_v2.json: recommended_profile motion_compensation_v2
```

- [ ] **Step 4: Run generator test**

Run: `python -m pytest tests/test_uav_gpr_motion_v2_sample_package.py -q`

### Task 2: CLI Acceptance

**Files:**
- Modify: `tests/test_uav_gpr_motion_v2_sample_package.py`

- [ ] **Step 1: Add CLI acceptance test**

Use `cli_batch.validate_config()` and `cli_batch.run_job()` against a generated temp package. Assert final status is `ok`, the single step is `motion_compensation_v2`, and output shape is nonempty.

- [ ] **Step 2: Run focused test**

Run: `python -m pytest tests/test_uav_gpr_motion_v2_sample_package.py -q`

### Task 3: Bundled Package Generation

**Files:**
- Create/update generated files under `sample_data/uav_gpr_motion_v2/`
- Create/update `config/uav_gpr_motion_v2_synthetic.json`

- [ ] **Step 1: Generate repo package**

Run: `python scripts/generate_uav_gpr_motion_v2_sample.py`

- [ ] **Step 2: Validate bundled config**

Run: `python cli_batch.py validate --config config/uav_gpr_motion_v2_synthetic.json`

- [ ] **Step 3: Run bundled config**

Run: `python cli_batch.py run --config config/uav_gpr_motion_v2_synthetic.json --force`

### Task 4: Verification

**Files:**
- All edited files.

- [ ] **Step 1: Syntax**

Run: `python -m py_compile scripts/generate_uav_gpr_motion_v2_sample.py`

- [ ] **Step 2: Focused tests**

Run: `python -m pytest tests/test_uav_gpr_motion_v2_sample_package.py tests/test_cli_batch_profiles.py tests/test_motion_compensation_v2.py -q`

- [ ] **Step 3: Full gate**

Run: `python -m pytest -q`

Run: `python scripts/preflight_check.py`

Run: `git diff --check`

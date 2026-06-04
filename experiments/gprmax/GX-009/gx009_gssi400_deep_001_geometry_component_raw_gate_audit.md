# GX-009-GSSI400-DEEP-001 Geometry/Component/Raw Gate Audit

## 1) Task ID
- `GX-009-GSSI400-DEEP-001-GEOMETRY-COMPONENT-RAW-GATE`

## 2) Branch / Base / New Commit
- Branch: `main`
- Base commit: `32fc97003bd0333776e757381e48082875ab1508`
- New commit: pending in this audit stage

## 3) Remote Verification
- `git rev-parse HEAD`: `32fc97003bd0333776e757381e48082875ab1508`
- `git rev-parse origin/main`: `32fc97003bd0333776e757381e48082875ab1508`
- `git ls-remote origin main`: `32fc97003bd0333776e757381e48082875ab1508 refs/heads/main`

## 4) Why GX-009 Was Created
- Start an independent feasibility line for a GSSI-like 400 MHz deep-target synthetic setup.
- Keep GX-008 (`scene_037`) unchanged as current primary active reference.

## 5) Relationship to GX-008
- GX-009 is a separate feasibility branch.
- No GX-008 scene/model/campaign_active changes were made.

## 6) GSSI400 Antenna Verification Result
- Verified local file: `E:\gprMax\gprMax-v.3.1.7\user_libs\antennas\GSSI.py`
- Verified function exists: `antenna_like_GSSI_400(x, y, z, resolution=0.001, rotate90=False)`
- Verified documented/implemented resolution support:
  - `0.5 mm`, `1 mm`, `2 mm`
- Geometry-only run error confirms hard constraint:
  - `CmdInputError: This antenna module can only be used with a spatial discretisation of 0.5mm, 1mm, 2mm`

## 7) Geometry Design (Drafted)
- Domain: `1.40 x 0.45 x 0.80 m`
- Grid (attempted): `0.005 x 0.005 x 0.005 m`
- Soil/air split:
  - dry sand `z=0.000..0.550`
  - air/free-space `z=0.550..0.800`
- Antenna input path:
  - `x = 0.300 + (current_model_run - 1) * 0.020`
- Target:
  - PEC cylinder, center `(x,z)=(0.700,0.320)`, radius `0.030`, cover depth `0.200`

## 8) Geometry-only Raw/Background Result
- Command (raw geometry-only): failed with `CmdInputError` due to unsupported `resolution=0.005`.
- Command (background geometry-only): failed with the same `CmdInputError`.
- Gate status: **FAILED at geometry-only stage**.

## 9) Component Audit Method and Result
- Planned method: small `n=7` raw run then inspect `Ex/Ey/Ez/Hx/Hy/Hz`.
- Actual result: **not executed** (blocked by geometry-only failure).

## 10) Selected Component Recommendation
- No recommendation yet.
- `selected_component = unknown_until_component_audit` remains valid.

## 11) Raw n31 Gate Status
- **Not run** (blocked by geometry-only failure and antenna resolution constraint).

## 12) Raw Visual Output Paths
- None generated.

## 13) Metrics
- No simulation arrays produced in this task, therefore no component/raw metrics.

## 14) Known Limitations
- Current draft uses `dx=dy=dz=0.005 m`, incompatible with local `antenna_like_GSSI_400`.
- Need a revised feasibility design using supported antenna discretisation (`0.0005/0.001/0.002 m`) and likely adjusted domain/runtime strategy before proceeding.

## 15) Claim Boundary
- GSSI400 deep-target feasibility gate only.
- Not a GX-008 replacement.
- Not exact CLT-GPR replication.
- Not finalized benchmark.
- Not field validation.
- Not AutoTune evaluation.

## 16) Recommended Next Task
- `GX-009-GSSI400-DEEP-002-RESOLUTION-COMPATIBILITY-REDESIGN`
- Focus:
  1. redesign scene_001 grid to a supported resolution (`0.002 m` first candidate),
  2. recompute feasible domain/aperture/runtime budget,
  3. rerun geometry-only gate before any component/raw run.

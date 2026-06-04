# GX-009-GSSI400-DEEP-002-RESOLUTION-COMPATIBILITY-REDESIGN Audit

## 1) Task ID
- `GX-009-GSSI400-DEEP-002-RESOLUTION-COMPATIBILITY-REDESIGN`

## 2) Branch / Base / New Commit
- Branch: `main`
- Base commit: `c5c3c85ffa4fa4ac8b6678bb2de7453f1d99fcde`
- New commit: pending

## 3) Remote Verification
- `git rev-parse HEAD`: `c5c3c85ffa4fa4ac8b6678bb2de7453f1d99fcde`
- `git rev-parse origin/main`: `c5c3c85ffa4fa4ac8b6678bb2de7453f1d99fcde`
- `git ls-remote origin main`: `c5c3c85ffa4fa4ac8b6678bb2de7453f1d99fcde refs/heads/main`

## 4) Why scene_002 Was Created
- `scene_001` used `dx=0.005 m` and failed geometry-only.
- Local `antenna_like_GSSI_400` requires spatial discretisation of `0.0005/0.001/0.002 m`.
- `scene_002` redesign uses `dx=dy=dz=0.002 m` as the coarsest supported option.

## 5) scene_001 Failure Summary
- Error from gprMax:
  - `CmdInputError: This antenna module can only be used with a spatial discretisation of 0.5mm, 1mm, 2mm`
- Therefore scene_001 stayed as failed resolution gate history.

## 6) Resolution Compatibility Decision
- Adopted `0.002 m` grid in new scene:
  - `scene_002_gssi400_deep20_radius03_air_sand_dx002_smoke_gate`
- Kept physical setup aligned with scene_001 concept (air/sand interface, deep target, 400 MHz antenna path).

## 7) Domain/Grid Cost Warning
- Domain `1.40 x 0.45 x 0.80 m` at `0.002 m`:
  - approx cells: `700 x 225 x 400`
  - approx total: `63,000,000`
- Full `n31/n80` intentionally deferred in this task.

## 8) Geometry-only raw/background Result
- Raw geometry-only: PASS
  - generated: `gx009_scene002_gssi400_raw.vti` (local generated artifact, not committed)
- Background geometry-only: PASS
  - generated: `gx009_scene002_gssi400_bg.vti` (local generated artifact, not committed)
- Key observed config:
  - mode: 3D
  - domain: `1.4 x 0.45 x 0.8 m`
  - discretisation: `0.002 m`
  - receiver component in antenna model: `Ey`
  - no geometry boundary error

## 9) n1 Smoke Result
- Command: `python -m gprMax raw_with_target.in -n 1 -gpu 0`
- Status: PASS
- Wall runtime (wrapper call): ~`356 s`
- Solver time (gprMax reported): ~`0:04:38.740313`
- Output count after run: produced base `raw_with_target.out`
- Runtime warning observed:
  - repeated `UserWarning: The CUDA compiler succeeded, but said the following: kernel.cu`
  - run still completed successfully.

## 10) n3 Component Audit Result
- Command: `python -m gprMax raw_with_target.in -n 3 -restart 2 -gpu 0`
- Status: PASS
- Wall runtime (wrapper call): ~`965.7 s` (includes model 2..4 execution in this run)
- Output files present:
  - `raw_with_target.out`
  - `raw_with_target2.out`
  - `raw_with_target3.out`
  - `raw_with_target4.out`
- Component stats evaluated on traces from:
  - `raw_with_target.out`, `raw_with_target2.out`, `raw_with_target3.out`
- Available receiver components under `rxs/rx1`:
  - `Ex, Ey, Ez, Hx, Hy, Hz`
- Quick stats (`shape=[9088,3]` for each component):
  - `Ex`: energy `4654.2773`, mean_abs `0.13456`, p95 `0.98790`, p99 `2.07258`
  - `Ey`: energy `44775.6611`, mean_abs `0.41402`, p95 `3.01842`, p99 `6.41465`
  - `Ez`: energy `6717.0816`, mean_abs `0.16705`, p95 `1.25218`, p99 `2.42501`
  - `Hx`: energy `9.4996e-04`
  - `Hy`: energy `5.3473e-07`
  - `Hz`: energy `1.7654e-04`

## 11) Component Recommendation
- Recommended component for next GX-009 raw gate: **`Ey`**
- Rationale:
  - highest energy and strongest amplitude distribution among E-field components,
  - directly aligned with GSSI module’s default output point orientation.

## 12) Files Generated but Excluded
- Not committed:
  - `.out` files (`raw_with_target*.out`)
  - `.vti` geometry outputs (`gx009_scene002_gssi400_raw.vti`, `gx009_scene002_gssi400_bg.vti`)
  - derived local JSON stats file (`scene002_component_audit_n3.json`)
  - any scratch/output artifacts

## 13) Known Limitations
- This is smoke-level feasibility only.
- No background time-series run, no pairing, no `target_response`.
- No `n31/n80` production run in this task.
- Runtime cost is already high at `dx=0.002` even for tiny run counts.

## 14) Claim Boundary
- GSSI400 resolution-compatibility and smoke feasibility gate only.
- Not GX-008 replacement.
- Not full raw/background pair.
- Not benchmark.
- Not field validation.
- Not AutoTune evaluation.

## 15) Recommended Next Task
- `GX-009-GSSI400-DEEP-003-RAW31-BUDGETED-GATE`
- Scope:
  1. keep `scene_002` geometry fixed,
  2. run **raw-only** `n31` with segmented budgeting and checkpointed progress,
  3. convert `Ey` and evaluate raw visibility before any background pair decision.

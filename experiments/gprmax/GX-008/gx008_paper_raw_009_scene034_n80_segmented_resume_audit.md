#!/usr/bin/env markdown
# GX-008-PAPER-RAW-009-SCENE034-N80-SEGMENTED-RESUME Audit

## 1) Task ID
- GX-008-PAPER-RAW-009-SCENE034-N80-SEGMENTED-RESUME

## 2) Branch / base / new commit
- Branch: `main`
- Base commit: `5da1b900e2d1a4dc910316966560cce6fd6ab9ed`
- New commit: pending this audit commit

## 3) Remote verification
- `git branch --show-current` -> `main`
- `git rev-parse HEAD` -> `5da1b900e2d1a4dc910316966560cce6fd6ab9ed`
- `git rev-parse origin/main` -> `5da1b900e2d1a4dc910316966560cce6fd6ab9ed`
- `git ls-remote origin main` -> `5da1b900e2d1a4dc910316966560cce6fd6ab9ed`

## 4) Existing output inspection (before resume)

### A. Manifest-recorded output directories
- raw output dir (manifest):  
  `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_034_gssi_ey_depth03_radius05_centered_n80_pair_gate\raw_with_target`
- background output dir (manifest):  
  `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_034_gssi_ey_depth03_radius05_centered_n80_pair_gate\background_only`

Observation: those folders contained logs/manifests but no `.out`. Actual `.out` files were generated in model folder:
- `D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_034_gssi_ey_depth03_radius05_centered_n80_pair_gate\`

### B. Pre-resume `.out` status (actual model folder)
- raw existing count: 62
- background existing count: 62
- raw existing trace numbers: `1..62`
- background existing trace numbers: `1..62`
- raw missing trace numbers: `63..80`
- background missing trace numbers: `63..80`
- highest completed trace: 62 (raw/background)
- partial/corrupt files: no obvious zero-length/corrupt `.out` detected in existing `1..62`

## 5) Resume strategy
- Confirmed local gprMax help includes `-restart`:
  - `E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe -m gprMax --help`
  - `-restart RESTART` = model number to restart from.
- Planned continuation for missing `63..80` using:
  - `-n 18 -restart 63`
- No rerun of `1..62` attempted.

## 6) Commands run
- Raw resume:
  - `D:\CDUT-UavGPR-Controller\MyGPR\scripts\run_gprmax_gpu_env.bat -- E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe -m gprMax raw_with_target.in -n 18 -restart 63 -gpu 0`
- Background resume:
  - `D:\CDUT-UavGPR-Controller\MyGPR\scripts\run_gprmax_gpu_env.bat -- E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe -m gprMax background_only.in -n 18 -restart 63 -gpu 0`

## 7) Runtime per segment
- Raw continuation segment (`63..80` attempt): ~852.7 s
- Background continuation segment (`63..80` attempt): ~850.9 s

## 8) Final output counts after resume attempt
- raw output count: 78
- background output count: 78
- raw highest trace: 78
- background highest trace: 78
- raw missing traces: `79, 80`
- background missing traces: `79, 80`

## 9) Failure root cause at trace 79/80
- Both raw/background fail at model 79 with domain violation from antenna geometry block:
  - `#box: 0.832 0.021 0.054 1.002 0.129 0.097 hdpe`
  - Error: upper x-coordinate `1.002m` not within domain `x<=1.0m`.
- This is a geometry-boundary limitation under current `scene_034` start position:
  - `antenna_like_GSSI_1500(0.137 + (current_model_run - 1) * 0.01, ...)`
- Result: n80 cannot complete under current scene_034 geometry script.

## 10) Merge/conversion method status
- Required complete pair (`80/80` raw + `80/80` background) not achieved.
- Therefore no valid n80 conversion/pairing/metrics/visual completion was executed in this task.
- Existing MyGPR conversion chain remains unchanged (no HDF5-reader modification).

## 11) Shape summary
- n80 shape unavailable (incomplete at `78/80`).
- Last completed validated pair remains n31 baseline from prior task.

## 12) Component summary
- Selected component target remains `Ey` (unchanged).

## 13) Metrics summary
- n80 metrics unavailable due incomplete pair.
- n31 baseline reference (from prior audit):
  - raw energy = 0.3989
  - background energy = 0.3281
  - target_response energy = 0.03836
  - target/surface energy ratio ~ 3.09

## 14) Visual output paths
- No new valid n80 paired visual outputs produced in this task (incomplete pair).

## 15) n31 vs n80 comparison
- n31: complete and auditable.
- n80 (scene_034 current geometry): blocked at 79/80 by antenna-geometry boundary overflow.
- Conclusion: not a compute-timeout issue at this stage; it is a geometry-domain constraint.

## 16) Known limitations
- `scene_034` cannot reach traces 79/80 with current antenna start formula and current GSSI geometry footprint.
- To complete n80, geometry script must be corrected (scan start / aperture mapping that keeps antenna internals inside domain through model 80).
- Any such correction changes the n80 trajectory definition and should be treated as new corrected gate scene, not a strict continuation of existing `1..78`.

## 17) Claim boundary
- This is a segmented resume attempt for scene_034 synthetic GSSI/Ey dry-sand PEC-cylinder n80 pair gate.
- It is not exact CLT-GPR replication.
- It is not a finalized paper benchmark.
- It is not field validation.
- It is not AutoTune evaluation.

## 18) Recommended next task
1. Add corrected n80 geometry scene (new scene ID), explicitly guaranteeing model 80 antenna geometry remains within domain.
2. Re-run raw/background n80 from scratch for corrected scene (do not mix with scene_034 `1..78` outputs).
3. Then execute Ey conversion/pairing/metrics/visual comparison against n31 baseline.

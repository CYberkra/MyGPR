# GX-008-PAPER-REPL-001 Audit

## Date
- 2026-05-25

## Branch
- main

## Base commit
- 19321ae86cfea775936ba798b356f33d6cd79a3e

## Remote verification
- `git rev-parse HEAD`: `19321ae86cfea775936ba798b356f33d6cd79a3e`
- `git rev-parse origin/main`: `19321ae86cfea775936ba798b356f33d6cd79a3e`
- `git ls-remote origin main`: `19321ae86cfea775936ba798b356f33d6cd79a3e`

## Paper source checked
- `C:/Users/17844/Desktop/Learning_to_Remove_Clutter_in_Real-World_GPR_Images_Using_Hybrid_Data-复制.pdf`

## Parameter extraction summary
- Extracted clear parameters from Section II-A:
  - domain: `100 x 15 x 40 cm3`
  - discretization: `0.2 cm` in x/y/z
  - antenna nominal: built-in GSSI, `1.5 GHz`
  - antenna scan direction: x
  - step size: `1 cm`
  - A-scans per B-scan: `80`
  - object type: cylinders, PEC/PVC
  - one-object position: x=50 cm, depth in 1–10 cm
  - clutter-free construction: `raw - background`, background without objects
- See parameter table:
  - `experiments/gprmax/GX-008/gx008_paper_repl_001_parameter_audit.md`

## Exact vs approximate status
- `paper_aligned_approximation_v1` (not exact replication).
- Reason: Table II exact values and exact built-in GSSI command details unresolved from available extracted text in this pass.

## Chosen scene rationale
- Chosen scene: `scene_020_paper_aligned_single_target_repl`
- Rationale:
  1. Single target subset of paper synthetic setup.
  2. Paired raw/background contract matches paper clutter-free construction.
  3. Minimal complexity for shape-calibration before any CR-Net / AutoTune continuation.

## Paper parameter table
- See `gx008_paper_repl_001_parameter_audit.md`.

## MyGPR current mismatch table
- Main mismatches in this attempt:
  - exact antenna model command (paper built-in GSSI vs current dipole approximation)
  - exact Table II material constants unresolved
  - run count (`80` in paper vs `61` gate run)
  - soil/surface diversity reduced to single-scene minimal approximation

## scene_020 model design
- domain: `1.0 x 0.15 x 0.40 m` (paper-aligned)
- dx/dy/dz: `0.002 m` (paper-aligned)
- waveform center frequency: `1.5 GHz` (paper-aligned approximation)
- step: `0.01 m` (paper-aligned)
- scan geometry:
  - src start x=0.10, rx start x=0.15
  - target center x=0.50, in scan window center region
- target: single PEC cylinder along y, radius 0.03 m
- pair contract: raw/background identical except target

## dry-run result
- `campaign_status: ready`
- `scene_020_paper_aligned_single_target_repl: ready`
- `invalid_count: 0`

## raw gate result
- Gate run command: raw-only, `num-runs=61` (no background before pass).
- Result: **timeout** at 1800 seconds.
- run manifest:
  - status/run_status: timeout
  - requested_num_runs: 61
  - gpu_flag_emitted: true
- partial outputs observed:
  - numbered raw files count: 59
- partial trace check on available outputs:
  - file1 rx/src vs file19 rx/src changed with step progression
  - Ez trace L2 difference (`file1` vs `file19`) > 0 (`6.46`)

## background run decision
- **Not run**.
- Reason: raw gate did not complete within timeout budget.

## conversion/pairing/preview result
- Not executed in this task due raw gate timeout.

## visual comparison with paper
- Not available for scene_020 in this task (no completed raw conversion/preview).
- Therefore no claim about closeness to paper B-scan morphology is made.

## unresolved parameters
- Table II exact material constants across all paper soils.
- Exact built-in GSSI antenna configuration details used by authors.
- Exact paper simulation `time_window`.

## files deliberately excluded
- Not committed:
  - `.out/.h5/.vti/.vtk/.vtu`
  - generated `.csv/.npy/.png`
  - local scratch artifacts
- No MyGPR-Evidence git operations.

## claim boundary
- one-scene paper-aligned synthetic replication attempt only
- not full CLT-GPR replication
- not CR-Net training
- not field validation
- not AutoTune evaluation
- not paper-candidate benchmark yet

## recommended next task
- `GX-008-PAPER-REPL-002-SPEED-TUNE`
  1. keep paper-aligned geometry but reduce raw gate load (`n=31`) for first completion;
  2. if raw completes, run background with same `n`;
  3. complete conversion/pairing/preview and compare shape against paper synthetic examples;
  4. optionally add a second scene using one paper rough-surface condition once baseline runtime is stable.

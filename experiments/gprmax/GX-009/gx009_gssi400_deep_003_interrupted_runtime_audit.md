# GX-009-GSSI400-DEEP-003-INTERRUPTED-RUNTIME-AUDIT

## Task
- `GX-009-GSSI400-DEEP-003-INTERRUPTED-RUNTIME-AUDIT`

## Scope Boundary
- No simulation resume.
- No background run.
- No `n31` completion run.
- No `target_response` computation.

## Repository State
- Branch: `main`
- Local HEAD: `02d85ff886c97422e553ff9c64fb002aaa83fcc7`
- `origin/main`: `02d85ff886c97422e553ff9c64fb002aaa83fcc7`
- `git ls-remote origin main`: `02d85ff886c97422e553ff9c64fb002aaa83fcc7 refs/heads/main`

## Interrupted Command
- Command:
  - `D:\CDUT-UavGPR-Controller\MyGPR\scripts\run_gprmax_gpu_env.bat -- E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe -m gprMax raw_with_target.in -n 10 -restart 5 -gpu 0`
- Scene:
  - `scene_002_gssi400_deep20_radius03_air_sand_dx002_smoke_gate`
- Interruption:
  - user-requested stop during long runtime.
- Elapsed runtime before interruption signal (wrapper observation):
  - ~148 seconds before turn abort event.

## Current Output Count / Existing Traces
- Current raw output files present:
  - `raw_with_target.out`
  - `raw_with_target2.out`
  - `raw_with_target3.out`
  - `raw_with_target4.out`
  - `raw_with_target5.out`
- Current output count: **5**
- Existing trace indices inferred from filenames: **1..5**
- Missing for raw `n31`: **6..31**

## Partial Output Validity
- HDF5 readability check on all existing files: **pass**
- Receiver group exists in each file: **`rxs/rx1`**
- Components present in each file:
  - `Ex, Ey, Ez, Hx, Hy, Hz`
- Conclusion:
  - current partial outputs are valid and reusable for future continuation.

## Runtime Cost Estimate
- From previous scene_002 raw runs in this thread:
  - n1 wrapper runtime: ~356 s
  - n3 continuation wrapper runtime: ~965.7 s (3 traces)
- Approximate per-trace runtime range (wrapper-level): ~300–355 s/trace.
- For remaining 26 traces, rough additional budget would still be multi-hour.

## Reason for Stopping
- Runtime too long / compute budget exceeded for current session constraints.

## Decision
- Mark `GX-009 scene_002 raw31` as **interrupted** due to excessive runtime.
- Recommend **pausing** GSSI400 deep-target line for now.

## Suggested Next Step
- If resumed later, continue from trace 6 with smaller segments (e.g., 2–3 traces per segment) and strict wall-time checkpoints.
- Alternative: first reduce computational burden via scoped redesign (domain/time-window/aperture budget) before attempting `n31`.

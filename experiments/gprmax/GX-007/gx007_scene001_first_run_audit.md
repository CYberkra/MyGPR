#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-007 scene_001 first local gprMax run audit."""

# GX-007-RUN-001 Scene_001 First Local Run Audit

Date: 2026-05-23  
Repo: `D:\CDUT-UavGPR-Controller\MyGPR`  
Branch: `main`  
Scene: `scene_001_single_shallow_pipe`

## 1. Summary

- Overall result: **failed**
- Raw run status: **failed**
- Background run status: **failed**
- Output directory: `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/`
- target_response generated: **no**

Failure stage:

- gprMax executable discovery was initially missing (`WinError 2`).
- After local executable routing was provided, both runs failed with a gprMax runtime/model error:
  - `gprMax.exceptions.GeneralError: Non-physical wave propagation: Material 'pipe_metal' has wavelength sampled by 0 cells, less than required minimum for physical wave propagation. Maximum significant frequency estimated as 2.49642e+09Hz`

## 2. Commands Run

Dry-run:

```bash
python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-007/campaign.yaml --dry-run
```

Run raw:

```bash
python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-007/campaign.yaml --run-scene scene_001_single_shallow_pipe --variant raw_with_target --timeout-seconds 900
```

Run background:

```bash
python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-007/campaign.yaml --run-scene scene_001_single_shallow_pipe --variant background_only --timeout-seconds 900
```

Environment troubleshooting commands:

```bash
where.exe gprMax
```

## 3. gprMax Environment

- Campaign configured executable name: `gprMax`
- `where.exe gprMax`: not found on current PATH
- User-provided gprMax workspace: `E:\gprMax\gprMax-v.3.1.7`
- Actual runtime banner observed from stdout:
  - gprMax version: `v3.1.6 (Big Smoke)`
  - Host OS: `Windows 10 (64-bit)` (as reported by gprMax)

Note:

- To complete this audit run, a local wrapper route to `python -m gprMax` was used during execution troubleshooting.
- This is an execution-environment bridge only, not a model/result claim change.

## 4. Raw Run Result

- Manifest path:  
  `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/raw_with_target/run_manifest.json`
- Status: `failed`
- Return code: `1`
- Runtime seconds: `~14.61`
- stdout summary:
  - Model parsed and geometry commands processed.
  - Material table included `background_soil` and `pipe_metal`.
- stderr summary:
  - `Non-physical wave propagation` error for `pipe_metal`.
- Produced files:
  - `run_manifest.json`
  - `stdout.log`
  - `stderr.log`
  - No `.out` / `.h5` generated.

## 5. Background Run Result

- Manifest path:  
  `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/background_only/run_manifest.json`
- Status: `failed`
- Return code: `1`
- Runtime seconds: `~14.61`
- stdout summary:
  - Model parsed and geometry commands processed.
  - Material table still included `pipe_metal`.
- stderr summary:
  - Same `Non-physical wave propagation` error for `pipe_metal`.
- Produced files:
  - `run_manifest.json`
  - `stdout.log`
  - `stderr.log`
  - No `.out` / `.h5` generated.

## 6. Pairing Status

- raw/background simulation outputs found: **no** (`.out/.h5` absent)
- Shape compatibility: **unknown** (no simulation arrays produced)
- target_response generated: **no**
- Reason:
  - Both runs terminated before simulation outputs were produced due to the same material-physics constraint error.

## 7. Repository Hygiene

- `.out/.h5` committed to MyGPR: **no**
- large generated CSV committed to MyGPR: **no**
- MyGPR-Evidence artifact side effects:
  - `run_manifest.json`, `stdout.log`, `stderr.log` were written under output root for scene_001 raw/background
- No MyGPR source commit includes generated simulation binaries.

## 8. Claim Boundary

- This is a first local synthetic execution audit only.
- Not field validation.
- Not AutoTune evaluation.
- Not paper-candidate benchmark result.
- ROI is still placeholder and has not been validated by previewed target response.

## 9. Next Step

Recommended next task: **GX-007-FIX-001**

Rationale:

- gprMax model execution failed deterministically on a material-physics constraint.
- First fix should target model/material parameterization (especially `pipe_metal` representation under current discretization/frequency assumptions) before retrying run and pairing.

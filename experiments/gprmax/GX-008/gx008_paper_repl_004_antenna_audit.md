#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-008 paper replication antenna model audit and GSSI gate."""

# GX-008-PAPER-REPL-004-ANTENNA-AUDIT

## Date
- 2026-05-25

## Branch
- main

## Base commit
- 045fe595d3ee467ee3d2c00404a5ad18194672c5

## Remote verification
- `git rev-parse HEAD` = `045fe595d3ee467ee3d2c00404a5ad18194672c5`
- `git rev-parse origin/main` = `045fe595d3ee467ee3d2c00404a5ad18194672c5`
- `git ls-remote origin main` = `045fe595d3ee467ee3d2c00404a5ad18194672c5`

## Paper antenna requirement
- Paper states built-in GSSI 1.5GHz antenna for synthetic subdataset setup.

## Current scene_021/023 antenna setting
- `scene_021` and `scene_023` both use:
  - `#waveform: ricker 1.0 1500e6 src_wave`
  - `#hertzian_dipole: z ... src_wave`
  - explicit `#rx: ...`
  - `#src_steps/#rx_steps = 0.01 0 0`
- Antenna-to-soil distance approximation in these scenes is from point source/receiver placement, not a full built-in antenna geometry.
- `current_antenna_type`: simplified source/rx approximation.
- `paper_alignment_status` (antenna): partial.
- Can be called exact antenna replication: **no**.

## Local gprMax antenna support search
- Runtime package checked:
  - `E:\gprMax\gprMax-v.3.1.7\gprMax\__init__.py`
- Searched under:
  - `E:\gprMax\gprMax-v.3.1.7\user_libs\antennas\`
  - `E:\gprMax\gprMax-v.3.1.7\user_models\`
  - `E:\gprMax\gprMax-v.3.1.7\docs\source\`
  - `E:\gprMax\gprMax-v.3.1.7\tests\experimental\`
- Found:
  - `user_libs/antennas/GSSI.py`
  - function: `antenna_like_GSSI_1500(x, y, z, resolution=0.001, rotate90=False)`
  - docstring note: supports `resolution=0.001` or `0.002`
  - coordinate convention in docstring:
    - `(x,y)` relative to geometric center in x-y plane
    - `z` relative to bottom of antenna skid
  - example usage:
    - `user_models/cylinder_Bscan_GSSI_1500.in`
    - `from user_libs.antennas.GSSI import antenna_like_GSSI_1500`
    - `antenna_like_GSSI_1500(..., ..., ..., 0.001)`

## Found GSSI command/source
- Confirmed command pattern:
  - `from user_libs.antennas.GSSI import antenna_like_GSSI_1500`
  - `antenna_like_GSSI_1500(x, y, z, resolution=0.002)`
- Source files:
  - `E:\gprMax\gprMax-v.3.1.7\user_libs\antennas\GSSI.py`
  - `E:\gprMax\gprMax-v.3.1.7\user_models\cylinder_Bscan_GSSI_1500.in`

## Unresolved antenna issues
- Paper exact antenna trajectory details and exact run count (80) were not used in this gate.
- Whether paper used additional acquisition constraints not encoded in current scene remains unresolved.
- Wrapper `--smoke` shows known post-success `UnicodeDecodeError` reader-thread noise in host Python (`gbk` decode), non-blocking for run result.

## Computation risk
- GSSI antenna model is heavier than simplified source/rx.
- Gate strategy is required; full high-trace runs should be deferred until feasibility confirmed.

## Recommended candidate scene design
- Added:
  - `scene_025_paper_aligned_gssi_antenna_gate_n15`
- Scope:
  - preserve scene_023 domain/grid/material/target/step core
  - replace simplified source/rx with `antenna_like_GSSI_1500(...)`
  - use `n=15` raw-only gate

## scene_025 design details
- `based_on_scene`: `scene_023_paper_aligned_tableii_material_gate_n31`
- `replication_type`: `paper_aligned_gssi_antenna_gate`
- kept:
  - `domain=1.0x0.15x0.40`
  - `dx=dy=dz=0.002`
  - dry sand Table II: `eps_r=3.0`, `sigma=0.001`
  - PEC cylinder target geometry
  - scan step concept: 0.01 m equivalent movement per run
- changed:
  - antenna insertion via Python block + GSSI module
  - gate run count `expected_num_runs=15`

## Dry-run result
- `campaign_status=ready`
- `scene_025_paper_aligned_gssi_antenna_gate_n15: ready`
- `invalid_count=0`

## GPU wrapper result
- `scripts/run_gprmax_gpu_env.bat --check`: pass
- `scripts/run_gprmax_gpu_env.bat --smoke`: pass

## n=1 raw smoke
- Command run: scene_025 raw-only `--num-runs 1`
- Result: success
- runtime_seconds: `41.783`
- output manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_025_paper_aligned_gssi_antenna_gate_n15\raw_with_target\run_manifest.json`

## n=15 raw gate
- Command run: scene_025 raw-only `--num-runs 15`
- Result: success
- runtime_seconds: `371.880`
- actual output count: 15
- position metadata (rx1 Position):
  - n1: `[0.318, 0.072, 0.054]`
  - n8: `[0.388, 0.072, 0.054]`
  - n15: `[0.458, 0.072, 0.054]`
- Ey trace variability:
  - `L2(1,8)=0.129359`
  - `L2(8,15)=0.111980`
  - `L2(1,15)=0.053505`

## Raw-only preview
- Local-only generated:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_025_paper_aligned_gssi_antenna_gate_n15\raw_with_target\raw_only_preview_ey_n15.png`
- No background run in this task by design.

## Runtime/cost observation
- GSSI gate n=15 runtime (`~372s`) is feasible for gate usage.
- n=1 syntax smoke (`~42s`) is a practical pre-check before larger runs.

## Claim boundary
- This task is antenna audit + raw-only feasibility gate.
- `scene_025` is not an exact paper replication.
- Not full CLT-GPR replication.
- Not CR-Net training.
- Not field validation.
- Not AutoTune evaluation.
- Not paper-candidate benchmark.

## Next recommended task
- `GX-008-PAPER-REPL-005-GSSI-PAIRED-N15`:
  - run `scene_025` background `n=15`,
  - do conversion/pairing/preview,
  - compare with `scene_023` to quantify antenna model impact.

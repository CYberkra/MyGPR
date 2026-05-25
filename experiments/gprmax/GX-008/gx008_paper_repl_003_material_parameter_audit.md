#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-008 paper replication material parameter audit (Table II confirmed values)."""

# GX-008-PAPER-REPL-003 Material Parameter Audit

## Source
- `Learning_to_Remove_Clutter_in_Real-World_GPR_Images_Using_Hybrid_Data-复制.pdf`
- page 3, Table II
- confirmation mode: user-provided PDF confirmation

## Confirmed Table II Parameters

| material | eps_r | sigma |
|---|---:|---:|
| Dry sand | 3.0 | 0.001 |
| Damp sand | 8.0 | 0.01 |
| Dry clay soil | 10.0 | 0.01 |
| Wet clay soil | 12.0 | 0.01 |
| Dry loam soil | 10.0 | 0.001 |
| PVC | 3.5 | 0 |

## Replication Use Policy
- `scene_021` remains baseline.
- `scene_023` is material-parameter calibration only.
- `scene_023` must prioritize `dry sand + PEC cylinder`.
- Do not switch multiple variables at once (no simultaneous changes in soil class, target class, depth class, scan geometry, and waveform).

## Controlled Variable Rule for scene_023
- Keep from baseline unless explicitly justified:
  - domain
  - dx/dy/dz
  - scan step and run count policy
  - source/receiver configuration
  - target geometry family (cylinder)
- Apply calibration focus:
  - use Table II dry sand EM values for soil
  - keep target as PEC cylinder

## Claim Boundary
- This file records material-parameter alignment inputs only.
- Not a full CLT-GPR replication.
- Not CR-Net training.
- Not field validation.
- Not AutoTune evaluation.

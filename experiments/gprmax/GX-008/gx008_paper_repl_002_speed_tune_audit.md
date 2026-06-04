#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-008 paper replication speed-tune audit (scene_021 centered gate)."""

# GX-008-PAPER-REPL-002-SPEED-TUNE Audit

## Date
- 2026-05-25

## Branch
- main

## Base commit
- 49a6aae3955285fb36b8d61d16abc980e0eb05db

## Remote verification
- `git rev-parse HEAD` = `49a6aae3955285fb36b8d61d16abc980e0eb05db`
- `git rev-parse origin/main` = `49a6aae3955285fb36b8d61d16abc980e0eb05db`
- `git ls-remote origin main` = `49a6aae3955285fb36b8d61d16abc980e0eb05db`

## scene_020 recap
- `replication_type`: `paper_aligned_approximation`
- Paper-aligned core kept: `domain=1.0x0.15x0.40 m`, `dx=dy=dz=0.002 m`, scan step `0.01 m`, x-direction scan, one-target setup.
- `expected_num_runs=61`.
- `step >= dx` satisfied (`0.01 >= 0.002`).
- Scan designed to cover target center.

## scene_020 timeout analysis
- Previous raw-only run timed out at `1800s` with `59/61` numbered outputs.
- Root cause is computational cost of fine grid (`~500x75x200`) rather than trace-identical stepping bug.

## scene_020 partial raw inspection result
- Checked local outputs only, no rerun.
- Numbered files present: `raw_with_target1.out`, `raw_with_target30.out`, `raw_with_target59.out`.
- Source/receiver positions vary with trace index.
- Ez column differences:
  - `L2(1,30)=44.67313`
  - `L2(30,59)=45.45308`
  - `L2(1,59)=5.84309`
- Partial preview generated locally as timeout diagnostic only (not full B-scan conclusion).

## scene_021 design rationale
- Add a centered short-gate variant to reduce runtime and preserve paper-aligned core parameters.
- Scene ID: `scene_021_paper_aligned_centered_gate_n31`.
- Goal: complete gate with `n=31` where target is at scan center, then decide paired run by gate outcome.

## scene_021 differences from scene_020
- `based_on_scene`: `scene_020_paper_aligned_single_target_repl`.
- `expected_num_runs`: `61 -> 31`.
- Scan window shifted/centered:
  - `rx_start_x=0.35`, `rx_end_x=0.65`
  - target center `x=0.50` (center trace neighborhood).
- `replication_type`: `paper_aligned_speed_gate`.
- Unchanged alignment: domain, `dx/dy/dz`, scan step, target category, paired contract.

## dry-run result
- Command: `python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run`
- Result:
  - `campaign_status=ready`
  - `total_scenes=12`
  - `ready_count=12`
  - `invalid_count=0`

## GPU wrapper result
- `scripts/run_gprmax_gpu_env.bat --check`: passed.
- `scripts/run_gprmax_gpu_env.bat --smoke`: passed.

## raw gate result (scene_021, raw only n=31)
- Status: success
- Return code: 0
- Runtime: `773.837s`
- Requested runs: 31
- Actual numbered outputs: 31
- Manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_021_paper_aligned_centered_gate_n31\raw_with_target\run_manifest.json`
- Position metadata:
  - n1: `rx=[0.35,0.074,0.1]`, `src=[0.30,0.074,0.1]`
  - n16: `rx=[0.50,0.074,0.1]`, `src=[0.45,0.074,0.1]`
  - n31: `rx=[0.65,0.074,0.1]`, `src=[0.60,0.074,0.1]`
- Column variability:
  - `L2(1,16)=288.718018`
  - `L2(16,31)=289.443634`
  - `L2(1,31)=18.629498`
- Gate decision: pass (raw success + metadata changes + trace variability).

## background decision
- Raw gate passed, so background was executed with same run count.
- Background run status: success
- Return code: 0
- Runtime: `777.467s`
- Requested runs: 31
- Actual numbered outputs: 31
- Manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_021_paper_aligned_centered_gate_n31\background_only\run_manifest.json`

## conversion / pairing / preview result
- Conversion: success
  - `raw shape=[3636,31]`
  - `background shape=[3636,31]`
  - summary:
    - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_021_paper_aligned_centered_gate_n31\convert_summary_31.json`
- Pairing: success
  - `target_response shape=[3636,31]`
  - `paired_metrics.json` and `paired_validation_summary.json` generated.
- Preview/report: success
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - `paired_target_response_report.md`
  - `paired_report_summary.json`

## visual comparison
- Against scene_020 partial timeout run:
  - scene_021 provides complete paired 31-trace diagnostic with centered aperture.
  - scene_020 remained partial raw-only due timeout.
- Against scene_010:
  - scene_021 is closer to paper-aligned parameter regime (fine grid `dx=0.002`, 1.5GHz-aligned setup).
- Curvature indicator on `target_response`:
  - peak index sequence shows symmetric V/U-like curvature around center traces.
  - preliminary hyperbola trend: yes.
- Conservative conclusion:
  - scene_021 is a complete paper-aligned speed-gate diagnostic candidate.
  - this is not exact paper replication and not benchmark completion.

## Table II status
- Added manual entry template:
  - `experiments/gprmax/GX-008/gx008_paper_tableii_manual_entry_template.md`
- Exact numeric EM values remain pending where not reliably extracted from currently available sources.

## generated local artifacts
- Root:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_021_paper_aligned_centered_gate_n31\`
- Generated (local-only, excluded from source git):
  - `raw_with_target/*.out`
  - `background_only/*.out`
  - converted arrays (`.csv`, `.npy`)
  - paired outputs (`.csv`, `.npy`, `.png`, `.json`, `.md`)
  - stdout/stderr logs and run manifests

## files deliberately excluded
- `*.out`
- `*.h5`
- `*.vti`
- `*.vtk`
- `*.vtu`
- generated `*.csv`
- generated `*.npy`
- generated `*.png`
- any `MyGPR-Evidence` git operations

## claim boundary
- One-scene paper-aligned speed-tune diagnostic only (`scene_021`).
- `paper_aligned_speed_gate`, not exact replication.
- Not full CLT-GPR replication.
- Not CR-Net training or replication.
- Not AutoTune evaluation.
- Not field validation.
- Not paper-candidate benchmark.

## recommended next task
- `GX-008-PAPER-REPL-003-TABLEII-MATERIAL-CALIBRATION`:
  1) fill Table II EM parameters with verifiable sources,
  2) rerun same centered gate with calibrated materials,
  3) compare curvature clarity and paired metrics deltas against current scene_021.

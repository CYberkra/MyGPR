#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-008 paper replication scene_023 Table II material gate audit."""

# GX-008-PAPER-REPL-003B-SCENE023-MATERIAL-GATE Audit

## Date
- 2026-05-25

## Branch
- main

## Base commit
- 942ba9cd2f63bf1b84a5f3263fc19330ee5101c9

## Remote verification
- `git rev-parse HEAD` = `942ba9cd2f63bf1b84a5f3263fc19330ee5101c9`
- `git rev-parse origin/main` = `942ba9cd2f63bf1b84a5f3263fc19330ee5101c9`
- `git ls-remote origin main` = `942ba9cd2f63bf1b84a5f3263fc19330ee5101c9`

## scene_021 baseline recap
- Scene: `scene_021_paper_aligned_centered_gate_n31`
- `replication_type`: `paper_aligned_speed_gate`
- shape: `[3636, 31]` for raw/background/target_response
- centered gate (`rx: 0.35 -> 0.65`, target `x=0.50`)
- target: PEC cylinder
- status: paired pipeline complete with preliminary curvature

## Table II material source
- `Learning_to_Remove_Clutter_in_Real-World_GPR_Images_Using_Hybrid_Data-复制.pdf`
- page 3, Table II
- values applied:
  - dry sand: `eps_r=3.0`, `sigma=0.001`
  - PVC reference: `eps_r=3.5`, `sigma=0.0` (listed only; not used as target in this scene)

## scene_023 design
- New scene: `scene_023_paper_aligned_tableii_material_gate_n31`
- intent: `material calibration diagnostic only`
- based_on_scene: `scene_021_paper_aligned_centered_gate_n31`
- replication_type: `paper_aligned_tableii_material_speed_gate`
- material_calibration_scope: `dry_sand_tableii_only`

## changed_from_scene_021
- Material label/traceability updated to explicit Table II names (`dry_sand_tableii`, `pvc_tableii`).
- PVC reference constants updated to Table II (`3.5`, `0.0`) in `materials.txt`.

## unchanged_from_scene_021
- domain / dx_dy_dz / time_window
- waveform / hertzian_dipole / rx
- src_steps / rx_steps
- expected_num_runs = 31
- scan start/end and centered aperture
- target geometry (PEC cylinder)
- ROI range semantics (`sample_range`, `trace_range`)
- raw/background pair contract

## dry-run result
- `campaign_status=ready`
- `total_scenes=13`
- `scene_023_paper_aligned_tableii_material_gate_n31: ready`
- `invalid_count=0`

## GPU wrapper result
- `scripts/run_gprmax_gpu_env.bat --check`: passed
- `scripts/run_gprmax_gpu_env.bat --smoke`: passed
- note: wrapper emits known non-blocking `UnicodeDecodeError` in reader threads after success (no impact on run success status).

## raw gate result
- command: scene_023 raw-only n=31
- status: success
- runtime: `833.774s`
- requested_num_runs: 31
- actual output count: 31
- manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_023_paper_aligned_tableii_material_gate_n31\raw_with_target\run_manifest.json`

## position metadata check
- n1: `rx=[0.35, 0.074, 0.1]`, `src=[0.30, 0.074, 0.1]`
- n16: `rx=[0.50, 0.074, 0.1]`, `src=[0.45, 0.074, 0.1]`
- n31: `rx=[0.65, 0.074, 0.1]`, `src=[0.60, 0.074, 0.1]`
- result: source/receiver stepping is active and consistent with gate design.

## column variability check
- `L2(1,16)=288.718018`
- `L2(16,31)=289.443634`
- `L2(1,31)=18.629498`
- result: trace variability present.

## background decision
- raw gate passed (success + stepping + variability + response trend), so background run executed.

## background run result
- status: success
- runtime: `818.222s`
- requested_num_runs: 31
- actual output count: 31
- manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_023_paper_aligned_tableii_material_gate_n31\background_only\run_manifest.json`

## conversion/pairing/preview result
- conversion: success
  - summary:
    - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_023_paper_aligned_tableii_material_gate_n31\convert_summary_31.json`
  - raw/background shapes: `[3636, 31]`
- pairing: success
  - `target_response shape=[3636,31]`
  - `paired_metrics.json` and `paired_validation_summary.json` generated
- preview/report: success
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - `paired_target_response_report.md`

## scene_021 vs scene_023 comparison
- Material parameter policy:
  - scene_021: paper-aligned speed-gate baseline with draft naming.
  - scene_023: explicit Table II material calibration tags and audit traceability.
- Runtime:
  - scene_021 raw/bg: `773.837s / 777.467s`
  - scene_023 raw/bg: `833.774s / 818.222s`
- Shapes:
  - both scenes: raw/background/target_response = `[3636,31]`
- Paired metrics (scene_021 -> scene_023):
  - `target_response_energy`: `1065433.8773554063 -> 1065433.8773273278`
  - `roi_energy_ratio`: `0.9548959974060669 -> 0.9548959974051046`
  - `raw_background_psnr`: `22.611657496459003 -> 22.611657496702136`
- Visual:
  - scene_023 is visually near-identical to scene_021 under current processing outputs.
  - curvature trend remains preliminary; no claim of exact paper morphology replication.
- Interpretation:
  - under current setup, Table II dry-sand calibration produces negligible observable difference vs scene_021 baseline.

## generated local artifacts
- Local-only outputs under:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_023_paper_aligned_tableii_material_gate_n31\`
- Includes:
  - raw/background run manifests and logs
  - converted arrays
  - paired outputs, metrics, preview figures, reports

## files deliberately excluded
- `*.out`, `*.h5`, `*.vti`, `*.vtk`, `*.vtu`
- generated `*.csv`, `*.npy`, `*.png`
- any MyGPR-Evidence git operations

## claim boundary
- scene_023 is a synthetic material calibration speed-gate diagnostic only.
- single-variable calibration attempt relative to scene_021 baseline.
- not exact replication.
- not full CLT-GPR replication.
- not CR-Net training.
- not field validation.
- not AutoTune evaluation.
- not paper-candidate benchmark.

## recommended next task
- `GX-008-PAPER-REPL-003C-TABLEII-SOIL-CONTRAST-GATE`:
  1) keep geometry and scan fixed,
  2) add damp-sand Table II gate counterpart for controlled cross-soil contrast,
  3) compare whether soil contrast materially changes curvature clarity or clutter dominance.

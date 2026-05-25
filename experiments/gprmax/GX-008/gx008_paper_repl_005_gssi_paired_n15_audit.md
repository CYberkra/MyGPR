#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-008 paper replication scene_025 GSSI paired n15 audit."""

# GX-008-PAPER-REPL-005-GSSI-PAIRED-N15 Audit

## Date
- 2026-05-25

## Branch
- main

## Base commit
- f1768893a7e2ba71610c6a72acdbcac91827d7fe

## Remote verification
- `git rev-parse HEAD` = `f1768893a7e2ba71610c6a72acdbcac91827d7fe`
- `git rev-parse origin/main` = `f1768893a7e2ba71610c6a72acdbcac91827d7fe`
- `git ls-remote origin main` = `f1768893a7e2ba71610c6a72acdbcac91827d7fe`

## Prior antenna audit recap
- `scene_025` confirmed to use local gprMax GSSI-like antenna model:
  - `from user_libs.antennas.GSSI import antenna_like_GSSI_1500`
- raw-only gates had already succeeded:
  - n=1 success (`~41.783s`)
  - n=15 success (`~371.880s`)

## Current scene_025 design
- Scene: `scene_025_paper_aligned_gssi_antenna_gate_n15`
- scope: GSSI-like antenna paired n=15 diagnostic
- kept from scene_023: domain/grid/material class/PEC target geometry core
- changed from scene_023: source-rx replaced by GSSI-like antenna insertion

## Raw output reuse result
- Existing raw outputs reused successfully.
- raw manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_025_paper_aligned_gssi_antenna_gate_n15\raw_with_target\run_manifest.json`
- raw status: success
- requested_num_runs: 15
- actual output count: 15

## Background run result
- Command: scene_025 background n=15
- status: success
- return_code: 0
- runtime_seconds: `389.456`
- actual output count: 15
- manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_025_paper_aligned_gssi_antenna_gate_n15\background_only\run_manifest.json`
- stdout:
  - `...\background_only\stdout.log`
- stderr:
  - `...\background_only\stderr.log`

## Conversion status
- conversion succeeded (`convert_summary_15.json`)
- raw shape: `[3636, 15]`
- background shape: `[3636, 15]`
- converter used existing default receiver component behavior (current outputs indicate Ez chain in converted files).

## Component used for conversion (Ey/Ez)
- Raw gate variability was inspected on `Ey` directly from HDF5 for antenna behavior checks.
- Campaign converter/pairing pipeline used its default output extraction path (current converted/pair metrics reflect existing pipeline behavior).
- this run did not require source code changes for component selection.

## Pairing status
- pairing: success
- target_response shape: `[3636, 15]`
- outputs:
  - `paired_metrics.json`
  - `paired_validation_summary.json`
  - `target_response.npy/csv`

## Metrics status
- paired metrics generated successfully.
- key values:
  - `target_response_energy`: `0.012152296873563428`
  - `raw_background_psnr`: `70.2012604481515`
  - `roi_energy_ratio`: `0.0009864274054182684`

## Preview status
- preview/report generated successfully:
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - `paired_target_response_report.md`
  - `paired_report_summary.json`

## Visual check (GSSI paired n=15)
- target_response is visible but low-energy.
- preliminary curvature in n=15 is weak and not clearly diagnostic as a full benchmark-quality hyperbola.
- horizontal/direct-wave-like structure still influences interpretability.
- conservative conclusion: GSSI paired n=15 workflow is feasible; interpretability at n=15 is limited.

## scene_023 vs scene_025 conservative comparison
- scene_023:
  - antenna/source: simplified waveform + hertzian_dipole + rx
  - traces: 31
  - target_response_energy: `1065433.8773273278`
  - psnr: `22.611657496702136`
  - roi_energy_ratio: `0.9548959974051046`
- scene_025:
  - antenna/source: GSSI-like `antenna_like_GSSI_1500`
  - traces: 15
  - target_response_energy: `0.012152296873563428`
  - psnr: `70.2012604481515`
  - roi_energy_ratio: `0.0009864274054182684`
- caution:
  - n differs (31 vs 15)
  - source model differs (simplified vs antenna model)
  - footprint/field component behavior differs
  - direct metric magnitude comparison is not sufficient for physical superiority claims.

## Runtime/cost observation
- scene_025 costs are manageable for gate diagnostics:
  - raw n=15: `~371.880s`
  - background n=15: `~389.456s`
- suitable for controlled paired gate usage before considering larger trace counts.

## Generated local artifacts
- root:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_025_paper_aligned_gssi_antenna_gate_n15\`
- includes run logs/manifests, converted arrays, paired outputs, previews, and report files.

## Files deliberately excluded
- no MyGPR-Evidence git operations
- excluded generated files:
  - `.out/.h5/.vti/.vtk/.vtu`
  - generated `.csv/.npy/.png`
  - scratch outputs

## Claim boundary
- GSSI-like antenna paired n=15 diagnostic only.
- not exact replication.
- not full 80 A-scan B-scan.
- not full CLT-GPR replication.
- not CR-Net training.
- not field validation.
- not AutoTune evaluation.
- not paper-candidate benchmark.

## Recommended next task
- `GX-008-PAPER-REPL-006-GSSI-COMPONENT-VALIDATION`:
  1) add explicit component-selection audit (Ey vs Ez) in conversion path for antenna scenes,
  2) re-run paired n=15 with explicit component control for apples-to-apples comparison quality.

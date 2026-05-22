#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-007 scene_001 conversion and pairing audit."""

# GX-007-CONVERT-001 Scene_001 Conversion and Pairing Audit

Date: 2026-05-23  
Repo: `D:\CDUT-UavGPR-Controller\MyGPR`  
Branch: `main`

## 1. Summary

- conversion: **success**
- pairing: **success**
- preview: **success**
- output root used: `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007`

## 2. Source Outputs Inspected

Inspected runtime directories:

- `.../scene_001_single_shallow_pipe/raw_with_target/`
- `.../scene_001_single_shallow_pipe/background_only/`

Observed files/sizes:

Raw directory:

- `run_manifest.json` (1439 bytes)
- `stdout.log` (37857 bytes)
- `stderr.log` (0 bytes)

Background directory:

- `run_manifest.json` (1440 bytes)
- `stdout.log` (37726 bytes)
- `stderr.log` (0 bytes)

Runtime-generated native gprMax `.out/.vti` files were produced in model folder during execution and then removed from MyGPR source tree after conversion.

## 3. Conversion Method

Converter investigation:

- Existing generic gprMax `.out` reader exists in `core.gpr_io.read_gprmax_out`.
- No direct GX-007 scene converter script existed; added minimal script:
  - `scripts/gprmax_campaign_convert_scene001.py`

Commands used:

1) Re-run scene_001 raw/background to regenerate `.out`:

```bash
python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-007/campaign.yaml --run-scene scene_001_single_shallow_pipe --variant raw_with_target --timeout-seconds 900
python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-007/campaign.yaml --run-scene scene_001_single_shallow_pipe --variant background_only --timeout-seconds 900
```

2) Convert `.out` to pairing-ready arrays:

```bash
python scripts/gprmax_campaign_convert_scene001.py --json D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/converted_summary.json
```

Conversion outputs:

- raw:
  - `.../raw_with_target/converted/raw_bscan.npy`
  - `.../raw_with_target/converted/raw_bscan.csv`
  - shape: `(936, 1)`, dtype: `float32`
- background:
  - `.../background_only/converted/background_bscan.npy`
  - `.../background_only/converted/background_bscan.csv`
  - shape: `(936, 1)`, dtype: `float32`

Warnings:

- none in conversion stage.

## 4. Pairing Result

Command:

```bash
python scripts/gprmax_campaign_runner.py --pair-outputs --campaign-id GX-007_paired_background_benchmark --scene-id scene_001_single_shallow_pipe --raw-output D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/raw_with_target/converted/raw_bscan.npy --background-output D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/background_only/converted/background_bscan.npy --output-dir D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs --source-format auto
```

Result:

- shape compatibility: **yes** (`936 x 1` vs `936 x 1`)
- target_response generated: **yes**
- validation summary:
  - `.../paired_outputs/paired_validation_summary.json`
- metrics:
  - `.../paired_outputs/paired_metrics.json`

Key metrics:

- `raw_energy`: `255703.86933204762`
- `background_energy`: `255693.32561147978`
- `target_response_energy`: `10.543924899213927`
- `target_to_background_energy_ratio`: `4.12366058988774e-05`

## 5. Preview/Report Result

Command:

```bash
python scripts/gprmax_campaign_runner.py --preview-pair --campaign-id GX-007_paired_background_benchmark --scene-id scene_001_single_shallow_pipe --raw-output D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/raw_with_target/converted/raw_bscan.npy --background-output D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/background_only/converted/background_bscan.npy --target-response D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs/target_response.npy --output-dir D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs --source-format auto
```

Generated preview/report files:

- `raw_preview.png`
- `background_preview.png`
- `target_response_preview.png`
- `paired_preview_panel.png`
- `paired_target_response_report.md`
- `paired_report_summary.json`

All files are under MyGPR-Evidence output root, not inside MyGPR source repo.

## 6. Repository Hygiene

- `.out/.h5/.vti` committed to MyGPR: **no**
- generated CSV/NPY committed to MyGPR: **no**
- MyGPR-Evidence changed locally: **yes**
  - scene_001 run manifests/logs
  - converted raw/background CSV/NPY
  - paired outputs, metrics, previews, lightweight report
- MyGPR-Evidence committed in this task: **no**

## 7. Claim Boundary

- Synthetic scene_001 conversion/pairing audit only.
- Not field validation.
- Not AutoTune evaluation.
- Not paper-candidate result.
- ROI remains placeholder; no ROI update is claimed in this task.

## 8. Next Step

Recommended next task: **GX-007-ROI-001**

Reason:

- target_response and preview assets are now available and can support a first ROI refinement pass for scene_001.

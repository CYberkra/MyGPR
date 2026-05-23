#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AT-BG-002 diagnostic harness usage and output draft."""

# AT-BG-002 Harness Draft

## Purpose

AT-BG-002 provides a diagnostic-only offline harness for synthetic paired background suppression candidate comparison.
It is not production AutoTune scoring and not UI integration.

## Script

- `scripts/autotune_background_suppression_diagnostic.py`
- core module: `core/autotune_background_suppression.py`

## Usage Example

```bash
python scripts/autotune_background_suppression_diagnostic.py \
  --raw D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008_scene001_flat_dry_sand_pec_shallow_paired_diagnostic/arrays/raw_bscan.csv \
  --target-response D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008_scene001_flat_dry_sand_pec_shallow_paired_diagnostic/arrays/target_response.csv \
  --output-dir D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/AT-BG-002/GX-008_scene001 \
  --artifact-id GX-008_scene001_flat_dry_sand_pec_shallow_paired_diagnostic \
  --scene-id scene_001_flat_dry_sand_pec_shallow
```

Optional:

- `--roi-json <path>`
- `--candidate-config <json>`
- `--write-arrays true|false` (default false)
- `--max-preview-candidates <int>`

## Candidate Methods (v1)

- mean background subtraction (global + moving window)
- median background subtraction (global + moving window)
- SVD background suppression (remove_rank 1/2/3)

## Scoring v1

Conservative rule:

- primary `mae` ascending
- tie-break `rmse` ascending
- then `psnr` descending
- warning/oversuppression penalties
- optional ROI preservation penalty when ROI is valid

## Outputs

- `trial_table.json`
- `trial_table.csv`
- `selected_parameters.json`
- `background_suppression_autotune_report.md`
- `background_suppression_autotune_manifest.json`

## Limitations

- No production scoring replacement.
- No workflow-wide AutoTune orchestration.
- No real no-prior ground-truth guarantee.
- No CR-Net/deep learning path in this phase.

## Claim Boundary

- Synthetic paired diagnostic comparison only.
- Not AutoTune superiority evidence.
- Not field validation.
- Not proof of closer-to-truth underground structure on no-prior real data.

## Next Task

- `AT-BG-003`: integrate trial outputs into lightweight viewer/report pipeline without changing production scoring.

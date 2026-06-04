#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-007 dry-run readiness audit (no real gprMax execution)."""

# GX-007 Dry-run Readiness Audit

Date: 2026-05-23  
Repo: `D:\CDUT-UavGPR-Controller\MyGPR`  
Branch: `main`

## 1. Summary

- Readiness conclusion: **ready_for_first_local_run**
- Scenes audited:
  - `scene_001_single_shallow_pipe`
  - `scene_002_single_deeper_pipe`
- Campaign dry-run: **pass**

Scope note: this is syntax/contract readiness only. No real gprMax run was performed.

## 2. Campaign Dry-run Result

Command:

```bash
python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-007/campaign.yaml --dry-run
```

Observed output:

- `campaign_status`: `ready`
- `total_scenes`: `2`
- `ready_count`: `2`
- `warning_count`: `0`
- `invalid_count`: `0`

## 3. gprMax Input Syntax Review

Reviewed files:

- `models/scene_001_single_shallow_pipe/raw_with_target.in`
- `models/scene_001_single_shallow_pipe/background_only.in`
- `models/scene_002_single_deeper_pipe/raw_with_target.in`
- `models/scene_002_single_deeper_pipe/background_only.in`

Per-file command presence check:

- `#domain`: present
- `#dx_dy_dz`: present
- `#time_window`: present
- `#waveform`: present
- `#hertzian_dipole`: present
- `#rx`: present
- `#src_steps`: present
- `#rx_steps`: present
- `#box`: present
- `#geometry_view`: present

Target-line check:

- `raw_with_target.in`: contains `#cylinder` target line
- `background_only.in`: does not contain `#cylinder` target line

No obvious command-form issues were found in this audit.

## 4. Material Definition Review

Initial risk found:

- Geometry used `background_soil` and `pipe_metal`, while material definitions existed only in companion `materials.txt`.
- This can be ambiguous for direct gprMax execution if `.in` does not import external material tables.

Fix applied in this task:

- Added explicit `#material` lines to all four `.in` files:
  - `#material: 9.0 0.01 1.0 0.0 background_soil`
  - `#material: 1000000.0 10000000.0 1.0 0.0 pipe_metal`

Result:

- Material use in geometry now has explicit in-file definitions.
- `materials.txt` is kept as human-readable companion documentation.

## 5. Pair Contract Audit

For each scene, `raw_with_target.in` vs `background_only.in` was checked for contract consistency.

Matched fields:

- `#domain`
- `#dx_dy_dz`
- `#time_window`
- `#waveform`
- `#hertzian_dipole`
- `#rx`
- `#src_steps`
- `#rx_steps`
- background `#box`
- `#geometry_view` domain/grid extents and resolution
- `#material` definitions (after fix)

Expected differences only:

- `#title` text
- `#geometry_view` output name (`scene_*_raw` vs `scene_*_background`)
- target line (`#cylinder`) exists only in `raw_with_target.in`

No unexpected pair-contract drift found.

## 6. ROI Placeholder Review

Checked files:

- `annotations/scene_001_single_shallow_pipe_roi.json`
- `annotations/scene_002_single_deeper_pipe_roi.json`

Findings:

- ROI JSON exists for both scenes.
- `roi_schema` is `gprmax_target_roi_v1`.
- `scene_id` matches campaign scene IDs.
- ROI entry `status` is `initial_placeholder`.
- Notes explicitly state placeholder nature; no ROI metrics claim is made.

## 7. Run-readiness Checklist

- [x] campaign dry-run ready
- [x] raw/background path pairs exist
- [x] `.in` commands structurally valid
- [x] material definitions addressed in `.in`
- [x] pair contract passes
- [x] ROI placeholders documented
- [x] no generated outputs staged
- [x] no Evidence repo changes

## 8. Required Fixes Before Run

Applied in this task:

1. File: `models/scene_001_single_shallow_pipe/raw_with_target.in`  
   Issue: materials used but not explicitly defined in `.in`  
   Fix: added `#material` lines for `background_soil` and `pipe_metal`

2. File: `models/scene_001_single_shallow_pipe/background_only.in`  
   Issue: same as above  
   Fix: added same `#material` lines

3. File: `models/scene_002_single_deeper_pipe/raw_with_target.in`  
   Issue: same as above  
   Fix: added same `#material` lines

4. File: `models/scene_002_single_deeper_pipe/background_only.in`  
   Issue: same as above  
   Fix: added same `#material` lines

Remaining risk:

- This audit does not replace actual parser/runtime verification by gprMax itself; first local run is still required in `GX-007-RUN-001`.

## 9. Claim Boundary

- This is run-readiness audit only.
- No real gprMax run was performed.
- No `.out`, `.h5`, or simulation CSV outputs were generated.
- No `target_response` artifact exists yet.
- No AutoTune evaluation was performed.
- No field validation claim is made.

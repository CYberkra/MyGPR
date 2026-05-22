#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-007 first paired gprMax benchmark design plan (docs-only)."""

# GX-007 Paired gprMax Benchmark Plan

## 1. Executive Summary

GX-007 is the first small paired gprMax benchmark set for MyGPR.

Its purpose is to evaluate background suppression behavior and AutoTune candidate behavior under controlled synthetic conditions using known:

- `raw_with_target`
- `background_only`
- `target_response = raw_with_target - background_only`

GX-007 is synthetic diagnostic benchmark design only. It is not field validation.

## 2. Scientific Rationale

Real field data (e.g., YingShan/YaAn) does not provide clutter-free ground truth, so direct full-reference scoring is limited.

Paired gprMax design provides:

- one scene with target (`raw_with_target`)
- one scene without target (`background_only`)
- a deterministic response reference:
  - `target_response = raw_with_target - background_only`

This enables controlled full-reference checks that no-prior field data cannot provide directly.

## 3. Minimum Scene Set

Planned small set (4–5 scenes):

1. `scene_001_single_shallow_pipe`
2. `scene_002_single_deeper_pipe`
3. `scene_003_two_targets_different_depths`
4. `scene_004_rough_surface_or_layer_clutter`
5. `scene_005_no_target_negative_control`

Per-scene intent:

### scene_001_single_shallow_pipe
- Purpose: baseline shallow single-target visibility and subtraction behavior.
- raw model: target included.
- background model: same scene without target.
- Intended target: one shallow pipe-like object.
- Expected ROI: one compact hyperbola region.
- Validation risk/purpose: shallow clutter suppression versus target preservation.

### scene_002_single_deeper_pipe
- Purpose: depth sensitivity and deep-energy preservation.
- raw model: same background, deeper target inserted.
- background model: same scene without target.
- Intended target: one deeper object.
- Expected ROI: deeper/lower SNR region.
- Validation risk/purpose: deep target attenuation versus over-suppression.

### scene_003_two_targets_different_depths
- Purpose: multi-target separation and unequal depth behavior.
- raw model: two targets at different depths.
- background model: same scene without targets.
- Intended target(s): two responses with different amplitudes and time positions.
- Expected ROI: two disjoint target regions.
- Validation risk/purpose: whether one target is suppressed while another remains.

### scene_004_rough_surface_or_layer_clutter
- Purpose: structured clutter/background challenge.
- raw model: layered or rough-surface clutter plus target.
- background model: same clutter without target.
- Intended target(s): one or more targets under stronger clutter.
- Expected ROI: target ROI overlapping high-clutter context.
- Validation risk/purpose: aggressive background suppression side effects.

### scene_005_no_target_negative_control
- Purpose: negative control and false-positive energy baseline.
- raw model: no target (control).
- background model: no target (identical contract except optional stochastic differences disabled).
- Intended target(s): none.
- Expected ROI: optional empty list.
- Validation risk/purpose: residual artifacts after subtraction should remain low.

## 4. Pair Contract

For each `raw_with_target` / `background_only` pair, the following must be identical:

- domain
- `dx/dy/dz`
- `time_window`
- waveform
- source/receiver configuration
- scan path
- material background
- output sampling

Only intentional difference:

- target exists only in `raw_with_target`

Contract implications:

- Shape mismatch is a hard failure.
- Metadata mismatch may invalidate the pair.
- `target_response` is meaningful only when pair contract is satisfied.

## 5. Proposed Directory Layout

Proposed local design layout:

```text
experiments/gprmax/GX-007/
  campaign.yaml
  models/
    scene_001_single_shallow_pipe/
      raw_with_target.in
      background_only.in
      materials.txt
    scene_002_single_deeper_pipe/
      raw_with_target.in
      background_only.in
      materials.txt
  annotations/
    scene_001_single_shallow_pipe_roi.json
    scene_002_single_deeper_pipe_roi.json
```

Output root should be outside MyGPR source repo or in policy-controlled evidence storage, e.g.:

```text
D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007
```

## 6. Campaign YAML Draft

Sample draft (two scenes):

```yaml
campaign_id: GX-007_first_paired_benchmark
output_root: D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007
gprmax_executable: gprMax

scenes:
  - scene_id: scene_001_single_shallow_pipe
    description: baseline shallow single target
    raw_model: models/scene_001_single_shallow_pipe/raw_with_target.in
    background_model: models/scene_001_single_shallow_pipe/background_only.in
    materials: models/scene_001_single_shallow_pipe/materials.txt
    target_roi: annotations/scene_001_single_shallow_pipe_roi.json
    expected_outputs:
      - raw_with_target
      - background_only
      - target_response
    tags:
      - gx007
      - shallow_target
      - paired_benchmark

  - scene_id: scene_002_single_deeper_pipe
    description: deeper single target
    raw_model: models/scene_002_single_deeper_pipe/raw_with_target.in
    background_model: models/scene_002_single_deeper_pipe/background_only.in
    materials: models/scene_002_single_deeper_pipe/materials.txt
    target_roi: annotations/scene_002_single_deeper_pipe_roi.json
    expected_outputs:
      - raw_with_target
      - background_only
      - target_response
    tags:
      - gx007
      - deeper_target
      - paired_benchmark
```

## 7. ROI Annotation Draft

Draft ROI JSON:

```json
{
  "roi_schema": "gprmax_target_roi_v1",
  "scene_id": "scene_001_single_shallow_pipe",
  "target_rois": [
    {
      "label": "pipe_1",
      "time_start_idx": 100,
      "time_end_idx": 180,
      "trace_start_idx": 40,
      "trace_end_idx": 90,
      "notes": "initial estimated hyperbola ROI"
    }
  ]
}
```

ROI is initially approximate and should be versioned/reviewed iteratively.

## 8. Expected Outputs

For each completed scene, expected artifacts:

- raw run manifest
- background run manifest
- raw output metadata
- background output metadata
- raw/background CSV or NPY converted arrays
- `target_response.npy`
- `target_response.csv`
- `paired_validation_summary.json`
- `paired_metrics.json`
- preview PNGs
- lightweight report markdown
- claim boundary note

## 9. Metrics Plan

Current GX-RUN-003 / GX-RUN-004 metrics:

- shape checks
- raw/background/target_response energy
- `target_to_background_energy_ratio`
- `abs_difference_mean`
- `abs_difference_max`

Future ROI metrics (planned, not implemented yet):

- target ROI energy preservation
- background ROI suppression
- false-positive energy outside ROI
- edge/shape preservation

## 10. Claim Boundaries

GX-007 must be interpreted under strict boundaries:

- GX-007 is synthetic diagnostic evidence.
- It is not real field validation.
- It does not prove AutoTune superiority.
- It does not prove target detection correctness in YingShan/YaAn field data.
- It can support controlled evaluation of background suppression and candidate scoring behavior.

## 11. Execution Readiness Checklist

Before running real gprMax:

- [ ] gprMax executable available.
- [ ] campaign YAML passes dry-run.
- [ ] all model/material/ROI files exist.
- [ ] output root writable.
- [ ] raw/background pair contract reviewed.
- [ ] no large outputs staged into MyGPR source repo.
- [ ] MyGPR-Evidence storage policy confirmed.

## 12. Recommended Next Steps

- `GX-007-MODEL-001`: create minimal model/material/ROI files for `scene_001` and `scene_002`.
- `GX-007-DRYRUN-001`: run campaign dry-run only.
- `GX-007-RUN-001`: run first raw/background pair locally.
- `GX-007-EVIDENCE-001`: save first valid paired artifact package in MyGPR-Evidence.
- `AT-022`: use GX-007 target_response to evaluate background suppression AutoTune behavior.

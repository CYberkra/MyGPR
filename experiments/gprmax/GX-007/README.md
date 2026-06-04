#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-007 paired gprMax benchmark preparation note."""

# GX-007 Minimal Paired gprMax Model Drafts

## Purpose

GX-007 is the first small paired synthetic benchmark preparation set for MyGPR.
This folder provides minimal model/config drafts for:

- `scene_001_single_shallow_pipe`
- `scene_002_single_deeper_pipe`

## Scene List

- `scene_001_single_shallow_pipe`
  - shallow single target sanity-check pair
- `scene_002_single_deeper_pipe`
  - deeper single target pair for weaker/deeper response behavior

## Pair Contract

For each scene:

- `raw_with_target.in` and `background_only.in` are intended to be identical except target object lines.
- domain / grid (`dx/dy/dz`) / time window / waveform / source-receiver / scan path / material background should match.
- `background_only.in` removes only the target object.

## Current Status

- Model drafts prepared: **yes**
- Campaign YAML prepared: **yes**
- ROI JSON drafts prepared: **yes (initial placeholders)**
- Real gprMax execution: **not run**
- target_response artifact: **not generated**

## Claim Boundary

- Synthetic benchmark preparation only.
- No real gprMax run has been executed yet.
- No target_response result exists yet.
- No AutoTune evaluation has been performed yet.
- No field validation claim is made.

## Dry-run Command

```bash
python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-007/campaign.yaml --dry-run
```

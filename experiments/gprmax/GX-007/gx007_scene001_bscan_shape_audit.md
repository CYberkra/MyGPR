#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-007 scene_001 B-scan shape diagnosis and mitigation audit."""

# GX-007-SCENE001-BSCAN-FIX Audit

Date: 2026-05-23  
Repo: `D:\CDUT-UavGPR-Controller\MyGPR`  
Branch: `main`

## 1. Summary

- outcome: **fixed_to_2d_bscan (diagnostic path)**
- old shape: `[936, 1]`
- new shape: `[936, 21]`
- target_response regenerated: **yes** (under new v2 output path)
- preview regenerated: **yes** (under new v2 output path)

Notes:

- The 2D result was achieved by changing run invocation (multi-run scan), not by changing scene_001 model geometry.
- The run timed out before completing all requested traces, but produced enough per-trace outputs to generate a 2D artifact.

## 2. Root Cause of `[936, 1]`

Root cause is **execution mode**, not converter math:

1. Scene model has stepping directives:
   - `#src_steps: 0.01 0 0`
   - `#rx_steps: 0.01 0 0`
2. In gprMax, these stepping directives only take effect across repeated model runs (`-n`), i.e., B-scan acquisition sequence.
3. GX-RUN-002 runner default scene execution does a single run (no `-n`), therefore only one trace is produced and conversion returns `[samples, 1]`.
4. Converter `scripts/gprmax_campaign_convert_scene001.py` correctly reads gprMax `.out` and can merge numbered per-trace files through `read_gprmax_out`, but there must be multiple run outputs available first.

## 3. Changes Made

Committed source changes in this task:

- none to processing algorithms
- none to model geometry for scene_001
- audit document only

Diagnostic runtime changes (not committed):

- temporarily routed `gprmax_executable` to local wrapper (`gprMax.cmd`) to call local gprMax Python environment
- executed scene_001 with extra args:
  - `--extra-arg=-n --extra-arg=101`
- reverted temporary executable routing and removed wrapper afterward

## 4. Re-run Result

### Raw / Background run status

- raw run: `timeout` at 1200s, progressed to model ~22/101
- background run: `timeout` at 1200s, progressed to model ~21/101

Even with timeout, both runs produced numbered per-trace outputs (`*1.out`, `*2.out`, ...), enabling partial 2D assembly.

### Conversion result

Command (diagnostic):

```bash
python scripts/gprmax_campaign_convert_scene001.py \
  --raw-out experiments/gprmax/GX-007/models/scene_001_single_shallow_pipe/raw_with_target1.out \
  --background-out experiments/gprmax/GX-007/models/scene_001_single_shallow_pipe/background_only1.out \
  --raw-converted-dir D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/raw_with_target/converted_v2 \
  --background-converted-dir D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/background_only/converted_v2
```

Conversion output shapes:

- raw: `[936, 21]`
- background: `[936, 21]`

### Pairing result

- status: success
- output dir:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs_v2`
- `target_response_shape`: `[936, 21]`
- key metric:
  - `target_response_energy`: `701.656608797365`

### Preview result

- status: success
- regenerated:
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - report/summary JSON

## 5. Evidence Handling

- New Evidence files generated: **yes**
- Location:
  - `.../raw_with_target/converted_v2/`
  - `.../background_only/converted_v2/`
  - `.../paired_outputs_v2/`
- MyGPR-Evidence committed in this task: **no** (generation only)

## 6. Claim Boundary

- no field validation
- no AutoTune evaluation
- not paper-candidate result yet
- this v2 artifact is still a diagnostic run outcome (partial trace count due timeout), not a final benchmark package

## 7. Next Step

Recommended next task: **GX-007-EVIDENCE-002**

With scope:

1. Persist `converted_v2` + `paired_outputs_v2` as a separate versioned Evidence artifact.
2. Record timeout/partial-trace context explicitly.
3. Add a stable scan-run policy (e.g., reduce domain/time-window or set feasible `-n`) to complete a full planned trace count without timeout.

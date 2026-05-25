# GX-008-HYPERBOLA-FAST-DIAG-002B-CONVERT-PREVIEW Audit

## Date
- 2026-05-25

## Branch
- main

## Base commit
- df33ecf0a66c06531e3899928478b6e2b0848aa7

## Remote verification
- `git rev-parse HEAD`: `df33ecf0a66c06531e3899928478b6e2b0848aa7`
- `git rev-parse origin/main`: `df33ecf0a66c06531e3899928478b6e2b0848aa7`
- `git ls-remote origin main`: `df33ecf0a66c06531e3899928478b6e2b0848aa7`

## FAST-DIAG-001 recap
- Previous diagnosis identified `step < dx` as the primary cause of no trace variation in scene_007/009.
- scene_010 introduced `src_steps=rx_steps=0.01`, aligned with `dx=0.01`, to avoid stepping collapse.

## step<dx root cause (recap)
- With `dx=0.01` and step `0.005`, source/receiver coordinates across runs collapse to the same grid indices.
- Converted columns became identical; preview cannot show curvature regardless of color scaling.

## scene_010 design
- scene_id: `scene_010_micro_hyperbola_step001_dry_sand_pec_sphere`
- domain: `1.2 x 0.6 x 0.4`
- `dx=dy=dz=0.01`
- `time_window=12e-9`
- waveform: `ricker 1.0 900e6`
- source start: `x=0.30`
- receiver start: `x=0.35`
- `src_steps=rx_steps=0.01 0 0`
- runs: `61`
- scan window: `rx x=0.35 -> 0.95`
- target center: `x=0.65` (scan center region), target: PEC sphere.

## scene_010 raw result
- run_status: success
- requested_num_runs: 61
- actual output count: 61
- runtime_seconds: ~279.749

## position metadata check
- raw trace positions vary as expected:
  - n1: rx 0.35 / src 0.30
  - n31: rx 0.65 / src 0.60
  - n61: rx 0.95 / src 0.90

## column variability check
- Raw Ez columns show clear non-zero differences:
  - L2(1,31)=93.76579
  - L2(31,61)=94.14606
  - L2(1,61)=17.07403
- Trace variability is restored.

## background run decision and result
- Decision: run background because raw variability gate passed.
- background run_status: success
- requested_num_runs: 61
- actual output count: 61
- runtime_seconds: ~279.487

## conversion status
- Tool: `scripts/gprmax_campaign_convert_scene001.py` with explicit raw/background base paths and run_count=61.
- Result: success.
- Summary path:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_010_micro_hyperbola_step001_dry_sand_pec_sphere\convert_summary_61.json`

## raw/background/target_response shapes
- raw: `[625, 61]`
- background: `[625, 61]`
- target_response: `[625, 61]`

## pairing status
- Tool: `scripts/gprmax_campaign_pair_converted.py`
- Result: success.
- Validation summary:
  - `...scene_010...\paired_outputs\paired_validation_summary.json`

## metrics status
- Result: success (`paired_metrics.json` generated).
- Notable values:
  - `target_response_std`: ~1.7347
  - `target_response_energy`: ~114735.14
- Warning:
  - `roi_missing_ranges` (ROI file uses `trace_window/depth_window`; current metric ROI path expects `trace_range/sample_range`).

## preview status
- Result: success.
- Generated:
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - `paired_target_response_report.md`
  - `paired_report_summary.json`
- Additional zoom/contrast:
  - No separate custom zoom/clipped PNG pipeline added in source.
  - Visual review done on existing target_response and panel previews.

## visual hyperbola check
- clear typical hyperbola: **no (not strictly confirmed)**
- preliminary curvature: **yes**
- trace variability visible in preview: **yes**
- target_response dominated by horizontal band: **no (dominant response is curved arch-like trend)**
- Interpretation:
  - scene_010 now shows meaningful curvature trend after restoring stepping.
  - Treat as micro hyperbola diagnostic candidate; further tuning still required before any benchmark-grade claim.

## generated local artifacts
- `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_010_micro_hyperbola_step001_dry_sand_pec_sphere\`
  - raw/background manifests and logs
  - converted arrays and summary
  - paired outputs (metrics/preview/report)

## files deliberately excluded
- Not committed:
  - `.out/.h5/.vti/.vtk/.vtu`
  - generated `.csv/.npy/.png`
  - scratch files
- No MyGPR-Evidence git operations.

## claim boundary
- synthetic fast diagnostic only
- scene_010 is not Evidence artifact yet
- not AutoTune evaluation
- not field validation
- not paper-candidate benchmark

## recommended next task
- `GX-008-HYPERBOLA-FAST-DIAG-003-TUNE`:
  1. keep `step >= dx` rule,
  2. add explicit ROI schema (`trace_range/sample_range`) for ROI-aware metrics,
  3. test 81-trace and/or slight depth/radius/frequency variants to improve “clear typical hyperbola” visibility.

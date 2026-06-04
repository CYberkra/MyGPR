#!/usr/bin/env markdown
# GX-RUN-CONVERT-GENERIC-001 Audit

Date: 2026-05-23  
Branch: `main`  
Base commit: `ec6aeac16e0c9a4a1938b5af29e3fc0c9c9441e2`

## Remote verification

- `git rev-parse HEAD`: `ec6aeac16e0c9a4a1938b5af29e3fc0c9c9441e2`
- `git rev-parse origin/main`: `ec6aeac16e0c9a4a1938b5af29e3fc0c9c9441e2`
- `git ls-remote origin main`: `ec6aeac16e0c9a4a1938b5af29e3fc0c9c9441e2`

## Root cause

Primary root cause was path input mismatch in pairing stage:

- conversion script outputs:
  - `raw_with_target/converted/raw_bscan.{npy,csv}`
  - `background_only/converted/background_bscan.{npy,csv}`
- previous failing pair commands referenced non-existent filenames:
  - `raw_with_target_bscan.npy`
  - `background_only_bscan.npy`

This made `--pair-outputs/--preview-pair` correctly report:
- `raw_missing`
- `background_missing`

Secondary robustness gap:

- existing CLI requires explicit raw/background paths; no scene-root discovery helper existed, so filename mismatches were easy to trigger.

## Fix implemented

1. Added converted discovery helper:
- `core/gprmax_campaign/pairing.py`
  - `discover_converted_pair_paths(scene_root, prefer_format, raw_path, background_path)`
  - supports Windows-style scene-root strings and CSV/NPY fallback.

2. Added generic pair+preview script:
- `scripts/gprmax_campaign_pair_converted.py`
  - inputs: `--scene-root`, `--output-dir`, `--campaign-id`, `--scene-id`
  - optional: `--raw`, `--background`, `--prefer-format`, `--json`
  - preserves old GX-007 converter script compatibility.

3. Added tests for discovery + generic script path:
- converted subdir discovery
- Windows-style scene root
- GX-008-like scene-root pair+preview flow

## Files changed

- `core/gprmax_campaign/pairing.py`
- `core/gprmax_campaign/__init__.py`
- `scripts/gprmax_campaign_pair_converted.py`
- `tests/test_gprmax_campaign_pairing.py`
- `tests/test_gprmax_campaign_preview.py`
- `experiments/gprmax/GX-008/gx008_convert_pair_001_audit.md`

## Compatibility notes

- `scripts/gprmax_campaign_convert_scene001.py` unchanged (backward compatible).
- existing `--pair-outputs` / `--preview-pair` interfaces unchanged.
- GX-007 flow remains supported.

## GX-008 scene_001 conversion input paths

- raw converted:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\raw_with_target\converted\raw_bscan.npy`
- background converted:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\background_only\converted\background_bscan.npy`

## Pairing result (executed)

Using:

```bat
python scripts\gprmax_campaign_pair_converted.py ^
  --scene-root D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow ^
  --output-dir D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\paired_outputs ^
  --campaign-id GX-008_paper_inspired_mini_benchmark_draft ^
  --scene-id scene_001_flat_dry_sand_pec_shallow ^
  --json D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\paired_outputs\pair_preview_generic_41.json
```

Result:

- pairing status: `success`
- shape: raw/background/target_response = `[936, 41]`
- generated:
  - `paired_validation_summary.json`
  - `paired_metrics.json`
  - `target_response.npy`
  - `target_response.csv`

## Preview result (executed)

- preview status: `success`
- generated:
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - `paired_target_response_report.md`
  - `paired_report_summary.json`

## Generated local artifact paths

- all generated outputs are under:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\paired_outputs\`

## Files deliberately excluded

Not committed to MyGPR source:

- `.out/.h5/.vti/.vtk/.vtu`
- generated `.csv/.npy/.png`
- MyGPR-Evidence git changes

## Tests run

- `python -m pytest tests\test_gprmax_campaign_pairing.py tests\test_gprmax_campaign_preview.py -q`
- `python -m pytest tests -q -k "gprmax"`
- `python scripts\preflight_check.py`
- `git diff --check`

## Claim boundary

- This fixes conversion/pairing/preview tooling robustness.
- GX-008 run remains synthetic scene_001 workflow only.
- Not a MyGPR-Evidence commit task.
- Not AutoTune evaluation.
- Not field validation.
- Not paper-candidate benchmark claim.

## Next task

- `GX-008-EVIDENCE-001` (curate successful GX-008 scene_001 paired artifacts into Evidence with claim-boundary metadata), or run `GX-008-RUN-002` for scene_002 using the same generic pairing entry.

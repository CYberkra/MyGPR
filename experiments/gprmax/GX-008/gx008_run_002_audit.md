# GX-008-RUN-002 Audit

## Date
- 2026-05-23

## Branch
- `main`

## Base Commit
- `817bf55f1e9cb3ed91da96693964b4492aa75d5a`

## Remote Verification
- `git rev-parse HEAD`: `817bf55f1e9cb3ed91da96693964b4492aa75d5a`
- `git rev-parse origin/main`: `817bf55f1e9cb3ed91da96693964b4492aa75d5a`
- `git ls-remote origin main`: `817bf55f1e9cb3ed91da96693964b4492aa75d5a`

## Scene Run
- Scene: `scene_002_flat_damp_sand_pec_shallow`
- Campaign: `experiments/gprmax/GX-008/campaign_draft.yaml`
- Variant runs executed:
  - `raw_with_target`
  - `background_only`

## Commands Run
- Dry-run:
  - `python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run`
- GPU wrapper check:
  - `scripts\run_gprmax_gpu_env.bat --check`
  - `scripts\run_gprmax_gpu_env.bat --smoke`
- Raw run:
  - `scripts\run_gprmax_gpu_env.bat -- python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --run-scene scene_002_flat_damp_sand_pec_shallow --variant raw_with_target --num-runs 41 --gpu-device 0 --gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe --timeout-seconds 1800`
- Background run:
  - `scripts\run_gprmax_gpu_env.bat -- python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --run-scene scene_002_flat_damp_sand_pec_shallow --variant background_only --num-runs 41 --gpu-device 0 --gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe --timeout-seconds 1800`
- Conversion (using existing converter with explicit `.out` roots):
  - `python scripts\gprmax_campaign_convert_scene001.py --raw-out D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_002_flat_damp_sand_pec_shallow\raw_with_target.out --background-out D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_002_flat_damp_sand_pec_shallow\background_only.out --raw-converted-dir D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\raw_with_target\converted --background-converted-dir D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\background_only\converted --raw-run-count 41 --background-run-count 41 --json D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\convert_summary_41.json`
- Pairing:
  - `python scripts\gprmax_campaign_runner.py --pair-outputs --campaign-id GX-008_paper_inspired_mini_benchmark_draft --scene-id scene_002_flat_damp_sand_pec_shallow --raw-output D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\raw_with_target\converted\raw_bscan.csv --background-output D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\background_only\converted\background_bscan.csv --output-dir D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\paired_outputs --source-format csv --json D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\pair_outputs_direct_41.json`
- Preview/report:
  - `python scripts\gprmax_campaign_runner.py --preview-pair --campaign-id GX-008_paper_inspired_mini_benchmark_draft --scene-id scene_002_flat_damp_sand_pec_shallow --raw-output D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\raw_with_target\converted\raw_bscan.csv --background-output D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\background_only\converted\background_bscan.csv --target-response D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\paired_outputs\target_response.npy --output-dir D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\paired_outputs --source-format auto --json D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\pair_preview_41.json`

## Dry-Run Result
- `campaign_status: ready`
- `total_scenes: 2`
- `ready_count: 2`
- `invalid_count: 0`
- `scene_002_flat_damp_sand_pec_shallow: ready`

## GPU Wrapper Check Result
- `--check`: success
- `--smoke`: success (with non-fatal Unicode decode noise in diagnostic subprocess output)

## GPU Smoke Result
- gprMax runtime GPU smoke: success
- Runtime Python:
  - `E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe`

## Raw Run Status
- `status: success`
- `return_code: 0`
- `runtime_seconds: 252.906`
- `requested_num_runs: 41`
- `gpu_flag_emitted: true`
- Manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\raw_with_target\run_manifest.json`

## Background Run Status
- `status: success`
- `return_code: 0`
- `runtime_seconds: 254.298`
- `requested_num_runs: 41`
- `gpu_flag_emitted: true`
- Manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\background_only\run_manifest.json`

## Requested Num-Runs
- `41`

## Actual Output Counts
- Raw `.out` count: `41`
- Background `.out` count: `41`
- Native outputs are generated in:
  - `D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_002_flat_damp_sand_pec_shallow\`

## Runtime Summary
- Raw runtime: ~252.9 s
- Background runtime: ~254.3 s

## Run Manifest Paths
- `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\raw_with_target\run_manifest.json`
- `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\background_only\run_manifest.json`

## Stdout/Stderr Paths
- Raw stdout:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\raw_with_target\stdout.log`
- Raw stderr:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\raw_with_target\stderr.log`
- Background stdout:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\background_only\stdout.log`
- Background stderr:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\background_only\stderr.log`

## Conversion Status
- `status: success`
- Summary:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\convert_summary_41.json`
- Shapes:
  - raw: `[936, 41]`
  - background: `[936, 41]`

## Pairing Status
- `status: success`
- Summary:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\pair_outputs_direct_41.json`
- Outputs:
  - `target_response.csv`
  - `target_response.npy`
  - `paired_validation_summary.json`
  - `paired_metrics.json`
- Shape:
  - target_response: `[936, 41]`

## Metrics Status
- `paired_metrics.json` generated
- Example key values:
  - `raw_energy: 10404565.8622`
  - `background_energy: 10404513.5348`
  - `target_response_energy: 52.3670`
  - `target_to_background_energy_ratio: 5.033e-06`

## Preview Status
- `status: success`
- Summary:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\pair_preview_41.json`
- Generated:
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - `paired_target_response_report.md`
  - `paired_report_summary.json`

## Generated Local Artifact Paths
- `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_002_flat_damp_sand_pec_shallow\`
  - `raw_with_target\...`
  - `background_only\...`
  - `paired_outputs\...`
  - `convert_summary_41.json`
  - `pair_outputs_direct_41.json`
  - `pair_preview_41.json`

## Files Deliberately Excluded
- Not committed to MyGPR source repo:
  - `.out`, `.h5`, `.vti`, `.vtk`, `.vtu`
  - generated `.csv`, `.npy`, `.png`
  - all local artifacts under `MyGPR-Evidence`

## Repository Hygiene
- This task commits only the source audit document.
- No MyGPR-Evidence git operations were performed.

## Claim Boundary
- GX-008 `scene_002` only.
- Synthetic paired run and diagnostic output only.
- Not an Evidence artifact commit yet.
- Not AutoTune evaluation.
- Not field validation.
- Not paper-candidate benchmark.

## Recommended Next Task
- `GX-008-EVIDENCE-002`: curated Evidence packaging for scene_002 outputs and manifest/index updates in MyGPR-Evidence.

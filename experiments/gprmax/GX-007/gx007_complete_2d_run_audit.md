#!/usr/bin/env markdown
# GX-007 Scene001 Complete 2D Run Audit

## 1. Metadata

- date: 2026-05-23
- branch: `main`
- base commit: `852e96121d16729529ff3c4bca0a775c1d50fd17`
- source commit: pending in this task until commit
- task: `GX-007-COMPLETE-2D-RUN-001`

## 2. Commands Run

Pre-check:

```text
git checkout main
git pull --ff-only origin main
git status --short
git log -1 --format=%H
python scripts/preflight_check.py
```

Environment detection:

```text
nvcc --version
python -c "import pycuda; print('pycuda available')"
```

Campaign dry-run:

```text
python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-007/campaign.yaml --dry-run
```

Runner execution (actual run used temporary campaign file with absolute scene paths and executable override):

```text
python scripts/gprmax_campaign_runner.py --campaign C:/Users/17844/AppData/Local/Temp/gx007_campaign_local_gpu_abs.yaml --run-scene scene_001_single_shallow_pipe --variant raw_with_target --num-runs 21 --gpu-device 0 --timeout-seconds 1200
python scripts/gprmax_campaign_runner.py --campaign C:/Users/17844/AppData/Local/Temp/gx007_campaign_local_gpu_abs.yaml --run-scene scene_001_single_shallow_pipe --variant raw_with_target --num-runs 21 --timeout-seconds 1200
python scripts/gprmax_campaign_runner.py --campaign C:/Users/17844/AppData/Local/Temp/gx007_campaign_local_gpu_abs.yaml --run-scene scene_001_single_shallow_pipe --variant background_only --num-runs 21 --timeout-seconds 1200
```

Conversion / pairing / preview:

```text
python scripts/gprmax_campaign_convert_scene001.py --raw-out experiments/gprmax/GX-007/models/scene_001_single_shallow_pipe/raw_with_target.out --background-out experiments/gprmax/GX-007/models/scene_001_single_shallow_pipe/background_only.out --raw-run-count 21 --background-run-count 21 --json D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/convert_summary_21.json
python scripts/gprmax_campaign_runner.py --pair-outputs --campaign-id GX-007_paired_background_benchmark --scene-id scene_001_single_shallow_pipe --raw-output D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/raw_with_target/converted/raw_bscan.npy --background-output D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/background_only/converted/background_bscan.npy --output-dir D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs --source-format auto --json D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs/pair_result_21.json
python scripts/gprmax_campaign_runner.py --preview-pair --campaign-id GX-007_paired_background_benchmark --scene-id scene_001_single_shallow_pipe --raw-output D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/raw_with_target/converted/raw_bscan.npy --background-output D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/background_only/converted/background_bscan.npy --target-response D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs/target_response.npy --output-dir D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs --source-format auto --json D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/paired_outputs/preview_result_21.json
```

## 3. GPU Environment and Usage

- `nvcc --version`: available (CUDA 11.8 shown)
- `pycuda` import: available
- first GPU attempt (`--gpu-device 0`, `num-runs=21`) result:
  - status: failed
  - return code: `3221226505`
  - stderr indicated `pycuda.driver.CompileError` and CUDA context cleanup abort
- action taken: downgraded to CPU path for stability

Conclusion:

- GPU environment exists, but this run did not complete reliably on GPU.
- final complete 2D artifact was generated with CPU run path.

## 4. Run Status and Trace Counts

Final successful manifests:

- raw manifest:
  - path: `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/raw_with_target/run_manifest.json`
  - status: `success`
  - requested_num_runs: `21`
  - gpu_requested: `false`
- background manifest:
  - path: `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/background_only/run_manifest.json`
  - status: `success`
  - requested_num_runs: `21`
  - gpu_requested: `false`

Actual trace count used for conversion:

- raw traces merged from `raw_with_target1.out` to `raw_with_target21.out`
- background traces merged from `background_only1.out` to `background_only21.out`
- actual raw trace count: `21`
- actual background trace count: `21`
- consistency: matched

## 5. Shapes

- raw shape: `936 x 21`
- background shape: `936 x 21`
- target_response shape: `936 x 21`
- paired validation status: `ready`
- preview status: `success`

This run is classified as:

- complete 2D diagnostic (small-scale, 21 traces)
- not a partial diagnostic

## 6. Output Paths

- conversion summary:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007/scene_001_single_shallow_pipe/convert_summary_21.json`
- converted arrays:
  - `.../raw_with_target/converted/raw_bscan.npy`
  - `.../raw_with_target/converted/raw_bscan.csv`
  - `.../background_only/converted/background_bscan.npy`
  - `.../background_only/converted/background_bscan.csv`
- paired outputs:
  - `.../paired_outputs/target_response.npy`
  - `.../paired_outputs/target_response.csv`
  - `.../paired_outputs/paired_validation_summary.json`
  - `.../paired_outputs/paired_metrics.json`
- preview/report:
  - `.../paired_outputs/raw_preview.png`
  - `.../paired_outputs/background_preview.png`
  - `.../paired_outputs/target_response_preview.png`
  - `.../paired_outputs/paired_preview_panel.png`
  - `.../paired_outputs/paired_target_response_report.md`
  - `.../paired_outputs/paired_report_summary.json`

## 7. Metrics Summary

From `paired_metrics.json`:

- raw_energy: `5370261.43814295`
- background_energy: `5369559.78566127`
- target_response_energy: `701.656608797365`
- target_to_background_energy_ratio: `0.000130673022893059`
- abs_difference_mean: recorded in metrics file
- abs_difference_max: recorded in metrics file

## 8. Repository Hygiene

- no `.out/.h5/.vti` generated files committed to MyGPR
- no generated `.csv/.npy/.png` committed to MyGPR
- MyGPR-Evidence repo was used as external output location only; not modified by git in this task
- no UI / AutoTune / motion compensation algorithm changes

## 9. Claim Boundary

- this is a synthetic small-scale scene_001 execution and conversion audit
- this is not field validation
- this is not AutoTune evaluation
- this is not a paper-candidate benchmark
- this does not support AutoTune superiority claims

## 10. Recommended Evidence Task

- recommended next task: `GX-007-EVIDENCE-002`
  - selectively package the successful `936x21` scene_001 artifact set into a new Evidence artifact path
  - keep claim boundary as small-scale synthetic diagnostic

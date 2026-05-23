#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GX-007 GPU Runbook

## Why VS x64 / cl.exe Is Required

gprMax GPU execution on Windows relies on CUDA/PyCUDA runtime compilation paths that require MSVC tools (`cl.exe`) to be available in the active shell environment. If `cl.exe` is missing, GPU compile usually fails even when `nvcc` and `nvidia-smi` are present.

## Host Python vs gprMax Runtime Python

- MyGPR host Python may not have `pycuda` or `gprMax`; this is acceptable.
- Actual GPU run should use external runtime python with:
  - `--gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe`
- GPU readiness is determined by runtime smoke result, not host import result alone.

## Standard GPU Entry

1. Environment check:
   - `scripts\run_gprmax_gpu_env.bat --check`
2. Diagnostic smoke:
   - `scripts\run_gprmax_gpu_env.bat --smoke`
3. Run command in wrapped environment:
   - `scripts\run_gprmax_gpu_env.bat -- python scripts\gprmax_campaign_runner.py ...`

## Runner Example

```bat
python scripts\gprmax_campaign_runner.py ^
  --campaign experiments/gprmax/GX-007/campaign.yaml ^
  --run-scene scene_001_single_shallow_pipe ^
  --variant raw_with_target ^
  --num-runs 21 ^
  --gpu-device 0 ^
  --gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe ^
  --timeout-seconds 1200
```

## How to Judge GPU Ready

Use `python scripts\diagnose_gprmax_gpu_env.py --gprmax-python <path> --gpu-device 0`.

Key readiness fields:
- `gprmax_help_ok`
- `minimal_smoke_ok`
- `gprmax_runtime_gpu_ready`
- `readiness_reason`

If host `pycuda` is missing but runtime smoke succeeds, readiness can still be `true`.

## GX-007 Current Results (Stabilization Phase)

- direct gprMax raw `-n 1`: run in this task, see diagnostic section below.
- runner raw `--num-runs 1`: run in this task, see diagnostic section below.
- runner raw `--num-runs 21`: run in this task, see diagnostic section below.
- background `--num-runs 21`: only run after raw 21 success.

## Failure Triage

On failure inspect:
- `run_manifest.json`
- `stderr.log`
- `stdout.log`

Look for:
- startup failure (`WinError 2`)
- `pycuda.driver.CompileError`
- `nvcc fatal`
- timeout/partial status

## Claim Boundary

GPU is an optional acceleration path only. GPU readiness and runtime stability do not change scientific conclusions, claim boundaries, or evidence interpretation.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GX-007 GPU Failure Diagnostic

## 1. Observed Failure

- Task: `GX-RUN-GPU-DIAG-001`
- Branch: `main`
- Base commit: `e76bcf95fe48eeefee3705f21ceb2b245aad3d6f`
- Symptom:
  - `nvcc --version` available
  - `pycuda` importable
  - GX-007 scene_001 GPU run (`--gpu-device 0 --num-runs 21`) failed
  - failure class: `pycuda.driver.CompileError`
  - return code observed: `3221226505`
- CPU fallback remained usable for GX-007 complete 2D diagnostic path (`[936, 21]`) and is unchanged by this task.

## 2. Environment Summary

From `python scripts/diagnose_gprmax_gpu_env.py --json output/gpu_diag_default.json`:

- OS/platform: `Windows-11-10.0.26200-SP0`
- Python executable: `D:\Miniconda3\python.exe`
- Python version: `3.13.12` (Conda base)
- Conda env: `base`
- `gprMax` import (current Python): **failed** (`ModuleNotFoundError`)
- `pycuda` import (current Python): **ok**
- `nvcc`: **available**, CUDA compilation tools `11.8`
- `nvidia-smi`: **available**
  - GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
  - Driver: `591.86`
  - CUDA runtime shown by driver: `13.1`
- `cl.exe`: **not found in current shell PATH**
- VS/MSVC key env vars: empty in this shell

## 3. Minimal PyCUDA Compile Result

Script internal minimal kernel compile/run:

- `minimal_pycuda_compile_ok`: **false**
- compile failure core message:
  - `nvcc fatal   : Cannot find compiler 'cl.exe' in PATH`
- captured as `pycuda.driver.CompileError`.

Interpretation: the CUDA/PyCUDA toolchain in the active shell is incomplete for runtime CUDA code compilation because MSVC compiler toolchain is not injected into PATH/env.

## 4. Minimal gprMax GPU Smoke Result

Used minimal fixture:

- `experiments/gprmax/smoke/minimal_gpu_smoke.in`

Smoke command used by diagnostic script:

- `E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe -m gprMax <tmp>/minimal_gpu_smoke.in -gpu 0`

Observed:

- gprMax starts, detects GPU, builds model pre-solve steps.
- GPU solve fails at PyCUDA compile/preprocess stage with `pycuda.driver.CompileError`.
- return code: `3221226505`.

This reproduces GPU failure on a minimal model, so failure is not specific to GX-007 scene complexity.

## 5. GX-007 GPU Result

From existing `experiments/gprmax/GX-007/gx007_complete_2d_run_audit.md`:

- first GX-007 GPU attempt (`--gpu-device 0 --num-runs 21`) failed
- return code: `3221226505`
- error family: `pycuda.driver.CompileError`
- CPU retry succeeded and produced complete small 2D diagnostic artifact.

## 6. Root-Cause Classification

Current classification: **GPU compiler-chain environment issue (high confidence)**.

Evidence:

1. Minimal PyCUDA compile fails before model-specific logic.
2. Minimal gprMax GPU smoke fails with same compile class.
3. Failure message directly points to missing `cl.exe` in PATH.

Secondary possibility (after fixing MSVC shell): gprMax + PyCUDA + CUDA version compatibility issue may still exist, but current blocker is earlier and deterministic.

## 6.1 Runner Startup Command Root Cause (latest check)

After confirming GPU smoke success in the gprMax venv, GX-007 runner failure was re-checked from run manifest:

- failing command (old behavior):
  - `["gprMax", "<scene_001 raw_with_target.in>", "-n", "21", "-gpu", "0"]`
- failure:
  - `status: failed`
  - `return_code: null`
  - `runtime_seconds: ~0.014`
  - `error_message: [WinError 2] 系统找不到指定的文件。`

This is a startup command resolution failure (`gprMax` not found in PATH for current shell), not a model-science failure.

Fix path added in runner:

- explicit runtime parameter: `--gprmax-python <path-to-venv-python>`
- command mode:
  - `<python.exe> -m gprMax <model.in> ...`
- intended usage:
  - `--gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe`

This decouples MyGPR host Python environment from gprMax runtime environment.

## 7. Attempted Fixes

- Added diagnostics script `scripts/diagnose_gprmax_gpu_env.py`.
- Added `--clear-pycuda-cache` option and executed it.
- Re-ran diagnostics with cache clear.
- Result: failure unchanged; root error still `Cannot find compiler 'cl.exe'`.

No system PATH mutation, CUDA reinstall, or package reinstall was performed automatically.

## 8. Manual Remediation Steps

Recommended conservative sequence:

1. Open a shell with MSVC toolchain loaded:
   - `x64 Native Tools Command Prompt for VS 2022` (or matching Build Tools prompt).
2. Verify compiler availability:
   - `where cl`
   - `cl`
3. Verify CUDA compiler in same shell:
   - `where nvcc`
   - `nvcc --version`
4. Activate gprMax Python environment used for GPU run.
5. Run diagnostics again:
   - `python scripts/diagnose_gprmax_gpu_env.py --clear-pycuda-cache`
6. Confirm:
   - `minimal_pycuda_compile_ok == true`
7. Re-run minimal gprMax GPU smoke.
8. Only after smoke succeeds, retry GX-007 GPU run.

If minimal PyCUDA still fails after MSVC shell fix, next manual step is compatibility alignment among:

- gprMax version
- pycuda version
- CUDA toolkit version
- active Python ABI

Do this as explicit environment maintenance, not in MyGPR source code.

## 9. GPU Readiness for GX-007

- Current status: **pending re-validation with `--gprmax-python`**.
- CPU path remains available and already produced complete small 2D diagnostic outputs.

## 10. Claim Boundary

- This document is an execution-environment diagnostic only.
- No new scientific benchmark claim is made.
- No field validation claim is made.
- No AutoTune superiority claim is made.
- No Evidence repository operation was performed in this task.

## 11. Next Recommended Action

- Run GPU workload from a verified MSVC-enabled shell and re-run:
  - `scripts/diagnose_gprmax_gpu_env.py`
  - minimal gprMax GPU smoke
  - GX-007 scene_001 GPU run
- If minimal compile passes but GX-007 still fails, open `GX-RUN-GPU-DIAG-002` focused on gprMax/PyCUDA/CUDA version compatibility and model-scale GPU compile path.

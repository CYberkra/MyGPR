#!/usr/bin/env markdown
# GX-007 GPU Run Notes

## Scope

This note documents optional GPU passthrough support for MyGPR `gprmax_campaign_runner`.

## Optional GPU Path

- GPU usage is optional and disabled by default.
- Supported GPU path targets NVIDIA CUDA devices through gprMax `-gpu`.
- Supported CLI forms:
  - `--gpu` -> emits `-gpu`
  - `--gpu-device 0` -> emits `-gpu 0`
  - `--gpu-devices 0 1 2 3` -> emits `-gpu 0 1 2 3`
- `--num-runs N` remains compatible with GPU flags.

## Environment Expectations

- CUDA Toolkit should be installed and `nvcc` should be available on PATH.
- Python module `pycuda` should be importable in the execution environment.
- Runner checks are lightweight (`nvcc --version`, import probe for `pycuda`).
- Missing checks never fail dry-run mode; they only produce warnings when GPU is requested.

## Manifest Contract

`run_manifest.json` records:

- `gpu_requested`
- `gpu_flag_emitted`
- `gpu_device_ids`
- `nvcc_available`
- `pycuda_available`
- `gpu_warning`
- `requested_num_runs`
- `command`
- `run_status`

## Out of Scope

- No GUI changes.
- No automatic MPI/GPU scheduling policy.
- No claim-boundary change for synthetic/field evidence interpretation.
- No automatic requirement that all machines must provide CUDA.

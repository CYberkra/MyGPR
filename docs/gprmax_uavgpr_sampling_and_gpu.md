# gprMax UAV-GPR Sampling and GPU Notes

## Sampling Decision

MyGPR field UAV-GPR SFCW data currently appears mainly as 501 or 701 samples per trace.
The gprMax airborne validation report should therefore feed MyGPR with the same A-scan
length when testing the processing chain and auto-tune behavior.

Do not simply set `#time_window: 501` as the default model command. The gprMax manual
allows an integer `#time_window` to mean FDTD iteration count, but that would shorten
the physical simulated time for the current 2 mm grid and can truncate the target return.

The current report strategy is:

- keep gprMax physical modelling at the configured time window;
- preserve the raw `.out` files and raw FDTD sample count;
- resample the BScan rows to 501 by default before running the MyGPR processing chain;
- allow `--ascan-samples 701` for the other current field-data length;
- allow `--ascan-samples 0` when raw gprMax time-step samples are needed.

This makes the processing and auto-tune validation face the same matrix height as real
field data without sacrificing the wavefield travel time needed by the synthetic model.

Official reference: https://docs.gprmax.com/en/latest/input.html

## GPU Execution

Official gprMax GPU usage requires:

- NVIDIA CUDA-enabled GPU;
- NVIDIA CUDA Toolkit installed;
- `nvcc` on `PATH`;
- `pycuda` installed in the gprMax Python environment.

Single-GPU command pattern:

```powershell
python -m gprMax user_models/cylinder_Ascan_2D.in -gpu
python -m gprMax user_models/cylinder_Ascan_2D.in -gpu 0
```

Multiple B-scan traces can combine MPI with GPU task farming. The gprMax manual example
uses one CPU master plus one worker per GPU:

```powershell
python -m gprMax user_models/cylinder_Bscan_2D.in -n 60 -mpi 5 -gpu 0 1 2 3
```

For MyGPR report runs, the equivalent interface is:

```powershell
python scripts\gprmax_benchmark\gprmax_multi_scenario_report.py `
  --gprmax-root E:\gprMax\gprMax-v.3.1.7 `
  --scenario airborne_single_cylinder_v1 `
  --runs 48 `
  --ascan-samples 501 `
  --geometry-fixed `
  --gpu 0
```

If GPU compilation fails even when `pycuda` and `nvcc` are present, keep CPU as the
trusted baseline and record the gprMax stderr path. Treat this as an environment/toolchain
issue until a minimal official test model succeeds with `-gpu`.

Current workstation check on 2026-05-11:

- `pycuda`: available in `E:\gprMax\gprMax-v.3.1.7\.venv`
- `nvcc`: found at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe`
- `mpi4py`: not available, so MPI task farm is not ready
- root cause found: ordinary PowerShell/Codex sessions did not load Visual Studio's
  C++ build environment, so `nvcc` could not find `cl.exe`
- `vcvars64.bat` is available at `E:\sisual stdio 2022\VC\Auxiliary\Build\vcvars64.bat`
- the report script now auto-loads `vcvars64.bat` for Windows `--gpu` runs when `cl.exe`
  is not already on `PATH`
- `--gpu 0` has been verified on the MyGPR airborne smoke after this environment fix

If auto-detection ever fails, pass the Visual Studio environment script explicitly:

```powershell
python scripts\gprmax_benchmark\gprmax_multi_scenario_report.py `
  --gprmax-root E:\gprMax\gprMax-v.3.1.7 `
  --scenario airborne_no_target_background_v1 `
  --runs 1 `
  --ascan-samples 501 `
  --geometry-fixed `
  --gpu 0 `
  --cuda-vcvars "E:\sisual stdio 2022\VC\Auxiliary\Build\vcvars64.bat"
```

Official references:

- https://docs.gprmax.com/en/latest/gpu.html
- https://docs.gprmax.com/en/latest/openmp_mpi.html

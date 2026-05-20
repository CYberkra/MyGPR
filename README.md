# MyGPR

MyGPR is a PyQt6 desktop application for UAV/GPR B-scan data inspection,
processing, auto-tuning, sidecar metadata integration, and deterministic
benchmark evidence export.

## Runtime

- Python 3.10+
- PyQt6 and PyQt6-Fluent-Widgets
- NumPy, Pandas, SciPy, Matplotlib, h5py, PyWavelets

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Main Entry Points

Run the GUI:

```bash
python app_qt.py
```

Windows shortcut:

```bash
启动GPR.bat
```

Run CLI batch validation and processing:

```bash
python cli_batch.py validate --config config/cli_batch_mvp_example.json
python cli_batch.py run --config config/cli_batch_mvp_example.json
```

Run deterministic motion-compensation benchmark evidence export:

```bash
python cli_batch.py validate --config config/motion_compensation_v1_benchmark.json
python cli_batch.py run --config config/motion_compensation_v1_benchmark.json
```

`cli_batch.py resume` is intentionally not implemented yet and exits non-zero.

## Continuous Integration

The `MyGPR Lightweight CI` GitHub Actions workflow runs on pushes and pull
requests for `codex/research-gprmax-autotune`. It installs the development
dependencies, runs `python scripts/preflight_check.py`, and runs the lightweight
gprMax/AutoTune pytest subset.

CI is a code smoke gate only. It does not run heavy gprMax simulations, does not
require local native `.out` files, and does not validate paper or Evidence
claims.

## Repository Map

- `app_qt.py` - main PyQt6 GUI entry point.
- `cli_batch.py` - batch processing and benchmark CLI.
- `core/` - shared runtime logic, I/O, method registry, processing engine,
  presets, sidecar integration, metrics, and evidence export.
- `ui/` - Qt pages, dialogs, workbench widgets, parameter editors, and logs.
- `PythonModule/` - ndarray algorithms plus legacy CSV wrapper compatibility.
- `tests/` - pytest unit and integration coverage.
- `scripts/preflight_check.py` - syntax plus GUI/runtime smoke gate.
- `config/` - runnable CLI and benchmark configs.
- `sample_data/` - bundled sidecar and benchmark-compatible sample data.
- `output/` - generated artifacts, ignored by git.

## Sample Data

Current bundled examples:

- `sample_data/gui_sidecar_all_data_main.csv` - airborne CSV with explicit
  trace timestamps.
- `sample_data/gui_sidecar_all_data_rtk.csv` - RTK sidecar.
- `sample_data/gui_sidecar_all_data_imu.csv` - IMU sidecar.
- `sample_data/gui_sidecar_all_data_README.md` - GUI sidecar verification notes.
- `sample_data/motion_compensation_v1/README.md` - deterministic motion
  benchmark semantics and expected artifacts.

## Processing Surface

The method registry and processing engine are the coordination points:

- `core/methods_registry.py` defines public methods, metadata, category labels,
  parameter schemas, and auto-tune stages.
- `core/processing_engine.py` runs ndarray methods and preserves runtime
  metadata/warnings.
- `core/preset_profiles.py` defines GUI presets and recommended CLI profiles.

Motion-compensation V1 currently uses this deterministic sequence:

```text
trajectory_smoothing
motion_compensation_speed
motion_compensation_attitude
motion_compensation_height
motion_compensation_vibration
```

## UAV-GPR Research Specs

- `docs/uav_gpr_standard_processing_flow.md` - recommended UAV measured-data
  processing flow and literature-backed ordering.
- `docs/auto_tune_research_comparison_design.md` - manual baseline vs
  auto-tuned comparison page contract for research evidence.
- `docs/motion_compensation_v2_design.md` - RTK/IMU/altimeter motion
  compensation rebuild plan.

## Verification

Focused smoke:

```bash
python scripts/preflight_check.py
```

Full test suite:

```bash
python -m pytest -q
```

Useful targeted checks:

```bash
python -m pytest tests/test_cli_batch_profiles.py -q
python -m pytest tests/test_runtime_warnings.py -q
python -m pytest tests/test_motion_compensation_pipeline_e2e.py -q
```

Fast syntax check for edited Python files:

```bash
python -m py_compile app_qt.py cli_batch.py core/processing_engine.py
```

## Packaging

```bash
build_exe.bat
pyinstaller gpr_gui.spec --clean --noconfirm
```

Run `python scripts/preflight_check.py` before packaging.

## Archiving Stable Checkpoints

For stable, user-meaningful checkpoints, prefer a descriptive commit and,
when the conclusion should survive future sessions, archive it with:

```bash
python scripts/archive_checkpoint.py --summary "checkpoint summary"
```

The default vault target is configured in `scripts/archive_checkpoint.py`.

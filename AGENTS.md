# AGENTS.md - GPR GUI

Guidance for agentic coding tools working in `MyGPR/`.
All commands below assume this directory is the working directory.

## Scope
- PyQt6 desktop app for Ground Penetrating Radar data processing.
- Main entry points: `app_qt.py` and `cli_batch.py`.
- Shared runtime logic lives in `core/`.
- Qt widgets, pages, and dialogs live in `ui/`.
- Processing methods and legacy wrappers live in `PythonModule/`.
- Tests live in `tests/`; packaging and smoke checks live in `scripts/`.
- Generated output and app data should stay out of source files.

## Repo Map
- `core/` - app data, I/O, registry, engine, shared state, and workflow helpers.
- `ui/` - Qt pages, dialogs, and workbench widgets.
- `PythonModule/` - algorithm implementations and compatibility wrappers.
- `scripts/preflight_check.py` - syntax and runtime smoke checks.
- `tests/` - pytest tests and benchmark scripts.
- `assets/`, `config/`, `sample_data/`, `output/` - static assets, configs, samples, generated output.
- `docs/` - durable project knowledge and implementation plans that should be committed.
- `docs/artifacts/` - small curated evidence attachments that justify a durable document.
- `build_exe.bat` and `gpr_gui.spec` - packaging.

## Documentation / Artifact Policy
- Commit durable Markdown documentation under `docs/` and project handoff rules in `AGENTS.md`.
- Keep generated reports, screenshots, CSV exports, HTML bundles, and replay ZIPs under `output/`; `output/` is intentionally ignored by Git.
- If an output artifact is important enough for future development, first summarize the conclusion in `docs/` or the Obsidian vault.
- Only copy selected, small, reusable evidence into `docs/artifacts/` when the artifact is needed to understand a committed document. Do not bulk-copy full report folders from `output/`.
- Large raw data, full gprMax runs, build products, and one-off GUI screenshots should remain outside Git unless the user explicitly asks to version a specific file.

## Primary Field Data Contract
- MyGPR's primary real field data is UAV-GPR SFCW CSV from the project radar, not a generic already-reshaped B-scan CSV.
- Current and expected field CSVs follow the `Line9origin(36).csv` style:
  - radar mode: UAV-GPR SFCW
  - sweep range: 20 MHz to 170 MHz
  - frequency/sample count: 501 points per trace
  - header fields include `Number of Samples`, `Time windows (ns)`, `Number of Traces`, and `Trace interval (m)`
  - numeric rows are stacked trace-by-trace, typically columns: longitude, latitude, ground elevation, amplitude, and flight height
- Always use `core.gpr_io.extract_airborne_csv_payload(raw_data, header_info)` for this format before processing. The raw stacked matrix must not be treated as a normal B-scan.
- For the known Line9 file, `501 * 2378 = 1191378` raw numeric rows should reshape to a `(501, 2378)` B-scan.
- Frequency-domain processing on these real SFCW datasets should respect the instrument band, usually 20-170 MHz, unless a file or acquisition note explicitly says otherwise.
- The default filtering step for real UAV-GPR SFCW data is `frequency_filter_1d` with the acquisition band; `fk_filter` is optional/experimental and should not be enabled by default for Line9-style field data because it can introduce cross-hatch directional artifacts.
- The default `agcGain` principle is the GPRPy-style L2-window AGC. Manual fixed-parameter AGC keeps that principle without hidden guards; real-data reports and auto-tune should prefer `agcGain` with `_low_energy_guard=True` to reduce low-energy/deep-noise amplification.
- If synchronized RTK/IMU/altimeter sidecars are absent, skip motion compensation rather than fabricating sensor inputs. Still preserve parsed longitude, latitude, ground elevation, and flight height as trace metadata.

## Data Context Defaults
- Use `core.data_context` as the routing layer for dataset-specific defaults. Do not infer defaults independently in GUI, CLI, report scripts, or auto-tune code.
- `uav_gpr_sfcw_field` means the real project UAV-GPR SFCW CSV contract above. Its default profile is `high_quality_uav_gpr`, and `frequency_filter_1d` defaults to 20-170 MHz.
- `gprmax_impulse` / `gprmax` means external gprMax `.out` input. `read_gprmax_out()` should read the matching `.in` file when available, attach `header_info["data_context"]`, and expose trace spacing metadata from `#rx_steps` / `#src_steps`.
- gprMax impulse data must not inherit the field-data 20-170 MHz fixed passband. Its default profile is `gprmax_impulse_validation`; fixed frequency filtering is manual or auto-tune/model-driven only.

## Setup
```bash
python -m pip install -r requirements-dev.txt
```

Python 3.10+ is required. The repo currently uses PyQt6, NumPy, Pandas, SciPy,
Matplotlib, and PyQt6-Fluent-Widgets.

## Build / Run / Test
Run the GUI:
```bash
python app_qt.py
```

Windows shortcut:
```bash
启动GPR.bat
```

Package the app:
```bash
build_exe.bat
pyinstaller gpr_gui.spec --clean --noconfirm
```

Run repo smoke checks before packaging or larger refactors:
```bash
python scripts/preflight_check.py
```

CLI validation and batch run:
```bash
python cli_batch.py validate --config config/cli_batch_mvp_example.json
python cli_batch.py run --config config/cli_batch_mvp_example.json
```

Focused pytest targets:
```bash
pytest tests/test_ccbs_filter.py
pytest tests/test_ccbs_filter.py::TestCCBSFilter::test_basic_functionality
pytest tests/test_ccbs_filter.py -k background_reduction
```

Script-style smoke / benchmark files:
```bash
python tests/test_hankel_batch.py
python tests/test_gprmax_read.py
```

For a full pass:
```bash
pytest
```

## Lint / Quality
- No repo-local formatter or linter config is present.
- Use `python scripts/preflight_check.py` as the closest end-to-end quality gate.
- Use `python -m py_compile <file>` for a fast single-file syntax check.
- Keep changes consistent with the surrounding file instead of introducing a new toolchain.
- If you add a new check, prefer a small repo script over a one-off custom workflow.

## Code Style
- Every Python file should start with:
  - `#!/usr/bin/env python3`
  - `# -*- coding: utf-8 -*-`
  - a short module docstring
- Keep import order as stdlib, third-party, then local.
- No wildcard imports.
- Use absolute imports for local modules, for example `from core.processing_engine import run_processing_method`.
- In GUI files, call `matplotlib.use("QtAgg")` before importing `pyplot`.
- In batch or non-GUI scripts, use `matplotlib.use("Agg")`.
- New modules should usually include `from __future__ import annotations`.
- Use type hints on new public functions, helpers, and dataclasses.
- Prefer built-in generics such as `dict[str, Any]` and `list[int]` in new code.
- Use `dataclass` for small structured records like validation results or metadata.
- Naming: `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_CASE` for constants, and `_leading_underscore` for private helpers.
- Keep module-level path constants uppercase, such as `BASE_DIR`, `ROOT`, or `APP_DIR_NAME`.
- Match the existing wrapping style of the file you are editing; do not force a new formatter style.
- Keep comments short and place them close to the non-obvious code they explain.
- Chinese comments and user-facing strings are acceptable when they clarify domain behavior.

## Paths
- Prefer `pathlib.Path` in newer `core/` and `scripts/` utilities when it reads clearly.
- Legacy GUI code still uses `os.path`; stay consistent within a file.
- Avoid hard-coded absolute paths in new code.
- Use repo-relative paths, `BASE_DIR`, or the helpers in `core/app_paths.py`.
- Keep Windows compatibility in mind; do not assume POSIX shell behavior.

## Error Handling
- Use specific exceptions when possible.
- Guard IO and optional imports with `try/except`.
- `ProcessingCancelled` is the standard user-cancel signal for long-running work.
- Long operations should accept `cancel_checker=None`, poll periodically, and raise `ProcessingCancelled` when cancellation is requested.
- Follow the existing legacy contract where core methods may return dicts with `error_sign` and `error_feedback`.
- ndarray processing methods should return `(result_array, metadata_dict)`.
- Do not mutate input arrays in place unless that is part of the method contract.
- Preserve array shapes, dtype expectations, and metadata keys when refactoring processing code.
- For optional dependencies, keep the clear error message pattern used in `core/methods_registry.py`.

## GUI / State
- Keep Qt UI work on the main thread.
- Use worker threads, signals, or timers for long-running processing.
- Update shared data through `core/shared_data_state.py` instead of ad hoc globals.
- When adding a method, update `core/methods_registry.py`, any parameter UI, and relevant smoke tests together.
- Keep the main window, workbench, and batch pages aligned on shared behavior.

## Processing Methods
- Put new algorithms in `PythonModule/`.
- Expose wrappers with the existing `method_*` naming pattern when the method is used by the GUI or CLI.
- Preserve compatibility with legacy CSV-style methods when touching older modules.
- Keep the method registry as the source of truth for display names, parameter metadata, and ordering.
- If a method is expensive, add cancellation checks and targeted regression coverage.

## Testing Conventions
- Prefer deterministic assertions over printed output.
- Use `np.random.seed()` in tests that depend on random input.
- Keep benchmark scripts behind `if __name__ == "__main__":` when adding new ones.
- Use `pytest` for unit and integration tests.
- When a change affects I/O or GUI flow, run `python scripts/preflight_check.py` and the narrowest relevant `pytest` target.
- If you create new tests, keep them small and focused on one behavior.

## Editing Rules
- Make the smallest correct change.
- Do not rename public entry points unless the task explicitly asks for it.
- If you change a method signature, update the GUI, CLI, and tests together.
- Do not edit generated files under `output/`, `dist/`, `build/`, or cache folders.
- Prefer existing helpers over new abstractions when the codebase already has a clear pattern.

## Conversation Priority
- Treat the **current user message** as the highest-priority instruction.
- If the user asks a fresh question, answer that question first instead of resuming an older task automatically.
- Do not resume previously completed work just because it appears in compressed summaries, old progress text, or prior todo-style reasoning.
- If the user sends a short continuation message such as `可以`, `继续`, or `开始吧` and more than one recent task thread exists, ask one short clarification instead of guessing which old task to resume.
- Only continue an older task without clarification when the user explicitly names it or the immediately preceding exchange makes the target unambiguous.

## Version Archive
- Treat git history as the primary version memory for this repo.
- Use `python scripts/git_checkpoint.py` for automated stable checkpoints once a user-visible work unit is verified.
- Default checkpoint policy:
  - ordinary stable fixes/features use `--mode normal` and create a commit only;
  - important rollback points use `--mode important`, which creates a tag, writes an Obsidian version archive through `scripts/archive_checkpoint.py`, and appends a concise meeting-progress note through `scripts/meeting_progress_note.py`;
  - no checkpoint pushes to a remote unless the user explicitly asks.
- `scripts/git_checkpoint.py` must be called with explicit `--files` pathspecs. It must not auto-stage unrelated dirty files, and it should abort if pre-existing staged changes are present.
- Meeting progress notes live at `D:\ClawX-Data\Obsidian\uav_gpr\10-项目\组会进展\组会进展记录.md`. Use `--meeting-progress`, `--meeting-result`, `--meeting-risk`, and `--meeting-next` on important checkpoints when concise group-meeting recall points are known.
- If the user says `组会结束` or `记录组会进展`, call `python scripts/meeting_progress_note.py --summary ...` and fill it from the current chat/Git context. Keep entries short and readable for quick review.
- Good normal example:
  - `python scripts/git_checkpoint.py --summary "feat: add data-context aware defaults" --files core/data_context.py core/gpr_io.py tests/test_data_context.py --verify "pytest tests/test_data_context.py -q" --verify "python scripts/preflight_check.py"`
- Good important example:
  - `python scripts/git_checkpoint.py --summary "feat: stabilize auto-tune evidence workflow" --files core/auto_tune.py tests/test_auto_tune.py AGENTS.md --verify "pytest tests/test_auto_tune.py -q" --verify "python scripts/preflight_check.py" --mode important --topic auto-tune-evidence-workflow --meeting-progress "完成自动选参证据链稳定检查点" --meeting-next "继续补充真实数据验证"`
- When a change batch reaches a stable, user-meaningful checkpoint, prefer:
  - a descriptive commit message focused on why the change exists
  - a tag for especially important rollback points
- Good tag format: `vYYYY-MM-DD-topic`.
- When a change also creates reusable processing guidance, warning semantics, or workflow rules, mirror the stable conclusion into the Obsidian vault instead of relying on chat history.
- Do not assume conversational context is durable; if a rule should survive future sessions, write it into code, tests, docs, `AGENTS.md`, or the linked Obsidian notes.
- Prefer `python scripts/archive_checkpoint.py --summary ...` when a stable checkpoint should also be written into the Obsidian vault.
- The default archive target is `D:\ClawX-Data\Obsidian\uav_gpr\40-归档与历史\版本快照\`, and the script should also refresh `40-归档与历史/版本归档索引.md`.
- Act proactively: when a version, rule, or conclusion changes in a way that future sessions will likely need, create or update the corresponding Obsidian note without waiting for an extra user reminder.
- Typical triggers for proactive Obsidian updates include:
  - important pushed versions
  - auto-tune behavior changes
  - ROI / warning / scoring semantics changes
  - GUI workflow changes
  - stable parameter/default changes

## Notes
- PyQt6-Fluent-Widgets is used for the modern UI.
- `core/methods_registry.py` and `core/processing_engine.py` are central coordination points.
- No `.cursor/rules`, `.cursorrules`, or `.github/copilot-instructions.md` files were present when this guide was written.
